import sys
sys.path.insert(1, "/home/public/work/freesound_curation/audio_text_grounding")

from pathlib import Path
import argparse
import json
import pickle
import math

import torch
import librosa
import pandas as pd
from tqdm import tqdm

import utils.train_util as train_util
import utils.eval_util as eval_util
from utils.build_vocab import Vocabulary


def print_pass(*args):
    pass


class InferDataset:

    def __init__(self, fin, fsid_to_fpath, sample_rate):
        self.fsid_to_fpath = fsid_to_fpath
        self.sample_rate = sample_rate
        self.fsids = []
        with open(fin, "r") as reader:
            for line in reader.readlines():
                fsid = line.strip()
                if "freesound_" + fsid not in self.fsid_to_fpath and \
                    fsid not in self.fsid_to_fpath:
                    print(fsid + " not found")
                    continue
                self.fsids.append(line.strip())

    def __getitem__(self, index):
        fsid = self.fsids[index]
        try:
            fpath = self.fsid_to_fpath["freesound_" + fsid]
        except KeyError:
            fpath = self.fsid_to_fpath[fsid]
        try:
            waveform, _ = librosa.core.load(fpath, sr=self.sample_rate)
        except Exception as e:
            print(fsid + " loading error: " + e.__str__())
            waveform = np.array([-1])
        duration = waveform.shape[0] / self.sample_rate
        return waveform, duration, fsid

    def __len__(self):
        return len(self.fsids)


def main(args):
    experiment_path = args.experiment_path
    text = args.text
    fin = args.fin
    fout = args.fout
    threshold = args.threshold
    sample_rate = args.sample_rate
    connect_dur = args.connect_duration
    fade_in_out = args.fade_in_out
    max_single_occurrence_duration = args.max_single_occurrence_duration
    min_single_occurrence_duration = args.min_single_occurrence_duration
    min_duration_after_padding = args.min_duration_after_padding


    meta_info = json.load(open(args.fmeta))
    meta_info.update({
        "single_occurrence_spliter": {
            "query": text,
            "threshold": threshold,
            "connect_duration": connect_dur,
            "min_single_occurrence_duration": min_single_occurrence_duration,
            "max_single_occurrence_duration": max_single_occurrence_duration,
            "min_duration_after_padding": min_duration_after_padding
        }
    })
    json.dump(meta_info, open(args.fmeta, "w"), indent=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    exp_dir = Path(experiment_path)
    config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml")

    model = train_util.init_obj_from_str(config["model"])
    ckpt = torch.load(exp_dir / "best.pth", "cpu")
    train_util.load_pretrained_model(model, ckpt, output_fn=print_pass)
    model = model.to(device)

    model.eval()
    
    vocab_path = config["data"]["train"]["collate_fn"]["tokenizer"]["args"]["vocabulary"]
    vocabulary = Vocabulary()
    vocabulary.load_state_dict(pickle.load(open(vocab_path, "rb")))

    tokens = [vocabulary(token) for token in text.split()]

    input_dict = {
        "text": torch.as_tensor(tokens).long().reshape(1, 1, -1).to(device),
        "text_len": torch.tensor(len(tokens)).reshape(1, 1),
        "specaug": False
    }

    fsd_df = pd.read_csv(args.fsid_to_fpath, sep="\t")
    fsid_to_fpath = dict(zip(fsd_df["audio_id"], fsd_df["file_name"]))

    time_resolution = model.audio_encoder.time_resolution
    n_connect = math.ceil(connect_dur / time_resolution)

    dataset = InferDataset(fin, fsid_to_fpath, sample_rate)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)

    with torch.no_grad(), open(fout, "w") as writer:
        for batch in tqdm(dataloader):
            waveform, duration, fsid = batch

            if waveform[0].shape[0] == 1 and waveform[0, 0] == -1:
                continue

            duration = duration[0].item()
            fsid = fsid[0]

            input_dict.update({
                "waveform": torch.as_tensor(waveform).to(device),
                "waveform_len": [len(waveform)]
            })
            model_output = model(input_dict)
            prob = model_output["frame_sim"][0, :, 0].cpu().numpy()
            filtered_prob = eval_util.median_filter(
                prob[None, :], window_size=1, threshold=threshold)[0]
            change_indices = eval_util.find_contiguous_regions(
                eval_util.connect_clusters(
                    filtered_prob,
                    n_connect
                )
            )

            num_occurrence = len(change_indices)
            
            if num_occurrence == 0:
                print(f"{fsid} has no occurrence of '{text}', excluded")
                continue
            elif num_occurrence > 1:
                segs = []
                for row in change_indices:
                    onset = row[0] * time_resolution
                    offset = row[1] * time_resolution
                    segs.append([onset, offset])
                
                writed_num = 0
                for seg_idx, segment in enumerate(segs):
                    onset, offset = segment
                    ori_duration = offset - onset
                    if ori_duration > max_single_occurrence_duration or \
                        ori_duration < min_single_occurrence_duration:
                        continue
                    if seg_idx == 0:
                        if onset - fade_in_out < 0:
                            start = 0
                        else:
                            start = onset - fade_in_out
                    else:
                        if onset - fade_in_out > segs[seg_idx - 1][1]:
                            start = onset - fade_in_out
                        else:
                            continue

                    if seg_idx == len(segs) - 1:
                        if offset + fade_in_out > duration:
                            end = duration
                        else:
                            end = offset + fade_in_out
                    else:
                        if offset + fade_in_out < segs[seg_idx + 1][0]:
                            end = offset + fade_in_out
                        else:
                            continue

                    seg_duration = end - start
                    if seg_duration < min_duration_after_padding:
                        pad = (min_duration_after_padding - seg_duration) / 2
                    else:
                        pad = 0
                    writer.write(f"{fsid} {start:.3f} {end:.3f} {pad:.3f}\n")
                    writed_num += 1

                if writed_num == 0:
                    print(f"{fsid} has no single occurrence of '{text}' to be splited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", "-exp", type=str, required=True)
    parser.add_argument("--fin", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--fout", type=str, required=True)
    parser.add_argument("--fmeta", type=str, required=True)
    parser.add_argument("--fade_in_out", type=float, default=0.2)
    parser.add_argument("--min_single_occurrence_duration", type=float, default=0.25,
                        help="minimum single occurrence duration in the original clip, for filtering out noisy output")
    parser.add_argument("--max_single_occurrence_duration", type=float, default=2.0,
                        help="maximumm single occurrence duration in the original clip, for filtering out multiple occurrences")
    parser.add_argument("--min_duration_after_padding", type=float, default=2.0,
                        help="minimum duration of single occurrence after padding")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--connect_duration", "-cd", type=float, default=0.5)
    parser.add_argument("--fsid_to_fpath", type=str,
                        default="/media/williamzhangsjtu/datastorage2/freesound/audio.csv")
    
    args = parser.parse_args()
    main(args)
