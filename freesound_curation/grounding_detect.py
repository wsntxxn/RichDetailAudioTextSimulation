import sys
import os
sys.path.insert(1, os.path.join(os.getcwd(), "./audio_text_grounding"))

from pathlib import Path
import json
import math
import argparse
import pickle
import torch
import librosa
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

import utils.train_util as train_util
import utils.eval_util as eval_util
from utils.build_vocab import Vocabulary


def print_pass(*args):
    pass


def inference(args):
    experiment_path = args.experiment_path
    audio = args.audio
    text = args.text
    output = args.output
    threshold = args.threshold
    sample_rate = args.sample_rate
    connect_dur = args.connect_duration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    exp_dir = Path(experiment_path)
    config = train_util.parse_config_or_kwargs(exp_dir / "config.yaml")

    model = train_util.init_obj_from_str(config["model"])
    ckpt = torch.load(exp_dir / "best.pth", "cpu")
    train_util.load_pretrained_model(model, ckpt, output_fn=print_pass)
    model = model.to(device)
    
    vocab_path = config["data"]["train"]["collate_fn"]["tokenizer"]["args"]["vocabulary"]
    vocabulary = Vocabulary()
    vocabulary.load_state_dict(pickle.load(open(vocab_path, "rb")))

    waveform, _ = librosa.core.load(audio, sr=sample_rate)
    duration = waveform.shape[0] / sample_rate
    text = [vocabulary(token) for token in text.split()]

    input_dict = {
        "waveform": torch.as_tensor(waveform).unsqueeze(0).to(device),
        "waveform_len": [len(waveform)],
        "text": torch.as_tensor(text).long().reshape(1, 1, -1).to(device),
        "text_len": torch.tensor(len(text)).reshape(1, 1),
        "specaug": False
    }

    model.eval()
    with torch.no_grad():
        model_output = model(input_dict)
    prob = model_output["frame_sim"][0, :, 0].cpu().numpy()

    time_resolution = model.audio_encoder.time_resolution
    n_connect = math.ceil(connect_dur / time_resolution)
    filtered_prob = eval_util.median_filter(
        prob[None, :], window_size=1, threshold=threshold)[0]
    change_indices = eval_util.find_contiguous_regions(
        eval_util.connect_clusters(
            filtered_prob,
            n_connect
        )
    )

    results = []
    total_seg_len = 0
    seg_lens = []
    nontarget_lens = []

    offset = 0.0

    for row in change_indices:
        onset = row[0] * time_resolution
        nontarget_len = onset - offset
        offset = row[1] * time_resolution
        results.append([onset, offset])
        seg_len = offset - onset
        total_seg_len += seg_len
        seg_lens.append(seg_len)
        nontarget_lens.append(nontarget_len)
    
    nontarget_lens.append(duration - offset)
    print("segments: ", results)
    print(f"total segment length: {total_seg_len}, proportion: {total_seg_len / duration:.3f}")
    if len(seg_lens) > 0:
        print(f"max segment length: {max(seg_lens)}")
    print(f"max nontarget duration: {max(nontarget_lens):.3f}")
    
    plt.figure(figsize=(14, 5))
    plt.plot(prob)
    plt.axhline(y=threshold, color='r', linestyle='--')
    xlabels = [f"{x:.2f}" for x in np.arange(0, duration, duration / 5)]
    plt.xticks(ticks=np.arange(0, len(prob), len(prob) / 5),
               labels=xlabels,
               fontsize=15)
    plt.xlabel("Time / second", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.ylim(0, 1)
    if not Path(output).parent.exists():
        Path(output).parent.mkdir(parents=True)
    plt.savefig(output, bbox_inches="tight", dpi=150)


class InferDataset:

    def __init__(self, fin, fsid_to_fpath, sample_rate):
        self.fsid_to_fpath = fsid_to_fpath
        self.sample_rate = sample_rate
        self.fsids = []
        with open(fin, "r") as reader:
            for line in reader.readlines():
                fsid = line.strip()
                if "freesound_" + fsid not in self.fsid_to_fpath and fsid not in self.fsid_to_fpath:
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


def filter_fids(args):
    experiment_path = args.experiment_path
    text = args.text
    fin = args.fin
    fout = args.fout
    threshold = args.threshold
    sample_rate = args.sample_rate
    connect_dur = args.connect_duration
    max_non_target_duration = args.max_non_target_duration
    max_total_duration = args.max_total_duration

    cut_segments = args.cut_segments
    fade_in_out = args.fade_in_out
    min_segment_duration = args.min_segment_duration

    meta_info = json.load(open(args.fmeta))
    meta_info.update({
        "grounding_filter": {
            "query": text,
            "threshold": threshold,
            "connect_duration": connect_dur,
            "max_non_target_duration": max_non_target_duration
        }
    })
    json.dump(meta_info, open(args.fmeta, "w"), indent=4)

    if max_non_target_duration is None:
        max_non_target_duration = float("inf")

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

    fil_fsids = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            waveform, duration, fsid = batch

            if waveform[0].shape[0] == 1 and waveform[0, 0] == -1:
                continue

            duration = duration[0]
            fsid = fsid[0]

            if duration > max_total_duration:
                print(f"{fsid} is too long, excluded")
                continue

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

            segs = []
            seg_lens = []
            nontarget_lens = []

            offset = 0.0

            for row in change_indices:
                onset = row[0] * time_resolution
                nontarget_len = onset - offset
                offset = row[1] * time_resolution
                segs.append([onset, offset])
                seg_len = offset - onset
                seg_lens.append(seg_len)
                nontarget_lens.append(nontarget_len)
            
            nontarget_lens.append(duration - offset)

            if len(segs) == 0:
                print(f"'{text}' is not detected in {fsid}, excluded")
            elif max(nontarget_lens) > max_non_target_duration:
                if cut_segments:
                    for seg in segs:
                        onset, offset = seg
                        ori_duration = offset - onset
                        if ori_duration < min_segment_duration:
                            continue
                        if onset - fade_in_out < 0:
                            start = 0
                        else:
                            start = onset - fade_in_out
                        if offset + fade_in_out > duration:
                            end = duration
                        else:
                            end = offset + fade_in_out
                        fil_fsids.append(f"{fsid} {start:.3f} {end:.3f} {0.000}")
                else:
                    print(f"{fsid} has non-target segments of > {max_non_target_duration}, excluded")
            else:
                fil_fsids.append(fsid)

            gc.collect()

    
    with open(fout, "w") as writer:
        for fsid in fil_fsids:
            writer.write(fsid + "\n")
    print(f"{len(fil_fsids)} audios left after filtering")


def filter_single_occurrence(args):
    experiment_path = args.experiment_path
    text = args.text
    fin = args.fin
    fout = args.fout
    fmeta = args.fmeta
    threshold = args.threshold
    sample_rate = args.sample_rate
    connect_dur = args.connect_duration
    max_single_occurrence_duration = args.max_single_occurrence_duration
    min_single_occurrence_duration = args.min_single_occurrence_duration
    max_total_duration = args.max_total_duration

    meta_info = json.load(open(fmeta))
    meta_info.update({
        "single_occurrence_filter": {
            "query": text,
            "threshold": threshold,
            "connect_duration": connect_dur,
            "max_single_occurrence_duration": max_single_occurrence_duration,
            "min_single_occurrence_duration": min_single_occurrence_duration
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

    fil_fsids = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            waveform, duration, fsid = batch

            if waveform[0].shape[0] == 1 and waveform[0, 0] == -1:
                continue

            duration = duration[0]
            fsid = fsid[0]

            if duration > max_total_duration:
                print(f"{fsid} is too long, excluded")
                continue

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

            if len(change_indices) == 0:
                print(f"{fsid} is not detected, excluded")
                continue

            if len(change_indices) > 1:
                print(f"{fsid} has more than one occurrence, excluded")
                continue

            row = change_indices[0]
            onset = row[0] * time_resolution
            offset = row[1] * time_resolution
            if offset - onset > max_single_occurrence_duration:
                print(f"{fsid} has single occurrence of > {max_single_occurrence_duration}, excluded")
                continue
            elif offset - onset < min_single_occurrence_duration:
                print(f"{fsid} has single occurrence of < {min_single_occurrence_duration}, excluded")
                continue
            
            fil_fsids.append(fsid)

    with open(fout, "w") as writer:
        for fsid in fil_fsids:
            writer.write(fsid + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_infer = subparsers.add_parser("infer")
    parser_infer.add_argument("--experiment_path", "-exp", type=str, required=True)
    parser_infer.add_argument("--audio", type=str, required=True)
    parser_infer.add_argument("--text", type=str, required=True)
    parser_infer.add_argument("--output", type=str, required=True)
    parser_infer.add_argument("--threshold", type=float, default=0.5)
    parser_infer.add_argument("--sample_rate", type=int, default=32000)
    parser_infer.add_argument("--connect_duration", "-cd", type=float, default=0.5)

    parser_filter = subparsers.add_parser("filter")
    parser_filter.add_argument("--experiment_path", "-exp", type=str, required=True)
    parser_filter.add_argument("--fin", type=str, required=True)
    parser_filter.add_argument("--text", type=str, required=True)
    parser_filter.add_argument("--fout", type=str, required=True)
    parser_filter.add_argument("--fmeta", type=str, required=True)
    parser_filter.add_argument("--max_non_target_duration", type=float, default=None)
    parser_filter.add_argument("--cut_segments", action="store_true", default=False)
    parser_filter.add_argument("--fade_in_out", type=float, default=0.5)
    parser_filter.add_argument("--min_segment_duration", type=float, default=2.0)
    parser_filter.add_argument("--threshold", type=float, default=0.5)
    parser_filter.add_argument("--sample_rate", type=int, default=32000)
    parser_filter.add_argument("--connect_duration", "-cd", type=float, default=0.5)
    parser_filter.add_argument("--fsid_to_fpath", type=str,
                               default="/media/williamzhangsjtu/datastorage2/freesound/audio.csv")
    parser_filter.add_argument("--max_total_duration",
                               help="maximum duration of the original audio clip, set to exclude extremely-long audio clips (e.g., 1 hour long)",
                               type=float,
                               default=600.0)

    parser_single = subparsers.add_parser("filter_single_occurrence")
    parser_single.add_argument("--experiment_path", "-exp", type=str, required=True)
    parser_single.add_argument("--fin", type=str, required=True)
    parser_single.add_argument("--text", type=str, required=True)
    parser_single.add_argument("--fout", type=str, required=True)
    parser_single.add_argument("--fmeta", type=str, required=True)
    parser_single.add_argument("--min_single_occurrence_duration", type=float, default=0.25)
    parser_single.add_argument("--max_single_occurrence_duration", type=float, default=2.0)
    parser_single.add_argument("--threshold", type=float, default=0.5)
    parser_single.add_argument("--sample_rate", type=int, default=32000)
    parser_single.add_argument("--connect_duration", "-cd", type=float, default=0.5)
    parser_single.add_argument("--fsid_to_fpath", type=str,
                               default="/media/williamzhangsjtu/datastorage2/freesound/audio.csv")
    parser_single.add_argument("--max_total_duration",
                               help="maximum duration of the original audio clip, set to exclude extremely-long audio clips (e.g., 1 hour long)",
                               type=float,
                               default=600.0)

    args = parser.parse_args()
    if args.mode == "infer":
        inference(args)
    elif args.mode == "filter":
        filter_fids(args)
    elif args.mode == "filter_single_occurrence":
        filter_single_occurrence(args)
