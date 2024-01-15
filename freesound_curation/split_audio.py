import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import laion_clap


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def main(args):

    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt() # download the default pretrained checkpoint.
    model.eval()

    # Get text embedings from texts:
    text_data = [f"This is a sound of {args.text}.", "dummy text"]

    fsd_df = pd.read_csv(args.fsid_to_fpath, sep="\t")
    fsid_to_fpath = dict(zip(fsd_df["audio_id"], fsd_df["file_name"]))
 
    lines = open(args.fin).readlines()
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        text_embed = model.get_text_embedding(text_data, use_tensor=True)
        for line in tqdm(lines):
            if len(line.strip().split()) < 4:
                continue 
            fsid, onset, offset, pad = line.strip().split()
            try:
                fpath = fsid_to_fpath["freesound_" + fsid]
            except KeyError:
                fpath = fsid_to_fpath[fsid]
            new_name = Path(fpath).stem + f"_{onset}_{offset}_{pad}.wav"
            new_path = output / new_name
            os.system(f"ffmpeg -loglevel quiet -i {fpath} -ac 1 -ar 16000 -ss {onset} -to {offset} ./split_tmp.wav")
            left_pad_len = int(float(pad) * 1000)
            os.system(f"ffmpeg -loglevel quiet -i ./split_tmp.wav -af 'adelay={left_pad_len},apad=pad_len=16000:pad_dur={pad}' {new_path.__str__()}")
            os.remove("./split_tmp.wav")
            audio_file = [new_path]
            audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
            score = audio_embed @ text_embed.transpose(0, 1)
            score = score.cpu().numpy()[0, 0]
            if score <= args.threshold:
                os.remove(new_path.__str__())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--fsid_to_fpath", type=str,
                        default="/media/williamzhangsjtu/datastorage2/freesound/audio.csv")
    args = parser.parse_args()
    main(args)

