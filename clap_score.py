import argparse
import numpy as np
import json
from pathlib import Path
import json

import librosa


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def inference(args):
    import torch
    import laion_clap
    import pandas as pd
    from tqdm import tqdm
    
    ftext = args.ftext
    fout = args.fout


    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt() # download the default pretrained checkpoint.
    model.eval()

    wav_df = pd.read_csv(args.wav_csv, sep="\t")
    aid_to_fpath = dict(zip(wav_df["audio_id"], wav_df["file_name"]))
    aid_to_cap = json.load(open(ftext))

    if not Path(fout).parent.exists():
        Path(fout).parent.mkdir(parents=True)

    
    with torch.no_grad(), open(fout, "w") as writer:
        for aid, cap in tqdm(aid_to_cap.items()):
            fpath = aid_to_fpath[aid + ".wav"]
            text_data = [cap, "dummy text"]
            text_embed = model.get_text_embedding(text_data, use_tensor=True)
            audio_file = [fpath]
            audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
            score = audio_embed @ text_embed.transpose(0, 1)
            score = score.cpu().numpy()[0, 0]
            writer.write(f"{aid}\t{score:.3f}\n")


def filter_aids(args):

    ftext = args.ftext
    fout = args.fout
    threshold = args.threshold
    pos_keywords = args.pos_keywords
    neg_keywords = args.neg_keywords

    aid_to_cap = json.load(open(ftext))

    aid_to_score = {}
    with open(args.score, "r") as f:
        for line in f.readlines():
            aid, score = line.strip().split()
            aid_to_score[aid] = float(score)

    
    filtered = {}
    for aid, cap in aid_to_cap.items():
        if len(pos_keywords) > 0:
            if not any([kw in cap for kw in pos_keywords]):
                continue
        if len(neg_keywords) > 0:
            if any([kw in cap for kw in neg_keywords]):
                continue
        if aid_to_score[aid] > threshold:
            filtered[aid] = cap

    with open(fout, "w") as writer:
        json.dump(filtered, writer, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    parser_infer = subparsers.add_parser("infer_score")
    parser_infer.add_argument("--wav_csv", type=str, required=True)
    parser_infer.add_argument("--ftext", type=str, required=True)
    parser_infer.add_argument("--fout", type=str, required=True)
    
    parser_filter = subparsers.add_parser("filter")
    parser_filter.add_argument("--score", type=str, required=True)
    parser_filter.add_argument("--fout", type=str, required=True)
    parser_filter.add_argument("--ftext", type=str, required=True)
    parser_filter.add_argument("--threshold", type=float, required=True)
    parser_filter.add_argument("--pos_keywords", type=str, nargs="+", default=[])
    parser_filter.add_argument("--neg_keywords", type=str, nargs="+", default=[])



    args = parser.parse_args()

    if args.mode == "infer_score":
        inference(args)
    elif args.mode == "filter":
        filter_aids(args)

