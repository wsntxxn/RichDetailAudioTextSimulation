import argparse
import numpy as np
import json
from pathlib import Path

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
    
    fin = args.fin
    text = args.text
    fout = args.fout
    fmeta = args.fmeta

    if Path(fmeta).exists():
        meta_info = json.load(open(fmeta))
        meta_info.update({
            "clap_filter": {
                "query": text
            }
        })
        json.dump(meta_info, open(fmeta, "w"), indent=4)

    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt() # download the default pretrained checkpoint.
    model.eval()

    fsd_df = pd.read_csv(args.fsid_to_fpath, sep="\t")
    fsid_to_fpath = dict(zip(fsd_df["audio_id"], fsd_df["file_name"]))

    # Get text embedings from texts:
    text_data = [f"This is a sound of {text}.", "dummy text"]

    if args.max_duration is None:
        max_duration = float("inf")
    else:
        max_duration = args.max_duration

    with torch.no_grad(), open(fin, "r") as reader, open(fout, "w") as writer:
        text_embed = model.get_text_embedding(text_data, use_tensor=True)
        for line in tqdm(reader.readlines()):
            try:
                fsid = line.strip()
                try:
                    fpath = fsid_to_fpath[fsid]
                except KeyError:
                    fpath = fsid_to_fpath["freesound_" + fsid]
                duration = librosa.get_duration(filename=fpath)
                if duration > max_duration:
                    print(f"{fsid} is too long, longer than {max_duration}, excluded")
                    continue
                audio_file = [fpath]
                audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
                score = audio_embed @ text_embed.transpose(0, 1)
                score = score.cpu().numpy()[0, 0]
                writer.write(f"{fsid}\t{score:.3f}\n")
            except Exception:
                continue


def filter_fsids(args):
    fin = args.fin
    fout = args.fout
    fmeta = args.fmeta
    percentile = args.percentile
    threshold = args.threshold

    if percentile is None and threshold is None:
        raise ValueError("percentile or threshold should be provided.")

    meta_info = json.load(open(fmeta))
    if percentile is not None:
        meta_info["clap_filter"].update({
            "percentile": percentile
        })
    else:
        meta_info["clap_filter"].update({
            "threshold": threshold   
        })
    json.dump(meta_info, open(fmeta, "w"), indent=4)

    scores = []
    with open(fin, "r") as reader:
        for line in reader.readlines():
            fsid, score = line.strip().split("\t")
            scores.append(float(score))
    scores = np.array(scores)

    if percentile is not None:
        threshold = np.percentile(scores, percentile)
    
    with open(fout, "w") as writer, open(fin, "r") as reader:
        for line in reader.readlines():
            fsid, score = line.strip().split("\t")
            if float(score) >= threshold:
                writer.write(fsid + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    parser_infer = subparsers.add_parser("infer_score")
    parser_infer.add_argument("--fin", type=str, required=True)
    parser_infer.add_argument("--text", type=str, required=True)
    parser_infer.add_argument("--fout", type=str, required=True)
    parser_infer.add_argument("--fmeta", type=str, required=True)
    parser_infer.add_argument("--fsid_to_fpath", type=str,
                              default="/media/williamzhangsjtu/datastorage2/freesound/audio.csv")
    parser_infer.add_argument("--max_duration", type=float, default=None)
    
    parser_filter = subparsers.add_parser("filter")
    parser_filter.add_argument("--fin", type=str, required=True)
    parser_filter.add_argument("--fout", type=str, required=True)
    parser_filter.add_argument("--fmeta", type=str, required=True)
    parser_filter.add_argument("--percentile", "-p", type=float, default=None)
    parser_filter.add_argument("--threshold", type=float, default=None)


    args = parser.parse_args()

    if args.mode == "infer_score":
        inference(args)
    elif args.mode == "filter":
        filter_fsids(args)
