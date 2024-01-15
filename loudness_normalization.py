from pathlib import Path
import argparse
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm


def normalize_volume(audio_file_path, target_dBFS):
    audio = AudioSegment.from_file(audio_file_path)
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)


def main(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flist = list(in_dir.rglob("*.wav"))
    for path in tqdm(flist):
        full_out_path = out_dir / path.relative_to(in_dir)
        if not full_out_path.parent.exists():
            full_out_path.parent.mkdir(parents=True)
        normalized_audio = normalize_volume(path, args.target_dbfs)
        normalized_audio.export(full_out_path, format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_dbfs", type=float, default=-20.0)

    args = parser.parse_args()
    main(args)
