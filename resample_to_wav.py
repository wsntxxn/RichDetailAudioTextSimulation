import argparse
import os
from pathlib import Path


def main(args):
    filelist = open(args.filelist).readlines()
    output = Path(args.output)
    output.mkdir(exist_ok=True, parents=True)
    for file in filelist:
        audio_path = file.strip()
        new_name = Path(audio_path).stem + ".wav"
        new_path = output / new_name
        os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {new_path.__str__()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", "-f", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()
    main(args)
