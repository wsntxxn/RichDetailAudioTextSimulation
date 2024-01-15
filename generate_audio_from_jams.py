import os
import argparse
from glob import glob
from pathlib import Path
import scaper
from tqdm import tqdm
from pypeln import process as pr


def generate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jams_dir = Path(args.jams_dir)
    num_workers = len(os.sched_getaffinity(0))


    def wrapper(fname):
        audio_id = Path(fname).stem + ".wav"
        if (output_dir / audio_id).exists():
            return
        audio, jams, anno_list, event_list = scaper.generate_from_jams(
            fname, str(output_dir / audio_id)
        )


    jams_files = glob(str(jams_dir / "*.jams"))
    with tqdm(total=len(jams_files)) as pbar:
        for result in pr.map(wrapper,
                             jams_files,
                             workers=num_workers,
                             maxsize=4):
            pbar.update()

    # for fname in tqdm(glob(str(jams_dir / "*.jams"))):
        # print(fname)
        # wrapper(fname)

    print("Finished synthesizing audio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jams_dir", "-j", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)

    args = parser.parse_args()
    generate(args)
