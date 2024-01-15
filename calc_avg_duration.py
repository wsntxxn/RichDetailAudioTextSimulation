import os
from pathlib import Path
import argparse
import librosa
import numpy as np
import pandas as pd


def main(args):
    sound_to_duration = {}
    for sound in os.listdir(args.audio_source_dir):
        sound_dir = Path(args.audio_source_dir) / sound
        durations = []
        for fname in os.listdir(sound_dir):
            fpath = sound_dir / fname
            duration = librosa.core.get_duration(filename=fpath.__str__())
            durations.append(duration)
        avg_duration = np.mean(durations)
        sound_to_duration[sound] = avg_duration
    df = pd.DataFrame({"sound": sound_to_duration.keys(),
                       "duration": sound_to_duration.values()})
    df.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_source_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)
