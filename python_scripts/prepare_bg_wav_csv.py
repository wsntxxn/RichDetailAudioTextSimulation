import pandas as pd
from pathlib import Path
import librosa


bg_types = ["wind_blowing", "crickets_chirping", "bird_chirping", "thunder", "traffic_noise"]


data = []

for bg_type in bg_types:
    bg_path = Path(f"./audio_source/normalized_audio/{bg_type}")
    for wav_path in bg_path.glob("*.wav"):
        data.append({
            "file_name": wav_path.absolute().__str__(),
            "type": bg_type,
            "duration": librosa.core.get_duration(filename=wav_path.absolute().__str__())
        })


for wav_path in Path("/mnt/lustre/sjtu/home/xnx98/data/shared_data/raa/musan/music").rglob("*/*.wav"):
    data.append({
        "file_name": wav_path.absolute().__str__(),
        "type": "music",
        "duration": librosa.core.get_duration(filename=wav_path.absolute().__str__())
    })


df = pd.DataFrame(data)
df.to_csv("./audio_source/metadata/bg_wav.csv", index=False, sep="\t")
