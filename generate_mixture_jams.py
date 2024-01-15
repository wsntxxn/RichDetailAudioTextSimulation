import glob
import os
import json
from pathlib import Path
import argparse
from typing import Dict, List
import numpy as np
import pandas as pd
import jams
from tqdm import trange, tqdm
import librosa
from scaper.core import _sample_trunc_norm


def load_dict_from_csv(csv, cols):
    df = pd.read_csv(csv, sep="\t")
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


def select_valid_sound_from_list(sound_files: List,
                                 aid_to_duration: Dict,
                                 random_state: np.random.RandomState):
    sound_file = random_state.choice(sound_files, 1)[0]
    duration = aid_to_duration[Path(sound_file).name]
    while duration < 0.01:
        sound_file = random_state.choice(sound_files, 1)[0]
        duration = aid_to_duration[Path(sound_file).name]
    return sound_file, duration


def select_snr(mean=10.0,
               std=5.0,
               min=-5.0,
               max=25.0,
               random_state=None):
    snr = _sample_trunc_norm(mean, std, min, max, random_state)
    return snr


def generate_one_sound(sound_files: List,
                       aid_to_duration: Dict,
                       max_num_occurrence: int,
                       cur_duration: float,
                       sound_type: str,
                       time_sensitive: bool,
                       id_sensitive: bool,
                       max_clip_duration: float,
                       max_interval: float,
                       sound_info_list: List,
                       repeat_single: bool,
                       times_control: bool,
                       random_state: np.random.RandomState,
                       loud_threshold: float=15.0,
                       low_threshold: float=2.5,):
    """
    Args:
        cur_duration (float): the current maximum offset of all sounds
        sound_info_list (list): output list for jams, will be modified in-place
    """

    # select a valid sound file
    sound_file, duration = select_valid_sound_from_list(sound_files,
                                                        aid_to_duration,
                                                        random_state)
    
    # TODO maximum tolerant single event duration
    if duration > 10.0:
        # trunc_duration = _sample_trunc_norm(5.0, 1.0, 3.0, 10.0, random_state)
        trunc_duration = random_state.uniform(5.0, 10.0)
        source_time = random_state.uniform(0, duration - trunc_duration)
        duration = trunc_duration
    else:
        source_time = 0.0

    # randomly sample a loudness
    snr = select_snr(random_state=random_state)

    if snr > loud_threshold:
        loudness = "loud"
    elif snr < low_threshold:
        loudness = "low"
    else:
        loudness = None

    # determine when to start: overlap / continuation
    continuation = False if random_state.rand() < 0.5 else True
    if continuation:
        interval = random_state.uniform(0, 1.0) * max_interval
        cur_duration_event = cur_duration + min(interval, max_clip_duration - cur_duration)
    else:
        # overlap with added sounds
        if cur_duration > duration:
            cur_duration_event = random_state.uniform(0, cur_duration - duration)
        else:
            cur_duration_event = 0.0

    metadata = {
        "sound_type": sound_type,
        "start": round(cur_duration_event, 3)
    }
    if loudness is not None:
        metadata["loudness"] = loudness

    if not time_sensitive and id_sensitive and not repeat_single:
        metadata_list = []
    
    cur_sound_infos = []

    for occur_i in range(max_num_occurrence):
        
        if cur_duration_event >= max_clip_duration:
            break

        if occur_i > 0 and not repeat_single:
            sound_files.remove(sound_file)
            sound_file, duration = select_valid_sound_from_list(sound_files,
                                                                aid_to_duration,
                                                                random_state)
            if duration > 10.0:
                # trunc_duration = _sample_trunc_norm(5.0, 1.0, 3.0, 10.0, random_state)
                trunc_duration = random_state.uniform(5.0, 10.0)
                source_time = random_state.uniform(0, duration - trunc_duration)
                duration = trunc_duration
            else:
                source_time = 0.0
            # TODO determine whether to resample snr in multiple files
            # snr = select_snr(random_state=random_state)

        occur_i_duration = min(duration, max_clip_duration - cur_duration_event)

        # if duration left for the current sound is too short, just stop, to avoid unnatural stop
        if occur_i_duration < duration and occur_i_duration < 2.5:
            break

        sound_info = {
            "time": cur_duration_event,
            "duration": occur_i_duration,
            "value": {
                "label": sound_type,
                "source_file": sound_file.__str__(),
                "source_time": source_time,
                "event_time": cur_duration_event,
                "event_duration": occur_i_duration,
                "snr": snr,
                "role": "foreground",
                "pitch_shift": None,
                "time_stretch": None
            }
        }
        # if sound_info["value"]["event_time"] < 0:
        #     import pdb; pdb.set_trace()
        cur_duration_event += occur_i_duration
        sound_info_list.append(sound_info)
        cur_sound_infos.append(sound_info)
        
        if not time_sensitive and id_sensitive and not repeat_single:
            cur_metadata = metadata.copy()
            cur_metadata["id"] = occur_i + 1
            cur_metadata["start"] = round(sound_info["time"], 3)
            cur_metadata["end"] = round(cur_duration_event, 3)
            metadata_list.append(cur_metadata)

        if cur_duration_event >= max_clip_duration:
            break
        
        interval = random_state.uniform(0, 1.0) * max_interval
        cur_duration_event += min(interval, max_clip_duration - cur_duration_event)


    if len(cur_sound_infos) == 0:
        metadata = None
    else:
        if time_sensitive:
            if times_control:
                if id_sensitive:
                    metadata.update({
                        "end": round(cur_duration_event, 3),
                        "times": len(cur_sound_infos),
                    })
                    if len(cur_sound_infos) > 1:
                        metadata["id"] = "single" if repeat_single else "multiple"
                else:
                    metadata.update({
                        "end": round(cur_duration_event, 3),
                        "times": len(cur_sound_infos)
                    })
            else:
                metadata.update({
                    "end": round(cur_duration_event, 3),
                })
        else:
            if id_sensitive and not repeat_single:
                # different ids (e.g., multiple speakers)
                metadata = metadata_list
            else:
                metadata.update({
                    "end": round(cur_duration_event, 3)
                })

    cur_duration = max(cur_duration, cur_duration_event)
    return cur_duration, metadata


def generate(args):
    audio_source_dir = Path(args.audio_source_dir)
    min_num_events = args.min_num_events
    max_num_events = args.max_num_events
    max_event_occurrence = args.max_event_occurrence
    init_sil = args.init_sil

    output_dir = Path(args.output_jams_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # read sound attributes
    bg_sound_types: List[str] = []
    with open("sound_attributes/bg_sounds.txt", "r") as f:
        for line in f.readlines():
            bg_sound_types.append(line.strip())


    must_audible_bg_sounds: List[str] = []
    with open("sound_attributes/must_audible_bg.txt", "r") as f:
        for line in f.readlines():
            must_audible_bg_sounds.append(line.strip())

    non_fg_sound_types: List[str] = []
    with open("sound_attributes/non_fg_sounds.txt", "r") as f:
        for line in f.readlines():
            non_fg_sound_types.append(line.strip())

    time_sensitive_sound_types: List[str] = []
    with open("sound_attributes/time_sensitive_sounds.txt", "r") as f:
        for line in f.readlines():
            time_sensitive_sound_types.append(line.strip())

    id_sensitive_sound_types: List[str] = []
    with open("sound_attributes/id_sensitive_sounds.txt") as f:
        for line in f.readlines():
            id_sensitive_sound_types.append(line.strip())

    sound_type_to_files = {}
    for sound_type in os.listdir(audio_source_dir):
        sound_dir = audio_source_dir / sound_type
        if sound_type in time_sensitive_sound_types:
            sound_type_to_files[sound_type] = {
                "single_occurrence": list((sound_dir / "single_occurrence").rglob("*.wav")),
                "non_single": list((sound_dir / "non_single").rglob("*.wav"))
            }
        else:
            sound_files = list(sound_dir.rglob("*.wav"))
            sound_type_to_files[sound_type] = sound_files
    sound_types = list(sound_type_to_files.keys())

    
    filter_sound_type = lambda x: [_ for _ in x if _ in sound_type_to_files.keys()]

    bg_sound_types = filter_sound_type(bg_sound_types)
    time_sensitive_sound_types = filter_sound_type(time_sensitive_sound_types)
    id_sensitive_sound_types = filter_sound_type(id_sensitive_sound_types)

    env_to_sound_types = {}
    with open("sound_attributes/env.json", "r") as f:
        env_to_sound_types = json.load(f)

    sound_type_to_env = {}
    for env, sounds in env_to_sound_types.items():
        for sound_type in sounds:
            sound_type_to_env[sound_type] = env

    aid_to_duration = load_dict_from_csv(args.duration_file,
                                         ("audio_id", "duration"))

    random_state = np.random.RandomState(args.seed)

    metas = {}

    if min_num_events == 1 and max_num_events == 3:
        num_events_weight = [0.6, 0.35, 0.05]
    else:
        num = max_num_events - min_num_events + 1
        num_events_weight = [1 / num] * num

    generated_num = 0
    with tqdm(total=args.syn_number) as pbar:
        while generated_num < args.syn_number:
            # for jams file
            """
            {
                "time": time,  # onset of this event
                "duration": duration,  # actual duration of this event, equals `event_duration * time_stretch`, the actual data in the mixture is `source[source_time: source_time + event_duration] * time_stretch` 
                "value": {
                    "label": label,
                    "source_file": source_file,
                    "source_time": source_time,  # onset of the source file
                    "event_time": event_time,  # onset of this event, same as outer "time" 
                    "snr": snr,
                    "role": "foreground" | "background",
                    "pitch_shift": pitch_shift
                    "time_stretch": time_stretch
                },
                ("condifence": 1.0) (for background)
            }
            """
            sound_info_list: List[Dict] = []
            metadata: List[Dict] = []

            # add background file
            # if background is "domestic"/"wild"/"music", exclude specific foreground sound types
            bg_sound_type: str = random_state.choice(bg_sound_types)
            if bg_sound_type in sound_type_to_env:
                env_type = sound_type_to_env[bg_sound_type]
                if env_type == "domestic":
                    exclude_sound_types = env_to_sound_types["wild"] + [bg_sound_type]
                elif env_type == "wild":
                    exclude_sound_types = env_to_sound_types["domestic"] + [bg_sound_type]
                elif env_type == "music":
                    exclude_sound_types = env_to_sound_types["instrument"] + env_to_sound_types["music"]
                exclude_sound_types += non_fg_sound_types.copy()
            else:
                exclude_sound_types = non_fg_sound_types.copy()
                if bg_sound_type not in exclude_sound_types:
                    exclude_sound_types.append(bg_sound_type)

            avail_sound_types = [sound_type for sound_type in sound_types if sound_type not in exclude_sound_types]
            
            bg_path: Path = random_state.choice(sound_type_to_files[bg_sound_type])
            bg_info = {
                "time": 0.0,
                "value": {
                    "label": bg_sound_type,
                    "source_file": bg_path.__str__(),
                    "event_time": 0.0,
                    "snr": 0, # `snr` here does not have any effect since the background energy (loudness) only depends on `ref_db`
                    "role": "background",
                    "pitch_shift": None,
                    "time_stretch": None
                },
                "confidence": 1.0
            }

            # add other files (foreground or background)
            # initial silence
            init_onset = random_state.uniform(0, init_sil)
            cur_duration = init_onset

            num_events = random_state.choice(
                np.arange(min_num_events, max_num_events + 1),
                p=num_events_weight)

            
            for event_i in range(num_events):
                # why don't we sample `num_events` sounds at first? because of the maximum duration, there may not be enough space for all sounds
                sound_type = random_state.choice(avail_sound_types)
                
                if sound_type == "motorboat" and bg_sound_type != "ocean_waves":
                    sound_type = random_state.choice(avail_sound_types)


                # if a sound is times-sensitive or identity-sensitive in descriptions, we may repeat the sound multiple times
                if sound_type in time_sensitive_sound_types:
                    if random_state.random() < args.times_desc_prob:
                        sound_files = sound_type_to_files[sound_type]["single_occurrence"].copy()
                        times_control = True
                        max_num_occurrence = random_state.randint(1, max_event_occurrence + 1)
                    else:
                        sound_files = sound_type_to_files[sound_type]["non_single"].copy()
                        times_control = False
                        max_num_occurrence = 1
                else:
                    sound_files = sound_type_to_files[sound_type].copy()
                    times_control = False
                    if sound_type in id_sensitive_sound_types:
                        max_num_occurrence = random_state.randint(1, args.max_distinct_identity + 1)
                    else:
                        max_num_occurrence = 1

                # if sound_type in time_sensitive_sound_types:
                    # max_num_occurrence = random_state.randint(1, max_event_occurrence + 1)
                    # if sound_type in id_sensitive_sound_types:
                        # repeat_single = random_state.rand() < 0.5
                    # else:
                        # repeat_single = True
                # else:
                    # if sound_type in id_sensitive_sound_types:
                        # max_num_occurrence = random_state.randint(1, args.max_distinct_identity + 1)
                        # repeat_single = False
                    # else:
                        # max_num_occurrence = 1
                        # repeat_single = False
                
                if len(sound_files) == 0:
                    print(sound_type)
                
                time_sensitive = sound_type in time_sensitive_sound_types
                id_sensitive = sound_type in id_sensitive_sound_types
                if sound_type in id_sensitive_sound_types:
                    repeat_single = random_state.rand() < 0.5
                    if sound_type not in time_sensitive_sound_types:
                        repeat_single = False
                else:
                    repeat_single = True 

                cur_duration, metadata_i = generate_one_sound(
                    sound_files=sound_files,
                    aid_to_duration=aid_to_duration,
                    max_num_occurrence=max_num_occurrence,
                    cur_duration=cur_duration,
                    sound_type=sound_type,
                    time_sensitive=time_sensitive,
                    id_sensitive=id_sensitive,
                    max_clip_duration=args.max_duration,
                    max_interval=args.max_interval,
                    sound_info_list=sound_info_list,
                    repeat_single=repeat_single,
                    times_control=times_control,
                    random_state=random_state,
                    loud_threshold=args.loud_threshold,
                    low_threshold=args.low_threshold
                )

                if metadata_i is not None:
                    if isinstance(metadata_i, dict):
                        metadata.append(metadata_i)
                    elif isinstance(metadata_i, list):
                        metadata.extend(metadata_i)
                avail_sound_types.remove(sound_type)

                if cur_duration >= args.max_duration:
                    break

            jam = jams.JAMS()
            jam.file_metadata.duration = cur_duration
            ann = jams.Annotation(namespace="scaper", time=0, duration=cur_duration,
                                  sandbox={
                                      "scaper": {
                                          "duration": cur_duration,
                                          "original_duration": cur_duration,
                                          "fg_path": args.fg_path,
                                          "bg_path": args.bg_path,
                                          "protected_labels": [],
                                          "sr": args.sample_rate,
                                          "ref_db": args.ref_db,
                                          "n_channels": 1,
                                          "fade_in_len": 0.01,
                                          "fade_out_len": 0.01,
                                          "reverb": args.reverb,
                                          "disable_sox_warnings": True
                                        }
                                  })


            # add background duration info
            bg_duration = librosa.core.get_duration(filename=bg_path)
            others_duration = cur_duration
            if bg_duration > others_duration:
                max_source_time = bg_duration - others_duration
                bg_source_time = random_state.uniform(0, max_source_time)
                bg_event_time = others_duration
            else:
                bg_source_time = 0
                bg_event_time = bg_duration
            bg_info["duration"] = bg_event_time
            bg_info["value"].update({
                "source_time": bg_source_time,
                "event_duration": bg_event_time
            })
            ann.append(**bg_info)
            
            all_metadata = {
                "background": bg_sound_type,
                "sounds": metadata
            }

            snrs = [sound_info["value"]["snr"] for sound_info in sound_info_list]
            if np.all(np.array(snrs) > args.no_bg_snr_threshold) and \
                bg_sound_type not in must_audible_bg_sounds and \
                len(metadata) > 0:
                del all_metadata["background"]

            for sound_info in sound_info_list:
                ann.append(**sound_info)

            jam.annotations.append(ann)
            
            generated_num += 1

            jam.save(str(output_dir / f"syn_{generated_num}.jams"))
            metas[f"syn_{generated_num}"] = all_metadata
            pbar.update()

    json.dump(metas, open(args.output_meta, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_duration", type=float, required=True)
    parser.add_argument("--syn_number", "-n", type=int, required=True)
    parser.add_argument("--audio_source_dir", type=str, required=True)
    parser.add_argument("--duration_file", type=str, required=True)
    parser.add_argument("--output_jams_dir", type=str, required=True)
    parser.add_argument("--output_meta", type=str, required=True)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min_num_events", type=int, default=1)
    parser.add_argument("--max_num_events", type=int, default=4)
    parser.add_argument("--max_event_occurrence", type=int, default=5)
    parser.add_argument("--max_interval", type=float, default=2.5)
    parser.add_argument("--max_distinct_identity", type=int, default=2)
    parser.add_argument("--times_desc_prob", type=float, default=0.8)

    # duration setting
    parser.add_argument("--init_sil", type=float, default=2.5,
                        help="Initial silence duration")
    parser.add_argument("--max_single_occurrence_duration",
                        type=float,
                        default=10.0,
                        help="Maximum duration for a single sound occurrence")
    
    # loudness setting
    parser.add_argument("--loud_threshold", type=float, default=17.5)
    parser.add_argument("--low_threshold", type=float, default=2.5)
    parser.add_argument("--no_bg_snr_threshold", type=float, default=15.0)

    # scaper argument
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--ref_db", type=float, default=-20.0)
    parser.add_argument("--reverb", type=float, default=None)
    parser.add_argument("--fg_path", type=str, default="audio_source")
    parser.add_argument("--bg_path", type=str, default="audio_source")

    args = parser.parse_args()

    job_name = Path(args.output_meta).stem
    synth_meta_fpath = Path(args.output_meta).with_name(job_name + "_synth_hp.json")
    with open(synth_meta_fpath, "w") as writer:
        json.dump(vars(args), writer, indent=4)

    generate(args)
