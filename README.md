# Audio-Text Simulation with Rich Details

## Single-event source preparation

We curate single-event audio source data from Freesound by writing related text queries.
Then we rely on [LAION-CLAP](https://github.com/LAION-AI/CLAP) and [TAG](https://github.com/wsntxxn/TextToAudioGrounding) to split and filter single-event source data.
Refer to [here](./freesound_curation/README.md) for details on data source curation.

The structure of single-event audio source data is shown below. For each `event` directory under `[audio_source_dir]`, if the event is typically short in duration (e.g., dog barking), there are two sub-directories "single_occurrence" and "non_single" to differentiate source audio with corresponding occurrence times.
Otherwise, all source audio files are listed directly in `event` directory.
```
[audio_source_dir]
├── bird_chirping
│   ├── bird_chirping_1.wav
│   ├── bird_chirping_2.wav
│   │   .
│   │   .
│   └── bird_chirping_n.wav
│
├── dog_barking
│   ├── single_occurrence
│   │   ├── dog_barking_single_1.wav
│   │   ├── dog_barking_single_2.wav
│   │   │   .
│   │   │   .
│   │   └── dog_barking_single_n.wav
│   │
│   └── non_single
│       ├── dog_barking_non_single_1.wav
│       ├── dog_barking_non_single_2.wav
│       │   .
│       │   .
│       └── dog_barking_non_single_n.wav
│
│   .
│   .
│   .
```

## Data simulation

Mixture audios are generated by first generating jams data and corresponding metadata via [generate_mixture_jams.py](./generate_mixture_jams.py), then generating audio and text based on the jams and metadata, respectively.

### JAMS generation

```bash
python ./python_scripts/generate_mixture_jams.py \
    --max_duration 15.0 \
    --syn_number 2000 \
    --audio_source_dir ${audio_source_dir} \
    --duration_file ${duration_file} \
    --output_jams_dir ${jams_dir} \
    --output_meta ${meta_file} \
```

`audio_source_dir` is the directory of single-event audio source files described in the previous part.
Most parameters are self-explained from their literal meanings.
Explanations for some parameters:

* `[min/max]_num_events`: the minimum / maximum event number in a single audio, except the background
* `max_event_occurrence`: the maximum occurrence number of a single event, e.g., if it is 5, the sound of dog barking can occur for at most **5** times in any generated audio
* `max_distinct_identity`: for identity-sensitive sounds (e.g., man speaking), the maximum unique identities in an audio, e.g., if it is 2, there are a maximum of **2** men speaking in a single audio
* `times_desc_prob`: the probability that the temporal relationship between sound events is explicitly described in the caption
* `[loud/low]_threshold`: the snr threshold that a sound event is described as *loudly/faintly*
* `no_bg_snr_threshold`: the snr threshold that all sound events are recognized as foreground events

### Audio generation from JAMS

```bash
python ./python_scripts/generate_audio_from_jams.py \
    --jams_dir ${jams_dir} \
    --output_dir ${audio_dir}
```

`jams_dir` is the one generated in the previous step while the generated audio clips are in `audio_dir`.

### Caption generation from metadata

```bash
python ./python_scripts/generate_caption_from_metadata.py \
    --in_json ${meta_file} \
    --out_json ${caption_file} \
    --one_request
```

`meta_file` is generated from the JAMS generation step and the generated caption is in `caption_file`.

## BibTeX

If you find this repository useful, please cite using this BibTeX:
```BibTeX
@inproceedings{xu2024detailed,
  title={A Detailed Audio-Text Data Simulation Pipeline Using Single-Event Sounds},
  author={Xu, Xuenan and Xu, Xiaohang and Xie, Zeyu and Zhang, Pingyue and Wu, Mengyue and Yu, Kai},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1091--1095},
  year={2024},
  organization={IEEE}
}
```