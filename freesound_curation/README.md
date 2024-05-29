# Single-source Data Curation from Freesound

This is a description of our pipeline on curating single-source sound event data, with little to no background noise, from [Freesound platform](https://freesound.org/).
The pre-processed IDs have been provided in `curated_audio`.
Our processing metadata such as the search query or the filtering threshold (will see later) are stored in `curated_audio/{sound}/meta.json` as reference.

The API interface may change. The pipeline may be not suitable for other sources. But **we hope the pipeline design and some components (CLAP filtering, TAG filtering and splitting) can be helpful**.

## Search queries

We first write specific Freesound queries semi-automatically for each sound event.
It contains positive (marked with "+") and negative (marked with "-") queries.
For example, "+a -b" means the returned audio clips must contain the keyword "a" but must not contain "b".

```bash
python freesound_search.py \
    --query "+bird +chirping -music -dog -car -water" \
    --save_path ./curated_audio/bird_chirping/searched_ids.txt \
    --meta ./curated_audio/bird_chirping/meta.json \
    --min_duration 30.0 \
    --max_duration 180.0 \
    -c $CREDENTAIL \
    -t $TOKEN
```
This will store searched Freesound IDs into `save_path` and write corresponding metadata (e.g., min_duration, query) into `meta`.
`CREDENTIAL` and `TOKEN` are information required by Freesound [OAuth2 authentication](https://freesound.org/docs/api/authentication.html#authentication).
They look like:
* `CREDENTIAL`
```json
{
    "client_id": "xxx",
    "client_secret": "xxx"
}
```
* `TOKEN`
```json
{
    "access_token": "xxx",
    "refresh_token": "xxx"
    // ... (useless part)
}
```
You need to generate them yourself if you want to use Freesound search API.

## Filtering

You may notice that the positive and negative keywords in the example query are far from complete.
Besides, keyword-based filtering do not guarantee that curated audio clips meet our requirements.
Therefore, we use further filtering steps to ensure the quality. 

### TAG filtering

[TAG (Text-to-Audio Grounding)](https://github.com/wsntxxn/TextToAudioGrounding) is used to detect the occurrence of a sound event described by natural language prompt in a given audio.
Since we will randomly select a segment from each clip, we use TAG to filter out audio clips with too long non-target segments to avoid selecting segments without target sound events.

Download the [checkpoint](https://drive.google.com/file/d/1xDQT_KQ6l9Hzcn4QkO1G3XBJmdw1LCVe/view?usp=drive_link). Unzip it into `$MODEL_DIR`:
```bash
unzip audiocaps_cnn8rnn_w2vmean_dp_ls_clustering_selfsup.zip -d $MODEL_DIR
```
Modify the training data vocabulary path in `$MODEL_DIR/config.yaml` (*data.train.collate_fn.tokenizer.args.vocabulary*) to `$MODEL_DIR/vocab.pkl`.
 
Then perform filtering, e.g., on bird chirping clips:
```bash
python grounding_detect.py filter \
    -exp $MODEL_DIR \
    --fin curated_audio/bird_chirping/searched_ids.txt \
    --text "bird chirping" \
    --fout curated_audio/bird_chirping/grounding_filtered.txt \
    --fmeta curated_audio/bird_chirping/meta.json \
    --max_non_target_duration 5.0 \
    --fsid_to_fpath $FSID_TO_FPATH
```
where `fin` is the text file listing bird chirping Freesound ids, `fout` is the file to write filtered ids.
Like before, `fmeta` is the metadata file to write configurations, and `fsid_to_fpath` is a tsv file to provide the mapping from Freesound ID to the real file path in this format:
| audio_id | file_name |
| --- | --- |
| 78952 | /path/to/78952.wav |
| ... | ... |

Some sound events may require single occurence segments to support simulation of audio with detailed occurrence numbers (e.g., dog barking).
This can also be done by the `filter_single_occurrence` function:
```bash
python grounding_detect.py filter_single_occurrence \
    -exp $MODEL_DIR \
    --fin curated_audio/dog_barking/searched_ids.txt \
    --text "dog barking" \
    --fout curated_audio/dog_barking/filtered_single.txt \
    --fmeta curated_audio/dog_barking/meta.json \
    --fsid_to_fpath $FSID_TO_FPATH
```
Then `fout` will contain Freesound IDs of those single dog barking occurrence sound.  


### CLAP filtering

TAG model is good at detecting the event temporally, but it is only trained on AudioCaps, where many sound types are excluded.
Its generalization ability to unseen sounds is limited. 
To further filter out sound clips unrelated to the description, we use [CLAP](https://github.com/LAION-AI/CLAP), a larger audio-text model trained on large-scale data.

First we calculate the audio-text similarity score:
```bash
python grounding_detect.py infer \
    --fin curated_audio/bird_chirping/grounding_filtered.txt \
    --text "bird chirping" \
    --fmeta curated_audio/bird_chirping/meta.json \
    --fout ./curated_audio/bird_chirping/clap_scores.txt
```

This will give the similarity score between each audio clip and the given prompt.
We filter the unrelated audio clips based on the score:
```bash
python clap_detect.py filter \
    --fin curated_audio/bird_chirping/clap_scores.txt \
    --fmeta curated_audio/bird_chirping/meta.json \
    --fout curated_audio/bird_chirping/clap_filtered.txt \
    --threshold 0.3
```

Finally, we obtain the filtered IDs in `curated_audio/bird_chirping/clap_filtered.txt`.

## Splitting

Some audio clips contain clean occuurences of sound events, but are discarded because we need single occurrence.
These clips can be leveraged by splitting into single occurrence segments:
```bash
python grounding_split_occurrence.py \
    --fin curated_audio/dog_barking/clap_filtered.txt \
    --text "dog barking" \
    --fmeta curated_audio/dog_barking/meta.json \
    --fout ./curated_audio/dog_barking/filtered_to_split.txt \
    --fsid_to_fpath $FSID_TO_FPATH \
    --min_segment_duration 2.0 \
    --fade_in_out 0.5 \
    --connect_duration 0.5
```
The detected segments with IDs, onsets and offsets are stored in `fout`.
Other parameters are similar to previous ones.
* `min_segment_duration` set the minimum length of the splitted segment
* `fade_in_out` means the final onset / offset will be set `fade_in_out` seconds before / after the detected onset / offset, to avoid abrupt starting or ending
* `connect_duration` is an important parameter to control the minimum silence duration between segments.
For example, `connect_duration = 0.5` means if the distance between two segments is less than 0.5s, the two segments will be treated as a single one.
Different sound events may require different settings.

The obtained file is in this format (a plain text with the column separator of space, without header, here we use table for better visualization):

| Freesound ID | start | end | pad |
| --- | --- | --- | --- |
| 413758 | 6.760 | 7.720 | 0.520 |
| 236038 | 1.720 | 2.200 | 0.760 |
| ... | ... | ... | ... |

`pad` is set to make the segment have the duration of at least `min_segment_duration` seconds.


Finally, we split the original ID file into single occurrence file and non-single occurrence one:
```bash
python split_single_multiple_txt.py \
    --fwhole curated_audio/dog_barking/clap_filtered.txt \
    --fsingle curated_audio/dog_barking/filtered_single.txt \
    --fsplit curated_audio/dog_barking/filtered_to_split.txt \
    --fout curated_audio/dog_barking/non_single.txt
```

