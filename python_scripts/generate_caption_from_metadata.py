import json
import requests
from pathlib import Path
import argparse
import os
import random
from typing import Dict, List
import openai
from tqdm import tqdm


API_KEY = os.environ["API_KEY"]
openai.api_key = API_KEY


def describe_sound_by_openai(metadata, prompt,
                             model_name="gpt-3.5-turbo",
                             temperature=0.5):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant to describe sounds."},
            {"role": "user", "content": prompt + json.dumps(metadata)}
        ],
        temperature=temperature
    )
    
    result = response.choices[0].message.content
    return result


def describe_sound_by_fetch(metadata, prompt,
                            model_name="gpt-3.5-turbo",
                            temperature=0.5):
    header = {
        "Content-Type":"application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    post_dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant to describe sounds."},
            {"role": "user", "content": prompt + json.dumps(metadata)}
        ],
        "temperature": temperature,
    }
 
    response = requests.post("https://frostsnowjh.com/v1/chat/completions",
                             json=post_dict,
                             headers=header)
    result = response.json()["choices"][0]["message"]["content"]
 
    return result


def convert_item(item, sound_type_to_dscrp, single_sound_prompt, no_request=False):
    """
    {
        'sound_type': 'man_speaking_male_voice',
        'id': 1,
        ...
    }
    ->
    {
        'id': 'man speaking 1',
        'sound': 'man speaking',
        ...
    }
    """
    result = {}
    dscrp = random.choice(sound_type_to_dscrp[item["sound_type"]])
    single_meta = {
        "sound": dscrp,
    }

    if "loudness" in item:
        single_meta["loudness"] = item["loudness"]

    if "times" in item:
        if isinstance(item["times"], int) and item["times"] > 3:
            single_meta["times"] = "multiple"
        else:
            single_meta["times"] = item["times"]

    if "id" in item:
        if isinstance(item["id"], int):
            result["id"] = dscrp + " " + str(item["id"])
        elif isinstance(item["id"], str):
            single_meta["id"] = item["id"]
            result["id"] = dscrp
    else:
        result["id"] = dscrp

    if no_request:
        result["text"] = dscrp
    else:
        result["text"] = describe_sound_by_fetch(single_meta, single_sound_prompt)

    # print(result["text"])

    request_data = single_sound_prompt + json.dumps(single_meta)

    return result, request_data


def convert_item_no_request(item, sound_type_to_dscrp):

    result = {}
    dscrp = random.choice(sound_type_to_dscrp[item["sound_type"]])

    if "loudness" in item:
        result["loudness"] = item["loudness"]

    if "times" in item:
        if isinstance(item["times"], int) and item["times"] > 3:
            result["times"] = "multiple"
        else:
            result["times"] = item["times"]

    if "id" in item:
        if isinstance(item["id"], int):
            result["sound"] = dscrp + " " + str(item["id"])
        elif isinstance(item["id"], str):
            result["id"] = item["id"] # single / multiple
            result["sound"] = dscrp
    else:
        result["sound"] = dscrp

    return result


def remove_longer_tuples(input_list):
    reachable_from = {}

    # Populate the mapping
    for a, b in input_list:
        if a not in reachable_from:
            reachable_from[a] = set()
        reachable_from[a].add(b)

    # Function to find all reachable nodes starting from a given node
    def find_reachable(node, visited):
        if node in visited:
            return set()
        visited.add(node)
        reachable = set()
        if node in reachable_from:
            for next_node in reachable_from[node]:
                reachable.add(next_node)
                reachable |= find_reachable(next_node, visited)
        return reachable

    all_reachable_from = {}

    for node in reachable_from:
        all_reachable_from[node] = find_reachable(node, set())

    to_keep = []
    # Check each tuple to see if it should be kept
    for a, b in input_list:
        keep = True
        for node in all_reachable_from:
            if node in all_reachable_from[a] and b in all_reachable_from[node]:
                keep = False
                break
        if keep:
            to_keep.append((a, b))
    
    return to_keep


def determine_relation(event1, event2):
    # overlap = max(0, min(event1["end"], event2["end"]) - max(event1["start"], event2["start"]))
    overlap = event1["end"] - event2["start"]
    duration = min(event1["end"] - event1["start"], event2["end"] - event1["start"])
    if overlap < 0.5 * duration:
        return "sequential"
    else:
        return "simultaneous"
    

def merge_simul_list(simul_list, merged_meta, id_key="id"):
    repr_to_list = {}
    for e1, e2 in simul_list:
        if e1 in repr_to_list:
            repr_to_list[e1].append(e2)
        else:
            matched = None
            for e_other in repr_to_list:
                if e1 in repr_to_list[e_other]:
                    matched = e_other
                    break
            if matched is None:
                repr_to_list[e1] = [e2]
            else:
                if e2 not in repr_to_list[matched]:
                    repr_to_list[matched].append(e2)

    event_to_grp = {}
    grp_to_event = {}
    for grp_idx, repr in enumerate(repr_to_list):
        event_to_grp[repr] = grp_idx
        grp_to_event[grp_idx] = [repr]
        for e in repr_to_list[repr]:
            event_to_grp[e] = grp_idx
            grp_to_event[grp_idx].append(e)

    grp_idx = len(simul_list)
    for item in merged_meta:
        if item[id_key] not in event_to_grp:
            event_to_grp[item[id_key]] = grp_idx
            grp_to_event[grp_idx] = [item[id_key]]
            grp_idx += 1

    return event_to_grp, grp_to_event


def remove_common_in_seq(seq_list: List[List[int,]],
                         grp_to_event: Dict[int, List[str]],):
    # output = []
    to_remove = []
    for start_grp, end_grp in seq_list:
        # output.append([grp_to_event[start_grp], grp_to_event[end_grp]])

        # remove common
        for event in grp_to_event[start_grp]:
            if event in grp_to_event[end_grp]:
                to_remove.append(event)
    
    for start_grp, end_grp in seq_list:
        # start_no_dup = []
        # end_no_dup = []
        for event in grp_to_event[start_grp]:
            if event in to_remove:
                grp_to_event[start_grp].remove(event)
            # if event not in to_remove:
                # start_no_dup.append(event)
        for event in grp_to_event[end_grp]:
            if event in to_remove:
                grp_to_event[end_grp].remove(event)
            # if event not in to_remove:
                # end_no_dup.append(event)
        # output.append([start_no_dup, end_no_dup])
    # return output


def flatten_seq_list(seq_list, grp_to_event):
    visited = set()
    grp_to_idx = {}


    while len(visited) < len(seq_list):
        for start_grp, end_grp in seq_list:
            if len(grp_to_idx) == 0:
                grp_to_idx[start_grp] = 0
                grp_to_idx[end_grp] = 1
                visited.add((start_grp, end_grp))
                continue

            if (start_grp, end_grp) in visited:
                continue
            if start_grp in grp_to_idx:
                grp_to_idx[end_grp] = grp_to_idx[start_grp] + 1
            elif end_grp in grp_to_idx:
                grp_to_idx[start_grp] = grp_to_idx[end_grp] - 1
            visited.add((start_grp, end_grp))

    sorted_grp = [k for k, v in sorted(grp_to_idx.items(), key=lambda item: item[1])]
    grp_strs = []
    for grp in sorted_grp:
        events = grp_to_event[grp]
        grp_str = " and ".join(["'" + event + "'" for event in events])
        grp_strs.append("(" + grp_str + ")")
    output = " then ".join(grp_strs)
    return output


def generate_temporal_list(merged_meta, id_key="id"):
    seq_list = []
    simul_list = []
    for i in range(len(merged_meta)):
        for j in range(len(merged_meta)):
            if i == j or merged_meta[i]["start"] >= merged_meta[j]["start"]:
                continue
            relation = determine_relation(merged_meta[i], merged_meta[j])
            if relation == "sequential":
                seq_list.append((merged_meta[i][id_key], merged_meta[j][id_key]))
            else:
                simul_list.append((merged_meta[i][id_key], merged_meta[j][id_key]))            

    event_to_grp, grp_to_event = merge_simul_list(simul_list, merged_meta, id_key)

    # print(seq_list)
    grped_seq_list = []
    for e1, e2 in seq_list:
        if (event_to_grp[e1], event_to_grp[e2]) not in grped_seq_list:
            grped_seq_list.append((event_to_grp[e1], event_to_grp[e2]))
    # print(grped_seq_list)

    seq_list_no_dup = remove_longer_tuples(grped_seq_list)
    remove_common_in_seq(seq_list_no_dup, grp_to_event)
    output = flatten_seq_list(seq_list_no_dup, grp_to_event)
    
    return output


def generate_sentence_with_temporal(metadata, sound_type_to_dscrp,
                                    single_sound_prompt, sentence_prompt,
                                    no_request):
    if "background" in metadata:
        meta_for_llm = {"background": random.choice(sound_type_to_dscrp[metadata["background"]])}
    else:
        meta_for_llm = {}

    meta_for_llm["sounds"] = []
    meta_for_temporal = []
    
    request_prompt = ""

    for item in metadata["sounds"]:
        converted_item, request_data = convert_item(item,
                                                    sound_type_to_dscrp,
                                                    single_sound_prompt,
                                                    no_request)
        request_prompt += request_data
        meta_for_llm["sounds"].append(converted_item)
        meta_for_temporal.append({
            "id": converted_item["id"],
            "start": item["start"],
            "end": item["end"]
        })

    meta_for_llm["temporal"] = generate_temporal_list(meta_for_temporal, "id")
    print(meta_for_llm)
    
    request_prompt += sentence_prompt + json.dumps(meta_for_llm)
    if no_request:
        return {"caption": "dummy text", "prompt": request_prompt}
    else:
        return {"caption": describe_sound_by_fetch(meta_for_llm, sentence_prompt)}


def generate_sentence_with_temporal_one_request(metadata, sound_type_to_dscrp, sentence_prompt):
    if "background" in metadata:
        meta_for_llm = {"background": random.choice(sound_type_to_dscrp[metadata["background"]])}
    else:
        meta_for_llm = {}
    meta_for_llm["sounds"] = []
    meta_for_temporal = []
    for item in metadata["sounds"]:
        converted_item = convert_item_no_request(
            item, sound_type_to_dscrp
        )
        meta_for_llm["sounds"].append(converted_item)
        meta_for_temporal.append({
            "sound": converted_item["sound"],
            "start": item["start"],
            "end": item["end"]
        })
    meta_for_llm["temporal"] = generate_temporal_list(meta_for_temporal, "sound")
    if meta_for_llm["temporal"] == "":
        del meta_for_llm["temporal"]
    # print("Metadata: ")
    # print(metadata)
    # print("Converted for LLM: ")
    # print(meta_for_llm)
    result = describe_sound_by_fetch(meta_for_llm, sentence_prompt)
    return {
        "caption": result
    }


def main(args):
    sound_type_to_dscrp = json.load(open(args.sound_descriptions, "r"))

    metas = json.load(open(args.in_json, "r"))


    with open(args.single_sound_prompt, "r") as f:
        single_sound_prompt = f.read()
    with open(args.sentence_prompt, "r") as f:
        sentence_prompt = f.read()


    if args.count_token:
        no_request = True
        with open(args.out_json, "r") as reader:
            aid_to_caption = json.load(reader)
    else:
        no_request = False

    job_name = Path(args.in_json).stem
    progress_file = Path(args.in_json).with_name(f"{job_name}_progress.txt")
    total_text = ""

    if progress_file.exists():
        progressed = []
        with open(progress_file, "r") as reader:
            for line in reader.readlines():
                progressed.append(line.strip())
    else:
        progressed = []

    progressed_num = 0
    

    for audio_id in tqdm(metas):
        
        if audio_id in progressed:
            continue

        success = False
        try:
            if args.one_request:
                response = generate_sentence_with_temporal_one_request(
                    metas[audio_id],
                    sound_type_to_dscrp,
                    sentence_prompt
                )
            else:
                response = generate_sentence_with_temporal(metas[audio_id],
                                                           sound_type_to_dscrp,
                                                           single_sound_prompt,
                                                           sentence_prompt,
                                                           no_request)

            if Path(args.out_json).exists():
                progressed_result = json.load(open(args.out_json))
            else:
                progressed_result = {}
            progressed_result.update({audio_id: response["caption"]})
            json.dump(progressed_result, open(args.out_json, "w"), indent=4)

            with open(progress_file, "a") as progress_writer:
                progress_writer.write(f"{audio_id}\n")
            success = True

        except Exception:
            continue

        if success:
            progressed_num += 1

        if args.first_n > 0 and progressed_num > args.first_n:
            break

        if args.count_token:
            total_text += response["prompt"]
            total_text += aid_to_caption[audio_id]

    if args.count_token:
        print("Token number: ", len(total_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_token", default=False, action="store_true")
    parser.add_argument("--in_json", "-j", required=True, type=str)
    parser.add_argument("--sound_descriptions", type=str, default="./sound_attributes/category_to_description.json")
    parser.add_argument("--single_sound_prompt", default="./chatgpt_prompts/single_sound.txt", type=str)
    parser.add_argument("--sentence_prompt", default="./chatgpt_prompts/temporal_caption_flatten.txt", type=str)
    parser.add_argument("--out_json", "-o", required=True, type=str)
    parser.add_argument("--one_request", default=False, action="store_true")
    parser.add_argument("--first_n", default=-1, type=int)


    args = parser.parse_args()
    main(args)
