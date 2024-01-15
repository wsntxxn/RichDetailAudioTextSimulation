import requests
import urllib
import json
import argparse
from requests_oauthlib import OAuth2Session
from bs4 import BeautifulSoup
import os
import requests
import freesound
import time
from pathlib import Path
from tqdm import tqdm


def search(query,
           min_duration=0.0,
           max_duration="*",):
    """
    Args:
        query: str, 
    """
    escaped_query = urllib.parse.quote(query)
    results = client.text_search(
        query=escaped_query,
        fields="id",
        page_size=150,
        filter=f"duration: [{min_duration} TO {max_duration}]"
    )
    fsd_ids = []
    with tqdm(total=results.count) as pbar:
        while results:
            for sound in results:
                if not sound:
                    break
                fsd_ids.append(sound.id)
                pbar.update()
            if not results.next:
                break
            results = results.next_page()
    return fsd_ids


def download(id, save_path):
    if len(list(Path(save_path).glob(f"{id}*"))) > 0:
        return
    sound = client.get_sound(id)
    sound.retrieve(save_path, name=f"{id}." + f"{sound.type}")
    time.sleep(1.0)


def download_from_idtxt(txt_path):
    with open(txt_path, 'r') as f:
        file_names = f.read().splitlines()
    for id in file_names:
        download(id, "./dog_bark/")


def download_id_notfound():
    with open("id_notfound.txt", 'r') as f:
        file_names = f.read().splitlines()
    for id in file_names:
        download(id, "./dog_bark")


if __name__== '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", default=10.0)
    parser.add_argument("--credential_file", "-c", type=str, required=True)
    parser.add_argument("--token_file", "-t", type=str, required=True)

    args = parser.parse_args()
    with open(args.token_file, "r") as f:
        token_data = json.load(f)
        access_token = token_data["access_token"]
        refresh_token = token_data["refresh_token"]
    
    with open(args.credential_file, "r") as f:
        credential_data = json.load(f)
        client_id = credential_data["client_id"]
        client_secret = credential_data["client_secret"]

    # check whether to refresh access token
    headers = {'Authorization': f'Bearer {access_token}'}
    get_url = "https://freesound.org/apiv2/sounds/632621/"
    response = requests.get(get_url, headers=headers)
    if response.status_code == 401:
        print("refresh token")
        refresh_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        response = requests.post("https://freesound.org/apiv2/oauth2/access_token/", data=refresh_data)
        token_data = json.loads(response.text)
        with open(args.token_file, "w") as f:
            json.dump(token_data, f)
        access_token = token_data["access_token"]

    client = freesound.FreesoundClient()
    client.set_token(access_token, "oauth")
    fsd_ids = search(query=args.query,
                     min_duration=args.min_duration,
                     max_duration=args.max_duration)
    
    existed_ids = []
    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            for line in f.readlines():
                existed_ids.append(line.strip())

    with open(args.save_path, "a") as f:
        for fsd_id in fsd_ids:
            if str(fsd_id) not in existed_ids:
                f.write(str(fsd_id) + "\n")

    if os.path.exists(args.meta):
        meta = json.load(open(args.meta, "r"))
    else:
        meta = {"queries": []}
    if args.query not in meta["queries"]:
        meta["queries"].append(args.query)
    meta.update({
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
    })
    json.dump(meta, open(args.meta, "w"), indent=4)
