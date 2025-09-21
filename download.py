import requests
import json
import pandas as pd
import os

#open desired match dataset as dataframe
matches_file = "matches_2_27.json" #Premier League 2015/16
with open(matches_file, 'r') as file:
    data = json.load(file)
df_matches = pd.json_normalize(data)


all_files = [] #contains all event file names from above match file
for file in df_matches["match_id"]:
    all_files.append(str(file)+'.json')

os.makedirs("Premier_League_1516", exist_ok=True)

select_files = all_files
print(select_files)
user = "statsbomb"
repo = "open-data"
branch = "master"
path = "data/events"

#downloads the files from github repository
for file in select_files:
    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}/{file}"
    r = requests.get(url)
    if r.status_code == 200:
        file_path = os.path.join("Premier_League_1516", file)
        with open(file_path, 'wb') as f:
            f.write(r.content)
    else:
        print(f"Failed to download {file}")
