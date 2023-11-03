import json

import requests
from process_kaiko import kaiko_df

from perp.constants import TIME_BLOCK_PATH

# get kaiko_df timestamp
time_block_dict: dict[int, dict[str, int]] = {}

# get unique kaiko_df["poll_timestamp"] / 1_000
time_block_list = kaiko_df["poll_timestamp"].unique() / 1_000

for i in time_block_list:
    # get https://coins.llama.fi/block/ethereum/{1698595200} result

    url = f"https://coins.llama.fi/block/ethereum/{int(i)}"
    response = requests.get(url)
    result = response.json()
    time_block_dict[int(i)] = result

# save result to json
with open(TIME_BLOCK_PATH, "w") as f:
    json.dump(time_block_dict, f)
