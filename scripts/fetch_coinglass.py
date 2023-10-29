import gzip
import json

import requests

from perp.constants import COINGLASS_PATH, SYMBOL_LIST
from perp.settings import COINGLASS_SECRET

COINGLASS_ENDPOINT = "https://open-api.coinglass.com/public/v2/funding_usd_history"
headers = {
    "accept": "application/json",
    "coinglasssecret": COINGLASS_SECRET,
}

# save result in jsonl.gz file
with gzip.open(COINGLASS_PATH, "wt") as f:
    for symbol in SYMBOL_LIST:
        params = {"symbol": symbol, "time_type": "h8"}
        # Make the request
        response = requests.get(
            COINGLASS_ENDPOINT, headers=headers, params=params, timeout=10
        )
        if response.status_code == 200:
            results = response.json()["data"]
            if results["dateList"]:
                # save results, which is a dict
                f.write(json.dumps({symbol: results}) + "\n")
