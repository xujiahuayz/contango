import json
import os

import requests

from perp.constants import COINGLASS_PATH

COINGLASS_SECRET = os.environ.get("COINGLASS_SECRET")

COINGLASS_ENDPOINT = "https://open-api.coinglass.com/public/v2/funding_usd_history"
params = {"symbol": "ETH", "time_type": "h8"}
# Set up headers
headers = {
    "accept": "application/json",
    "coinglasssecret": COINGLASS_SECRET,
}

# Make the request
response = requests.get(COINGLASS_ENDPOINT, headers=headers, params=params)


if not COINGLASS_PATH.exists():
    # Check if the request was successful
    if response.status_code == 200:
        results = response.json()["data"]
        # save results, which is a list to COINGLASS_PATH
        with open(COINGLASS_PATH, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(f"Error {response.status_code}: {response.text}")
