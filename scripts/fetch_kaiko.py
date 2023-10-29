import gzip
import json

import pandas as pd
import requests

from perp.constants import KAIKO_EXCHANGE_PATH, KAIKO_SLIPPAGE_PATH, SYMBOL_LIST
from perp.settings import KAIKO_API_KEY

exchange_df = pd.read_pickle(KAIKO_EXCHANGE_PATH)


headers = {
    "Accept": "application/json",
    "X-Api-Key": KAIKO_API_KEY,
}

with gzip.open(KAIKO_SLIPPAGE_PATH, "wt") as f:
    for symbol in SYMBOL_LIST:
        print(symbol)
        for s in [1e2, 1e4, 1e6]:
            print(s)
            params = {
                "slippage": s,
                "page_size": 100,
                "interval": "8h",
                "start_time": "2023-04-01T00:00:00.000Z",
                # "end_time": "2023-10-29T00:00:00.000Z",
                "sort": "asc",
            }

            for e in ["Binance", "OkEX", "Huobi", "CoinEx"]:
                e_code = exchange_df[exchange_df["name"] == e]["code"].values[0]
                response = requests.get(
                    url=f"https://us.market-api.kaiko.io/v2/data/order_book_snapshots.latest/exchanges/{e_code}/spot/{symbol.lower()}-usdt/ob_aggregations/slippage",
                    headers=headers,
                    params=params,
                    timeout=100,
                )
                if response.status_code != 200:
                    print(response.status_code)
                    print(response.text)
                    continue
                print(response.url)
                result = response.json()
                f.write(json.dumps(result) + "\n")

                # result = response.json()
                # f.write(json.dumps(result) + "\n")

            # data = result["data"]

            # next_url = result["next_url"]
            # while next_url:
            #     print(next_url)
            #     response = requests.get(
            #         url=next_url,
            #         headers=headers,
            #         timeout=100,
            #     )
            #     result = response.json()
            #     next_url = result["next_url"]
            #     data.extend(result["data"])
