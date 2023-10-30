import gzip
import json

import pandas as pd
import requests

from perp.constants import (
    KAIKO_EXCHANGE_PATH,
    KAIKO_SLIPPAGE_PATH,
    SYMBOL_LIST,
    TRADE_SIZE_LIST,
)
from perp.settings import KAIKO_API_KEY

exchange_df = pd.read_pickle(KAIKO_EXCHANGE_PATH)


headers = {
    "Accept": "application/json",
    "X-Api-Key": KAIKO_API_KEY,
}

with gzip.open(KAIKO_SLIPPAGE_PATH, "wt") as f:
    for symbol in SYMBOL_LIST:
        print(symbol)
        for s in TRADE_SIZE_LIST:
            print(s)
            params = {
                "slippage": s,
                "page_size": 100,
                "interval": "8h",
                "start_time": "2023-04-01T00:00:00.000Z",
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
