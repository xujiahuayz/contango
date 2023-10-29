from pprint import pprint

import requests

from perp.constants import SYMBOL_LIST
from perp.settings import KAIKO_API_KEY

exchanges = requests.get(
    url="https://reference-data-api.kaiko.io/v1/exchanges", timeout=10
).json()["data"]

# change structure to dict with 'name' as key and 'code' as value and sort by name
exchange_dict = {e["name"]: e["code"] for e in exchanges}
exchange_dict = dict(sorted(exchange_dict.items()))

headers = {
    "Accept": "application/json",
    "X-Api-Key": KAIKO_API_KEY,
}

for symbol in SYMBOL_LIST:
    print(symbol)
    for s in [1, 100, 10000]:
        print(s)
        params = {
            "slippage": s,
            "page_size": 1,
            "interval": "8h",
        }

        for e in ["Binance", "OkEX", "Huobi", "CoinEx"]:
            response = requests.get(
                url=f"https://us.market-api.kaiko.io/v2/data/order_book_snapshots.latest/exchanges/{exchange_dict[e]}/spot/{symbol.lower()}-usdt/ob_aggregations/slippage",
                headers=headers,
                params=params,
                timeout=100,
            )

            print(e)

            pprint(response.json()["data"])
