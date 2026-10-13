import gzip
import json
import requests
from datetime import datetime, timezone

from perp.constants import COINGLASS_PATH, PRODUCT_LIST, SYMBOL_LIST
from perp.settings import COINGLASS_API_KEY

API_KEY = COINGLASS_API_KEY
BASE_URL = "https://open-api-v4.coinglass.com"
HEADERS = {"accept": "application/json", "CG-API-KEY": API_KEY}


def get_timestamp_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def fetch_funding_rates(symbol):
    url = f"{BASE_URL}/api/futures/funding-rate/exchange-list"
    response = requests.get(url, headers=HEADERS, params={"symbol": symbol}, timeout=15)
    data = response.json()

    if response.status_code != 200 or data.get("code") != "0":
        raise RuntimeError(f"{symbol}: HTTP {response.status_code} code={data.get('code')} msg={data.get('msg')}")

    return data["data"][0] if data.get("data") else None


def fetch_prices():
    url = f"{BASE_URL}/api/futures/pairs-markets"
    response = requests.get(url, headers=HEADERS, timeout=15)
    data = response.json()

    assert data.get("code") == "0", data

    price_map = {}
    for pair in data["data"]:
        exchange = pair["exchange_name"]
        symbol = pair["symbol"]  # e.g., "BTC/USDT"
        base, quote = symbol.split("/")

        if exchange in PRODUCT_LIST and quote in {"USDT", "USD"}:
            price_map[(exchange, base)] = {
                "current_price": float(pair["current_price"]),
                "index_price": float(pair["index_price"])
            }

    return price_map


def main():
    timestamp = get_timestamp_ms()
    price_map = fetch_prices()
    rows = []

    for symbol in SYMBOL_LIST:
        funding_data = fetch_funding_rates(symbol)
        if not funding_data:
            continue

        for item in funding_data.get("stablecoin_margin_list", []):
            exchange = item["exchange"]
            if exchange not in PRODUCT_LIST:
                continue

            prices = price_map.get((exchange, symbol), {})
            rows.append({
                "t": timestamp,
                "symbol": symbol,
                "exchange": exchange,
                "funding_rate": item["funding_rate"],
                "interval_h": item["funding_rate_interval"],
                "next_funding_time": item["next_funding_time"],
                "price_usd": prices.get("current_price"),
                "index_price_usd": prices.get("index_price"),
            })

    # Append mode for time series accumulation
    with gzip.open(COINGLASS_PATH, "at") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()