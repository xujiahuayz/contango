import json

import requests

from perp.constants import DATA_PATH

BINANCE_PATH = DATA_PATH / "binance_ethusd.json"
BINANCE_PATH_FULL = DATA_PATH / "binance_ethusd_full.csv"

# check if BINANCE_PATH exists
# if not, fetch data from binance

# read blob:https://www.binance.com/3c0fea36-ca61-4d67-b7e6-c11b7ae3e9a4


if not (BINANCE_PATH_FULL.exists() or BINANCE_PATH.exists()):
    # fetch data from binance, API info https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
    results = requests.get(
        "https://www.binance.com/fapi/v1/fundingRate?symbol=ETHUSDT&limit=1000"
    ).json()
    # save results, which is a list to BINANCE_PATH
    with open(BINANCE_PATH, "w") as f:
        json.dump(results, f, indent=2)


DYDX_PATH = DATA_PATH / "dydx_ethusd.json"

if not DYDX_PATH.exists():
    # get funding rates from dydx https://api.dydx.exchange/v3/historical-funding/ETH-USD?effectiveBeforeOrAt=2022-01-05
    last_date = "2023-11-01T00:00:00.000Z"
    dydx_url = (
        lambda date: f"https://api.dydx.exchange/v3/historical-funding/ETH-USD?effectiveBeforeOrAt={date}"
    )
    results = requests.get(dydx_url(last_date)).json()["historicalFunding"]
    new_last_date = results[-1]["effectiveAt"]

    while last_date > new_last_date:
        last_date = new_last_date
        results += requests.get(dydx_url(last_date)).json()["historicalFunding"][1:]
        new_last_date = results[-1]["effectiveAt"]
        print(last_date)

    with open(DYDX_PATH, "w") as f:
        json.dump(results, f, indent=2)
