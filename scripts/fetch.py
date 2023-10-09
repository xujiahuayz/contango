import json
import pandas as pd

import requests

from perp.constants import DATA_PATH

BINANCE_PATH = DATA_PATH / "binance_ethusd.json"
BINANCE_PATH_FULL = DATA_PATH / "binance_ethusd_full.csv"


# can download from blob:https://www.binance.com/3c0fea36-ca61-4d67-b7e6-c11b7ae3e9a4

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


def interpolate_df(df_path: str, rate_column: str, new_rate_column: str) -> pd.Series:
    # read aave usdc borrow csv and remove the last row, convert first column to timestamp
    df = pd.read_csv(DATA_PATH / df_path).iloc[:-1]
    df["timestamp_pd"] = pd.to_datetime(df["date"])
    # convert from percentage value to decimal value
    df[new_rate_column] = df[rate_column] / 100
    df.set_index("timestamp_pd", inplace=True)

    # resample to 8-hour interval and make sure the new indices are in the range
    return (
        pd.concat(
            [
                df[new_rate_column],
                df.resample("8H").asfreq()[new_rate_column],
            ]
        )
        .sort_index()
        .interpolate()
        .iloc[1:]
    )


aave_usdc_borrow = interpolate_df(
    "aave_usdc_borrow.csv", "Variable borrow rate", "usdc_borrow_apy"
)
aave_eth_deposit = interpolate_df(
    "aave_eth_deposit.csv", "Deposit rate", "eth_deposit_apy"
)

# merge two series into a dataframe and keep only the rows where both series have values
aave_df = pd.concat([aave_usdc_borrow, aave_eth_deposit], axis=1).dropna()

binance_df = pd.read_json(BINANCE_PATH).rename(
    columns={
        "fundingRate": "binance_funding_rate",
    }
)
# set fundingTime to the nearest 8-hour interval
binance_df["fundingTime"] = (
    pd.to_datetime(binance_df["fundingTime"], unit="ms")
    .dt.tz_localize("UTC")
    .dt.round("8H")
)
# set fundingTime as index
binance_df.set_index("fundingTime", inplace=True)


# read dydx data as a dataframe
dydx_df = pd.read_json(DYDX_PATH, convert_dates=["effectiveAt"]).set_index(
    "effectiveAt"
)

# merge aave_df with dydx_df[price] based on the index
aave_binance_df = (
    aave_df.merge(dydx_df[["price"]], left_index=True, right_index=True)
    .merge(binance_df[["binance_funding_rate"]], left_index=True, right_index=True)
    .dropna()
)
assert sum(aave_binance_df.resample("8H").asfreq().index != aave_binance_df.index) == 0
