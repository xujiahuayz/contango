import json

import pandas as pd

from perp.constants import COINGLASS_PATH, DATA_PATH

with open(COINGLASS_PATH, "r") as f:
    results = json.load(f)
dataMap = pd.DataFrame(results["dataMap"])
frDataMap = pd.DataFrame(results["frDataMap"])
dataMap == frDataMap
# merge the two dataframes
coinglass_df = dataMap.merge(
    frDataMap, left_index=True, right_index=True, suffixes=("", "_fr")
)
coinglass_df["timestamp"] = pd.to_datetime(results["dateList"], unit="ms")
coinglass_df["timestamp"] = coinglass_df["timestamp"].dt.tz_localize("UTC")
coinglass_df.set_index("timestamp", inplace=True)

# change the percentage to decimal
coinglass_df = coinglass_df / 100
coinglass_df["price"] = results["priceList"]


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

# merge aave_df with dydx_df[price] based on the index
coinglass_aave_df = aave_df.merge(
    coinglass_df, left_index=True, right_index=True
).sort_index()
