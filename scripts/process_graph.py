import gzip
import json

import pandas as pd

from perp.constants import AAVE_V3_PARAM_PATH


def reshape_reserve_item(line: str):
    reservepara = json.loads(line)
    reservepara["reserve"] = reservepara["reserve"]["symbol"]
    return reservepara


# # read the jsonl file as a list
with gzip.open(AAVE_V3_PARAM_PATH, "rt") as f:
    reservepara_list = [reshape_reserve_item(line) for line in f]

# # convert the list to a dataframe
indices = ["timestamp", "reserve"]
reservepara_df = (
    pd.DataFrame(reservepara_list).drop_duplicates(subset=indices).set_index(indices)
)[["variableBorrowRate", "liquidityRate", "utilizationRate"]].astype(float)

reservepara_df[["variableBorrowRate", "liquidityRate"]] = (
    reservepara_df[["variableBorrowRate", "liquidityRate"]] / 1e27
)


reservepara_df = reservepara_df.reset_index("reserve")
# change timestamp "1674806411" type to datetime
reservepara_df.index = pd.to_datetime(reservepara_df.index, unit="s")


def interpolate_df(asset_name: str, col_name: str) -> pd.DataFrame:
    df = reservepara_df[reservepara_df["reserve"] == asset_name][[col_name]]
    sampled_ts = df.resample("8H").asfreq()
    return (
        pd.concat([df, sampled_ts])
        .sort_index()
        .interpolate()
        .merge(sampled_ts[[]], how="right", left_index=True, right_index=True)
    )


def aave_rates_df(risk_asset: str, usd_asset: str, long_risk: bool) -> pd.DataFrame:
    if long_risk:
        borrow_rates = interpolate_df(usd_asset, "variableBorrowRate")
        lend_rates = interpolate_df(risk_asset, "liquidityRate")
    else:
        borrow_rates = interpolate_df(risk_asset, "variableBorrowRate")
        lend_rates = interpolate_df(usd_asset, "liquidityRate")

    return pd.concat([borrow_rates, lend_rates], axis=1).dropna()


if __name__ == "__main__":
    aave_df = aave_rates_df(risk_asset="WETH", usd_asset="USDC", long_risk=True)
    print(aave_df)
