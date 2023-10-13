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
)[["variableBorrowRate", "liquidityRate"]].astype(float) / 1e27


reservepara_df = reservepara_df.reset_index("reserve")
# change timestamp "1674806411" type to datetime
reservepara_df.index = pd.to_datetime(reservepara_df.index, unit="s")


def aave_rates_df(risk_asset: str, usd_asset: str, long_risk: bool) -> pd.DataFrame:
    def interpolate_df(asset_name: str, rate_name: str) -> pd.DataFrame:
        df = reservepara_df[reservepara_df["reserve"] == asset_name][[rate_name]]
        sampled_ts = df.resample("8H").asfreq()
        return (
            pd.concat([df, sampled_ts])
            .sort_index()
            .interpolate()
            .merge(sampled_ts[[]], how="right", left_index=True, right_index=True)
        )

    rate_names = ["liquidityRate", "variableBorrowRate"]
    usd_df = interpolate_df(
        risk_asset, rate_names[long_risk]
    )  # USD borrowed if long_risk
    risk_df = interpolate_df(
        usd_asset, rate_names[not long_risk]
    )  # risk asset lent if long_risk

    return pd.concat([usd_df, risk_df], axis=1).dropna()


if __name__ == "__main__":
    aave_df = aave_rates_df(risk_asset="WETH", usd_asset="USDC", long_risk=True)
    print(aave_df)
