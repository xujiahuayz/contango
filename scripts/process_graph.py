import gzip
import json

import pandas as pd

from perp.constants import AAVE_V3_PARAM_PATH, DAI_PATH
from decimal import Decimal, InvalidOperation

RAY = Decimal(10) ** 27

def reshape_reserve_item(line: str):
    reservepara = json.loads(line)
    reservepara["reserve"] = reservepara["symbol"]
    if 'ts' in reservepara.keys():
        reservepara['timestamp'] = pd.to_datetime(reservepara['ts'], unit="s", utc=True)
        reservepara['timestamp'] = reservepara['timestamp'].round("8H")
    else:
        reservepara['timestamp'] = pd.to_datetime(reservepara['sample_timestamp'], unit="s", utc=True)
    return reservepara



def ray_to_apr(x):
    try:
        return float(Decimal(str(x)) / RAY)
    except (InvalidOperation, TypeError):
        return float('nan')




# # read the jsonl file as a list
with gzip.open(AAVE_V3_PARAM_PATH, "rt") as f:
    reservepara_list = [reshape_reserve_item(line) for line in f]


# # convert the list to a dataframe
indices = ["timestamp", "reserve"]

reservepara_df = (
    pd.DataFrame(reservepara_list).drop_duplicates(subset=indices).set_index(indices)
)[
    [
        "variableBorrowRatePct",
        "liquidityRatePct",
        "utilization",
        # "totalATokenSupply",
        # "totalCurrentVariableDebt",
    ]
].astype(
    float
)

reservepara_df = reservepara_df.reset_index("reserve")
reservepara_df.index = pd.to_datetime(reservepara_df.index, unit="s")


with gzip.open(DAI_PATH, "rt") as f:
    for line in f:
        dai = [reshape_reserve_item(line) for line in f]
dai_df = pd.DataFrame(dai).drop_duplicates(subset=indices).set_index('timestamp')
dai_df["variableBorrowRatePct"] = dai_df["variableBorrowRate"].map(ray_to_apr) * 100
dai_df["liquidityRatePct"] = dai_df["liquidityRate"].map(ray_to_apr) * 100
dai_df['utilization'] = 1- dai_df['availableLiquidity'].map(ray_to_apr)/dai_df['totalLiquidity'].map(ray_to_apr)
clms = ["reserve", "variableBorrowRatePct", "liquidityRatePct", "utilization"]
reservepara_df = pd.concat([reservepara_df[clms], dai_df[clms]])


def interpolate_df(asset_name: str, col_name: list[str]) -> pd.DataFrame:
    df = reservepara_df[reservepara_df["reserve"] == asset_name][col_name]
    sampled_ts = df.resample("8H").last()
    return (
        df
        .sort_index()
        .reindex(df.index.union(sampled_ts.index))
        .interpolate()
        .merge(sampled_ts[[]], how="right", left_index=True, right_index=True)
    )


def aave_rates_df(risk_asset: str, usd_asset: str, long_risk: bool) -> pd.DataFrame:
    if long_risk:
        borrow_rates = interpolate_df(usd_asset, ["variableBorrowRatePct"])
        lend_rates = interpolate_df(risk_asset, ["liquidityRatePct"])
    else:
        borrow_rates = interpolate_df(risk_asset, ["variableBorrowRatePct"])
        lend_rates = interpolate_df(usd_asset, ["liquidityRatePct"])
    borrow_rates = borrow_rates[~borrow_rates.index.duplicated(keep="last")]
    return pd.concat([borrow_rates, lend_rates], axis=1).dropna()


if __name__ == "__main__":
    aave_df = aave_rates_df(risk_asset="WETH", usd_asset="DAI", long_risk=True)
    print(aave_df)
