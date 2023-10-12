import gzip
import json

import pandas as pd

from perp.constants import AAVE_V3_PARAM_PATH


def aave_rates_df(long_asset: str, short_asset: str) -> pd.DataFrame:
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
        pd.DataFrame(reservepara_list)
        .drop_duplicates(subset=indices)
        .set_index(indices)
    )[["variableBorrowRate", "liquidityRate"]].astype(float) / 1e27

    reservepara_df = reservepara_df.reset_index("reserve")
    # change timestamp "1674806411" type to datetime
    reservepara_df.index = pd.to_datetime(reservepara_df.index, unit="s")

    def interpolate_df(asset_name: str, variable_name: str) -> pd.DataFrame:
        df = reservepara_df[reservepara_df["reserve"] == asset_name][[variable_name]]
        sampled_ts = df.resample("8H").asfreq()
        return (
            pd.concat([df, sampled_ts])
            .sort_index()
            .interpolate()
            .merge(sampled_ts[[]], how="right", left_index=True, right_index=True)
        )

    long_df_int = interpolate_df(long_asset, "liquidityRate")
    short_df_int = interpolate_df(short_asset, "variableBorrowRate")

    return pd.concat([long_df_int, short_df_int], axis=1).dropna()


if __name__ == "__main__":
    LONG_ASSET = "WETH"
    SHORT_ASSET = "USDC"
    df = aave_rates_df(LONG_ASSET, SHORT_ASSET)
    print(df)
    from scripts.process_coinglass import coinglass_df

    coinglass_df.merge(df, how="inner", left_index=True, right_index=True)
