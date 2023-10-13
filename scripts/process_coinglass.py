import gzip
import json

import pandas as pd

from perp.constants import COINGLASS_PATH

# load COINGLASS_PATH which is jsonl.gz file
results = {}
with gzip.open(COINGLASS_PATH, "rt") as f:
    for line in f:
        results.update(json.loads(line))


def coinglass_fr_df(risk_asset: str) -> pd.DataFrame:
    long_dict = results[risk_asset]

    dataMap = pd.DataFrame(long_dict["dataMap"])
    frDataMap = pd.DataFrame(long_dict["frDataMap"])
    # merge the two dataframes
    df = dataMap.merge(
        frDataMap, left_index=True, right_index=True, suffixes=("", "_fr")
    )
    df["timestamp"] = pd.to_datetime(long_dict["dateList"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # change the percentage to decimal
    df = df / 100
    df["price"] = long_dict["priceList"]
    return df.iloc[:-1]


if __name__ == "__main__":
    coinglass_df = coinglass_fr_df(risk_asset="ETH")
    print(coinglass_df)
