import gzip
import json

import pandas as pd

from perp.constants import KAIKO_SLIPPAGE_PATH, KAIKO_EXCHANGE_PATH

exchange_df = pd.read_pickle(KAIKO_EXCHANGE_PATH)

# load COINGLASS_PATH which is jsonl.gz file
result_df = pd.DataFrame()
with gzip.open(KAIKO_SLIPPAGE_PATH, "rt") as f:
    for line in f:
        # results.update(json.loads(line))
        result = json.loads(line)
        df = pd.DataFrame(result["data"])
        print(len(df))
        e = result["query"]["exchange"]
        df["exchange"] = exchange_df[exchange_df["code"] == e]["name"].values[0]
        # 'instrument': 'eth-usdt', take the chars before '-'
        df["risk_asset"] = result["query"]["instrument"].split("-")[0].upper()
        df["trade_size"] = result["query"]["slippage"]
        result_df = pd.concat(
            [result_df, df], ignore_index=True
        )  # use pd.concat to combine dataframes
