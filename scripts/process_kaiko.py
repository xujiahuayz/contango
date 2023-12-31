import gzip
import json

import pandas as pd

from perp.constants import KAIKO_EXCHANGE_PATH, KAIKO_SLIPPAGE_PATH

exchange_df = pd.read_pickle(KAIKO_EXCHANGE_PATH)

# load COINGLASS_PATH which is jsonl.gz file
kaiko_df = pd.DataFrame()
with gzip.open(KAIKO_SLIPPAGE_PATH, "rt") as f:
    for line in f:
        result = json.loads(line)
        df = pd.DataFrame(result["data"])
        # print(len(df))
        e = result["query"]["exchange"]
        df["exchange"] = exchange_df[exchange_df["code"] == e]["name"].values[0]
        # 'instrument': 'eth-usdt', take the chars before '-'
        df["risk_asset"] = result["query"]["instrument"].split("-")[0].upper()
        df["trade_size"] = result["query"]["slippage"]
        kaiko_df = pd.concat(
            [kaiko_df, df], ignore_index=True
        )  # use pd.concat to combine dataframes

# set 'ask_slippage' and 'bid_slippage' to numeric
kaiko_df["ask_slippage"] = pd.to_numeric(kaiko_df["ask_slippage"])
kaiko_df["bid_slippage"] = pd.to_numeric(kaiko_df["bid_slippage"])
