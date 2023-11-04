import re

import numpy as np
import pandas as pd
from perp.constants import UNISWAP_TIME_SERIES_PATH
import gzip
import json


def parse_output(output: str) -> dict:
    if output is not None:
        # Remove ANSI color codes
        output = re.sub(r"\x1b\[\d+m", "", output)
        # Split the output into lines
        lines = output.strip().split("\n")
        if len(lines) <= 1:
            return {"Best Route": lines[0]}
    else:
        return {}

    # Initialize an empty dictionary to store the parsed data
    data = {}

    # Initialize a variable to keep track of the current key
    current_key = None

    # Iterate over each line
    for line in lines:
        # Split the line into key and value
        if ":" in line:
            key, value = map(str.strip, line.split(":", 1))
            data[key] = value.strip('\n"')
            current_key = key
        else:
            # If the line does not contain a colon, it is a continuation of the previous line
            data[current_key] += ("\n" + line.strip()).strip('\n"')

    return data


quotes = []
with gzip.open(UNISWAP_TIME_SERIES_PATH, "rt") as f:
    for line in f:
        result = json.loads(line)
        output = parse_output(output=result["output"])

        result.update(
            {
                "raw_quote": float(output["Raw Quote Exact In"])
                if "Raw Quote Exact In" in output
                else np.nan,
                "gas_quote": float(output["Gas Used Quote Token"])
                if "Gas Used Quote Token" in output
                else np.nan,
                "gas_adjusted_quote": float(output["Gas Adjusted Quote In"])
                if "Gas Adjusted Quote In" in output
                else np.nan,
                "output": output,
            }
        )
        quotes.append(result)

uniswap_df = pd.DataFrame(quotes)
uniswap_df["quote_price"] = uniswap_df["raw_quote"] / uniswap_df["trade_size"]
uniswap_df["gas_adjusted_price"] = (
    uniswap_df["raw_quote"]
    + uniswap_df["gas_quote"] * (-1) ** uniswap_df["buy_risk_asset"]
) / uniswap_df["trade_size"]

# turn timestamp into int
uniswap_df["timestamp"] = uniswap_df["timestamp"].astype(int)
indices = ["timestamp", "risk_asset"]
#  get highest quote_price for each risk_asset at each timestamp if buy_risk_asset is True
best_buy = (
    uniswap_df[uniswap_df["buy_risk_asset"]].groupby(indices).max()["quote_price"]
)

#  get lowest quote_price for each risk_asset at each timestamp if buy_risk_asset is False
best_sell = (
    uniswap_df[~uniswap_df["buy_risk_asset"]].groupby(indices).min()["quote_price"]
)

uniswap_df["best_buy"] = uniswap_df.set_index(indices).index.map(best_buy)
uniswap_df["best_sell"] = uniswap_df.set_index(indices).index.map(best_sell)
uniswap_df["mid_price"] = (uniswap_df["best_buy"] + uniswap_df["best_sell"]) / 2

# "slippage_starting_price" is the highest of the two: "mid_price" and 'best_buy' if buy_risk_asset is True
#  and the lowest of the tow: "mid_price" and 'best_sell' if buy_risk_asset is False
uniswap_df["slippage_starting_price"] = np.where(
    uniswap_df["buy_risk_asset"],
    np.maximum(uniswap_df["mid_price"], uniswap_df["best_buy"]),
    np.minimum(uniswap_df["mid_price"], uniswap_df["best_sell"]),
)


uniswap_df["slippage_unadjusted"] = (
    (uniswap_df["quote_price"] - uniswap_df["mid_price"])
    / uniswap_df["mid_price"]
    * (-1) ** uniswap_df["buy_risk_asset"]
)

uniswap_df["slippage"] = (
    (uniswap_df["quote_price"] - uniswap_df["slippage_starting_price"])
    / uniswap_df["slippage_starting_price"]
    * (-1) ** uniswap_df["buy_risk_asset"]
)
