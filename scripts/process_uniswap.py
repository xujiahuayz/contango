import gzip
import json
from perp.constants import (
    UNISWAP_PATH,
    SYMBOL_LIST,
    USD_STABLECOIN,
    TOKEN_DICT,
    TRADE_SIZE_LIST,
)
import pandas as pd

quotes = []
with gzip.open(UNISWAP_PATH, "rt") as f:
    for line in f:
        result = json.loads(line)
        # amount = int(round(float(result["allQuotes"][1]["quote"]["amountDecimals"])))
        quote_adjusted_amount = result["allQuotes"][1]["quote"][
            "quoteGasAndPortionAdjustedDecimals"
        ]
        quote_amount = result["allQuotes"][1]["quote"]["quoteDecimals"]
        this_quote = {
            "buy_risk_asset": result["buy_risk_asset"],
            "trade_size": result["trade_size"],
            "risk_asset": result["risk_asset"],
            "quote_adjusted_amount": float(quote_adjusted_amount),
            # "quote_amount": float(quote_amount),
        }
        quotes.append(this_quote)

uniswap_df = pd.DataFrame(quotes)
uniswap_df["quote_price"] = (
    uniswap_df["quote_adjusted_amount"] / uniswap_df["trade_size"]
)
# get highest quote_price for each risk_asset if buy_risk_asset is True
best_buy = (
    uniswap_df[uniswap_df["buy_risk_asset"]].groupby("risk_asset").max()["quote_price"]
)
# get lowest quote_price for each risk_asset if buy_risk_asset is False
best_sell = (
    uniswap_df[~uniswap_df["buy_risk_asset"]].groupby("risk_asset").min()["quote_price"]
)
# get mid price
mid_price = (best_buy + best_sell) / 2
uniswap_df["mid_price"] = uniswap_df["risk_asset"].map(mid_price)

uniswap_df["slippage"] = (
    (uniswap_df["quote_price"] - uniswap_df["mid_price"])
    / uniswap_df["mid_price"]
    * (-1) ** uniswap_df["buy_risk_asset"]
)
