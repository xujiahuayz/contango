import pandas as pd

from perp.constants import SYMBOL_LIST, TABLE_PATH
from scripts.process_kaiko import kaiko_df
from scripts.process_uniswap import uniswap_df

# take kaiko latest poll_timestamp
kaiko_snapshot = kaiko_df[
    kaiko_df["poll_timestamp"] == kaiko_df["poll_timestamp"].max()
].reset_index(drop=True)

# restructure kaiko_snapshot such that 'ask_slippage' and 'bid_slippage' are in the same column 'slippage' with 'buy_risk_asset' column, and 'exchange' values become column names


melted = pd.melt(
    kaiko_snapshot,
    id_vars=["exchange", "risk_asset", "trade_size"],
    value_vars=["ask_slippage", "bid_slippage"],
    var_name="buy_risk_asset",
    value_name="slippage",
)

# Map 'ask_slippage' and 'bid_slippage' to 'buy' and 'sell'
melted["buy_risk_asset"] = melted["buy_risk_asset"].map(
    {"ask_slippage": True, "bid_slippage": False}
)


# Pivot the table so that 'exchange' values become column names
indices = ["risk_asset", "trade_size", "buy_risk_asset"]

kaiko_pivoted = melted.pivot_table(
    index=indices,
    columns="exchange",
    values="slippage",
)

# rename colume Okex to OKX
kaiko_pivoted = kaiko_pivoted.rename(columns={"OkEX": "OKX"})

# change uniswap_df index to ["risk_asset", "trade_size", "buy_risk_asset"]
uniswap_df = uniswap_df.set_index(indices)[["slippage"]]
# change WBTC to BTC and WETH to ETH, and change 'slippage' column name to 'uniswap'
uniswap_df = uniswap_df.rename(index={"WBTC": "BTC", "WETH": "ETH"})
uniswap_df.columns = ["Uniswap"]

# merge kaiko_pivoted and uniswap_df
result = kaiko_pivoted.merge(uniswap_df, how="outer", left_index=True, right_index=True)

# make index trade_size with thousands separator and without decimal places
result.index = result.index.set_levels(
    result.index.levels[1].map(lambda x: "{:,}".format(x).replace(".0", "")),
    level=1,
)


# sort risk_asset by SYMBOL_LIST and display in percentage
result = result.loc[SYMBOL_LIST] * 100
# sort buy_risk_asset by True and False
result = result.sort_index(level="buy_risk_asset", ascending=False)


# rename index
result.index.names = ["risk asset", "trade size ($)", "long risk"]

# save result to latex with exactly 2 decimal places, merge cells where possible
# replace NaN with -
result.to_latex(
    TABLE_PATH / "slippage.tex",
    float_format="{:0.2f}".format,
    na_rep="---",
    multirow=True,
    multicolumn=True,
)
