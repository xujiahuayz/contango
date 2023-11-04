import pandas as pd

from perp.constants import SYMBOL_LIST, TABLE_PATH
from scripts.process_kaiko import kaiko_df
from scripts.process_router import uniswap_df


kaiko_df["timestamp"] = (kaiko_df["poll_timestamp"] / 1000).astype(int)
# restructure kaiko_snapshot such that 'ask_slippage' and 'bid_slippage' are in the same column 'slippage' with 'buy_risk_asset' column, and 'exchange' values become column names
slippage_df = pd.melt(
    kaiko_df,
    id_vars=["timestamp", "exchange", "risk_asset", "trade_size"],
    value_vars=["ask_slippage", "bid_slippage"],
    var_name="buy_risk_asset",
    value_name="slippage",
)

# Map 'ask_slippage' and 'bid_slippage' to 'buy' and 'sell'
slippage_df["buy_risk_asset"] = slippage_df["buy_risk_asset"].map(
    {"ask_slippage": True, "bid_slippage": False}
)

# replace Okex to OKX in column 'exchange'
slippage_df["exchange"] = slippage_df["exchange"].replace("OkEX", "OKX")

uniswap_df["exchange"] = "Uniswap"
# concatenate kaiko_df and uniswap_df
# change WBTC to BTC and WETH to ETH
uniswap_df["risk_asset"] = uniswap_df["risk_asset"].replace(
    {"WBTC": "BTC", "WETH": "ETH"}
)

slippage_df = pd.concat(
    [slippage_df, uniswap_df[slippage_df.columns]], ignore_index=True
)

if __name__ == "__main__":
    # take 3 snapshots of slippage_df, lowest timestamp, median timestamp, highest timestamp
    all_times = slippage_df["timestamp"].unique()
    for time in [all_times.min(), all_times[len(all_times) // 2], all_times.max()]:
        # Pivot the table so that 'exchange' values become column names
        indices = ["risk_asset", "buy_risk_asset", "trade_size"]

        kaiko_pivoted = slippage_df[slippage_df["timestamp"] == time].pivot_table(
            index=indices,
            columns="exchange",
            values="slippage",
        )

        # merge kaiko_pivoted and uniswap_df
        result = kaiko_pivoted.sort_index(
            level=["buy_risk_asset", "trade_size"], ascending=[False, True]
        )

        # make index trade_size with thousands separator and without decimal places
        result.index = pd.MultiIndex.from_tuples(
            [
                (risk_asset, long_risk, "{:,}".format(trade_size).replace(".0", ""))
                for risk_asset, long_risk, trade_size in result.index
            ],
            names=["risk asset", "long risk", "trade size ($)"],
        )

        # sort risk_asset by SYMBOL_LIST and display in percentage
        result = result.loc[SYMBOL_LIST] * 100

        # save result to latex with exactly 3 decimal places, merge cells where possible
        # replace NaN with -
        result.to_latex(
            TABLE_PATH / f"slippage{time}.tex",
            float_format="{:0.3f}".format,
            na_rep="---",
            multirow=True,
            multicolumn=True,
        )
