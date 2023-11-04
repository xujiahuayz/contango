import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from perp.constants import FIGURE_PATH
from scripts.tabulate_slippage import slippage_df

colors = ["C0", "C1", "C2", "C3"]

x_column = "time"

slippage_df[x_column] = pd.to_datetime(
    slippage_df["timestamp"], unit="s", origin="unix"
)


# set slippage to be nan if slippage is 0
slippage_df["slippage"] = slippage_df["slippage"].replace(0, float("nan"))

# take the smallest of as_slippage and bid_slippage, as well as the largest of ask_slippage and bid_slippage, as y-axis limits
y_min = slippage_df["slippage"].min() / 1.05
y_max = slippage_df["slippage"].max() * 1.05

exchange_list = slippage_df["exchange"].unique()

# for each risk_asset, exchange,
for risk_asset in slippage_df["risk_asset"].unique():
    # set subplots with one subplot per exchange sharing the same y-axis
    fig, axs = plt.subplots(
        1,
        len(exchange_list),
        sharey=True,
        figsize=(len(exchange_list) * 3, 3),
    )
    for j, exchange in enumerate(exchange_list):
        df = slippage_df[
            (slippage_df["risk_asset"] == risk_asset)
            & (slippage_df["exchange"] == exchange)
        ]
        df = df.set_index(x_column)
        # plot the slippage time series with each trade_size as a line
        for i, trade_size in enumerate(df["trade_size"].unique()):
            df_long_short = df[df["trade_size"] == trade_size]
            # ask_slippage solid line, bid_slippage dashed line
            # only label lines for the first exchange
            df_new = df_long_short[df_long_short["buy_risk_asset"]]
            axs[j].plot(
                df_new["slippage"],
                label=f"long {int(trade_size)}" if j == 0 else None,
                marker="o",
                color=colors[i],
                alpha=0.5,
                markersize=3,
            )
            df_new = df_long_short[~df_long_short["buy_risk_asset"]]
            axs[j].plot(
                df_new["slippage"],
                label=f"short {int(trade_size)}" if j == 0 else None,
                linestyle="--",
                marker="x",
                color=colors[i],
                alpha=0.8,
                markersize=3,
            )
            # log y-axis
            axs[j].set_ylim(y_min, y_max)
            axs[j].set_yscale("log")
            # only show date on x axis without time or year
            axs[j].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            # Set locator for x-axis to show fewer dates
            locator = mdates.DayLocator(interval=7)  # Show date every 3 days
            axs[j].xaxis.set_major_locator(locator)
        # change OkEX to OKX
        # axs[j].set_title(exchange.replace("OkEX", "OKX"))
    # add a common legend horizontally centered above the subplots with one row
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
    # save to pdf
    fig.savefig(FIGURE_PATH / f"slippage_{risk_asset}.pdf", bbox_inches="tight")
