from matplotlib import pyplot as plt
import pandas as pd

from perp.constants import FIGURE_PATH, KAIKO_EXCHANGES
from scripts.process_kaiko import kaiko_df
from matplotlib import dates as mdates

colors = ["C0", "C1", "C2", "C3"]
# take the smallest of as_slippage and bid_slippage, as well as the largest of ask_slippage and bid_slippage, as y-axis limits
y_min = kaiko_df[["ask_slippage", "bid_slippage"]].min().min() / 2
y_max = kaiko_df[["ask_slippage", "bid_slippage"]].max().max() * 5

# convert poll_timestamp 1696003200000 to datetime
kaiko_df["poll_timestamp"] = pd.to_datetime(
    kaiko_df["poll_timestamp"], unit="ms", origin="unix"
)

# for each risk_asset, exchange,
for risk_asset in kaiko_df["risk_asset"].unique():
    # set subplots with one subplot per exchange sharing the same y-axis
    fig, axs = plt.subplots(
        1,
        len(KAIKO_EXCHANGES),
        sharey=True,
        figsize=(len(KAIKO_EXCHANGES) * 3, 3),
    )
    for j, exchange in enumerate(KAIKO_EXCHANGES):
        df = kaiko_df[
            (kaiko_df["risk_asset"] == risk_asset) & (kaiko_df["exchange"] == exchange)
        ]
        df = df.set_index("poll_timestamp")
        # plot the slippage time series with each trade_size as a line
        for i, trade_size in enumerate(df["trade_size"].unique()):
            df_new = df[df["trade_size"] == trade_size]
            # ask_slippage solid line, bid_slippage dashed line
            # only label lines for the first exchange
            axs[j].plot(
                df_new["ask_slippage"],
                label=f"long {int(trade_size)}" if j == 0 else None,
                marker="o",
                color=colors[i],
                alpha=0.5,
                markersize=4,
            )
            axs[j].plot(
                df_new["bid_slippage"],
                label=f"short {int(trade_size)}" if j == 0 else None,
                linestyle="--",
                marker="x",
                color=colors[i],
                alpha=0.8,
                markersize=4,
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
        axs[j].set_title(exchange.replace("OkEX", "OKX"))
    # add a common legend horizontally centered above the subplots with one row
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
    # save to pdf
    fig.savefig(FIGURE_PATH / f"slippage_{risk_asset}.pdf", bbox_inches="tight")
