import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from perp.constants import CONTANGO_NAME, FIGURE_PATH, SYMBOL_LIST, USD_STABLECOIN
from scripts.process_graph import interpolate_df
from scripts.simluate import c_perp_position_change

leverage_color = ["C0", "C1", "C2", "C3"]
plt.rcParams.update({"font.size": 20})

rates_cols = ["variableBorrowRate", "liquidityRate", "utilizationRate"]

stable_util = interpolate_df(USD_STABLECOIN, rates_cols)

for symbol in SYMBOL_LIST:
    # initiate the plot with 3 subplots that share the same x-axis
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    risk_asset = ("W" if symbol in ["BTC", "ETH"] else "") + symbol
    risk_util = interpolate_df(risk_asset, rates_cols)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.4)
    for i, leverage in enumerate([2, 5, 20]):
        for long_risk in [True, False]:
            df = c_perp_position_change(
                risk_asset=symbol,
                usd_asset=USD_STABLECOIN,
                long_risk=long_risk,
                leverage_multiplier=leverage,
            )

            fr = (df[CONTANGO_NAME] * 100).rolling(7 * 3, center=True).mean()

            ax.set_xlim([df.index[0], df.index[-1]])

            # plot with different line style for long and short
            ax.plot(
                df.index,
                fr,
                label=f"{leverage}x {'long' if long_risk else 'short'}",
                linewidth=1.5,
                color=leverage_color[i],
                alpha=0.9,
                linestyle="-" if long_risk else "--",
            )

    for plot_risk in [True, False]:
        ax_to_use = ax2 if plot_risk else ax3
        df_to_use = risk_util if plot_risk else stable_util
        ax_to_use.plot(
            df_to_use.index,
            df_to_use["utilizationRate"],
            linewidth=1.5,
            alpha=0.9,
            color="C4",
            label=None if plot_risk else "utilization rate",
        )
        ax_to_use.plot(
            df_to_use.index,
            df_to_use["variableBorrowRate"],
            linewidth=1.5,
            alpha=0.9,
            color="C5",
            label=None if plot_risk else "borrow APY",
            linestyle="--",
        )
        ax_to_use.plot(
            df_to_use.index,
            df_to_use["liquidityRate"],
            linewidth=1.5,
            alpha=0.9,
            color="C6",
            label=None if plot_risk else "lend APY",
            linestyle=":",
        )

    # set ax2 y label
    ax2.set_ylabel("risk \n asset")
    ax3.set_ylabel("stable \n coin")

    ax2.set_ylim([-0.1, 1.2])
    ax3.set_ylim([-0.1, 1.2])

    # Add a legend on top of the plot outside the frame without border
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 2), frameon=False)
    ax3.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -1.1), frameon=False)
    # Format the x-axis to show only the month without year
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # save to file
    plt.savefig(
        FIGURE_PATH / f"leverage_{symbol}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    # Show the plot
    plt.show()
    plt.close()
