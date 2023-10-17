import numpy as np
from perp.constants import FIGURE_PATH, SYMBOL_LIST
from scripts.simluate import c_perp_position_change

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

leverage_color = ["C0", "C1", "C2"]

for symbol in SYMBOL_LIST:
    plt.rcParams.update({"font.size": 20})
    # initiate the plot with two subplots that share the same x-axis, for ax2, we need 2 y-axis, one on the left and one on the right
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    # ax3 = ax2.twinx()
    # ax3.set_ylim([-2, 52])

    ax.axhline(y=1, color="k", linestyle="-", linewidth=0.4)
    # yaxis range from 0.5 to 5.5
    ax.set_ylim([0.3, 5.7])
    # ax2.set_ylim([-0.24, 0.24])
    # set ylabel
    ax.set_ylabel("Health factor")
    ax2.set_ylabel("Price in USD")
    for i, leverage in enumerate([2, 5, 20]):
        for long_risk in [True, False]:
            df = c_perp_position_change(
                risk_asset=symbol,
                usd_asset="DAI",
                long_risk=long_risk,
                leverage_multiplier=leverage,
            )
            health_factors = df["cperp_health"]
            # .rolling(7 * 3, center=True)
            # .mean()

            # plot with different line style for long and short
            ax.plot(
                health_factors.index,
                health_factors,
                label=f"{leverage}x {'long' if long_risk else 'short'}",
                linewidth=1.5,
                color=leverage_color[i],
                alpha=0.9,
                linestyle="-" if long_risk else "--",
            )

            if long_risk:
                # plot price log return on ax2
                ax2.plot(
                    df.index,
                    df["price"],
                    # .apply(np.log).diff(),
                    linewidth=0.5,
                    alpha=0.9,
                    color="k",
                )
            # plot interest rate on ax2 on the y-axis on the right

            # ax3.plot(
            #     df.index,
            #     df["liquidityRate"] * 100,
            #     linewidth=1.5,
            #     alpha=0.9,
            #     color="C4",
            #     linestyle="-" if long_risk else "--",
            # )

            # ax3.plot(
            #     df.index,
            #     df["variableBorrowRate"] * 100,
            #     linewidth=1.5,
            #     alpha=0.9,
            #     color="C5",
            #     linestyle="-" if long_risk else "--",
            # )

    # Add a legend on top of the plot outside the frame without border
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.6), frameon=False)

    # Format the x-axis to show only the month without year
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # save to file
    plt.savefig(
        FIGURE_PATH / f"health_{symbol}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    # Show the plot
    plt.show()
    plt.close()
