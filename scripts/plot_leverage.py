from perp.constants import FIGURE_PATH, SYMBOL_LIST
from scripts.simluate import c_perp_position_change

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

leverage_color = ["C0", "C1", "C2", "C3"]
plt.rcParams.update({"font.size": 20})


for symbol in SYMBOL_LIST:
    # initiate the plot with 3 subplots that share the same x-axis
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.4)
    for i, leverage in enumerate([2, 5, 20]):
        for long_risk in [True, False]:
            df = c_perp_position_change(
                risk_asset=symbol,
                usd_asset="DAI",
                long_risk=long_risk,
                leverage_multiplier=leverage,
            )

            fr = (df["Contango"] * 100).rolling(7 * 3, center=True).mean()

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

            ax_to_plot = ax2 if long_risk else ax3

            ax_to_plot.plot(
                df.index,
                df["liquidityRate"],
                linewidth=1.5,
                alpha=0.9,
                color="C4",
            )

            ax_to_plot.plot(
                df.index,
                df["variableBorrowRate"],
                linewidth=1.5,
                alpha=0.9,
                color="C5",
            )
            # set y-axis range
            ax_to_plot.set_ylim([0, 1.4])
            # log y-axis
            # ax_to_plot.set_yscale("log")

    # Add a legend on top of the plot outside the frame without border
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 2), frameon=False)

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
