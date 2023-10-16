from perp.constants import FIGURE_PATH, SYMBOL_LIST
from scripts.simluate import c_perp_position_change

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

leverage_color = ["C0", "C1", "C2", "C3"]

for symbol in SYMBOL_LIST:
    plt.rcParams.update({"font.size": 20})
    # initiate the plot with dark gray background in the plot area
    fig, ax = plt.subplots()
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.4)
    for i, leverage in enumerate([2, 5, 20]):
        for long_risk in [True, False]:
            fr = (
                (
                    c_perp_position_change(
                        risk_asset=symbol,
                        usd_asset="DAI",
                        long_risk=long_risk,
                        leverage_multiplier=leverage,
                    )["Contango"]
                    * 100
                )
                .rolling(7 * 3, center=True)
                .mean()
            )
            # plot with different line style for long and short
            ax.plot(
                fr.index,
                fr,
                label=f"{leverage}x {'long' if long_risk else 'short'}",
                linewidth=1.5,
                color=leverage_color[i],
                alpha=0.9,
                linestyle="-" if long_risk else "--",
            )

    # Add a legend on top of the plot outside the frame without border
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.3), frameon=False)

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
