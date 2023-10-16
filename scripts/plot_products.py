from perp.constants import FIGURE_PATH, SYMBOL_LIST
from scripts.simluate import c_perp_position_change

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

products = ["dYdX", "Huobi", "Binance", "Contango"]
# set plot color with python default color
product_color = ["C0", "C1", "C2", "C3"]


# product_color = ["#68228B", "#FF3030", "#B8860B", "#00CED1"]

for symbol in SYMBOL_LIST:
    coinglass_aave_df = c_perp_position_change(
        risk_asset=symbol, usd_asset="DAI", long_risk=True, leverage_multiplier=5
    )

    coinglass_aave_df = coinglass_aave_df[products] * 100

    plt.rcParams.update({"font.size": 20})
    # initiate the plot with dark gray background in the plot area
    fig, ax = plt.subplots()
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.4)
    for i, product in enumerate(products):
        # add alpha with the same color
        ax.plot(
            coinglass_aave_df.index,
            coinglass_aave_df[product],
            color=product_color[i],
            alpha=0.45,
            linewidth=0.45,
            marker=".",
            markersize=1,
        )
        # add 7 days rolling average with the same color
        ax.plot(
            coinglass_aave_df.index,
            coinglass_aave_df[product].rolling(7 * 3, center=True).mean(),
            color=product_color[i],
            label=product,
            linewidth=1.5,
            alpha=0.9,
        )

    # Add a legend on top of the plot outside the frame without border
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.3), frameon=False)

    # Format the x-axis to show only the month without year
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # save to file
    plt.savefig(
        FIGURE_PATH / f"funding_rates_{symbol}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    # Show the plot
    plt.show()
    plt.close()
