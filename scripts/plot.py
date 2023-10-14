from perp.constants import FIGURE_PATH, SYMBOL_LIST
from scripts.simluate import c_perp_position_change

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

products = ["dYdX", "Huobi", "Binance", "Contango"]

for symbol in SYMBOL_LIST:
    coinglass_aave_df = c_perp_position_change(
        risk_asset=symbol, usd_asset="USDC", long_risk=True, leverage_multiplier=5
    )

    coinglass_aave_df = coinglass_aave_df[products] * 100

    # plot with large font size

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots()

    # Plot the data
    for product in products:
        ax.plot(coinglass_aave_df.index, coinglass_aave_df[product], label=product)

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
