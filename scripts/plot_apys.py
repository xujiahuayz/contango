import numpy as np
from matplotlib import pyplot as plt
from process_graph import reservepara_df

from perp.constants import FIGURE_PATH, SYMBOL_LIST


def inverse_log1p(y: float) -> float:
    return np.exp(y) - 1


for symbol in SYMBOL_LIST:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_xlim([-0.01, 0.57])

    # Move x-axis ticks to the top and remove unnecessary tick labels
    ax1.xaxis.tick_top()
    ax1.tick_params(labelbottom=False, bottom=False, labeltop=True)
    ax2.xaxis.tick_top()
    ax2.tick_params(labelbottom=False, bottom=False, labeltop=False)

    risk_asset = ("W" if symbol in ["BTC", "ETH"] else "") + symbol
    risk_df = reservepara_df[reservepara_df["reserve"] == risk_asset]

    # Log-transform x-axis data
    log_variableBorrowRate = np.log1p(risk_df["variableBorrowRate"])
    log_liquidityRate = np.log1p(risk_df["liquidityRate"])

    # Scatter plots
    ax1.scatter(
        log_variableBorrowRate,
        risk_df["totalCurrentVariableDebt"],
        s=1,
        alpha=0.5,
    )
    ax2.scatter(
        log_liquidityRate,
        risk_df["totalATokenSupply"],
        s=1,
        alpha=0.5,
    )

    # Update x-tick labels after plotting
    for ax in [ax1, ax2]:
        ax.set_xlim([-0.01, 0.57])
        current_ticks = ax.get_xticks()
        ax.set_xticklabels(
            [f"{inverse_log1p(tick):.2f}" for tick in current_ticks if tick >= 0]
        )

    # Setting axis labels
    ax1.set_xlabel("Borrow APY")
    ax1.set_ylabel("Total Borrow")

    ax2.set_xlabel("Deposit APY")
    ax2.set_ylabel("Total Deposit")

    # Optional: Save the figure
    plt.savefig(FIGURE_PATH / f"apys_{risk_asset}.pdf")

    # Show the plot
    plt.show()
