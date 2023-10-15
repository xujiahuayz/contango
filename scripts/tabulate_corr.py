from matplotlib.colors import LinearSegmentedColormap

from perp.constants import PRODUCT_LIST, SYMBOL_LIST, TABLE_PATH
from scripts.simluate import c_perp_position_change

# define for background_gradient the cmap argument (type Colormap) that is light red for 1 and light blue for -1 and white for 0
colors = [(0.5, 0.5, 1), (1, 1, 1), (1, 0.5, 0.5)]  # Light red, white, light blue
color_map = LinearSegmentedColormap.from_list("custom", colors)

for symbol in SYMBOL_LIST:
    coinglass_aave_df = c_perp_position_change(
        risk_asset=symbol, usd_asset="USDC", long_risk=True, leverage_multiplier=5
    )
    sum_df = (coinglass_aave_df[PRODUCT_LIST].iloc[1:-1]) * 100  # convert to percentage

    corr = sum_df.corr()
    # get significance level
    corr_significance = sum_df.corr(method="spearman")

    corr_with_color = corr.style.background_gradient(
        cmap=color_map, vmin=-1, vmax=1
    ).format(
        "{:.2f}"
    )  # format to 2 decimals

    # save corr_with_color to latex and preserve the color
    corr_latex = corr_with_color.to_latex(
        convert_css=True, column_format="@{}l*{9}{R{9.8mm}}@{}"
    ).replace("Contango", "{\\bf Contango}")
    with open(TABLE_PATH / f"corr_{symbol}.tex", "w") as f:
        f.write(corr_latex)
