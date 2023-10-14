from perp.constants import SYMBOL_LIST, TABLE_PATH
from scripts.simluate import c_perp_position_change

for symbol in SYMBOL_LIST:
    coinglass_aave_df = c_perp_position_change(
        risk_asset=symbol, usd_asset="USDC", long_risk=True, leverage_multiplier=5
    )
    sum_df = (
        (
            (
                coinglass_aave_df[
                    [
                        # "Bitmex",
                        "Binance",
                        "Bybit",
                        "OKX",
                        "Huobi",
                        "Gate",
                        "Bitget",
                        "dYdX",
                        "CoinEx",
                        # "BingX",
                        "Contango",
                    ]
                ].iloc[1:-1]
            )
            * 100
        )  # convert to percentage
        .describe()
        .T.sort_values("std", ascending=True)[
            ["std", "mean", "min", "25%", "50%", "75%", "max", "count"]
        ]
    )

    sum_df["count"] = sum_df["count"].astype(int)
    # reformat "mean", "min", "25%", "50%", "75%", "max", as $0.000$
    for col in ["mean", "min", "25%", "50%", "75%", "max"]:
        sum_df[col] = sum_df[col].apply(lambda x: f"${x:.4f}$")
        # if colname has %, rename to xx percentile
        if "%" in col:
            sum_df.rename(columns={col: f"{col[:-1]}\%"}, inplace=True)

    max_std = sum_df["std"].max().round(4)
    sum_df["std"] = sum_df["std"].apply(lambda x: f"\databar{{{x:.4f}}}")

    # replace contango with {\bf Contango}
    sum_df.rename(index={"Contango": "{\\bf Contango}"}, inplace=True)

    # turn to latex table
    latex_table = f"\\renewcommand{{\\maxnum}}{{{max_std}}}\n" + sum_df.to_latex(
        column_format="@{}l@{\hspace{3mm}}r*{6}{R{10mm}}r@{}", escape=False
    )
    #  save to file
    with open(TABLE_PATH / f"funding_rates_{symbol}.tex", "w") as f:
        f.write(latex_table)
