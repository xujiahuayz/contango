from perp.constants import DATA_PATH, SYMBOL_LIST
from scripts.simluate import c_perp_position_change

import matplotlib.pyplot as plt

for symbol in SYMBOL_LIST:
    print(symbol)
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
    print(sum_df)


# plt.plot(coinglass_aave_df["dYdX"], label="dYdX")

# plt.plot(coinglass_aave_df["OKX"], label="OKX")
# plt.plot(coinglass_aave_df["Binance"], label="Binance")

# plt.plot(coinglass_aave_df["Contango"], label="Contango")
# plt.legend()


plt.plot(coinglass_aave_df["cperp_health"])
# draw a horizontal line at 1
plt.axhline(y=1, color="r", linestyle="-")

coinglass_aave_df.to_excel(DATA_PATH / "coinglass_aave_df.xlsx")