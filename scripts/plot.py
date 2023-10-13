from perp.constants import DATA_PATH
from scripts.simluate import c_perp_position_change

import matplotlib.pyplot as plt

coinglass_aave_df = c_perp_position_change(
    risk_asset="ETH", usd_asset="USDC", long_risk=True, leverage_multiplier=10
)


# plt.plot(coinglass_aave_df["dYdX"], label="dYdX")

# plt.plot(coinglass_aave_df["OKX"], label="OKX")
# plt.plot(coinglass_aave_df["Binance"], label="Binance")

# plt.plot(coinglass_aave_df["Contango"], label="Contango")
# plt.legend()


plt.plot(coinglass_aave_df["cperp_health"])
# draw a horizontal line at 1
plt.axhline(y=1, color="r", linestyle="-")

coinglass_aave_df.to_excel(DATA_PATH / "coinglass_aave_df.xlsx")
