import numpy as np
from matplotlib import pyplot as plt

from perp.constants import DATA_PATH
from perp.env import DefiEnv, PlfPool, User, cPerp, cPool
from perp.utils import PriceDict
from scripts.process_coinglass import coinglass_aave_df

env = DefiEnv(
    prices=PriceDict({"usdc": 1.0, "eth": coinglass_aave_df.iloc[0]["price"]}),
)
market_user = User(
    env=env,
    name="MarketUser",
    funds_available={"usdc": 999_999_999_999, "eth": 999_999_999_999},
)

INITIAL_FUNDS = 1_000_000
PERIODIC_EXPONENT = 1 / (3 * 365)
charlie = User(env=env, name="Charlie", funds_available={"usdc": INITIAL_FUNDS})
plf_eth = PlfPool(
    env=env,
    initiator=market_user,
    initial_starting_funds=1_000_000,
    initial_borrowing_funds=0,
    asset_name="eth",
    collateral_factor=0.7,
    liquidation_threshold=0.8,
    flashloan_fee=0,
    periodic_exponent=PERIODIC_EXPONENT,
)
plf_usdc = PlfPool(
    env=env,
    initiator=market_user,
    initial_starting_funds=1_000_000,
    initial_borrowing_funds=0,
    asset_name="usdc",
    collateral_factor=0.7,
    liquidation_threshold=0.8,
    flashloan_fee=0,
    periodic_exponent=PERIODIC_EXPONENT,
)
c_eth = cPool(
    env=env,
    asset_name="eth",
    funds_available=1_000_000,
    c_ratio=0.2,
    periodic_exponent=PERIODIC_EXPONENT,
)
c_usdc = cPool(
    env=env,
    asset_name="usdc",
    funds_available=1_000_000,
    c_ratio=0.2,
    periodic_exponent=PERIODIC_EXPONENT,
)

cperp1 = cPerp(
    env=env,
    position_name="cperp1",
    initiator_name="Charlie",
    init_asset="usdc",
    target_asset="eth",
    target_quantity=1,
    target_collateral_factor=0.8,
    trading_slippage=0,
)


# initiate a new float column in coinglass_aave_df
coinglass_aave_df["cperp_health"] = np.nan
coinglass_aave_df["cperp_pnl"] = np.nan
coinglass_aave_df["cperp_value"] = np.nan
coinglass_aave_df["cperp_principal"] = np.nan

# get index and row values in coinglass_aave_df
for index, row_values in coinglass_aave_df.iterrows():
    plf_eth.supply_apy = row_values["eth_deposit_apy"]
    plf_usdc.borrow_apy = row_values["usdc_borrow_apy"]
    env.prices["eth"] = row_values["price"]
    # assign values to the row
    coinglass_aave_df.at[index, "cperp_health"] = cperp1.plf_health
    coinglass_aave_df.at[index, "cperp_pnl"] = charlie.wealth - INITIAL_FUNDS
    coinglass_aave_df.at[index, "cperp_value"] = cperp1.value
    coinglass_aave_df.at[index, "cperp_principal"] = cperp1.funds_available[
        plf_eth.interest_token_name
    ]

    env.accrue_interest()

# rolling diff of cperp_value
coinglass_aave_df["cperp_value_diff"] = coinglass_aave_df["cperp_value"].diff()

coinglass_aave_df["cperp_principal_value_change"] = coinglass_aave_df[
    "cperp_principal"
].shift(1) * coinglass_aave_df["price"].diff(1)

coinglass_aave_df["cperp_funding_payment"] = (
    coinglass_aave_df["cperp_value_diff"]
    - coinglass_aave_df["cperp_principal_value_change"]
)

coinglass_aave_df["Contango"] = coinglass_aave_df["cperp_funding_payment"] / (
    coinglass_aave_df["price"] * coinglass_aave_df["cperp_principal"]
)


plt.plot(coinglass_aave_df["Binance"], label="Binance")
plt.plot(coinglass_aave_df["OKX"], label="OKX")


plt.plot(coinglass_aave_df["dYdX"], label="dYdX")
plt.plot(coinglass_aave_df["Contango"], label="Contango")
plt.legend()

# save aaave_binance_df to excel
aave_binance_excel = coinglass_aave_df.copy()
aave_binance_excel.index = aave_binance_excel.index.tz_localize(None)
aave_binance_excel.to_excel(DATA_PATH / "coinglass_aave_df.xlsx")
