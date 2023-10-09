from matplotlib import pyplot as plt

from perp.env import DefiEnv, PlfPool, User, cPerp, cPool
from perp.utils import PriceDict
from scripts.fetch import aave_binance_df

env = DefiEnv(
    prices=PriceDict({"usdc": 1.0, "eth": 1000.0}),
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
    target_quantity=3,
    target_collateral_factor=0.8,
    trading_slippage=0,
)

health_series = []
pnl_series = []
c_perp_value_series = []
# get eth_supply_apy etc from aave_binance_df
for row in aave_binance_df.iterrows():
    health_series.append(cperp1.plf_health)
    pnl_series.append(charlie.wealth - INITIAL_FUNDS)
    c_perp_value_series.append(cperp1.value)
    row_values = row[1]
    plf_eth.supply_apy = row_values["eth_deposit_rate"]
    plf_usdc.borrow_apy = row_values["usdc_borrow_rate"]
    env.prices["eth"] = row_values["price"]
    env.accrue_interest()


plt.plot(c_perp_value_series)

plt.plot(health_series)
plt.plot(pnl_series)
