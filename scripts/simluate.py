from matplotlib import pyplot as plt

from perp.env import DefiEnv, PlfPool, User, cPerp, cPool
from perp.utils import PriceDict

env = DefiEnv(
    prices=PriceDict({"dai": 1.0, "eth": 1000.0}),
)
market_user = User(
    env=env,
    name="MarketUser",
    funds_available={"dai": 999_999_999_999, "eth": 999_999_999_999},
)

INITIAL_FUNDS = 1_000_000
charlie = User(env=env, name="Charlie", funds_available={"dai": INITIAL_FUNDS})
plf_eth = PlfPool(
    env=env,
    initiator=market_user,
    initial_starting_funds=1_000_000,
    initial_borrowing_funds=0,
    asset_name="eth",
    collateral_factor=0.7,
    liquidation_threshold=0.8,
    flashloan_fee=0,
)
plf_dai = PlfPool(
    env=env,
    initiator=market_user,
    initial_starting_funds=1_000_000,
    initial_borrowing_funds=0,
    asset_name="dai",
    collateral_factor=0.7,
    liquidation_threshold=0.8,
    flashloan_fee=0,
)
c_eth = cPool(env=env, asset_name="eth", funds_available=1_000_000, c_ratio=0.2)
c_dai = cPool(env=env, asset_name="dai", funds_available=1_000_000, c_ratio=0.2)

cperp1 = cPerp(
    env=env,
    position_name="cperp1",
    initiator_name="Charlie",
    init_asset="dai",
    target_asset="eth",
    target_quantity=3,
    target_collateral_factor=0.8,
    trading_slippage=0,
)
# pre-determined market conditions
eth_supply_apy = [0.1, 0.2, 0.3, 0.4, 0.5]
eth_borrow_apy = [0.1, 0.2, 0.3, 0.4, 0.5]
dai_supply_apy = [0.1, 0.2, 0.3, 0.4, 0.5]
dai_borrow_apy = [0.1, 0.2, 0.3, 0.4, 0.5]
eth_price = [1000, 2000, 3000, 4000, 500]


health_series = []
pnl_series = []
for i in range(5):
    plf_eth.supply_apy = eth_supply_apy[i]
    plf_eth.borrow_apy = eth_borrow_apy[i]
    plf_dai.supply_apy = dai_supply_apy[i]
    plf_dai.borrow_apy = dai_borrow_apy[i]
    env.prices["eth"] = eth_price[i]
    env.accrue_interest()
    health_series.append(cperp1.plf_health)
    pnl_series.append(charlie.wealth - INITIAL_FUNDS)

plt.plot(health_series)
plt.plot(pnl_series)
