"""
Environment for the DeFi simulation.
"""


from __future__ import annotations

import logging

from perp.constants import (
    AAVE_TOKEN_PREFIX,
    CONTANGO_TOKEN_PREFIX,
    DEBT_TOKEN_PREFIX,
    INTEREST_TOKEN_PREFIX,
)
from perp.utils import PriceDict


class DefiEnv:
    """
    DeFi environment containing protocols and users.
    """

    def __init__(
        self,
        users: dict[str, User] | None = None,
        prices: PriceDict | None = None,
        plf_pools: dict[str, PlfPool] | None = None,
        c_pools: dict[str, cPool] | None = None,
    ):
        if users is None:
            users = {}

        if plf_pools is None:
            plf_pools = {}
        if c_pools is None:
            c_pools = {}

        if prices is None:
            prices = PriceDict({"dai": 1.0})

        self.prices = prices
        self.users = users
        self.plf_pools = plf_pools
        self.c_pools = c_pools
        self.timestamp: float = 0

    @property
    def prices(self) -> PriceDict:
        return self._prices

    @prices.setter
    def prices(self, value: PriceDict):
        if not isinstance(value, PriceDict):
            raise TypeError("must use PriceDict type")
        self._prices = value


class PlfPool:
    """
    Reference pool (e.g. Aave) for interest rates
    """

    def __init__(
        self,
        env: DefiEnv,
        initiator: User,
        initial_starting_funds: float,
        initial_borrowing_funds: float,
        asset_name: str,
        collateral_factor: float,
        flashloan_fee: float = 0.0,
    ):
        assert 0 <= collateral_factor <= 1, "collateral_factor must be between 0 and 1"
        assert (
            initial_borrowing_funds < initial_starting_funds * collateral_factor
        ), "initial_borrowing_funds must be less than initial_starting_funds * collateral_factor"

        self.env = env
        self.initiator = initiator
        self.asset_name = asset_name
        self.env.plf_pools[self.asset_name] = self
        self.collateral_factor = collateral_factor
        self.flashloan_fee = flashloan_fee

        self.interest_token_name = (
            INTEREST_TOKEN_PREFIX + AAVE_TOKEN_PREFIX + self.asset_name
        )
        self.borrow_token_name = DEBT_TOKEN_PREFIX + AAVE_TOKEN_PREFIX + self.asset_name

        self.initiator.funds_available[self.asset_name] += (
            initial_borrowing_funds - initial_starting_funds
        )

        # actual underlying that's still available, not the interest-bearing tokens
        self.total_available_funds = initial_starting_funds - initial_borrowing_funds

        # add interest-bearing token into initiator's wallet
        self.initiator.funds_available[
            self.interest_token_name
        ] = initial_starting_funds
        self.initiator.funds_available[self.borrow_token_name] = initial_borrowing_funds

        self.user_i_tokens: dict[str, float] = {
            self.initiator.name: initial_starting_funds
        }
        self.user_b_tokens: dict[str, float] = {
            self.initiator.name: initial_borrowing_funds
        }

    @property
    def lending_apy(self) -> float:
        return 0.0

    @property
    def borrowing_apy(self) -> float:
        return 0.0

    @property
    def utilization_ratio(self) -> float:
        if self.total_i_tokens == 0:
            return 0
        util_rate = self.total_b_tokens / self.total_i_tokens
        # TODO: understand utilization ratio
        return max(0, min(util_rate, 0.97))

    @property
    def total_i_tokens(self) -> float:
        return sum(self.user_i_tokens.values())

    @property
    def total_b_tokens(self) -> float:
        return sum(self.user_b_tokens.values())

    @property
    def reserve(self) -> float:
        return self.total_b_tokens + self.total_available_funds - self.total_i_tokens


class cPool:
    """
    methods for c pool
    """

    def __init__(
        self,
        env: DefiEnv,
        asset_name: str,
        funds_available: float = 0.0,
        c_ratio: float = 0.2,
        fee: float = 0.0,
    ):
        self.env = env
        self.asset_name = asset_name
        self.env.c_pools[self.asset_name] = self
        self.funds_available = funds_available
        self.user_i_tokens: dict[str, float] = {}
        self.user_b_tokens: dict[str, float] = {}
        self.c_ratio = c_ratio
        self.fee = fee

        self.interest_token_name = (
            INTEREST_TOKEN_PREFIX + CONTANGO_TOKEN_PREFIX + self.asset_name
        )
        self.borrow_token_name = (
            DEBT_TOKEN_PREFIX + CONTANGO_TOKEN_PREFIX + self.asset_name
        )

    @property
    def total_i_tokens(self) -> float:
        return sum(self.user_i_tokens.values())

    @property
    def total_b_tokens(self) -> float:
        return sum(self.user_b_tokens.values())

    @property
    def reserve(self) -> float:
        return self.total_b_tokens + self.funds_available - self.total_i_tokens

    @property
    def lending_apy(self) -> float:
        """
        use the reference pool for lending apy
        """
        return self.env.plf_pools[self.asset_name].lending_apy

    @property
    def borrowing_apy(self) -> float:
        """
        use the reference pool for borrowing apy
        """
        return self.env.plf_pools[self.asset_name].borrowing_apy


class User:
    def __init__(
        self, env: DefiEnv, name: str, funds_available: dict[str, float] | None = None
    ):
        if funds_available is None:
            funds_available = {"dai": 0.0, "eth": 0.0}
        self.env = env
        self.funds_available = funds_available
        self.name = name
        self.env.users[self.name] = self

    @property
    def wealth(self) -> float:
        user_wealth = sum(
            value * self.env.prices[asset_name]
            for asset_name, value in self.funds_available.items()
        )
        logging.debug(f"{self.name}'s wealth in USD: {user_wealth}")

        return user_wealth

    def open_contango(
        self,
        init_asset: str,
        target_asset: str,
        target_quantity: float,
        target_collateral_factor: float,
        trading_slippage: float = 0.0,  # hard code slippage for now without using AMM
    ):
        """
        Open a contango position.
        """
        if init_asset not in self.funds_available.keys():
            raise Exception("init_asset not in funds_available")
        if target_quantity <= 0:
            raise Exception("target_quantity must be greater than zero")

        plf_pool_init = self.env.plf_pools[init_asset]
        plf_pool_target = self.env.plf_pools[target_asset]
        c_pool_init = self.env.c_pools[init_asset]
        c_pool_target = self.env.c_pools[target_asset]

        plf_pool_target.user_i_tokens.setdefault(self.name, 0)
        plf_pool_init.user_b_tokens.setdefault(self.name, 0)
        c_pool_init.user_b_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf_pool_target.interest_token_name, 0)
        self.funds_available.setdefault(plf_pool_init.borrow_token_name, 0)
        self.funds_available.setdefault(c_pool_init.borrow_token_name, 0)

        init_quantity = (
            self.env.prices[target_asset]
            * target_quantity
            / self.env.prices[init_asset]
        )

        # Begin with $(1-\theta^0)P_0$ DAI, take the amount out of user's wallet
        self.funds_available[init_asset] -= (
            1 - target_collateral_factor
        ) * init_quantity

        # Borrow Cθ 0P0 DAI from Contango, take the amount out of contango pool
        borrow_amount_contango = (
            c_pool_init.c_ratio * target_collateral_factor * init_quantity
        )

        c_pool_init.funds_available -= borrow_amount_contango
        c_pool_init.user_b_tokens[self.name] += borrow_amount_contango
        self.funds_available[c_pool_init.borrow_token_name] += borrow_amount_contango

        # Get (1 −C)θ 0P0 DAI using flashloan (to be paid back within one block)
        flashloan_amount = (
            (1 - c_pool_init.c_ratio) * target_collateral_factor * init_quantity
        )

        plf_pool_init.total_available_funds -= flashloan_amount

        # Put together $(1-\theta^0)P_0$, $C\theta^0P_0$ , and $(1 - C)\theta^{0}P_0$ to swap in total $P_0$ DAI for $(1-\epsilon)$ ETH, where $\epsilon$ is the effect of price movement and slippage (can be positive or negative)
        # Deposit swapped $(1-\epsilon)(1-f^C)$ ETH as collateral on Aave and start earning interest according to $(1-\epsilon)(1-f^C)e^{r^c t}$
        deposit_amount = (
            (1 - trading_slippage) * (1 - c_pool_target.fee) * target_quantity
        )

        plf_pool_target.total_available_funds += deposit_amount
        plf_pool_target.user_i_tokens[self.name] += deposit_amount
        self.funds_available[plf_pool_target.interest_token_name] += deposit_amount

        # Borrow $(1 - C)\theta^{0}(1+f^F)P_0$ DAI against the collateral on Aave
        borrow_amount_aave = flashloan_amount * (1 + plf_pool_init.flashloan_fee)

        # update liquidity pool
        plf_pool_init.total_available_funds -= borrow_amount_aave
        # update b tokens of the user in the pool registry
        plf_pool_init.user_b_tokens[self.name] += borrow_amount_aave

        # matching balance in user's account to pool registry record
        self.funds_available[
            plf_pool_init.borrow_token_name
        ] = plf_pool_init.user_b_tokens[self.name]

        # repay flashloan
        plf_pool_init.total_available_funds += borrow_amount_aave


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    env = DefiEnv(
        prices=PriceDict({"dai": 1.0, "eth": 1000.0}),
    )
    market_user = User(
        env=env,
        name="MarketUser",
        funds_available={"dai": 999_999_999_999, "eth": 999_999_999_999},
    )
    charlie = User(env=env, name="Charlie", funds_available={"dai": 1_000_000})
    plf_eth = PlfPool(
        env=env,
        initiator=market_user,
        initial_starting_funds=1_000_000,
        initial_borrowing_funds=0,
        asset_name="eth",
        collateral_factor=0.7,
        flashloan_fee=0,
    )
    plf_dai = PlfPool(
        env=env,
        initiator=market_user,
        initial_starting_funds=1_000_000,
        initial_borrowing_funds=0,
        asset_name="dai",
        collateral_factor=0.7,
        flashloan_fee=0,
    )
    c_eth = cPool(env=env, asset_name="eth", funds_available=1_000_000, c_ratio=0.2)
    c_dai = cPool(env=env, asset_name="dai", funds_available=1_000_000, c_ratio=0.2)

    charlie.open_contango(
        init_asset="dai",
        target_asset="eth",
        target_quantity=3,
        target_collateral_factor=0.8,
        trading_slippage=0,
    )
    print(charlie.funds_available)
    print(plf_eth.total_available_funds)
    print(plf_eth.user_i_tokens)
    print(plf_dai.user_b_tokens)
    print(c_dai.user_b_tokens)
