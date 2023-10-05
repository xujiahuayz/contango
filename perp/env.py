"""
Environment for the DeFi simulation.
"""


from __future__ import annotations
import logging

import numpy as np

from perp.constants import (
    DEBT_TOKEN_PREFIX,
    INTEREST_TOKEN_PREFIX,
    CONTANGO_TOKEN_PREFIX,
    AAVE_TOKEN_PREFIX,
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
        amm_pools: dict[str, AmmPool] | None = None,
        plf_pools: dict[str, PlfPool] | None = None,
        c_pools: dict[str, cPool] | None = None,
    ):
        if users is None:
            users = {}
        if amm_pools is None:
            amm_pools = {}
        if plf_pools is None:
            plf_pools = {}
        if c_pools is None:
            c_pools = {}

        if prices is None:
            prices = PriceDict({"dai": 1.0})

        self.users = users
        self.amm_pools = amm_pools
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


class AmmPool:
    def __init__(self, env: DefiEnv, pool: dict[str, list[float]], fee: float = 0.0):
        # pool: {"asset": [qty, weight]}
        # assert the weights sum to 1
        if np.abs(sum([w[1] for w in pool.values()]) - 1) > 1e-6:
            raise Exception("asset weights do not sum to 1.0")
        self.env = env
        self.pool = pool
        self.fee = fee
        # concatenate the names of the assets in the pool alphabetically
        self.name = "".join(sorted([k for k in self.pool.keys()]))
        self.env.amm_pools[self.name] = self

    @property
    def invariant(self) -> float:
        """
        Compute the invariant by multiplying the pool's reserves raised to their respective weights.
        """
        C = 1
        for _, w in self.pool.items():
            C *= w[0] ** w[1]
        return C

    def _swap_in(self, asset_in: str, quantity_in: float, asset_out: str) -> float:
        """
        Swap `quantity_in` of `asset_in` for `asset_out` theoretically without updating the states
        """
        if asset_in not in self.pool.keys():
            raise Exception("asset_in not in pool")
        if asset_out not in self.pool.keys():
            raise Exception("asset_out not in pool")
        if quantity_in <= 0:
            raise Exception("quantity must be greater than zero")

        new_in_quantity = self.pool[asset_in][0] + quantity_in
        theoretical_new_out_quantity = (
            self.invariant / new_in_quantity ** self.pool[asset_in][1]
        ) ** (1 / self.pool[asset_out][1])
        quantity_out = (self.pool[asset_out][0] - theoretical_new_out_quantity) * (
            1 - self.fee
        )
        return quantity_out

    def _swap_out(self, asset_out: str, quantity_out: float, asset_in: str) -> float:
        """
        Swap `quantity_out` of `asset_out` for `asset_in` theoretically without updating the states
        """
        if asset_in not in self.pool.keys():
            raise Exception("asset_in not in pool")
        if asset_out not in self.pool.keys():
            raise Exception("asset_out not in pool")
        if quantity_out <= 0:
            raise Exception("quantity must be greater than zero")

        theoretical_new_out_quantity = self.pool[asset_out][0] - quantity_out / (
            1 - self.fee
        )
        new_in_quantity = (
            self.invariant / theoretical_new_out_quantity ** self.pool[asset_out][1]
        ) ** (1 / self.pool[asset_in][1])
        return new_in_quantity - self.pool[asset_in][0]

    def swap_in(self, asset_in: str, quantity_in: float, asset_out: str):
        """
        Swap `quantity_in` of `asset_in` for `asset_out` and update the states
        """
        quantity_out = self._swap_in(asset_in, quantity_in, asset_out)
        self.pool[asset_in][0] += quantity_in
        self.pool[asset_out][0] -= quantity_out

    def swap_out(self, asset_out: str, quantity_out: float, asset_in: str):
        """
        Swap out `quantity_out` of `asset_out` with `asset_in` and update the states
        """
        quantity_in = self._swap_out(asset_out, quantity_out, asset_in)
        self.pool[asset_in][0] += quantity_in
        self.pool[asset_out][0] -= quantity_out

    def update_balanced_liquidity(self, asset: str, quantity_change: float):
        old_quantity = self.pool[asset][0]
        self.pool[asset][0] += quantity_change
        multiplier = self.pool[asset][0] / old_quantity
        for k, _ in self.pool.items():
            if k != asset:
                self.pool[k][0] *= multiplier

    def spot_price(self, asset: str, denominator: str) -> float:
        """
        Compute the spot price of `asset` in terms of `denominator`.
        """
        if asset not in self.pool.keys():
            raise Exception("asset not in pool")
        if denominator not in self.pool.keys():
            raise Exception("denominator not in pool")
        return (self.pool[denominator][0] / self.pool[denominator][1]) / (
            self.pool[asset][0] / self.pool[asset][1]
        )


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

    @property
    def existing_borrow_value(self) -> float:
        return sum(
            self.funds_available[plf.borrow_token_name]
            * self.env.prices[plf.asset_name]
            for plf in self.env.plf_pools.values()
        )

    @property
    def existing_supply_value(self) -> float:
        return sum(
            self.funds_available[plf.interest_token_name] * self.env.prices[name]
            for name, plf in self.env.plf_pools.items()
        )

    @property
    def max_borrowable_value(self) -> float:
        return sum(
            self.funds_available[plf.interest_token_name]
            * self.env.prices[plf.asset_name]
            * plf.collateral_factor
            for plf in self.env.plf_pools.values()
        )

    def _borrow_repay(self, amount: float, plf: PlfPool) -> float:
        # set default values for user_b_tokens and funds_available if they don't exist

        plf.user_b_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf.borrow_token_name, 0)
        self.funds_available.setdefault(plf.asset_name, 0)
        self.funds_available.setdefault(plf.interest_token_name, 0)

        if amount >= 0:
            # borrow case
            # will never borrow EVERYTHING - always leave some safety margin
            additional_borrowable_amount = (
                self.max_borrowable_value - self.existing_borrow_value
            ) / self.env.prices[plf.asset_name]
            amount = max(
                min(
                    amount,
                    plf.total_available_funds,
                    additional_borrowable_amount,
                ),
                0,
            )
            if 0 <= amount < 1e-9:  # if amount is too small,
                return 0
        else:
            # repay case
            amount = max(
                amount,
                -plf.user_b_tokens[self.name],
                -self.funds_available[plf.asset_name],
            )

        logging.debug(
            f"borrowing {amount} {plf.borrow_token_name}"
            if amount > 0
            else f"repaying {-amount} {plf.borrow_token_name}"
        )
        # update liquidity pool
        plf.total_available_funds -= amount

        # update b tokens of the user in the pool registry
        plf.user_b_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.borrow_token_name] = plf.user_b_tokens[self.name]

        self.funds_available[plf.asset_name] += amount

        assert plf.total_available_funds >= 0, (
            "total available funds cannot be negative at \n %s" % plf
        )

        return amount

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
        c_pool_init.funds_available -= (
            c_pool_init.c_ratio * target_collateral_factor * init_quantity
        )

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
        self.funds_available[plf_pool_target.borrow_token_name] += deposit_amount

        # Borrow $(1 - C)\theta^{0}(1+f^F)P_0$ DAI against the collateral on Aave
        borrow_amount = flashloan_amount * (1 + plf_pool_init.flashloan_fee)

        # update liquidity pool
        plf_pool_init.total_available_funds -= borrow_amount
        # update b tokens of the user in the pool registry
        plf_pool_init.user_b_tokens[self.name] += borrow_amount

        # matching balance in user's account to pool registry record
        self.funds_available[
            plf_pool_init.borrow_token_name
        ] = plf_pool_init.user_b_tokens[self.name]

        # repay flashloan
        plf_pool_init.total_available_funds += borrow_amount


if __name__ == "__main__":
    pass
