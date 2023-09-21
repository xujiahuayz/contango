"""
Environment for the DeFi simulation.
"""


from __future__ import annotations

import numpy as np


class DefiEnv:
    """
    DeFi environment containing protocols and users.
    """

    def __init__(
        self,
        users: dict[str, User] | None = None,
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

        self.users = users
        self.amm_pools = amm_pools
        self.plf_pools = plf_pools
        self.c_pools = c_pools


class PlfPool:
    def __init__(self, env: DefiEnv, asset_name: str, collateral_factor: float):
        assert 0 <= collateral_factor <= 1, "collateral_factor must be between 0 and 1"
        self.env = env
        self.name = asset_name
        self.env.plf_pools[self.name] = self
        self.collateral_factor = collateral_factor

    @property
    def lending_apy(self) -> float:
        return 0.0

    @property
    def borrowing_aoy(self) -> float:
        return 0.0


class cPool:
    def __init__(self, env: DefiEnv, asset_name: str):
        self.env = env
        self.name = asset_name
        self.env.c_pools[self.name] = self


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

    def wealth(self, denominator: str) -> float:
        # TODO: this is not accurate - need to come up with a better price oracle
        return sum(
            [
                self.env.amm_pools["".join(sorted([k, denominator]))].spot_price(
                    k, denominator
                )
                * v
                for k, v in self.funds_available.items()
            ]
        )

    def open_contango(
        self,
        init_asset: str,
        target_asset: str,
        target_quantity: float,
    ):
        """
        Open a contango position.
        """
        if init_asset not in self.funds_available.keys():
            raise Exception("init_asset not in funds_available")
        if target_quantity <= 0:
            raise Exception("target_quantity must be greater than zero")

        amm_pool = self.env.amm_pools["".join(sorted([init_asset, target_asset]))]
        plf_pool_init = self.env.plf_pools[init_asset]

        swap_in_quantity = amm_pool._swap_out(
            asset_out=target_asset, quantity_out=target_quantity, asset_in=init_asset
        )

        init_quantity = swap_in_quantity * (1 - plf_pool_init.collateral_factor)

        if self.funds_available[init_asset] < init_quantity:
            raise Exception("insufficient funds")
        self.funds_available[init_asset] -= init_quantity
        # TODO: to continue here


if __name__ == "__main__":
    pass
