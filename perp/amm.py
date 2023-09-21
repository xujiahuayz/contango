"""
Amm-based DEX.
"""


class Amm:
    def __init__(self, reserves: list[float], weights: list[float]):
        for qty in reserves:
            self._validate_reserves(qty)

        self.reserves = reserves
        self.initial_reserves = reserves

        self._validate_len_of_reserves_and_weights(reserves, weights)
        self._validate_weights(weights)
        self.weights = weights

    def spot_price(self, asset_in_ix: float, asset_out_ix: float):
        """Gives you the current spot price of asset out denominated
        in asset_in.

        Args:
            asset_in_ix (float): index of the asset in which to denominate the output asset
            asset_out_ix (float): index of the output asset

        Raises:
            Exception: if this function is not implemented by inheriting class
        """
        # todo: these validations have to be enforced in the implementing functions
        # todo: use metaclass
        self._validate_asset_ix(asset_in_ix)
        self._validate_asset_ix(asset_out_ix)
        raise Exception("must be implemented")

    def trade(self, qty_in: float, asset_in_ix: int, asset_out_ix: int):
        raise Exception("must be implemented")

    def _trade(self, qty_in: float, asset_in_ix: int, asset_out_ix: int):
        """Simulates a DEX trade

        Args:
            qty_in (float): Qty of the traded asset. If positive, you are swapping
            this asset in return for asset at asset_ix. Similarly, if this is
            negative, then this is the amount you will get OUT of the DEX and you
            will be paying with the asset at asset_ix in the self.reserves.
            asset_in_ix (float): This is the index of the asset associated with the
            qty_in parameter, if qty_in is positive. asset_out_ix is then the index
            of the asset that you wish to get out of the pool.
            asset_out_ix (float): This is the index of the asset you will be getting
            out of the pool. If qty_in is negative, then asset_out_ix must the index
            of this asset that you are depleting the pool of.
        """
        self._validate_trade(qty_in, asset_in_ix, asset_out_ix)
        return self.trade(qty_in, asset_in_ix, asset_out_ix)

    def invariant(self):
        raise Exception("must be implemented")

    def slippage(self, qty_in: float, asset_in_ix: int, asset_out_ix: int):
        """Computes slippage due to protocol's design (invariant). Unlike the trade
        function, this function will not update the state. It merely simulates
        what would have happened if qty_in was traded in the protocol.

        Args:
            qty_in (float): qty of asset_in_ix to deplete (if negatie) or add to
            (if positive) to the pool.
            asset_in_ix (float): index of the asset we are adding to the pool
            (if qty_in is positive) or that we are removing (if qty_in is negative)
            asset_out_ix (float): index of the asset we are getting out of the pool
            (if qty_in is positive) or that we are putting in to pay for what we are
            getting out (qty_in is negatie, we are getting this much out of the pool)

        Raises:
            Exception: [description]
        """
        self._validate_trade(qty_in, asset_in_ix, asset_out_ix)
        raise Exception("must be implemented")

    def _value_pool(
        self, pct_change: float, asset_in_ix: int, asset_out_ix: int
    ) -> float:
        self._validate_pct_change(pct_change)
        self._validate_asset_ix(asset_in_ix)
        self._validate_asset_ix(asset_out_ix)
        return self.value_pool(pct_change, asset_in_ix, asset_out_ix)

    def value_pool(
        self, pct_change: float, asset_in_ix: int, asset_out_ix: int
    ) -> float:
        raise NotImplementedError("must be implemented")

    def value_hold(self, pct_change: float, asset_in_ix: int, asset_out_ix: int):
        self._validate_pct_change(pct_change)
        self._validate_asset_ix(asset_in_ix)
        self._validate_asset_ix(asset_out_ix)
        # equation no: 32 and 33 in the paper
        V = self.reserves[asset_in_ix] / self.weights[asset_in_ix]
        V2 = V * self.weights[asset_out_ix]
        V_held = V + V2 * pct_change
        return V_held

    def divergence_loss(self, pct_change: float, asset_in_ix: int, asset_out_ix: int):
        return (
            self.value_pool(pct_change, asset_in_ix, asset_out_ix)
            / self.value_hold(pct_change, asset_in_ix, asset_out_ix)
            - 1
        )

    def _validate_asset_ix(self, asset_ix: float):
        if asset_ix < 0 or asset_ix >= len(self.reserves):
            raise Exception("invalid asset ix")

    def _validate_trade(self, qty_in: float, asset_in_ix: int, asset_out_ix: int):
        self._validate_asset_ix(asset_in_ix)
        self._validate_asset_ix(asset_out_ix)

        if qty_in < 0:
            if self.reserves[asset_in_ix] < -qty_in:
                raise Exception("you cannot remove this much from the liquidity pool")

    @staticmethod
    def _validate_pct_change(pct_change: float):
        if pct_change < -1:
            raise Exception("pct. change cannot be less than -100% (-1)")

    @staticmethod
    def _validate_len_of_reserves_and_weights(
        reserves: list[float], weights: list[float]
    ):
        if len(reserves) != len(weights):
            raise Exception("reserves length and weights length must be the same")

    @staticmethod
    def _validate_reserves(qty: float):
        if qty <= 0:
            raise Exception("asset quantity must be greater than zero")

    @staticmethod
    def _validate_weights(weights: list[float]):
        if not sum(weights) == 1:
            raise Exception("asset weights do not sum to 1.0")

    def __repr__(self):
        return f"Amm(assets_qty={self.reserves},assets_weights={self.weights})"


class Balancer(Amm):
    def __init__(self, reserves: list[float], weights: list[float]):
        super().__init__(reserves, weights)

    def invariant(self) -> float:
        # todo: validations to be enforced in the inherited class
        C = 1
        for i, qty in enumerate(self.reserves):
            C *= qty ** self.weights[i]
        return C

    def spot_price(self, asset_in_ix: int, asset_out_ix: int) -> float:
        # todo: validations to be enforced by the inherited class
        return (self.reserves[asset_in_ix] * self.weights[asset_out_ix]) / (
            self.reserves[asset_out_ix] * self.weights[asset_in_ix]
        )

    def _compute_trade_qty_out(
        self, qty_in: float, asset_in_ix: int, asset_out_ix: int
    ) -> tuple[float, float]:
        pre_trade_reserves_in_ix = self.reserves[asset_in_ix]
        pre_trade_reserves_out_ix = self.reserves[asset_out_ix]
        updated_reserves_in_ix = pre_trade_reserves_in_ix + qty_in
        updated_reserves_out_ix = pre_trade_reserves_out_ix * (
            pre_trade_reserves_in_ix / updated_reserves_in_ix
        ) ** (self.weights[asset_in_ix] / self.weights[asset_out_ix])
        return updated_reserves_in_ix, updated_reserves_out_ix

    def trade(self, qty_in: float, asset_in_ix: int, asset_out_ix: int) -> float:
        pre_trade_reserves_out_ix = self.reserves[asset_out_ix]
        # todo: common step & validations to be enforced by the inherited class
        (
            updated_reserves_in_ix,
            updated_reserves_out_ix,
        ) = self._compute_trade_qty_out(qty_in, asset_in_ix, asset_out_ix)
        self.reserves[asset_in_ix] = updated_reserves_in_ix
        self.reserves[asset_out_ix] = updated_reserves_out_ix
        return pre_trade_reserves_out_ix - self.reserves[asset_out_ix]

    def slippage(self, qty_in: float, asset_in_ix: int, asset_out_ix: int) -> float:
        x_1 = qty_in
        _, r_2_prime = self._compute_trade_qty_out(qty_in, asset_in_ix, asset_out_ix)
        x_2 = self.reserves[asset_out_ix] - r_2_prime
        p = self.spot_price(asset_in_ix, asset_out_ix)
        return (x_1 / x_2) / p - 1

    def value_pool(
        self, pct_change: float, asset_in_ix: int, asset_out_ix: int
    ) -> float:
        V = self.reserves[asset_in_ix] / self.weights[asset_in_ix]
        V_prime = V * (1 + pct_change) ** self.weights[asset_out_ix]
        return V_prime


class Uniswap(Balancer):
    def __init__(self, reserves: list[float]):
        uniswap_equal_weights = [0.5, 0.5]
        super().__init__(reserves, uniswap_equal_weights)
