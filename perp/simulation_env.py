import numpy as np


def get_gbm(
    mu: float = 0.0,
    sigma: float = 0.1,
    dt: float = 1,
    p0: float = 2_000,
    n_steps: int = 100,  # number of price values to be generated
    n_mc: int = 1,
    seed: int = 1,
) -> np.ndarray:
    """Get gbm paths"""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_mc, n_steps - 1))
    normalized_path = (
        np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    ).cumprod(axis=1)
    return np.insert(normalized_path, 0, 1, axis=1) * p0


def sigmoid(nu: float | np.ndarray) -> float | np.ndarray:
    """
    map mu to utilisation
    """
    return 1 / (1 + np.exp(-nu))


def reciprocal_sigmoid(u: float | np.ndarray) -> float | np.ndarray:
    """
    map utilisation to mu
    """
    return np.log(u / (1 - u))


def irm(
    utilisation: float,
    u_optimal: float,
    r_0: float,
    r_1: float,
    r_2: float,
    collateral: bool,
) -> float:
    """
    Interest rate model
    """
    r = r_0 + (
        r_1 * utilisation / u_optimal
        if utilisation < u_optimal
        else r_1 + r_2 * (utilisation - u_optimal) / (1 - u_optimal)
    )
    return r * (utilisation if collateral else 1)


vect_irm = np.vectorize(irm, excluded=("u_optimal", "r_0", "r_1", "r_2", "collateral"))


class SimulationEnv:
    def __init__(
        self,
        price_paths: np.ndarray,
        alpha_eth: float = -0.15,
        alpha_dai: float = 0.05,
        u0_eth: float = 0.4,
        u0_dai: float = 0.4,
        r_0: float = 0.0,
        r_1: float = 0.04,
        r_2: float = 2.5,
        r: float = 0.05,  # r dai fixed
        u_optimal: float = 0.45,
        lt: float = 0.85,
        ltv0: float = 0.75,
        dt: float = 0.01,
        seed: int = 1,
        kappa: float = 1,
    ) -> None:

        self.p0 = price_paths[0, 0]
        self.n_mc, self.n_steps = price_paths.shape
        self.price_paths = price_paths
        self.time_array = np.tile(np.arange(self.n_steps) * dt, (self.n_mc, 1))

        self.u_eth = sigmoid(
            alpha_eth * (price_paths - self.p0) / price_paths.std()
            + reciprocal_sigmoid(u0_eth)
        )
        self.u_dai = sigmoid(
            alpha_dai * (price_paths - self.p0) / price_paths.std()
            + reciprocal_sigmoid(u0_dai)
        )

        self.r_collateral_eth = vect_irm(
            utilisation=self.u_eth,
            u_optimal=u_optimal,
            r_0=r_0,
            r_1=r_1,
            r_2=r_2,
            collateral=True,
        )
        self.r_debt_dai = vect_irm(
            utilisation=self.u_dai,
            u_optimal=u_optimal,
            r_0=r_0,
            r_1=r_1,
            r_2=r_2,
            collateral=False,
        )

        self.dt = dt  # time step
        self.lt = lt  # liquidation threshold
        self.ltv0 = ltv0  # initial loan-to-value ratio
        self.seed = seed
        self.kappa = kappa
        self.r = r

    @property
    def liquidation_times(self) -> np.ndarray:
        # get true / false mask for liquidation call
        liquidation_call_mask = (
            self.price_paths
            * np.exp((self.r_collateral_eth - self.r_debt_dai) * self.time_array)
            <= 1 / self.lt * self.ltv0 * self.p0
        )
        return (
            np.where(
                np.any(liquidation_call_mask, axis=1),
                np.argmax(liquidation_call_mask, axis=1),
                # if not liquidation, assign a time out of step range
                (self.n_steps + 1),
            )
            * self.dt
        ).reshape(-1, 1)

    @property
    def pnl_lending_position(self) -> np.ndarray:
        payoff = self.price_paths * np.exp(self.r_collateral_eth * self.time_array) - (
            self.ltv0 * self.p0 * np.exp(self.r_debt_dai * self.time_array)
        )
        # after liquidation no more payoff, it flattens out
        # TODO: check if it should flatten out or be NAN
        for i, _ in enumerate(payoff):
            payoff[i, self.time_array[i] > self.liquidation_times[i]] = payoff[
                i, self.time_array[i] == self.liquidation_times[i]
            ]
        return payoff - self.p0 * (1 - self.ltv0)

    @property
    def pnl_perps_position(self) -> tuple[np.ndarray, np.ndarray]:
        return self.price_paths - self.p0, self.price_paths * (
            1 - np.exp(self.r_collateral_eth * self.time_array)
        ) - self.ltv0 * self.p0 * (1 - np.exp(self.r_debt_dai * self.time_array))

    def perps_price_mean_rev(
        self,
        lambda_: float,
        sigma_f: float,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        f0 = self.p0 * (1 + self.r / self.kappa)
        f = np.ones_like(self.price_paths) * f0
        for step in range(1, self.n_steps):
            f[:, step] = (
                f[:, step - 1]
                + lambda_ * (self.price_paths[:, step - 1] - f[:, step - 1]) * self.dt
                + sigma_f * rng.normal(loc=0, scale=np.sqrt(self.dt), size=self.n_mc)
            )
        return f

    def perps_price_realistic(
        self,
        sigma_f: float,
        sigma_noise: float,
        window_length: int,
        delta: float,
        lambda_: int,
    ) -> np.ndarray:

        # moving average
        window_view = np.lib.stride_tricks.sliding_window_view(
            self.price_paths, window_length, axis=1
        )
        moving_average = np.mean(window_view, axis=-1)
        change = np.diff(moving_average, n=1, axis=-1)

        mean_rev = np.ones_like(self.price_paths) * self.p0
        rng = np.random.default_rng(self.seed)
        for step in range(1, self.n_steps):

            mean_rev[:, step] = (
                mean_rev[:, step - 1]
                + lambda_
                * (self.price_paths[:, step - 1] - mean_rev[:, step - 1])
                * self.dt
                + sigma_f * rng.normal(loc=0, scale=np.sqrt(self.dt), size=self.n_mc)
            )

        f = np.copy(self.price_paths)
        for i, step in enumerate(range(window_length, self.n_steps)):
            z = rng.random(size=self.n_mc)
            # bullish
            mask_bullish = change[:, i] > delta
            f[mask_bullish, step] += np.exp(sigma_noise * z[mask_bullish])
            # bearish
            mask_bearish = change[:, i] < -delta
            f[mask_bearish, step] -= np.exp(sigma_noise * z[mask_bearish])
            # neutral
            mask_neutral = (mask_bullish + mask_bearish) == 0
            f[mask_neutral, step] = mean_rev[mask_neutral, step]

        return f

    def get_funding_fee_perps(
        self,
        perps_price_paths: np.array,
    ):
        funding_fee = np.zeros(shape=(self.n_mc, self.n_steps))
        time_array = np.arange(self.n_steps) * self.dt
        for step in range(1, self.n_steps):
            t = time_array[step]
            funding_fee[:, step] = self.kappa * np.sum(
                self.dt
                * np.exp((t - time_array[:step]) * self.r_debt_dai[:, :step])
                * (perps_price_paths[:, :step] - self.price_paths[:, :step]),
                axis=1,
            )

        return funding_fee

    def get_pnl_perps(
        self,
        perps_price_paths: np.array,
    ):
        funding_fee = self.get_funding_fee_perps(perps_price_paths)
        pnl = np.zeros_like(self.price_paths)
        pnl = (
            perps_price_paths
            - perps_price_paths[:, 0].reshape(self.n_mc, 1)
            - funding_fee
        )
        coef = 1 + self.r / self.kappa
        return 1 / coef * pnl

    def liquidation_times_perp(
        self,
        perps_price_paths: np.array,
        lt_f: float,
    ):
        pnl = self.get_pnl_perps(perps_price_paths)
        maintenance_margin = self.price_paths * lt_f
        initial_margin = self.p0 * (1 - self.ltv0)
        mask = maintenance_margin >= initial_margin + pnl

        idx_liquidation_time = np.argmax(mask.astype(int), axis=1, keepdims=True)
        liquidation_time = idx_liquidation_time * self.dt
        liquidation_time[liquidation_time == 0] = self.dt * self.n_steps + 0.1
        return liquidation_time

    def pnl_perps_after_liquidation(self, perps_price_paths: np.array, lt_f: float):
        pnl = self.get_pnl_perps(perps_price_paths)
        time_array = self.time_array

        liquidation_times = self.liquidation_times_perp(
            perps_price_paths=perps_price_paths,
            lt_f=lt_f,
        )
        pnl = pnl * (time_array <= liquidation_times)
        for i, _ in enumerate(pnl):
            pnl[i, time_array[i] >= liquidation_times[i]] = pnl[
                i, time_array[i] == liquidation_times[i]
            ]

        return pnl
