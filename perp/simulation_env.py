import numpy as np
from perp.simulated_asset import SimulatedAssets


class SimulationEnv:
    def __init__(
        self,
        price_paths: np.ndarray,
        r_collateral_eth: float,
        r_debt_dai: float,
        dt: float = 0.01,
        r: float = 0.005,  # r dai fixed
        lt: float = 0.85,
        ltv0: float = 0.75,
        seed: int = 1,
        kappa: float = 1,
        sigma_f: float = 0.1,
        lambda_: float = 100,
        lt_f: float = 0.0875,
    ) -> None:

        self.p0 = price_paths[0, 0]
        self.n_mc, self.n_steps = price_paths.shape
        self.price_paths = price_paths
        self.time_array = np.tile(np.arange(self.n_steps) * dt, (self.n_mc, 1))
        self.r_collateral_eth = r_collateral_eth
        self.r_debt_dai = r_debt_dai

        self.dt = dt  # time step
        self.lt = lt  # liquidation threshold
        self.ltv0 = ltv0  # initial loan-to-value ratio
        self.seed = seed
        self.kappa = kappa
        self.r = r
        self.sigma_f = sigma_f
        self.lambda_ = lambda_
        self.lt_f = lt_f

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

    def pnl_lending_position(self) -> np.ndarray:
        payoff = self.price_paths * np.exp(self.r_collateral_eth * self.time_array) - (
            self.ltv0 * self.p0 * np.exp(self.r_debt_dai * self.time_array)
        )
        liquidation_times = self.liquidation_times()
        # after liquidation no more payoff, it flattens out
        # TODO: check if it should flatten out or be NAN
        for i, _ in enumerate(payoff):
            payoff[i, self.time_array[i] > liquidation_times[i]] = payoff[
                i, self.time_array[i] == liquidation_times[i]
            ]
        return payoff - self.p0 * (1 - self.ltv0)

    def implied_funding_fee(self) -> np.ndarray:
        return self.price_paths * (
            1 - np.exp(self.r_collateral_eth * self.time_array)
        ) - self.ltv0 * self.p0 * (1 - np.exp(self.r_debt_dai * self.time_array))

    def perps_price_mean_rev(
        self,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        f0 = self.p0 * (1 + self.r / self.kappa)
        f = np.ones_like(self.price_paths) * f0
        for step in range(1, self.n_steps):
            f[:, step] = (
                f[:, step - 1]
                + self.lambda_
                * (self.price_paths[:, step - 1] - f[:, step - 1])
                * self.dt
                + self.sigma_f
                * rng.normal(loc=0, scale=np.sqrt(self.dt), size=self.n_mc)
            )
        return f

    def perps_price_realistic(
        self,
        sigma_noise: float,
        window_length: int,
        delta: float,
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
                + self.lambda_
                * (self.price_paths[:, step - 1] - mean_rev[:, step - 1])
                * self.dt
                + self.sigma_f
                * rng.normal(loc=0, scale=np.sqrt(self.dt), size=self.n_mc)
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
    ) -> np.ndarray:
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
    ) -> np.ndarray:
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
        perps_price_paths: np.ndarray,
    ) -> np.ndarray:
        pnl = self.get_pnl_perps(perps_price_paths)
        maintenance_margin = self.price_paths * self.lt_f
        initial_margin = self.p0 * (1 - self.ltv0)
        mask = maintenance_margin >= initial_margin + pnl

        return (
            # if not liquidation, assign a time out of step range
            np.where(
                np.any(mask, axis=1),
                np.argmax(mask, axis=1),
                (self.n_steps + 1),
            )
            * self.dt
        ).reshape(-1, 1)

    def pnl_perps_after_liquidation(self, perps_price_paths: np.ndarray) -> np.ndarray:
        pnl = self.get_pnl_perps(perps_price_paths)
        time_array = self.time_array

        liquidation_times = self.liquidation_times_perp(
            perps_price_paths=perps_price_paths
        )
        for i, _ in enumerate(pnl):
            pnl[i, time_array[i] >= liquidation_times[i]] = pnl[
                i, time_array[i] == liquidation_times[i]
            ]

        return pnl


if __name__ == "__main__":
    pass
    # dai_backed_eth = SimulatedAssets(
    #     n_steps=101, seed=1, mu=0, sigma=0.3, n_mc=500, dt=0.01
    # )

    # self = SimulationEnv(
    #     price_paths=dai_backed_eth.price_paths,
    #     r_collateral_eth=dai_backed_eth.r_collateral_eth,
    #     r_debt_dai=dai_backed_eth.r_debt_dai,
    #     dt=dai_backed_eth.dt,
    #     lambda_=50,
    #     sigma_f=0.3,
    #     kappa=1,
    #     r=0.005,
    # )

    # perps_price_paths = self.perps_price_realistic(
    #     sigma_noise=6.5, window_length=5, delta=5
    # )

    # dai_backed_eth = SimulatedAssets(n_steps=10_000, seed=2)
    # sim_env = SimulationEnv(
    #     price_paths=dai_backed_eth.price_paths,
    #     r_collateral_eth=dai_backed_eth.r_collateral_eth,
    #     r_debt_dai=dai_backed_eth.r_debt_dai,
    #     dt=dai_backed_eth.dt,
    # )
    # print(sim_env.liquidation_times)
