import numpy as np


def get_gbm(
    mu: float = 0.0,
    sigma: float = 0.1,
    dt: float = 0.01,
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


class SimulatedAssets:
    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.1,
        dt: float = 0.01,
        p0: float = 2_000,
        n_steps: int = 100,  # number of price values to be generated
        n_mc: int = 1,
        seed: int = 1,
        alpha_eth: float = -0.15,
        alpha_dai: float = 0.05,
        u0_eth: float = 0.4,
        u0_dai: float = 0.4,
        r_0: float = 0.0,
        r_1: float = 0.04,
        r_2: float = 2.5,
        u_optimal: float = 0.45,
    ):
        self.price_paths = price_paths = get_gbm(mu, sigma, dt, p0, n_steps, n_mc, seed)
        self.p0 = p0
        self.dt = dt

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
