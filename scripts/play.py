# # plot e^X where X~N(0,1)

# import numpy as np
# import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

sizes_only = go.layout.Template(
    layout=go.Layout(
        font=dict(size=22),
        title=dict(font=dict(size=24)),
        legend=dict(font=dict(size=20)),
        xaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
        yaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
    )
)
pio.templates["sizes_only"] = sizes_only
pio.templates.default = "plotly+sizes_only"

# X = np.random.randn(100_000)
# Y = np.exp(X)

import argparse, streamlit as st

@st.cache_resource  # parse once per session
def _parse_cli():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.9)
    p.add_argument("--time", type=float, default=0.25)
    # long-run basis level m; omit for the baseline m = r / kappa, pass 0.0001 for the bull scenario
    p.add_argument("--premium", type=float, default=None)
    args, _ = p.parse_known_args()  # ignore Streamlit’s own flags
    return args

args = _parse_cli()
mu, sigma, t = args.mu, args.sigma, args.time
st.write("CLI:", mu, sigma, t)


dt = 1 / (3 * 365)  # 8-hour step, the funding settlement interval
p0 = 2_000
n_steps = 3 * 365  # one year of 8-hour steps
n_mc = 10000
# seed = 8765
# Funding coefficient per year: funding settles every 8 hours (3 x 365 periods per year),
# so the basis is paid off once per period (Ackerer et al., 2024).
KAPPA = 3 * 365
# Relative basis b = (F - P) / P, calibrated on 8-hour ETH data (scripts/calibrate_basis.py).
LAMBDA_B = 539.0  # mean-reversion speed per year (half-life about 11 hours)
SIGMA_B = 0.0072  # volatility per square-root year
BULL_PREMIUM = 0.0001  # 90th percentile of Binance ETHUSDT 8-hour funding rates
# Aave v3 interest rate strategies, per pool (published parameters, 2025).
# WETH, Ethereum Core: Aave governance forum, Risk Stewards WETH/wstETH update (19 July 2025).
IRM_ETH = dict(u_optimal=0.90, r_0=0.0, r_1=0.027, r_2=0.80, reserve_factor=0.15)
# DAI, Arbitrum: slope 1 = 6% (Chaos Labs Risk Stewards, 28 March 2025); optimal usage 90% and
# slope 2 of about 40% from the Aave app's interest rate model (read in October 2026); reserve
# factor 25% implied by the supply and borrow rates in data/aave_v3_arbitrum_dai_8h_180d.jsonl.gz.
IRM_DAI = dict(u_optimal=0.90, r_0=0.0, r_1=0.06, r_2=0.40, reserve_factor=0.25)
# Utilisation map d nu = alpha d log P, U = sigmoid(nu) (scripts/calibrate_utilisation.py):
# alpha from weekly changes of logit(U) on ETH log returns, u0 = sample mean utilisation.
alpha_eth = 1.6  # WETH, Aave v3 Ethereum
alpha_dai = 2.15  # DAI, Aave v3 Arbitrum
u0_eth = 0.88
u0_dai = 0.81
r_0 = 0.0
r_1 = 0.04
r_2 = 2.5
u_optimal = 0.45

lt = 0.95
ltv0 = 0.75

from typing import Tuple

import numpy as np


def get_gbm(
    mu: float, sigma: float, dt: float, n_steps: int, p0: float, seed: int, n_mc
) -> np.array:
    """Get gbm paths"""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_mc, n_steps))

    paths = np.ones((n_mc, n_steps + 1)) * p0
    paths[:, 1:] = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    paths = paths.cumprod(axis=1)
    return paths


def sigmoid(x: np.array):
    return 1 / (1 + np.exp(-x))


def reciprocal_sigmoid(x: np.array):
    return np.log(x / (1 - x))


def get_utilisation(
    price_paths: np.array,
    u0: float,
    a: float,
    b: float = 0,
) -> np.array:
    latent_u = np.zeros_like(price_paths)
    log_price_paths = np.log(price_paths)  # d nu = a d log P (Reviewer 2, comment 2.4e)
    latent_u[:, 0] = reciprocal_sigmoid(u0)

    for i in range(1, latent_u.shape[1]):
        delta_log_P = log_price_paths[:, i] - log_price_paths[:, i - 1]
        latent_u[:, i] = latent_u[:, i - 1] + (a + b * latent_u[:, i - 1]) * delta_log_P

    u = sigmoid(latent_u)
    return u


def irm(
    u_optimal: float,
    r_0: float,
    r_1: float,
    r_2: float,
    utilisation: float,
    collateral: bool,
    reserve_factor: float = 0.0,
) -> float:
    if utilisation < u_optimal:
        r = r_0 + r_1 * utilisation / u_optimal
    else:
        r = r_0 + r_1 + r_2 * (utilisation - u_optimal) / (1 - u_optimal)
    if collateral:
        # suppliers receive the borrow interest net of the protocol's reserve factor
        return utilisation * r * (1 - reserve_factor)
    else:
        return r


vect_irm = np.vectorize(
    irm, excluded=("u_optimal", "r_0", "r_1", "r_2", "collateral", "reserve_factor")
)



def get_liquidation_call_mask(
    price_paths: np.array,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
) -> np.array:
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))
    mask = (
        price_paths * np.exp((np.cumsum(r_collateral_eth, axis=1) - r_collateral_eth) * dt - (np.cumsum(r_debt_dai, axis=1) - r_debt_dai) * dt)
        <= 1 / lt * ltv0 * p0
    )
    return mask


price_paths = get_gbm(mu = 0.0,
    sigma = 0.1,
    dt = 0.01,
    p0 = 2_000,
    n_steps = 9999,  # number of price values to be generated
    n_mc = 1,
    seed = 2)

u_eth = get_utilisation(price_paths=price_paths, u0=u0_eth, a=alpha_eth)
u_dai = get_utilisation(price_paths=price_paths, u0=u0_dai, a=alpha_dai)
r_collateral_eth = vect_irm(
    **IRM_ETH, utilisation=u_eth, collateral=True
)
r_debt_dai = vect_irm(
    **IRM_DAI, utilisation=u_dai, collateral=False
)


def get_liquidation_times(
    price_paths: np.array,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
):
    mask = get_liquidation_call_mask(
        price_paths, dt, lt, ltv0, r_collateral_eth, r_debt_dai
    )
    mask = mask.astype(int)
    idx_liquidation_time = np.argmax(mask, axis=1, keepdims=True)
    liquidation_time = idx_liquidation_time * dt
    T = dt * price_paths.shape[1]
    liquidation_time[liquidation_time == 0] = (
        T + 0.1
    )  # big number out of considered time range
    return liquidation_time


def get_pnl_lending_position(
    price_paths,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
    flashloan_fee: float = 0.0,
    liquidation_penalty: float = 0.0,
):
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))
    liquidation_times = get_liquidation_times(
        price_paths, dt, lt, ltv0, r_collateral_eth, r_debt_dai
    )

    collateral_value = price_paths * np.exp((np.cumsum(r_collateral_eth, axis=1) - r_collateral_eth) * dt)
    debt_value = ltv0 * p0 * np.exp((np.cumsum(r_debt_dai, axis=1) - r_debt_dai) * dt)
    payoff = collateral_value - debt_value
    # Eq. (1): full liquidation (close factor 1); the liquidation penalty is charged on
    # the repaid debt and the borrower keeps the residual, frozen at tau^B and floored at 0.
    residual = np.maximum(collateral_value - (1 + liquidation_penalty) * debt_value, 0.0)
    for i, _ in enumerate(payoff):
        at_liquidation = time_array[i] == liquidation_times[i]
        if at_liquidation.any():
            payoff[i, time_array[i] >= liquidation_times[i]] = residual[i, at_liquidation][0]
    payoff = payoff * (1 - flashloan_fee)
    pnl = payoff - p0 * (1 - ltv0)
    return pnl


def decompose_pnl_lending_position(
    price_paths,
    dt: float,
    lt: float,
    ltv0: float,
    r_collateral_eth: np.array,
    r_debt_dai: float,
) -> Tuple[np.array, np.array]:
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))
    return price_paths - p0, p0 * ((
        1 - np.exp((np.cumsum(r_collateral_eth, axis=1) - r_collateral_eth) * dt)
    ) - ltv0 * (1 - np.exp((np.cumsum(r_debt_dai, axis=1) - r_debt_dai) * dt)))


@st.cache_data  # -- Magic command to cache data
def get_perps_price_mean_rev(
    price_paths: np.array,
    lambda_: float,
    sigma: float,
    dt: float,
    r: float,
    kappa: float = KAPPA,
    seed: int = 8000,
) -> np.array:
    n_mc = price_paths.shape[0]
    p0 = price_paths[0, 0]
    f0 = p0 * (1 + r / kappa)
    f = np.ones_like(price_paths) * f0
    rng = np.random.default_rng(seed)
    for step in range(1, f.shape[1]):
        z = rng.normal(size=n_mc)
        f[:, step] = (
            f[:, step - 1]
            + lambda_ * ((1 + r / kappa) * price_paths[:, step - 1] - f[:, step - 1]) * dt
            + np.sqrt(dt) * sigma * z
        )
    return f


@st.cache_data  # -- Magic command to cache data
def get_perps_price_mean_rev_to_non_arb(
    price_paths: np.array,
    lamda_: float,
    sigma: float,
    dt: float,
    r: float,
    kappa: float = KAPPA,
    seed: int = 8767,
) -> np.array:
    n_mc = price_paths.shape[0]
    p0 = price_paths[0, 0]
    f0 = p0 * (1 + r / kappa)
    f = np.ones_like(price_paths) * f0
    rng = np.random.default_rng(seed)
    coef = 1 + r / kappa
    for step in range(1, f.shape[1]):
        z = rng.normal(size=n_mc)
        f[:, step] = (
            f[:, step - 1]
            + lambda_ * (coef * price_paths[:, step - 1] - f[:, step - 1]) * dt
            + np.sqrt(dt) * sigma * z
        )
    return f


@st.cache_data
def get_perps_price_realistic(
    price_paths: np.array,
    sigma: float,
    sigma_noise: float,
    dt: float,
    window_length: int,
    delta: float,
    lambda_: int,
    seed: int,
) -> np.array:

    n_mc = price_paths.shape[0]
    p0 = price_paths[0, 0]

    # moving average
    window_view = np.lib.stride_tricks.sliding_window_view(
        price_paths, window_length, axis=1
    )
    moving_average = np.mean(window_view, axis=-1)
    change = np.diff(moving_average, n=1, axis=-1)

    mean_rev = np.ones_like(price_paths) * p0
    rng = np.random.default_rng(seed)
    for step in range(1, mean_rev.shape[1]):
        z = rng.normal(size=n_mc)
        mean_rev[:, step] = (
            mean_rev[:, step - 1]
            + lambda_ * (price_paths[:, step - 1] - mean_rev[:, step - 1]) * dt
            + np.sqrt(dt) * sigma * z
        )

    f = np.copy(price_paths)
    for i, step in enumerate(range(window_length, mean_rev.shape[1])):
        z = rng.random(size=n_mc)
        # bullish
        mask_bullish = change[:, i] > delta
        f[mask_bullish == 1, step] += np.exp(sigma_noise * z[mask_bullish == 1])
        # bearish
        mask_bearish = change[:, i] < -delta
        f[mask_bearish == 1, step] -= np.exp(sigma_noise * z[mask_bearish == 1])
        # neutral
        mask_neutral = (mask_bullish + mask_bearish) == 0
        f[mask_neutral == 1, step] = mean_rev[mask_neutral == 1, step]

    return f


def get_perps_price_basis_ou(
    price_paths: np.array,
    dt: float,
    lambda_: float,
    sigma_b: float,
    m: float,
    b0: float,
    seed: int,
) -> np.array:
    """F_t = P_t (1 + b_t) with db_t = lambda (m - b_t) dt + sigma_b dW_t, simulated exactly."""
    n_mc, n_steps = price_paths.shape
    rng = np.random.default_rng(seed)
    a = np.exp(-lambda_ * dt)
    s = sigma_b * np.sqrt((1 - a**2) / (2 * lambda_))
    b = np.empty_like(price_paths)
    b[:, 0] = b0
    for step in range(1, n_steps):
        b[:, step] = m + (b[:, step - 1] - m) * a + s * rng.standard_normal(n_mc)
    return price_paths * (1 + b)


def get_perps_price_non_arb(price_paths: np.array, r: float, kappa: float = KAPPA):
    f = price_paths * (1 + r / kappa)
    return f


def get_funding_fee_perps(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    dt: float,
    kappa: float = KAPPA,
):
    n_mc, n_steps = price_paths.shape
    funding_fee = np.zeros(shape=(n_mc, n_steps))
    time_array = np.arange(n_steps) * dt
    for i, step in enumerate(range(1, n_steps)):
        t = time_array[step]
        funding_fee[:, step] = kappa * np.sum(
            dt
            * np.exp((t - time_array[:step]) * r_debt_dai[:, :step])
            * (perps_price_paths[:, :step] - price_paths[:, :step]),
            axis=1,
        )

    return funding_fee


def get_pnl_perps(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    dt: float,
    r: float,
    kappa: float = KAPPA,
    trading_fee: float = 0.0,
):
    n_mc, n_steps = perps_price_paths.shape
    funding_fee = get_funding_fee_perps(
        price_paths, perps_price_paths, r_debt_dai, dt, kappa
    )
    pnl = np.zeros_like(price_paths)
    coef = 1 + r / kappa
    pnl = (1 / coef * (perps_price_paths * (1-trading_fee) - perps_price_paths[:, 0].reshape(n_mc, 1) )
           - funding_fee)

    return pnl


def get_liquidation_times_perp(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    lt_f: float,
    ltv0: float,
    dt: float,
    r: float,
    kappa: float = KAPPA,
    trading_fee: float = 0.0,
):
    pnl = get_pnl_perps(price_paths, perps_price_paths, r_debt_dai, dt, r, kappa, trading_fee)
    p0 = price_paths[0, 0]
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))

    maintenance_margin = price_paths * lt_f  # price_paths * (1 - max_ltv0) * lt_f
    initial_margin = p0 * (1 - ltv0)

    mask = maintenance_margin >= initial_margin + pnl

    T = dt * price_paths.shape[1]
    idx_liquidation_time = np.argmax(mask.astype(int), axis=1, keepdims=True)
    liquidation_time = idx_liquidation_time * dt
    liquidation_time[liquidation_time == 0] = T + 0.1
    return liquidation_time


def get_pnl_perps_after_liquidation(
    price_paths: np.array,
    perps_price_paths: np.array,
    r_debt_dai: np.array,
    lt_f: float,
    ltv0: float,
    dt: float,
    r: float,
    kappa: float = KAPPA,
    liquidation_clearance_fee: float = 0.0,
    trading_fee: float = 0.0,
):
    pnl = get_pnl_perps(price_paths, perps_price_paths, r_debt_dai, dt, r, kappa, trading_fee)
    n_mc, n_steps = price_paths.shape
    time_array = np.arange(n_steps) * dt
    time_array = np.tile(time_array, (n_mc, 1))

    liquidation_times = get_liquidation_times_perp(
        dt=dt,
        kappa=kappa,
        lt_f=lt_f,
        ltv0=ltv0,
        perps_price_paths=perps_price_paths,
        price_paths=price_paths,
        r=r,
        r_debt_dai=r_debt_dai,
        trading_fee = trading_fee
    )
    # Residual margin (P0(1 - theta0) + PnL - f_clr * F)^+ at tau^F, frozen afterwards.
    initial_margin = price_paths[0, 0] * (1 - ltv0)
    residual_pnl = np.maximum(
        initial_margin + pnl - liquidation_clearance_fee * perps_price_paths, 0.0
    ) - initial_margin
    for i, _ in enumerate(pnl):
        at_liquidation = time_array[i] == liquidation_times[i]
        if at_liquidation.any():
            pnl[i, time_array[i] >= liquidation_times[i]] = residual_pnl[i, at_liquidation][0]
    return pnl


# ------------
# GLOBAL vars
# ------------
def get_idx_from_t(t):
    return int(round(t / dt))


dt = 1 / (3 * 365)  # 8-hour step, the funding settlement interval
n_steps = 3 * 365  # one year of 8-hour steps
p0 = 2000
seed = 8700
n_mc = 10000 #500

r = 0.05
kappa = KAPPA
t = round(t / dt) * dt  # snap the reporting time to the simulation grid
# --------------

# -- Set page config
apptitle = "Leveraged trading"
st.set_page_config(
    layout="wide",
    page_title=apptitle,
)


# header_col1, header_col2 = st.columns([0.2, 0.8], gap="medium")
# with header_col1:
#     st.image("logo.jpeg", width=100)
# with header_col2:
#     st.title("Simtopia")

# --------
# INTRO
# --------
st.header("Loan positions vs perpetual futures")
st.markdown(
    """
    This dashboard allows users to compare loan contracts offered by lending protocols
    and perpetual futures, which are two main mechanisms for trading cryptocurrencies
    on margins. The user can study PnLs, the likelihood
    of liquidation for loan positions, and the close-out margin for both contracts.

    Users can study the Profit-and-Loss (PnL) of these positions and the implied
    funding fee/rate and maintenance margin rule for loan contracts and contrast these
    to perpetual futures under varied market conditions.

    This dashboard allows users to set the relevant parameters: maxLTV, LLTV,
    and initial and maintenance margin and run simulations under varied market conditions.
    This, for example, can help users study potential statistical arbitrage
    opportunities that may arise between futures markets and lending platforms or hedge
    the risk arising from liquidity provision on lending protocols.

    A thorough analysis underpinned with market data and simulations can be found in
    our paper: [Leveraged Trading via Lending Platforms](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4713126).
    Our [blog post](https://medium.com/@simtopia/a76d6239600d) summarises the results.
    """
)


st.subheader("Statistical model for the risky assets")
expander = st.expander("model")
with expander:
    st.markdown(
        """
        We model the price of ETH-DAI as a Geometric Brownian Process
        $$
            \\text{d} P_t=\mu P_t \\text{d}t + \sigma P_t \\text{d}W_t, \,\,P_0 = p_0\,.
        $$
        The drift coefficient $\mu$ corresponds to market trend, while the diffusion coefficient $\sigma$
        corresponds to market volatility. We provide two models for the price of the perpetual futures $F_t$:
        1. A mean-reverting process around the price of EHT-DAI,
        $$
            \\text{d}F_t = \lambda (P_t - F_t)\\text{d}t + \sigma^F \\text{d}W_t^F,\,\, F_0 = f_0, \,\, [W^F, W]_t = \\rho  t.
        $$
        Here $\lambda$ dictates the strength of the mean reversion, $\sigma^F$ is the intrinsic volatility of
        the price perpetual futures, while
        $\\rho$ is a correlation coefficient between noise processes driving the price of ETH-DAI and the Perp.

        2. Historical data suggests that the funding rate of perps futures is correlated to market sentiment.
        See for example [this post](https://blog.kraken.com/product/quick-primer-on-funding-rates)

        We detect the trend of the market by comparing the average price over consecutive windows of time. Let $\Delta t$ be the time intervals at which $P_t$ is recorded, then we define
        $$
        \\bar P_t^k := \\frac1{k}\sum_{j=0}^{k-1} P_{t-j \cdot \Delta t}.
        $$
        For a fixed tolerance parameter $\delta > 0$,
        - The market is bearish when $\\bar P_t^k - \\bar P_{t-1}^k < -\delta$.
        - The market is neutral when $\lvert\\bar P_t^k - \\bar P_{t-1}^k \rvert < \delta$.
        - The market is bullish when $\\bar P_t^k - \\bar P_{t-1}^k > \delta$.

        The perpetual futures price is then modelled by

        - $F_t = P_t + e^{\sigma^{exp}\Delta W^F_t}, \,\,$ if the market is bullish
        - $F_t = P_t - e^{\sigma^{exp}\Delta W^F_t}, \,\,$ if the market is bearish
        - $F_t$ is mean-reverting to $P_t$ if the market is neutral

        """
    )

# ----------
# Price processes
# -----------

st.subheader("Price process of underlying and price process of perps option")
col1, col2, col3 = st.columns([0.3, 0.3, 0.3], gap="large")
with col1:
    st.write("Price process")
    # mu = st.slider("$\mu$", -1.2, 1.2, 0.0, step=0.3)  # min, max, default
    # sigma = st.slider("$\sigma$", 0.1, 0.5, 0.3, 0.2)  # min, max, default

with col2:
    st.write("Perps Price process")
    select_perp = st.selectbox(
        "How do we define the Perp price process?",
        [
            "Mean-reversion to P",
            "Funding rates correlated with market sentiment",
        ],
        index=1,
    )
    lambda_ = st.slider(
        "$\lambda$ mean-reversion parameter", 1, 200, 120
    )  # min, max, default
    sigma_f = st.slider(
        "$\sigma^F$, vol mean reverting", 0.01, 500.0, 2400.0
    )  # min, max, default
    st.write("Noise std dev when perp funding rate is correlated with market sentiment")
    sigma_noise = st.slider(
        "$\sigma^{exp}$", 1.0, 8.0, 6.5, step=0.5
    )  # min, max, default


with col3:
    st.write(f"Number of Monte Carlo simulations: {n_mc}")


price_paths = get_gbm(mu, sigma, dt, n_steps, p0, seed, n_mc)
time = np.arange(0, n_steps + 1) * dt
df_price_paths = pd.DataFrame(
    price_paths.T, columns=["mc{}".format(i) for i in range(price_paths.shape[0])]
)
df_price_paths = df_price_paths.assign(time=time, underlying="spot")
# perps_price_paths = get_perps_price_realistic(
#         price_paths=price_paths,
#         sigma=sigma_f,
#         sigma_noise=sigma_noise,
#         dt=dt,
#         window_length=5,
#         delta=5,
#         lambda_=lambda_,
#     )
# if select_perp == "Mean-reversion to P":
r_perp = 0.005
basis_mean = args.premium if args.premium is not None else r_perp / kappa
perps_price_paths = get_perps_price_basis_ou(
    price_paths, dt=dt, lambda_=LAMBDA_B, sigma_b=SIGMA_B,
    m=basis_mean, b0=r_perp / kappa, seed=seed + 1,
)
# else:
# perps_price_paths = get_perps_price_realistic(
#     price_paths=price_paths,
#     sigma=sigma_f,
#     sigma_noise=sigma_noise,
#     dt=dt,
#     window_length=5,
#     delta=5,
#     lambda_=lambda_,
#     seed=seed,
# )
# elif select_perp == "Mean-reversion to P":
#     perps_price_paths = get_perps_price_mean_rev(
#         price_paths,
#         lambda_=lambda_,
#         sigma=sigma_f,
#         dt=dt,
#         r=0.01,
#         kappa=kappa,
#     )
# elif select_perp == "Mean-reversion to non-arbitrage price":
#     perps_price_paths = get_perps_price_mean_rev_to_non_arb(
#         price_paths,
#         lambda_=lambda_,
#         sigma=sigma_f,
#         dt=dt,
#         r=0.01,
#         kappa=kappa,
#     )


df_perps_price_paths = pd.DataFrame(
    perps_price_paths.T,
    columns=["mc{}".format(i) for i in range(perps_price_paths.shape[0])],
)
df_perps_price_paths = df_perps_price_paths.assign(time=time, underlying="perp")

df_price = pd.concat([df_price_paths, df_perps_price_paths])
df_price_melted = pd.melt(
    df_price,
    id_vars=["underlying", "time"],
    value_vars=["mc{}".format(i) for i in range(perps_price_paths.shape[0])],
    value_name="price",
)
_, price_plot_col, _ = st.columns([0.1, 0.8, 0.1])


COLOR_MAP = {"spot": "#636EFA",  # plotly blue
             "perp": "#EF553B"}

with price_plot_col:
    samples = ["mc{}".format(i) for i in range(20)]
    fig_price = px.line(
        df_price_melted[df_price_melted["variable"].isin(samples)],
        x="time",
        y="price",
        line_group="variable",
        color="underlying",  # use "variable" here if you want each mc# a different color
        color_discrete_map=COLOR_MAP,
    )

    fig_price.write_image(f"results/price_mu{mu}_sigma{sigma}.png", engine="kaleido", scale=2)
    st.plotly_chart(fig_price, use_container_width=True)

st.markdown("""---""")
st.markdown("Difference of price, $F_t - P_t$, proportional to perps fee rate. ")
col_seed, col, _ = st.columns([0.1, 0.8, 0.1], gap="medium")
with col_seed:
    seed0 = st.slider("seed price sample", 0, n_mc, 8700, step=1)
with col:
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=time, y=price_paths[seed0, :], name="ETH price"), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=perps_price_paths[seed0, :] - price_paths[seed0],
            name="F<sub>t</sub> - P<sub>t</sub>",
            fill="tozeroy",
        ),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="time")

    # Set y-axes titles
    fig.update_yaxes(title_text="F<sub>t</sub> - P<sub>t</sub>", secondary_y=True)
    fig.update_yaxes(title_text="ETH-USD price", secondary_y=False)
    fig.write_image(f"results/funding_rate_sentiment_mu{mu}_sigma{sigma}.png", engine="kaleido", scale=2)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------
# Utilisation of collateral Pool
# ----------------------------------
st.subheader("Correlation between price and Utilisation of collateral pool")
with st.expander("Impact of lending on the spot"):
    st.markdown(
        """
        If market participants are bullish on the price of ETH/DAI and wish to enter
        a leveraged position via a lending protocol,
        they are depositing ETH as collateral and borrowing DAI.

        On the other hand, opening a long ETH loan position, ceteris paribus decreases
        utilisation and hence interest rate for lending ETH,
        and increases utilisation and therefore interest rate for borrowing DAI
        (how these rates change depends on the elasticity of market supply and demand curves
        and interest rate mechanism).
        This implies that the PnL of a loan position decreases, and the liquidation risk increases.

        We model model the utilisation of the DAI and ETH pool in terms of the ETH price as

        $$
        \mathcal U_t^{ABC} = g(\\nu_t^{ABC}), \quad \\text{where }
        \quad d\\nu^{ABC}_t = \\alpha^{ABC} dP_t, g(\\nu^{ABC}_0) = u_0^{ABC}
        $$
        where $g(x) = (1+e^{-x})^{-1}$ is the sigmoid function and $ABC$ denotes ETH or DAI.

        In other words $\\alpha^{ABC}$ measures relative change of $\\nu$ with respect
        to change of $P$ and use sigmoid function $g$ to map
        $\\nu$ to the utilisation $U\in[0,1]$. We use the sigmoid function as it is
        convenient to bring any real value to the interval $(0,1)$.
        When $\\alpha^{ETH}$ is negative and $\\alpha^{DAI}$ is positive, an increase in
        the price of ETH decreases the utilisation of ETH and
        increases the utilisation of DAI, as speculated above.
        """
    )

col1, col2, col3 = st.columns([0.2, 0.2, 0.8], gap="medium")
with col1:
    st.write("Impact of ETH price on ETH utilisation")
    alpha_eth = st.slider("$\\alpha^{ETH}$", -3.0, 3.0, 1.6, step=0.01)
    u0_eth = st.slider("$u_0^{ETH}$", 0.0, 1.0, 0.88, step=0.01)
    st.write("Random seed")
    seed_ = st.slider("seed", 0, n_mc, 0, step=1)
with col2:
    st.write("Impact of ETH price on DAI utilisation")
    alpha_dai = st.slider("$\\alpha^{DAI}$", -3.0, 3.0, 2.15, step=0.01)
    u0_dai = st.slider("$u_0^{DAI}$", 0.0, 1.0, 0.81, step=0.01)

u_eth = get_utilisation(price_paths=price_paths, u0=u0_eth, a=alpha_eth)
u_dai = get_utilisation(price_paths=price_paths, u0=u0_dai, a=alpha_dai)
r_collateral_eth = vect_irm(
    **IRM_ETH, utilisation=u_eth, collateral=True
)
r_debt_dai = vect_irm(
    **IRM_DAI, utilisation=u_dai, collateral=False
)
# Create figure with secondary y-axis
_, col, _ = st.columns([0.1, 0.8, 0.1])
with col:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=time,
            y=price_paths[seed_, :],
            name="ETH price",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=time, y=u_eth[seed_, :], name="ETH pool utilisation"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=time, y=u_dai[seed_, :], name="DAI pool utilisation"),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="time")

    # Set y-axes titles
    fig.update_yaxes(title_text="pool utilisation", secondary_y=True)
    fig.update_yaxes(title_text="ETH-USD price", secondary_y=False)
    # fig.write_image(f"results/price_utilisation_mu{mu}_sigma{sigma}.png")

    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# Interest rate model
# --------------------------
st.markdown("---")
st.write("Interest rate model")
r_0 = 0
r_1 = 0.04
r_2 = 2.5
u_optimal = 0.45

utilisation = np.linspace(0, 1, 100)

rate = vect_irm(
    u_optimal=u_optimal,
    r_0=r_0,
    r_1=r_1,
    r_2=r_2,
    utilisation=utilisation,
    collateral=False,
)

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.markdown(
        """
        At this point we define the interest rate defined by the protocol. In Aave, this is given by

        $$r_{IRM} (\mathcal U) = r_0 + r_1 \cdot \\frac{\mathcal U}{\mathcal U^*} \cdot \, \mathbf 1_{\mathcal U \leq \mathcal U^*} + \left(r_1 + r_2 \cdot \\frac{\mathcal U - \mathcal U^*}{1-\mathcal U^*}\\right) \cdot \, \mathbf 1_{\mathcal U > \mathcal U^*}$$

        for $r_0,r_1,r_2 >0$ and $\mathcal U^* \in [0,1)$ the targeted pool utilisation by the protocol.

        We set $r_0 = %.2f, r_1 = %.2f, r_2 = %.2f, U^*=%.2f$

        """
        % (r_0, r_1, r_2, u_optimal)
    )
with col2:
    fig = px.line(x=utilisation, y=rate, labels={"x": "utilisation", "y": "rate"})
    st.plotly_chart(fig)

r_collateral_eth = vect_irm(
    **IRM_ETH, utilisation=u_eth, collateral=True
)
r_debt_dai = vect_irm(
    **IRM_DAI, utilisation=u_dai, collateral=False
)

st.markdown("""---""")
st.write("Price and Interest rates")
_, col, _ = st.columns([0.1, 0.8, 0.1])
with col:
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=time, y=price_paths[seed_, :], name="ETH price"), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=time, y=r_collateral_eth[seed_, :], name="ETH collateral rate"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=time, y=r_debt_dai[seed_, :], name="DAI debt rate"),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="time")

    # Set y-axes titles
    fig.update_yaxes(title_text="interest rates", secondary_y=True)
    fig.update_yaxes(title_text="ETH-USD price", secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# Loan position and perps params
# --------------------------------
st.markdown("""---""")
st.subheader("Loan position and long Perps position parameters")
col1, col2, _ = st.columns([0.2, 0.2, 0.8], gap="medium")
with col1:
    st.write("Initial Loan-To-Value")
    ltv0 = st.slider(r"$\theta^0$", 0.5, 1.0, 0.75)  # min, max, default
    st.write("Liquidation threshold in Lending protocol")
    lt = st.slider(r"$\theta$", 0.5, 1.0, 0.95)  # min, max, default
with col2:
    st.write("Maintenance margin account size in Perps option trading")
    lt_f = st.slider(r"$\theta^F$", 0.0, 1.0, 0.05)  # min, max, default


# ------------------------------------
# Liquidations - PnL and Stopping time
# ------------------------------------
st.markdown("""---""")
st.subheader("PnL - funding fee - liquidation times")

liquidation_times_lending = get_liquidation_times(
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    price_paths=price_paths,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
liquidation_times_perps = get_liquidation_times_perp(
    dt=dt,
    kappa=kappa,
    lt_f=lt_f,
    ltv0=ltv0,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r=r_debt_dai,
    r_debt_dai=r_debt_dai,
)
pnl_lending_position = get_pnl_lending_position(
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    price_paths=price_paths,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
pnl_perps = get_pnl_perps_after_liquidation(
    dt=dt,
    kappa=kappa,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r=0,  # r_debt_dai,
    lt_f=lt_f,
    ltv0=ltv0,
    r_debt_dai=r_debt_dai,
)


df_pnl_cperp = pd.DataFrame(
    pnl_lending_position.T,
    columns=["mc{}".format(i) for i in range(pnl_lending_position.shape[0])],
)
df_pnl_cperp = df_pnl_cperp.assign(time=time, derivative="cPerp")
df_pnl_perp = pd.DataFrame(
    pnl_perps.T, columns=["mc{}".format(i) for i in range(pnl_perps.shape[0])]
)
df_pnl_perp = df_pnl_perp.assign(time=time, derivative="Perp")
df_pnl = pd.concat([df_pnl_perp, df_pnl_cperp])
df_pnl_melted = pd.melt(
    df_pnl,
    id_vars=["time", "derivative"],
    value_vars=["mc{}".format(i) for i in range(pnl_lending_position.shape[0])],
    value_name="PnL",
)
df_ = df_pnl_melted[df_pnl_melted["time"] == t]
dct = {'mu':mu, 'sigma':sigma, 'time':t, }

st.markdown("Comparison of PnL at time {}".format(t))
_, price_col, _ = st.columns([0.1, 0.8, 0.1])
with price_col:
    pnl_t = df_pnl_melted[df_pnl_melted["time"] == t]
    # result = pd.DataFrame(mu,sigma,t,pnl_t.mean(),pnl_t.std())
    fig = px.histogram(
        pnl_t,
        x="PnL",
        color="derivative",
        opacity=0.5,
        nbins=200,
        barmode="overlay",
        # range_x=(-400, 200),
    )
    names = {"Perp": "Perp", "cPerp": "Loan position"}
    fig.for_each_trace(
        lambda x: x.update(name=names[x.name], legendgroup=names[x.name])
    )
    # fig.write_image(f"results/pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}.png")
    # if "realistic":
    fig.write_image(f"results/pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}_time{t}.png")
    st.plotly_chart(fig, use_container_width=True)
    table = (
        df_pnl_melted[df_pnl_melted["time"] == t][["derivative", "PnL"]]
        .groupby("derivative")
        .agg({"PnL": [np.mean, np.std, np.median,
                      lambda x: -np.quantile(x, 0.05),
                      lambda x: -x[x <= np.quantile(x, 0.05)].mean()]})
        .reset_index()
    )
    table = table.replace(to_replace="cPerp", value="Loan position")
    table.columns = ["derivative", "mean PnL", "std dev PnL", "median PnL", "VaR 5%", "ES 5%"]
    st.dataframe(table, hide_index=True, use_container_width=True)
    # if realistic:
    table.to_csv(f'results/pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}_time{t}.csv',
                 index=False,
                 mode="a",  # write header only once
                 float_format="%.6f",
                 )
    # table.to_csv(f'results/pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}.csv',
    #     index=False,
    #     mode="a",  # write header only once
    #     float_format="%.6f",
    # )
    # df_pnl_melted[df_pnl_melted["time"] == t][["derivative", "PnL"]].groupby(
    #     "derivative"
    # ).agg({"PnL": [np.mean, np.std]}).reset_index().to_csv(
    #     f"results/pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}.csv"
    # )

    cols = ["mc{}".format(i) for i in range(pnl_perps.shape[0])]
    pnl_diff = df_pnl_cperp[cols] - df_pnl_perp[cols]
    pnl_diff = pnl_diff.assign(time=time)
    pnl_diff_melted = pd.melt(
        pnl_diff, id_vars=["time"], value_vars=cols, value_name="diff_pnl"
    )
    fig = px.histogram(
        pnl_diff_melted[pnl_diff_melted["time"] == t],
        x="diff_pnl",
        opacity=0.5,
        nbins=200,
        labels={"diff_pnl": "(PnL loan position) - (PnL long perp position)"},
    )
    fig.write_image(f"results/diff_pnl_mu{mu}_sigma{sigma}_thetaF{lt_f}.png")
    st.plotly_chart(fig, use_container_width=True)
    table = (
        pnl_diff_melted[pnl_diff_melted["time"] == t]["diff_pnl"]
        .describe()
        .loc[["mean", "std"]]
    )
    table = table.replace(to_replace="cPerp", value="Loan position")
    st.dataframe(table, use_container_width=True)


# -------------
# Funding fee
# -------------
st.markdown("""---""")
st.markdown("Comparison of Funding Fees at time {}".format(t))
funding_fee_perp = get_funding_fee_perps(
    dt=dt,
    kappa=kappa,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r_debt_dai=r_debt_dai,
)
_, funding_fee_lending = decompose_pnl_lending_position(
    price_paths=price_paths,
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
df_funding_fee_cperp = pd.DataFrame(
    funding_fee_lending.T,
    columns=["mc{}".format(i) for i in range(funding_fee_lending.shape[0])],
)
df_funding_fee_cperp = df_funding_fee_cperp.assign(time=time, derivative="cPerp")
df_funding_fee_perp = pd.DataFrame(
    funding_fee_perp.T,
    columns=["mc{}".format(i) for i in range(funding_fee_perp.shape[0])],
)
df_funding_fee_perp = df_funding_fee_perp.assign(time=time, derivative="Perp")
df_funding_fee = pd.concat([df_funding_fee_perp, df_funding_fee_cperp])
df_funding_fee_melted = pd.melt(
    df_funding_fee,
    id_vars=["time", "derivative"],
    value_vars=["mc{}".format(i) for i in range(funding_fee_lending.shape[0])],
    value_name="funding_fee",
)


_, price_col, _ = st.columns([0.1, 0.8, 0.1])
with price_col:
    fig = px.histogram(
        df_funding_fee_melted[df_funding_fee_melted["time"] == t],
        x="funding_fee",
        color="derivative",
        opacity=0.5,
        nbins=200,
        barmode="overlay",
        # range_x=(-15, 15),
        labels={"cPerp": "Loan position", "funding_fee": "funding fee"},
    )
    names = {"Perp": "Perp", "cPerp": "Loan position"}
    fig.for_each_trace(
        lambda x: x.update(name=names[x.name], legendgroup=names[x.name])
    )
    # if realistic:
    fig.write_image(f"results/funding_fee_mu{mu}_sigma{sigma}_thetaF{lt_f}_time{t}.png")
    # fig.write_image(f"results/funding_fee_mu{mu}_sigma{sigma}_thetaF{lt_f}.png")
    st.plotly_chart(fig, use_container_width=True)

    table = (
        df_funding_fee_melted[df_pnl_melted["time"] == t][["derivative", "funding_fee"]]
        .groupby("derivative")
        .agg({"funding_fee": [np.mean, np.std]})
        .reset_index()
    )
    table.columns = ["derivative", "mean funding fee", "std dev funding fee"]
    table = table.replace(to_replace="cPerp", value="Loan position")
    # table.to_csv(f'results/funding_fee_mu{mu}_sigma{sigma}_thetaF{lt_f}.csv',
    #              index=False,
    #              mode="a",
    #              float_format="%.6f",
    #              )
    # st.dataframe(table, hide_index=True, use_container_width=True)

    table.to_csv(f'results/funding_fee_mu{mu}_sigma{sigma}_thetaF{lt_f}_time{t}.csv',
                 index=False,
                 mode="a",
                 float_format="%.6f",
                 )
    st.dataframe(table, hide_index=True, use_container_width=True)

    # df_funding_fee_melted[df_pnl_melted["time"] == t][
    #     ["derivative", "funding_fee"]
    # ].groupby("derivative").agg(
    #     {"funding_fee": [np.mean, np.std]}
    # ).reset_index().to_csv(
    #     f"results/funding_fee_mu{mu}_sigma{sigma}_thetaF{lt_f}.csv"
    # )


# --------------
# Stopping time
# --------------
st.markdown("""---""")
st.markdown("CDF of Liquidation time")
liquidation_times_lending = get_liquidation_times(
    dt=dt,
    lt=lt,
    ltv0=ltv0,
    price_paths=price_paths,
    r_collateral_eth=r_collateral_eth,
    r_debt_dai=r_debt_dai,
)
liquidation_times_perps = get_liquidation_times_perp(
    dt=dt,
    r=r_debt_dai,
    kappa=kappa,
    lt_f=lt_f,
    ltv0=ltv0,
    perps_price_paths=perps_price_paths,
    price_paths=price_paths,
    r_debt_dai=r_debt_dai,
)
df_liquidation_times_cperps = pd.DataFrame(
    {"liquidation_times": liquidation_times_lending.flatten(), "mc": np.arange(n_mc)}
)
df_liquidation_times_cperps = df_liquidation_times_cperps.assign(derivative="cPerp")
df_liquidation_times_perps = pd.DataFrame(
    {"liquidation_times": liquidation_times_perps.flatten(), "mc": np.arange(n_mc)}
)
df_liquidation_times_perps = df_liquidation_times_perps.assign(derivative="Perp")
df = pd.concat([df_liquidation_times_perps, df_liquidation_times_cperps])
_, col, _ = st.columns([0.1, 0.8, 0.1])
with col:
    fig = px.ecdf(
        df,
        x="liquidation_times",
        color="derivative",
        labels={"liquidation_times": "liquidation times"},
        range_x=(0, 1),
    )
    names = {"Perp": "Perp", "cPerp": "Loan position"}
    fig.for_each_trace(
        lambda x: x.update(name=names[x.name], legendgroup=names[x.name])
    )
    fig.write_image(f"results/liquidation_times_mu{mu}_sigma{sigma}.png")
    st.plotly_chart(fig, use_container_width=True)
