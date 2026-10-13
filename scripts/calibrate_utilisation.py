"""Calibrate the utilisation map (Reviewer 2, comment 2.4e).

Model (simulation, Section 5):
    U_t = g(nu_t),  g(x) = 1 / (1 + exp(-x)),  d nu_t = alpha d log P_t,  nu_0 = logit(u0).

On the 8-hour grid (16 March to 12 September 2025):
  * alpha: OLS slope of the change in logit(U_t) on the ETH log return over 8 hours,
    1 day and 1 week (the weekly estimate is used in the simulations),
  * u0: sample mean utilisation.
Pools: WETH on Aave v3 Ethereum (rows with chain == "mainnet-v3") and DAI on Aave v3 Arbitrum.
Utilisation is the usage ratio debt / (debt + available liquidity), the ratio Aave's interest
rate strategy uses: with it, the supply and borrow rates in the data imply reserve factors of
15% (WETH) and 25% (DAI), matching the published values, whereas the "utilization" field and
1 - available / total imply 16.7% and 20.6%.
Existing scripts are not modified.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from perp.constants import AAVE_V3_PARAM_PATH, DAI_PATH, OUT_PRICE_PATH


def to_8h(ts_seconds: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(ts_seconds, unit="s", utc=True)).round("8h")


def load_weth_utilisation() -> pd.Series:
    rows = [json.loads(line) for line in gzip.open(AAVE_V3_PARAM_PATH, "rt")]
    df = pd.DataFrame(rows)
    df = df[(df["symbol"] == "WETH") & (df["chain"] == "mainnet-v3")]
    debt = df["totalCurrentVariableDebt"].astype(float)
    u = debt / (debt + df["availableLiquidity"].astype(float))
    s = pd.Series(u.values, index=to_8h(df["ts"]))
    return s.groupby(level=0).last().rename("U_ETH")


def load_dai_utilisation() -> pd.Series:
    rows = [json.loads(line) for line in gzip.open(DAI_PATH, "rt")]
    df = pd.DataFrame(rows)
    debt = df["totalCurrentVariableDebt"].astype(float)
    u = debt / (debt + df["availableLiquidity"].astype(float))
    s = pd.Series(u.values, index=to_8h(df["sample_timestamp"]))
    return s.groupby(level=0).last().rename("U_DAI")


def load_spot(base: str = "ETH") -> pd.Series:
    rows = [json.loads(line) for line in gzip.open(OUT_PRICE_PATH, "rt")]
    df = pd.DataFrame([r for r in rows if r["base"] == base])
    s = pd.Series(df["price"].astype(float).values, index=to_8h(df["t"]))
    return s.groupby(level=0).last().rename("P")


def logit(u: pd.Series) -> pd.Series:
    u = u.clip(1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


def fit_alpha(u: pd.Series, p: pd.Series) -> dict:
    df = pd.concat([u, p], axis=1, join="inner").dropna().sort_index()
    df = df[df.index.to_series().diff().eq(pd.Timedelta("8h")) | (df.index == df.index[0])]
    d_nu = logit(df.iloc[:, 0]).diff()
    d_logp = np.log(df["P"]).diff()
    data = pd.concat([d_nu, d_logp], axis=1).dropna()
    y, x = data.iloc[:, 0].values, data.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return {
        "alpha": coef[1],
        "alpha_se": se,
        "intercept": coef[0],
        "r2": 1 - resid.var() / y.var(),
        "u0_mean": df.iloc[:, 0].mean(),
        "u_std": df.iloc[:, 0].std(),
        "n_obs": len(x),
        "sample": [str(df.index.min()), str(df.index.max())],
    }


def fit_alpha_horizons(u: pd.Series, p: pd.Series) -> dict:
    """alpha from non-overlapping changes over 8 hours, 1 day and 1 week.

    Utilisation responds slowly to prices, so the weekly estimate is used in the simulations;
    the shorter horizons are reported for robustness.
    """
    df = pd.concat([logit(u), np.log(p)], axis=1, join="inner").dropna().sort_index()
    df.columns = ["nu", "log_p"]
    out = {}
    for label, step in [("8h", 1), ("1d", 3), ("1w", 21)]:
        d = df.iloc[::step].diff().dropna()
        x, y = d["log_p"].values, d["nu"].values
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        out[label] = {
            "alpha": coef[1],
            "alpha_se": np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum()),
            "r2": 1 - resid.var() / y.var(),
            "n_obs": len(x),
        }
    out["u0_mean"] = float(u.reindex(df.index).mean())
    return out


if __name__ == "__main__":
    p = load_spot()
    out = {
        "ETH": fit_alpha_horizons(load_weth_utilisation(), p),
        "DAI": fit_alpha_horizons(load_dai_utilisation(), p),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/utilisation_calibration.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
