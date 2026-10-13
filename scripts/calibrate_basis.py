"""Calibrate the relative perpetual basis b_t = (F_t - P_t) / P_t (Reviewer 2, comment 2.4b-d).

Model (simulation, Section 5):
    db_t = lambda (m - b_t) dt + sigma_b dW_t,   F_t = P_t (1 + b_t).

Outputs, on the 8-hour grid of the composite ETH spot price:
  * empirical moments of b_t,
  * exact OU (AR(1)) estimates of lambda, the long-run mean and sigma_b,
  * the 90th percentile of the 8-hour Binance ETHUSDT funding rate, used as the
    fixed premium m in the bull scenario of Section 5.2.
Existing scripts are not modified; data loading follows calibrate_parameters.py.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from perp.constants import COINGLASS_PATH, DATA_PATH, OUT_PRICE_PATH

DT = 1 / (3 * 365)  # 8 hours in years
SAMPLE_START = pd.Timestamp("2025-04-17 16:00", tz="UTC")  # empirical sample, Section 6.1
SAMPLE_END = pd.Timestamp("2025-09-12 00:00", tz="UTC")
FUNDING_EXCHANGE, FUNDING_SYMBOL = "Binance", "ETHUSDT_PERP.A"
BULL_QUANTILE = 0.90


def load_perp_price() -> pd.Series:
    """Binance coin-margined ETH perpetual (ETHUSD_PERP.A) from Coinalyze 4-hour candles.

    Coinalyze stamps a candle with its start time, so the price at time t is the candle's
    open ("o"); its close ("c") is 4 hours later and would misalign F_t with the spot P_t.
    """
    with gzip.open(DATA_PATH / "coinalyze.jsonl.gz", "rt", encoding="utf-8") as f:
        history = json.loads(f.readline())["history"]
    df = pd.DataFrame(history)[["t", "o"]].set_index("t")
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    return df["o"].rename("F")


def load_spot(base: str = "ETH") -> pd.Series:
    rows = []
    with gzip.open(OUT_PRICE_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["base"] == base:
                rows.append(d)
    s = pd.DataFrame(rows).set_index("t")["price"]
    s.index = pd.to_datetime(s.index, unit="s", utc=True)
    return s.rename("P")


def load_funding(exchange: str, symbol: str) -> pd.Series:
    rows = []
    with gzip.open(COINGLASS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["exchange_name"] == exchange and d["symbol"] == symbol:
                rows.append(d)
    s = pd.DataFrame(rows).set_index("t")["funding_rate"] / 100  # percent -> decimal
    s.index = pd.to_datetime(s.index, unit="s", utc=True)
    return s.groupby(level=0).last().rename("R")


def fit_ou(b: pd.Series, dt: float = DT) -> dict:
    """Exact discretisation of the OU process: b_{t+dt} = a + phi b_t + e_t."""
    x, y = b.values[:-1], b.values[1:]
    X = np.column_stack([np.ones_like(x), x])
    (a, phi), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - (a + phi * x)
    var_e = resid.var(ddof=2)
    if not 0 < phi < 1:
        return {"phi": phi, "warning": "phi outside (0, 1): no mean-reverting fit"}
    lam = -np.log(phi) / dt
    return {
        "phi": phi,
        "lambda": lam,
        "half_life_hours": np.log(2) / lam * 365 * 24,
        "long_run_mean": a / (1 - phi),
        "sigma_b": np.sqrt(2 * lam * var_e / (1 - phi**2)),
        "n_obs": len(x),
    }


def moments(b: pd.Series) -> dict:
    return {
        "mean": b.mean(),
        "std": b.std(),
        "acf1": b.autocorr(1),
        "q05": b.quantile(0.05),
        "q50": b.quantile(0.50),
        "q95": b.quantile(0.95),
        "n_obs": int(b.count()),
    }


if __name__ == "__main__":
    spot = load_spot()
    df = pd.concat([spot, load_perp_price()], axis=1, join="inner")
    df = df[(df.index >= SAMPLE_START) & (df.index <= SAMPLE_END)]
    b = (df["F"] / df["P"] - 1).rename("basis")

    funding = load_funding(FUNDING_EXCHANGE, FUNDING_SYMBOL).reindex(spot.index)
    funding = funding[(funding.index >= SAMPLE_START) & (funding.index <= SAMPLE_END)].dropna()

    out = {
        "sample": [str(df.index.min()), str(df.index.max())],
        "basis_moments": moments(b),
        "ou_fit": fit_ou(b),
        "funding_8h": {
            "exchange": FUNDING_EXCHANGE,
            "symbol": FUNDING_SYMBOL,
            "mean": funding.mean(),
            "q50": funding.quantile(0.5),
            f"q{int(BULL_QUANTILE * 100)}": funding.quantile(BULL_QUANTILE),
            "n_obs": int(funding.count()),
        },
    }
    out["bull_premium_m"] = out["funding_8h"][f"q{int(BULL_QUANTILE * 100)}"]

    Path("results").mkdir(exist_ok=True)
    with open("results/basis_calibration.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
