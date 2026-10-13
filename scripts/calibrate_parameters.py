import gzip
import json

from perp.constants import OUT_PRICE_PATH


def calibrate_gbm(price_series, steps_per_day=3, days_per_year=365):

    prices = pd.Series(price_series).dropna().astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    dt = 1 / (steps_per_day * days_per_year)

    mean_r = log_returns.mean()
    var_r = ((log_returns - mean_r) ** 2).mean()  # MLE variance, divide by n

    sigma_hat = np.sqrt(var_r / dt)
    mu_hat = mean_r / dt + 0.5 * sigma_hat**2

    return {
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "mean_log_return_per_step": mean_r,
        "var_log_return_per_step": var_r,
        "n_returns": len(log_returns)
    }


def calibrate_perp_mr_exact(df, spot_col="spot", perp_col="perp", dt=1/1095):
    """
    Approximate exact discretization under piecewise-constant spot over each interval:
        F_{t+dt} = exp(-lambda dt) F_t + (1-exp(-lambda dt)) P_t + u_t

    Equivalently:
        Y_t = F_{t+dt} - P_t = phi (F_t - P_t) + u_t
        phi = exp(-lambda dt)
    """
    data = df[[spot_col, perp_col]].dropna().copy()

    data["x"] = data[perp_col] - data[spot_col]
    data["y"] = data[perp_col].shift(-1) - data[spot_col]
    data = data.dropna()

    x = data["x"].values
    y = data["y"].values

    # OLS without intercept
    phi_hat = np.dot(x, y) / np.dot(x, x)
    residuals = y - phi_hat * x
    var_u = np.mean(residuals**2)

    # numerical guard
    if phi_hat <= 0 or phi_hat >= 1:
        return {
            "phi_hat": phi_hat,
            "lambda_hat": np.nan,
            "sigma_F_hat": np.nan,
            "residual_var": var_u,
            "n_obs": len(data),
            "warning": "phi_hat outside (0,1), implying no valid mean-reverting lambda."
        }

    lambda_hat = -np.log(phi_hat) / dt
    sigma_F_hat = np.sqrt(2 * lambda_hat * var_u / (1 - np.exp(-2 * lambda_hat * dt)))

    return {
        "phi_hat": phi_hat,
        "lambda_hat": lambda_hat,
        "sigma_F_hat": sigma_F_hat,
        "residual_var": var_u,
        "n_obs": len(data)
    }

def beta_no_intercept(df, y_col, x_col):
    x = df[x_col]
    y = df[y_col]

    # drop missing values pairwise
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    beta = (x * y).sum() / (x**2).sum()
    return beta

import numpy as np
import pandas as pd

def calibrate_rho(df, spot_col="P", perp_col="F",
                  mu=None, sigma=None, lambda_=None, sigmaF=None,
                  dt=1/1095):

    if any(x is None for x in [mu, sigma, lambda_, sigmaF, dt]):
        raise ValueError("mu, sigma, lambda_, sigmaF, and dt must all be provided.")

    data = df[[spot_col, perp_col]].dropna().copy()
    data = data.rename(columns={spot_col: "P", perp_col: "F"}).reset_index(drop=True)

    P = data["P"][:-1].reset_index(drop=True)
    F = data["F"][:-1].reset_index(drop=True)

    dP = data["P"].diff().dropna().reset_index(drop=True)
    dF = data["F"].diff().dropna().reset_index(drop=True)

    dW_spot_hat = (dP - mu * P * dt) / (sigma * P)
    dW_perp_hat = (dF - lambda_ * (P - F) * dt) / sigmaF

    rho_hat = np.corrcoef(dW_spot_hat, dW_perp_hat)[0, 1]

    return rho_hat

if __name__ == '__main__':
    results = []
    with gzip.open('../data/coinalyze.jsonl.gz', "rt", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)['history']
            results = pd.DataFrame(line)[['t','c']]
    results = results.set_index(['t'])
    results.index = pd.to_datetime(results.index, unit="s", utc=True).tz_convert("UTC")
    results.columns = ['priceF']

    prices = []
    with gzip.open(OUT_PRICE_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if line['base'] == 'ETH':
                prices.append(line)
    prices = pd.DataFrame(prices).set_index(['t'])['price']
    prices.index = pd.to_datetime(prices.index, unit="s", utc=True).tz_convert("UTC")

    para = calibrate_gbm(prices)
    print(para['mu_hat'],para['sigma_hat'])

    results['spot'] = prices

    para_perps = calibrate_perp_mr_exact(results, spot_col='spot', perp_col='priceF')
    beta = beta_no_intercept(results, y_col='priceF', x_col='spot')

    print(beta)
    print(para_perps['lambda_hat'],para_perps['sigma_F_hat'])
    #
    # rho = calibrate_rho(results, spot_col='spot', perp_col='priceF',
    #                     mu=para['mu_hat'], sigma=para['sigma_hat'],
    #                     lambda_=para_perps['lambda_hat'],sigmaF=para_perps['sigma_F_hat'])
    # print(rho)