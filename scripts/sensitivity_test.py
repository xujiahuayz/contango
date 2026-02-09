import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scripts.play import (
    decompose_pnl_lending_position,
    get_funding_fee_perps,
    get_gbm,
    get_perps_price_mean_rev,
    get_perps_price_realistic,
    get_pnl_lending_position,
    get_pnl_perps_after_liquidation,
    get_utilisation,
    vect_irm,
    get_idx_from_t
)

def simulate_results(mu = 0, sigma = 0.9,
                    lambda_ = 100,
                    sigma_f = 100,
                    kappa = 1,
                    ):
    # ----------
    # Price processes
    # -----------
    price_paths = get_gbm(mu, sigma, dt, n_steps, p0, seed0, n_mc)
    time = np.arange(0, n_steps + 1) * dt
    if select_perp == "Mean-reversion to P":
        perps_price_paths = get_perps_price_mean_rev(
            price_paths, dt=dt, kappa=kappa, sigma=sigma_f, lambda_=lambda_, r=0.005
        )
    else:
        perps_price_paths = get_perps_price_realistic(
            price_paths=price_paths,
            sigma=sigma_f,
            sigma_noise=sigma_noise,
            dt=dt,
            window_length=5,
            delta=5,
            lambda_=lambda_,
        )

# ------------------------------------
# Liquidations - PnL and Stopping time
# ------------------------------------
    u_eth = get_utilisation(price_paths=price_paths, u0=u0_eth, a=alpha_eth)
    u_dai = get_utilisation(price_paths=price_paths, u0=u0_dai, a=alpha_dai)
    r_collateral_eth = vect_irm(
        u_optimal=u_optimal, r_0=r_0, r_1=r_1, r_2=r_2, utilisation=u_eth, collateral=True
    )
    r_debt_dai = vect_irm(
        u_optimal=u_optimal, r_0=r_0, r_1=r_1, r_2=r_2, utilisation=u_dai, collateral=False
    )
    pnl_lending_position = get_pnl_lending_position(
        dt=dt,
        lt=lt,
        ltv0=ltv0,
        price_paths=price_paths,
        r_collateral_eth=r_collateral_eth,
        r_debt_dai=r_debt_dai,
    )
    pnl_lending_position = pd.DataFrame(pnl_lending_position).T.assign(time=time)
    pnl_perps = get_pnl_perps_after_liquidation(
        dt=dt,
        kappa=kappa,
        perps_price_paths=perps_price_paths,
        price_paths=price_paths,
        r=0,
        lt_f=lt_f,
        ltv0=ltv0,
        r_debt_dai=r_debt_dai,
    )
    pnl_perps = pd.DataFrame(pnl_perps).T.assign(time=time)


    # -------------
    # Funding fee
    # -------------
    funding_fee_perp = get_funding_fee_perps(
        dt=dt,
        kappa=kappa,
        perps_price_paths=perps_price_paths,
        price_paths=price_paths,
        r_debt_dai=r_debt_dai,
    )
    funding_fee_perp = pd.DataFrame(funding_fee_perp).T.assign(time=time)
    _, funding_fee_lending = decompose_pnl_lending_position(
        price_paths=price_paths,
        dt=dt,
        lt=lt,
        ltv0=ltv0,
        r_collateral_eth=r_collateral_eth,
        r_debt_dai=r_debt_dai,
    )
    funding_fee_lending = pd.DataFrame(funding_fee_lending).T.assign(time=time)
    result = {'mu':mu, 'sigma':sigma, 't': t, 'sigma_f':sigma_f, 'lambda_':lambda_,
              'kappa': kappa}
    result.update({'pnl_perps': np.mean(pnl_perps[pnl_perps['time']==t].drop('time', axis=1))})
    result.update({'pnl_lending': np.mean(pnl_lending_position[pnl_lending_position['time']==t].drop('time', axis=1))})
    result.update({'pnl_perps_std': np.std(pnl_perps[pnl_perps['time']==t].drop('time', axis=1).values)})
    result.update({'pnl_lending_std': np.std(pnl_lending_position[pnl_lending_position['time']==t].drop('time', axis=1).values)})
    result.update({'funding_fee_perps': np.mean(funding_fee_perp[funding_fee_perp['time']==t].drop('time', axis=1))})
    result.update({'funding_fee_lending': np.mean(funding_fee_lending[funding_fee_lending['time']==t].drop('time', axis=1))})
    result.update({'funding_fee_perps_std': np.std(funding_fee_perp[funding_fee_perp['time'] == t].drop('time', axis=1).values)})
    result.update(
        {'funding_fee_lending_std': np.std(funding_fee_lending[funding_fee_lending['time'] == t].drop('time', axis=1).values)})

    return pd.DataFrame([result])

def heat_map(results, col1, col2, title, x_name1, y_name2,extent):

    results = results.set_index([y_name2,x_name1])
    colors = ["#EF553B", "#FFFFFF", "#636EFA"]
    cmap = LinearSegmentedColormap.from_list("custom", colors)
    fig, ax = plt.subplots()
    im = ax.imshow(
        (results[col1]-results[col2]).unstack(),
        origin="lower",
        aspect="auto",
        extent= extent,
        cmap = cmap,
    )

    ax.set_xlabel(x_name1)
    ax.set_ylabel(y_name2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title)
    plt.savefig(f'../figures/sensitivity_{title}.png')
    plt.show()

if __name__ == '__main__':
    dt = 0.01
    p0 = 2_000
    n_steps = 100
    seed = 8765
    alpha_eth = -0.15
    alpha_dai = 0.05
    u0_eth = 0.4
    u0_dai = 0.4
    r_0 = 0.0
    r_1 = 0.04
    r_2 = 2.5
    u_optimal = 0.45
    lt = 0.85
    ltv0 = 0.75
    n_mc = 10000
    r = 0.05
    kappa = 1
    sigma_noise = 6.5
    select_perp = "Mean-reversion to P"
    t = 0.25
    seed0 = 8767
    lt_f = 0.5

    ls_mu = np.linspace(-0.3, 0.3, 20)
    ls_sigma = np.linspace(0.1, 1.3, 20)
    ls_lambda = np.linspace(50, 150, 20)
    ls_sigmaF = np.linspace(50, 150, 20)
    ls_kappa = np.linspace(0.1, 10, 20)
    results = pd.DataFrame()
    for mu in ls_mu:
        for sigma in ls_sigma:
            result = simulate_results(mu=mu, sigma=sigma)
            results = pd.concat([results, result], axis=0,ignore_index=True)
    # results.to_csv('../results/simulation_results_muSigma.csv')

    results = pd.read_csv('../results/simulation_results_muSigma.csv')
    extent = [ls_mu.min(), ls_mu.max(), ls_sigma.min(), ls_sigma.max()]
    heat_map(results, 'pnl_lending','pnl_perps', "mean PnL difference (loan-perps)",
             'mu','sigma',extent)
    heat_map(results, 'pnl_lending_std', 'pnl_perps_std', 'std PnL difference (loan-perps)',
             'mu','sigma',extent)
    heat_map(results, 'funding_fee_lending','funding_fee_perps', "funding fee difference (loan-perps)",
             'mu','sigma',extent)
    heat_map(results, 'funding_fee_lending_std','funding_fee_perps_std', "funding fee std difference (loan-perps)",
             'mu','sigma',extent)

