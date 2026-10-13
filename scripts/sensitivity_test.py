import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
from scripts.play import (
    decompose_pnl_lending_position,
    get_funding_fee_perps,
    get_gbm,
    get_perps_price_basis_ou,
    get_pnl_lending_position,
    get_pnl_perps_after_liquidation,
    get_utilisation,
    vect_irm,
    IRM_ETH,
    IRM_DAI,
    KAPPA,
    LAMBDA_B,
    SIGMA_B,
)

def simulate_results(mu = 0, sigma = 0.7,
                    lambda_ = LAMBDA_B,
                    sigma_b = SIGMA_B,
                    kappa = KAPPA,
                    lt = 0.95, ltv0 = 0.75, alpha_eth = 1.6, alpha_dai = 2.15,
                    ):
    lt_f = 1 - lt  # perp maintenance margin matched to the loan liquidation threshold
    # ----------
    # Price processes
    # -----------
    price_paths = get_gbm(mu, sigma, dt, n_steps, p0, seed0, n_mc)
    time = np.arange(0, n_steps + 1) * dt
    perps_price_paths = get_perps_price_basis_ou(
        price_paths, dt=dt, lambda_=lambda_, sigma_b=sigma_b,
        m=0.005 / kappa, b0=0.005 / kappa, seed=seed0 + 1,
    )

# ------------------------------------
# Liquidations - PnL and Stopping time
# ------------------------------------
    u_eth = get_utilisation(price_paths=price_paths, u0=u0_eth, a=alpha_eth)
    u_dai = get_utilisation(price_paths=price_paths, u0=u0_dai, a=alpha_dai)
    r_collateral_eth = vect_irm(
        **IRM_ETH, utilisation=u_eth, collateral=True
    )
    r_debt_dai = vect_irm(
        **IRM_DAI, utilisation=u_dai, collateral=False
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
    result = {'mu':mu, 'sigma':sigma, 't': t, 'sigma_b':sigma_b, 'lambda_':lambda_,
              'kappa': kappa, 'theta': lt, 'theta0': ltv0, 'alpha_eth': alpha_eth, 'alpha_dai': alpha_dai}
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
    diff = (results[col1]-results[col2]).unstack()
    fig, ax = plt.subplots()
    im = ax.imshow(
        diff,
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
    # Same model and parameters as the baseline in play.py (Section 5.1). Paths are simulated on
    # the 8-hour grid up to the reporting time t = 0.25, at which all statistics are measured.
    dt = 1 / (3 * 365)
    p0 = 2_000
    t = 0.25
    n_steps = round(t / dt)
    t = n_steps * dt  # reporting time on the simulation grid
    alpha_eth = 1.6
    alpha_dai = 2.15
    u0_eth = 0.88
    u0_dai = 0.81
    lt = 0.95
    lt_f = 0.05
    ltv0 = 0.75
    n_mc = 10000
    seed0 = 8767

    ls_mu = np.linspace(-0.3, 0.3, 20)
    ls_sigma = np.linspace(0.1, 1.3, 20)
    ls_lambda = np.linspace(0.5 * LAMBDA_B, 2 * LAMBDA_B, 20)  # half to twice the estimates
    ls_sigma_b = np.linspace(0.5 * SIGMA_B, 2 * SIGMA_B, 20)
    ls_theta = np.linspace(0.80, 0.98, 20)
    ls_theta0 = np.linspace(0.50, 0.78, 20)
    ls_alpha_eth = np.linspace(0, 2 * alpha_eth, 20)  # zero to twice the estimates
    ls_alpha_dai = np.linspace(0, 2 * alpha_dai, 20)
    task = sys.argv[1] if len(sys.argv) > 1 else 'plot'

    if task == 'eth':
        results = pd.DataFrame()
        for mu in ls_mu:
            for sigma in ls_sigma:
                result = simulate_results(mu=mu, sigma=sigma)
                results = pd.concat([results, result], axis=0, ignore_index=True)
                results.to_csv('../results/simulation_results_muSigma.csv')

    if task == 'perp':
        results = pd.DataFrame()
        for lambda_ in ls_lambda:
            for sigma_b in ls_sigma_b:
                result = simulate_results(lambda_=lambda_, sigma_b=sigma_b)
                results = pd.concat([results, result], axis=0, ignore_index=True)
                results.to_csv('../results/simulation_results_lambdaSigmab.csv')

    if task == 'liq':
        results = pd.DataFrame()
        for theta in ls_theta:
            for theta0 in ls_theta0:
                result = simulate_results(lt=theta, ltv0=theta0)
                results = pd.concat([results, result], axis=0, ignore_index=True)
                results.to_csv('../results/simulation_results_thetaTheta0.csv')

    if task == 'util':
        results = pd.DataFrame()
        for a_eth in ls_alpha_eth:
            for a_dai in ls_alpha_dai:
                result = simulate_results(alpha_eth=a_eth, alpha_dai=a_dai)
                results = pd.concat([results, result], axis=0, ignore_index=True)
                results.to_csv('../results/simulation_results_alphaEthDai.csv')

    if task == 'plot':
        results = pd.read_csv('../results/simulation_results_muSigma.csv')
        extent = [ls_mu.min(), ls_mu.max(), ls_sigma.min(), ls_sigma.max()]
        heat_map(results, 'pnl_lending','pnl_perps', "mean PnL difference (loan-perps) for ETH parameters",
                 'mu','sigma',extent)
        heat_map(results, 'pnl_lending_std', 'pnl_perps_std', 'std PnL difference (loan-perps) for ETH parameters',
                 'mu','sigma',extent)
        heat_map(results, 'funding_fee_lending','funding_fee_perps', "funding fee difference (loan-perps) for ETH parameters",
                 'mu','sigma',extent)
        heat_map(results, 'funding_fee_lending_std','funding_fee_perps_std', "funding fee std difference (loan-perps) for ETH parameters",
                 'mu','sigma',extent)

        results = pd.read_csv('../results/simulation_results_lambdaSigmab.csv')
        results = results.rename(columns={'lambda_': 'lambda'})
        extent = [ls_lambda.min(), ls_lambda.max(), ls_sigma_b.min(), ls_sigma_b.max()]
        heat_map(results, 'pnl_lending', 'pnl_perps', "mean PnL difference (loan-perps) for perp parameters",
                 'lambda', 'sigma_b', extent)
        heat_map(results, 'pnl_lending_std', 'pnl_perps_std', 'std PnL difference (loan-perps) for perp parameters',
                 'lambda', 'sigma_b', extent)
        heat_map(results, 'funding_fee_lending', 'funding_fee_perps', "funding fee difference (loan-perps) for perp parameters",
                 'lambda', 'sigma_b', extent)
        heat_map(results, 'funding_fee_lending_std', 'funding_fee_perps_std', "funding fee std difference (loan-perps) for perp parameters",
                 'lambda', 'sigma_b', extent)

        results = pd.read_csv('../results/simulation_results_thetaTheta0.csv')
        extent = [ls_theta0.min(), ls_theta0.max(), ls_theta.min(), ls_theta.max()]
        heat_map(results, 'pnl_lending', 'pnl_perps', "mean PnL difference (loan-perps) for liquidation parameters",
                 'theta0', 'theta', extent)
        heat_map(results, 'pnl_lending_std', 'pnl_perps_std', 'std PnL difference (loan-perps) for liquidation parameters',
                 'theta0', 'theta', extent)
        heat_map(results, 'funding_fee_lending', 'funding_fee_perps', "funding fee difference (loan-perps) for liquidation parameters",
                 'theta0', 'theta', extent)
        heat_map(results, 'funding_fee_lending_std', 'funding_fee_perps_std', "funding fee std difference (loan-perps) for liquidation parameters",
                 'theta0', 'theta', extent)

        results = pd.read_csv('../results/simulation_results_alphaEthDai.csv')
        extent = [ls_alpha_eth.min(), ls_alpha_eth.max(), ls_alpha_dai.min(), ls_alpha_dai.max()]
        heat_map(results, 'pnl_lending', 'pnl_perps', "mean PnL difference (loan-perps) for utilisation parameters",
                 'alpha_eth', 'alpha_dai', extent)
        heat_map(results, 'pnl_lending_std', 'pnl_perps_std', 'std PnL difference (loan-perps) for utilisation parameters',
                 'alpha_eth', 'alpha_dai', extent)
        heat_map(results, 'funding_fee_lending', 'funding_fee_perps', "funding fee difference (loan-perps) for utilisation parameters",
                 'alpha_eth', 'alpha_dai', extent)
        heat_map(results, 'funding_fee_lending_std', 'funding_fee_perps_std', "funding fee std difference (loan-perps) for utilisation parameters",
                 'alpha_eth', 'alpha_dai', extent)
