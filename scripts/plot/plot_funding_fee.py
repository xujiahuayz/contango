import numpy as np
from perp.simulation_env import SimulationEnv, SimulatedAssets
import matplotlib.pyplot as plt

dai_backed_eth = SimulatedAssets(
    n_steps=101, seed=1, mu=0, sigma=0.3, n_mc=500, dt=0.01
)

sim_env = SimulationEnv(
    price_paths=dai_backed_eth.price_paths,
    r_collateral_eth=dai_backed_eth.r_collateral_eth,
    r_debt_dai=dai_backed_eth.r_debt_dai,
    dt=dai_backed_eth.dt,
    lambda_=50,
    sigma_f=0.3,
    kappa=1,
    r=0.005,
)


# plot funding fee histogram


for perps_price_path in [
    sim_env.perps_price_mean_rev(),
    sim_env.perps_price_realistic(sigma_noise=6.5, window_length=5, delta=5),
]:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    funding_fee = sim_env.get_funding_fee_perps(perps_price_path)[:, 10]
    implies_funding_rate = sim_env.implied_funding_fee()[:, 10]
    # histogram for funding_fee and implied_funding_rate sharing the same bins
    bins = np.linspace(-20, 20, 80)
    ax.hist(
        funding_fee,
        bins=bins,
        color="blue",
        alpha=0.3,
    )

    ax.hist(
        implies_funding_rate,
        bins=bins,
        color="red",
        alpha=0.3,
    )

    # close the figure
    plt.show()
