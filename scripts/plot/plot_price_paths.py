"""
Fig 2 Simulated samples of ETH and perps price processes for μ = 0, σ = 0.3
"""

from perp.simulation_env import SimulationEnv, SimulatedAssets
import matplotlib.pyplot as plt

# plot the price paths and sim_env.perps_price_paths

dai_backed_eth = SimulatedAssets(
    n_steps=101, seed=1, mu=0, sigma=0.3, n_mc=500, dt=0.01
)


sim_env = SimulationEnv(
    price_paths=dai_backed_eth.price_paths[:20, :],
    r_collateral_eth=dai_backed_eth.r_collateral_eth[:20, :],
    r_debt_dai=dai_backed_eth.r_debt_dai[:20, :],
    dt=dai_backed_eth.dt,
    lambda_=100,
    sigma_f=100,
    kappa=1,
)


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(
    sim_env.time_array[:20, :].T, sim_env.price_paths[:20, :].T, color="blue", alpha=0.8
)
ax.plot(
    sim_env.time_array[:20, :].T,
    sim_env.perps_price_mean_rev()[:20, :].T,
    color="red",
    alpha=0.8,
)
