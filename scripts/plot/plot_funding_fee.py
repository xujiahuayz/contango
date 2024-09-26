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

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

funding_fee = sim_env.get_funding_fee_perps(sim_env.perps_price_mean_rev())[:, 1]
implies_funding_rate = sim_env.implied_funding_fee[:, 1]
ax.hist(
    funding_fee,
    bins=60,
    color="blue",
    alpha=0.3,
)
# another histogram for implied funding rate with the same ax
ax.hist(
    implies_funding_rate,
    bins=60,
    color="red",
    alpha=0.3,
)
