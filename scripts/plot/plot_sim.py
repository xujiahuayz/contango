from perp.simulation_env import SimulationEnv, SimulatedAssets
import matplotlib.pyplot as plt

dai_backed_eth = SimulatedAssets(
    n_steps=100, seed=1, mu=0, sigma=0.5, n_mc=10_000, dt=0.01
)

sim_env = SimulationEnv(
    price_paths=dai_backed_eth.price_paths,
    r_collateral_eth=dai_backed_eth.r_collateral_eth,
    r_debt_dai=dai_backed_eth.r_debt_dai,
    dt=dai_backed_eth.dt,
    lambda_=100,
    sigma_f=100,
    kappa=1,
)


# plot funding fee histogram

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

funding_fee = sim_env.get_funding_fee_perps(sim_env.perps_price_mean_rev())
implies_funding_rate = sim_env.implied_funding_fee
ax.hist(
    funding_fee.flatten(),
    bins=100,
    color="blue",
    alpha=0.3,
)
# another histogram for implied funding rate with the same ax
ax.hist(
    implies_funding_rate.flatten(),
    bins=100,
    color="red",
    alpha=0.3,
)


# plot the price paths and sim_env.perps_price_paths

dai_backed_eth = SimulatedAssets(n_steps=100, seed=1, mu=0, sigma=0.3, n_mc=20, dt=0.01)

sim_env = SimulationEnv(
    price_paths=dai_backed_eth.price_paths,
    r_collateral_eth=dai_backed_eth.r_collateral_eth,
    r_debt_dai=dai_backed_eth.r_debt_dai,
    dt=dai_backed_eth.dt,
    lambda_=100,
    sigma_f=100,
    kappa=1,
)


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(sim_env.time_array.T, sim_env.price_paths.T, color="blue", alpha=0.8)
ax.plot(
    sim_env.time_array.T,
    sim_env.perps_price_mean_rev().T,
    color="red",
    alpha=0.8,
)

"""
fig 3
"""

dai_backed_eth = SimulatedAssets(
    n_steps=100,
    seed=1,
    mu=0,
    sigma=0.1,
    n_mc=10_000,
    dt=0.01,
    alpha_eth=-0.15,
    alpha_dai=0.05,
    u0_eth=0.4,
    u0_dai=0.4,
    r_0=0.0,
    r_1=0.04,
    r_2=2.5,
    u_optimal=0.45,
)

sim_env = SimulationEnv(
    price_paths=dai_backed_eth.price_paths,
    r_collateral_eth=dai_backed_eth.r_collateral_eth,
    r_debt_dai=dai_backed_eth.r_debt_dai,
    dt=dai_backed_eth.dt,
    lambda_=100,
    sigma_f=100,
    kappa=1,
)
perp_price_paths = sim_env.perps_price_mean_rev()

# plot the distribution of pnl
# get pnl at t = 0.25
pnl_perp = sim_env.get_pnl_perps(sim_env.get_pnl_perps(perp_price_paths))[:, 25]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(
    pnl_perp,
    bins=200,
    color="blue",
    alpha=0.3,
)
# plot implied funding rate
