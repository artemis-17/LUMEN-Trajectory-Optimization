
import numpy as np
import spiceypy as spice
import neat

from simulation import simulate_trajectory
from ephemerides import PLANET_IDS, earth_state, gm_si, load_kernels
from solar_sail import AU
from plotting import plot_trajectory
from neat_train import train_neat
from neat_guidance import make_neat_control

load_kernels()

et0 = spice.str2et("2035-01-01T00:00:00")

mu_sun = gm_si(10)
mu_planets = np.array([gm_si(i) for i in PLANET_IDS])

# Initial spacecraft state
earth0 = earth_state(et0)
er = earth0[:3] / np.linalg.norm(earth0[:3])

r0 = earth0[:3] + 0.02 * AU * er
v0 = earth0[3:]

x0 = np.hstack((r0, v0))

# Solar sail parameters
A = 340.0
m = 5.0
rho = 0.9

# Simulation parameters
params = {
    "et0": et0,
    "mu_sun": mu_sun,
    "mu_planets": mu_planets,
    "sail_params": (A, m, rho),
    "T": 5 * 365.25 * 86400.0,
    "r_min": 0.2 * AU,
    "r_goal": 5.0 * AU,
    "dt_control": 86400.0,
    "max_step": 86400.0,
    "plot_dt": 6 * 3600.0,
    "rtol": 1e-10,
    "atol": [0.1, 0.1, 0.1, 1e-6, 1e-6, 1e-6]
}

if __name__ == "__main__":

    winner, config = train_neat(x0, params, generations=100)

    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    best_control = make_neat_control(best_net)

    result = simulate_trajectory(
        best_control, x0, params, save_history=True
    )

    print(f"Best fitness: {winner.fitness:.4f}")
    print(f"Mission status: {result['status']}")
    print(f"Flight time: {result['t_final']/86400:.2f} days")
    print(f"Minimum distance: {result['r_min']/AU:.4f} AU")
    print(f"Final distance: {np.linalg.norm(result['x_final'][:3])/AU:.4f} AU")

    plot_trajectory(result["X"][:3, :].T, et0)

