
import pickle
import neat
import numpy as np
from pathlib import Path

from neat_guidance import make_neat_control
from simulation import simulate_trajectory
from fitness import trajectory_fitness
from plotting import plot_trajectory
from solar_sail import AU
from main import x0, params, et0

# Load NEAT configuration
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    str(Path(__file__).with_name("neat_config.txt"))
)

# Load best genome
with open(Path(__file__).with_name("best_genome.pkl"), "rb") as f:
    winner = pickle.load(f)

# Construct neural network
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Construct guidance law
nn_control = make_neat_control(net)

# Simulate trajectory
result = simulate_trajectory(
    nn_control, x0, params, save_history=True
)

# Evaluate fitness
fitness = trajectory_fitness(result, params)

print(f"Fitness: {fitness:.4f}")
print(f"Status: {result['status']}")
print(f"Flight time: {result['t_final']/86400:.2f} days")
print(f"Minimum distance: {result['r_min']/AU:.4f} AU")
print(f"Final distance: {np.linalg.norm(result['x_final'][:3])/AU:.4f} AU")

# Visualize trajectory
plot_trajectory(result["X"][:3, :].T, et0)
