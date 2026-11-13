
from pathlib import Path
import neat
import pickle

from neat_guidance import make_neat_control
from simulation import simulate_trajectory
from fitness import trajectory_fitness
    

def train_neat(x0, params, generations=2):

    # Load NEAT configuration
    config_path = Path(__file__).with_name("neat_config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path)
    )

    # Evaluate every genome in the population
    def eval_genomes(genomes, config):

        for genome_id, genome in genomes:

            # Construct neural network from genome
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # Construct guidance law
            nn_control = make_neat_control(net)

            # Simulate trajectory
            result = simulate_trajectory(
                nn_control, x0, params, save_history=False
            )

            # Assign fitness
            genome.fitness = trajectory_fitness(result, params)

    # Initialize population
    population = neat.Population(config)

    # Print evolutionary progress
    population.add_reporter(neat.StdOutReporter(True))

    # Run NEAT
    winner = population.run(eval_genomes, generations)

    with open(Path(__file__).with_name("best_genome.pkl"), "wb") as f:
        pickle.dump(winner, f)

    return winner, config
