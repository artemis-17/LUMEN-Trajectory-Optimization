
import pickle
import neat
from pathlib import Path
from graphviz import Digraph

# File paths
folder = Path(__file__).parent
config_path = folder / "neat_config.txt"
genome_path = folder / "best_genome.pkl"

# Load NEAT configuration
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    str(config_path)
)

# Load saved genome
with open(genome_path, "rb") as f:
    genome = pickle.load(f)

# Create network diagram
dot = Digraph("NEAT_Network", format="svg")
dot.attr(rankdir="LR", bgcolor="#151515")
dot.attr("node", fontcolor="white", color="white")
dot.attr("edge", fontcolor="white")

inputs = config.genome_config.input_keys
outputs = config.genome_config.output_keys

input_names = ["r/AU", "vr/vref", "vt/vref", "t/Tref"]
output_names = ["alpha", "delta"]

# Input nodes
for i, node in enumerate(inputs):
    dot.node(
        str(node), input_names[i],
        shape="circle", style="filled",
        fillcolor="#2878B5"
    )

# Output nodes
for i, node in enumerate(outputs):
    dot.node(
        str(node), output_names[i],
        shape="circle", style="filled",
        fillcolor="#239B72"
    )

# Hidden nodes
for node in genome.nodes:
    if node not in outputs:
        dot.node(
            str(node), str(node),
            shape="circle", style="filled",
            fillcolor="#D49135"
        )

# Connections
for (source, target), conn in genome.connections.items():

    weight = conn.weight

    color = "#4285F4" if weight >= 0 else "#EA4335"
    width = min(5.0, 1.0 + abs(weight))

    dot.edge(
        str(source),
        str(target),
        label=f"{weight:.2f}",
        color=color,
        penwidth=str(width),
        style="solid" if conn.enabled else "dashed"
    )

# Save and display
dot.render(
    filename=str(folder / "best_network"),
    view=True,
    cleanup=True
)

print("Network saved as best_network.svg")
print(f"Hidden neurons: {len(genome.nodes) - len(outputs)}")
print(f"Connections: {len(genome.connections)}")
