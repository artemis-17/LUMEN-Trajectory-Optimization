# LUMEN — Neuroevolutionary Solar-Sail Trajectory Design

A Python research prototype for **offline solar-sail trajectory design** using **NeuroEvolution of Augmenting Topologies (NEAT)**. The goal is to identify heliocentric guidance histories that take a lightweight solar-sail spacecraft toward the inner solar system and then accelerate it outward. Future work will incorporate requested science-target flybys and the sail-to-dish transition proposed for LUMEN.

> **Scope:** This is a trajectory-design and simulation tool, **not** a flight-ready real-time attitude controller. The learned policy is evaluated offline to generate candidate sail cone-angle and clock-angle command histories. Mission feasibility and spacecraft constraints require further validation.

## Current capabilities

- Heliocentric spacecraft propagation with solar point-mass gravity and third-body perturbations from eight planetary systems.
- Planetary positions from NASA/JPL **DE440s** SPICE ephemerides.
- Solar-radiation-pressure acceleration with sail area, spacecraft mass, and simplified specular reflectivity.
- Sail steering using cone angle **α** and clock angle **δ**.
- A configurable simulation with fixed guidance update intervals, event-based termination, and trajectory metrics.
- NEAT-Python training and saving a best genome (`best_genome.pkl`).
- Interactive Plotly trajectory visualization and an optional Graphviz network diagram.

The current training task focuses on a close-Sun approach and subsequent outbound travel, **not yet planetary flyby optimization**.

## Preliminary spacecraft and mission assumptions

| Parameter | Value | Status |
| --- | --- | --- |
| Sail area | 340 m² | Current LUMEN design input |
| Spacecraft mass | 5 kg | Current model input |
| Specular reflectivity | 0.9 | Assumed |
| Start epoch | 2035-01-01 00:00:00 | Simulation scenario |
| Initial state | Heliocentric state offset 0.02 AU outward from Earth, with Earth's heliocentric velocity | Simplified post-Earth-departure condition |
| Minimum solar distance | 0.2 AU | Current constraint |
| Initial outbound goal | 5 AU | Training milestone |
| Simulation horizon | 5 years | Configurable |

The simulation does **not** yet establish the feasibility of escaping from a specific cislunar or high-Earth orbit. A preliminary trajectory should not be interpreted as a validated LUMEN mission design.

Feel free to change any of these constants and see how the sims go!

## Repository structure

The project is organized around the following modules (names reflect the current development setup):

```text
LUMEN/
├── main.py               # Scenario parameters and NEAT training entry point
├── dynamics.py           # Spacecraft state derivatives
├── solar_sail.py         # Sail acceleration model
├── ephemerides.py        # SPICE loading, gravitational parameters, planet states
├── control.py            # Hand-designed baseline guidance
├── simulation.py         # Trajectory propagation and mission events
├── fitness.py            # NEAT fitness function
├── neat_guidance.py      # Convert a neural network to a guidance function
├── neat_train.py         # NEAT evolutionary training loop
├── neat_config.txt       # NEAT settings
├── test_neat.py          # Evaluate a saved genome
├── plotting.py           # Interactive trajectory visualization
├── visualize_network.py # Optional network-topology visualization
├── kernels/
│   ├── de440s.bsp        # JPL DE440s planetary ephemerides
│   ├── naif0012.tls      # Leap-seconds kernel
│   └── gm_de440.tpc      # Planetary gravitational parameters
├── requirements.txt
└── README.md
```

Some output files are created only after a training run. The `kernels/` directory is **intended to be tracked in Git**, so a cloned repository can run without downloading SPICE data separately.

## Installation (Windows / VS Code)

Requires a supported Python installation and Git.

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you have not created `requirements.txt` yet, install the core packages instead:

```powershell
python -m pip install numpy scipy matplotlib plotly spiceypy neat-python graphviz
```

The `graphviz` Python package is not the same as the **Graphviz system application** required for network diagram rendering. On Windows it can be installed with `winget install --id Graphviz.Graphviz -e`; `dot -V` should then work in a new terminal.

If PowerShell blocks environment activation, use the Windows Command Prompt activation script (`.venv\Scripts\activate.bat`) or adjust the execution policy for the current PowerShell process.

## SPICE data

For reproducibility, this repository includes the following kernels in `kernels/`:

- `de440s.bsp`: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
- `naif0012.tls`: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
- `gm_de440.tpc`: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc

The authoritative source is NASA/JPL's [NAIF generic kernels archive](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/). When updating kernels, check coverage, body IDs, gravitational parameters, and consistency with `ephemerides.py`.

**Ephemeris conventions:** The model uses SI units for integration and a J2000-oriented heliocentric inertial frame; SPICE geometric queries use ephemeris time (TDB seconds past J2000). Positions returned by SPICE in kilometers must be converted to meters. For precise planetary flybys, replace planetary-system barycenters with appropriate planet-center ephemerides and validate close-approach accuracy.

## Running the project

Train NEAT and inspect its best trajectory:

```powershell
python main.py
```

If you have already trained and saved `best_genome.pkl`, evaluate it without retraining:

```powershell
python test_neat.py
```

If Graphviz is installed, draw the saved genome's topology:

```powershell
python visualize_network.py
```

Training is stochastic: different random seeds or configuration settings may produce different networks and trajectories. To reproduce a specific result, preserve the NEAT configuration, simulation parameters, dependency versions, and random seed along with the saved genome.

## Model overview

The spacecraft state is **x = [r, v]**, with heliocentric equations of motion:

\[
\dot{\mathbf r}=\mathbf v,\qquad
\dot{\mathbf v}=\mathbf a_{\odot}+\mathbf a_{\mathrm{third\ body}}+\mathbf a_{\mathrm{sail}}.
\]

The sail model uses the solar distance and steering angles **α** (cone) and **δ** (clock). A goal-conditioned targeting policy is planned, but the current neural-network inputs are compact trajectory features such as heliocentric distance, radial velocity, transverse velocity, and mission time. The current outputs are steering angles rather than Cartesian thrust commands.

## Planned work

- Validate convergence and compare NEAT trajectories with hand-designed steering baselines.
- Implement the proposed **outward 4 AU sail-to-dish transition** and ballistic continuation.
- Add planet-center encounter geometry and scientific flyby feasibility checks.
- Add target and encounter-sequence conditioning to the neural-network inputs.
- Evaluate Jupiter/Uranus flyby requests and the tradeoff between science return, flight time, and escape energy.
- Refine Earth–Moon departure assumptions, shadowing, thermal limits, and attitude/slew feasibility.

## Mission context

LUMEN architecture page: http://soral.as.arizona.edu/LUMEN/#architecture

This repository is an independent preliminary simulation and optimization framework. It should not be presented as an officially validated mission trajectory or operational spacecraft guidance implementation.
