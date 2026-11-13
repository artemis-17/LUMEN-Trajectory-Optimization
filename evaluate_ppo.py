"""Replay a PPO policy with accurate LUMEN propagation and plot results."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from ppo_env import SolarSailPPOEnv, make_initial_state, DAY, AU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ppo_runs/lumen_ppo")
    parser.add_argument("--planets", action="store_true", help="Use the original full planetary dynamics")
    parser.add_argument("--plot-3d", action="store_true", help="Open the existing Plotly solar-system figure")
    args = parser.parse_args()

    x0, params = make_initial_state()
    env = SolarSailPPOEnv(x0, params, include_planets=args.planets,
                          accurate=True, record=True)
    policy = PPO.load(args.model, device="cpu")
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    t = np.asarray(env.history_t)
    X = np.asarray(env.history_x)
    U = np.asarray(env.history_u)
    radius = np.linalg.norm(X[:, :3], axis=1) / AU
    energy = np.sum(X[:, 3:]**2, axis=1) / 2.0 - params["mu_sun"] / (radius * AU)

    print(f"Model: {'Full planetary gravity' if args.planets else 'Sun-only gravity'}")
    print(f"Reached escape milestone: {info['is_success']}")
    print(f"Flight time: {info['day']:.1f} days")
    print(f"Final radius: {info['r_au']:.3f} AU")
    print(f"Minimum radius: {info['perihelion_au']:.3f} AU")
    print(f"Final osculating v-infinity: {info['v_inf_km_s']:.3f} km/s")

    folder = Path(args.model).parent
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "planets" if args.planets else "sun_only"
    np.savez_compressed(folder / f"ppo_trajectory_{suffix}.npz",
                        t=t, X=X.T, commands=U)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(t / DAY, radius)
    axs[0].axhline(0.2, linestyle="--", label="Thermal boundary")
    axs[0].axhline(5.0, linestyle=":", label="Distance milestone")
    axs[0].set_ylabel("Heliocentric distance (AU)")
    axs[0].legend()
    axs[1].plot(t / DAY, energy / 1e6)
    axs[1].axhline(0, linestyle="--")
    axs[1].set_ylabel("Solar two-body energy (km²/s²)")
    if len(U):
        axs[2].step(U[:, 0] / DAY, np.rad2deg(U[:, 1]), where="post", label="Cone α")
        axs[2].step(U[:, 0] / DAY, np.rad2deg(U[:, 2]), where="post", label="Clock δ")
    axs[2].set_ylabel("Command (degrees)")
    axs[2].set_xlabel("Mission elapsed time (days)")
    axs[2].legend()
    fig.tight_layout()
    output = folder / f"ppo_results_{suffix}.png"
    fig.savefig(output, dpi=180)
    print(f"Saved plots to {output}")
    plt.show()

    if args.plot_3d:
        from plotting import plot_trajectory
        plot_trajectory(X[:, :3], params["et0"])


if __name__ == "__main__":
    main()
