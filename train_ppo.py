"""Train the first LUMEN PPO policy on the unchanged sail physics."""

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from ppo_env import SolarSailPPOEnv, make_initial_state

# comment
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--planets", action="store_true", help="Train with full planetary perturbations (slower)")
    parser.add_argument("--no-shaping", action="store_true", help="Pure arrival/time/thermal reward, with no timeout shaping")
    parser.add_argument("--output", default="ppo_runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    x0, params = make_initial_state()

    env = SolarSailPPOEnv(x0, params, include_planets=args.planets,
                          terminal_shaping=not args.no_shaping)
    check_env(env, warn=True)
    env = Monitor(env, filename=str(folder / "training"),
                  info_keywords=("is_success", "r_au", "perihelion_au", "v_inf_km_s"))

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log=str(folder / "tb"),
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    checkpoint = CheckpointCallback(save_freq=20_000,
                                     save_path=str(folder / "checkpoints"),
                                     name_prefix="lumen_ppo")
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    except KeyboardInterrupt:
        print("Training interrupted; saving current model.")
    finally:
        model.save(str(folder / "lumen_ppo"))
        env.close()
        print(f"Saved model to {folder / 'lumen_ppo.zip'}")


if __name__ == "__main__":
    main()
