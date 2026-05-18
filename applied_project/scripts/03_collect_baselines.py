"""Compute expert-PPO and random-policy baselines for CartPole and Acrobot.

Saves one JSON per (env, seed) for the expert, and one JSON per env for random.
Results land in results/raw/baselines/{env}/.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def _patch_numpy_core():
    """Shim numpy._core.* → numpy.core.* for models saved with numpy ≥ 2.0."""
    import importlib
    import numpy.core as nc
    _core = types.ModuleType("numpy._core")
    _core.__path__ = []
    sys.modules.setdefault("numpy._core", _core)
    for name in ["numeric", "multiarray", "fromnumeric", "shape_base",
                 "function_base", "arrayprint", "defchararray", "umath"]:
        full = f"numpy._core.{name}"
        if full not in sys.modules:
            try:
                src = importlib.import_module(f"numpy.core.{name}")
                sys.modules[full] = src
            except ImportError:
                pass


def load_ppo(model_path) -> PPO:
    """Load a PPO model robustly across numpy and Python versions."""
    _patch_numpy_core()
    custom_objects = {
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2,
    }
    return PPO.load(str(model_path), custom_objects=custom_objects)


ENV_IDS = ["CartPole-v1", "Acrobot-v1"]
SEEDS = [0, 1, 2, 3, 4]
N_EPISODES = 20
EXPERT_MODEL_DIR = Path("models/experts")
RESULTS_DIR = Path("results/raw/baselines")


def evaluate_random(env_id: str, n_episodes: int, seed: int = 0) -> tuple[float, float]:
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False
        ep_return = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_ppo(
    env_id: str,
    model_path: Path,
    n_episodes: int,
    seed: int = 42,
) -> tuple[float, float]:
    env = gym.make(env_id)
    model = load_ppo(model_path)
    rng = np.random.default_rng(seed)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--envs", nargs="+", default=ENV_IDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    for env_id in args.envs:
        print(f"\n=== {env_id} ===")

        # Random baseline (one run per env, averaged over many episodes)
        random_path = RESULTS_DIR / env_id / "random.json"
        if random_path.exists():
            print(f"  [SKIP] random baseline already exists")
        else:
            mean, std = evaluate_random(env_id, n_episodes=args.n_episodes * len(args.seeds))
            save_json(random_path, {
                "env": env_id, "method": "random",
                "mean_return": mean, "std_return": std,
                "n_episodes": args.n_episodes * len(args.seeds),
            })
            print(f"  Random: {mean:.2f} ± {std:.2f}")

        # Expert PPO baseline (one run per seed)
        for seed in args.seeds:
            expert_result_path = RESULTS_DIR / env_id / f"expert_seed{seed}.json"
            if expert_result_path.exists():
                print(f"  [SKIP] expert seed {seed} already exists")
                continue

            model_path = EXPERT_MODEL_DIR / env_id / f"ppo_expert_seed_{seed}.zip"
            if not model_path.exists():
                print(f"  [SKIP] expert model not found: {model_path}")
                continue

            mean, std = evaluate_ppo(env_id, model_path, n_episodes=args.n_episodes, seed=seed)
            save_json(expert_result_path, {
                "env": env_id, "method": "expert", "seed": seed,
                "mean_return": mean, "std_return": std,
                "n_episodes": args.n_episodes,
            })
            print(f"  Expert seed {seed}: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()

# python scripts/03_collect_baselines.py
