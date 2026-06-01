import argparse
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


def load_ppo(model_path: str) -> PPO:
    """Load a PPO model robustly across numpy and Python versions."""
    _patch_numpy_core()
    custom_objects = {
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2,
    }
    return PPO.load(model_path, custom_objects=custom_objects)


def generate_expert_dataset(
    env_id: str,
    model_path: str,
    n_trajectories: int,
    save_path: str,
    deterministic: bool = True,
) -> None:
    env = gym.make(env_id)
    model = load_ppo(model_path)

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    episode_returns = []
    episode_lengths = []

    for _ in range(n_trajectories):
        obs, _ = env.reset()
        done = False

        ep_return = 0.0
        ep_length = 0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            all_states.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)
            all_dones.append(done)

            obs = next_obs
            ep_return += reward
            ep_length += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

    env.close()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        states=np.asarray(all_states, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.int64),
        rewards=np.asarray(all_rewards, dtype=np.float32),
        next_states=np.asarray(all_next_states, dtype=np.float32),
        dones=np.asarray(all_dones, dtype=np.bool_),
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
    )

    print(f"Saved dataset to {save_path}")
    print(f"Transitions: {len(all_states)}")
    print(
        "Expert return: "
        f"{np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-trajectories", type=int, default=10)
    parser.add_argument("--save-path", type=str, required=True)

    args = parser.parse_args()

    generate_expert_dataset(
        env_id=args.env_id,
        model_path=args.model_path,
        n_trajectories=args.n_trajectories,
        save_path=args.save_path,
    )

# python scripts/02_generate_expert_dataset.py --env-id CartPole-v1 --model-path models/experts/CartPole-v1/ppo_expert_seed_0.zip --n-trajectories 10 --save-path data/expert/CartPole-v1/expert_K10_seed0.npz
# repeat for: K = 1, 10, 50 and seed = 0, 1, 2