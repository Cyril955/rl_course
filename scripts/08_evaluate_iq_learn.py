"""Evaluate a trained IQ-Learn model.

Stores per-eval-seed means so 12_plot_comparison.py can aggregate over
both training seeds and evaluation seeds.

Results → results/evaluation/iq_learn/{env}/
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class OfflineQNetwork(nn.Module):
    """Mirror of IQ-Learn's OfflineQNetwork (64-ELU-64-ELU → action_dim)."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.elu(self.fc2(self.elu(self.fc1(x)))))


def evaluate_iq_model(
    env_id: str,
    model_path: Path,
    n_episodes: int,
    alpha: float,
    eval_seeds: list[int],
) -> tuple[float, float, list[float]]:
    """Returns (mean_return, std_return, per_eval_seed_means)."""
    env = gym.make(env_id)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = OfflineQNetwork(obs_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    q_net.eval()
    alpha_t = torch.tensor(np.log(alpha)).exp()

    per_seed_means = []
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        seed_returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
            done, ep_return = False, 0.0
            while not done:
                with torch.no_grad():
                    q = q_net(torch.FloatTensor(obs).unsqueeze(0))
                    action = Categorical(F.softmax(q / alpha_t, dim=1)).sample().item()
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += r
            seed_returns.append(ep_return)
        per_seed_means.append(float(np.mean(seed_returns)))

    env.close()
    return float(np.mean(per_seed_means)), float(np.std(per_seed_means)), per_seed_means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id",      type=str, required=True)
    parser.add_argument("--model-path",  type=str, required=True)
    parser.add_argument("--n-episodes",  type=int, default=20)
    parser.add_argument("--alpha",       type=float, default=0.001)
    parser.add_argument("--eval-seeds",  type=int, nargs="+", default=None)
    parser.add_argument("--K",           type=int, default=None)
    parser.add_argument("--train-seed",  type=int, default=None)
    parser.add_argument("--save-json",   type=str, default=None)
    args = parser.parse_args()

    eval_seeds = args.eval_seeds or [42]
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    mean_ret, std_ret, per_seed_means = evaluate_iq_model(
        args.env_id, model_path, args.n_episodes, args.alpha, eval_seeds)

    print(f"[{args.env_id}] IQ-Learn  K={args.K}  train_seed={args.train_seed}")
    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  "
          f"({len(eval_seeds)} eval seeds × {args.n_episodes} eps)")

    if args.save_json:
        result = {
            "env":                args.env_id,
            "method":             "iq_learn",
            "K":                  args.K,
            "train_seed":         args.train_seed,
            "mean_return":        mean_ret,
            "std_return":         std_ret,
            "per_eval_seed_means": per_seed_means,
            "eval_seeds":         eval_seeds,
            "n_episodes_per_seed": args.n_episodes,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()