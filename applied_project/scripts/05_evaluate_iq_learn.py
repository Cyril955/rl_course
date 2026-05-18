"""Evaluate a trained IQ-Learn SoftQ model (OfflineQNetwork architecture).

No hydra dependency — loads the raw state-dict file produced by train_iq.py.
Architecture matches external/IQ-Learn/iq_learn/agent/softq_models.py:OfflineQNetwork.
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
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        return self.fc3(x)


def evaluate_iq_model(
    env_id: str,
    model_path: Path,
    n_episodes: int = 20,
    alpha: float = 0.001,
    eval_seed: int = 42,
) -> tuple[float, float]:
    """Run n_episodes with the trained Q-net, return (mean_return, std_return)."""
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = OfflineQNetwork(obs_dim, action_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    q_net.load_state_dict(state_dict)
    q_net.eval()

    log_alpha = torch.tensor(np.log(alpha))
    alpha_t = log_alpha.exp()

    rng = np.random.default_rng(eval_seed)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False
        ep_return = 0.0

        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(obs).unsqueeze(0)
                q = q_net(state_t)
                dist = Categorical(F.softmax(q / alpha_t, dim=1))
                action = dist.sample().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained IQ-Learn model and optionally save results to JSON."
    )
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved Q-net state dict (.pt file).")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Temperature (must match agent.init_temp used during training).")
    parser.add_argument("--eval-seed", type=int, default=42)
    # Metadata stored in the JSON (not used by eval logic)
    parser.add_argument("--K", type=int, default=None,
                        help="Number of expert demos used for training.")
    parser.add_argument("--train-seed", type=int, default=None,
                        help="Random seed used during IQ-Learn training.")
    parser.add_argument("--save-json", type=str, default=None,
                        help="If given, save results dict to this JSON file.")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    mean_ret, std_ret = evaluate_iq_model(
        env_id=args.env_id,
        model_path=model_path,
        n_episodes=args.n_episodes,
        alpha=args.alpha,
        eval_seed=args.eval_seed,
    )

    print(f"[{args.env_id}] IQ-Learn  K={args.K}  train_seed={args.train_seed}")
    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  (n={args.n_episodes})")

    if args.save_json:
        result = {
            "env": args.env_id,
            "method": "iq_learn",
            "K": args.K,
            "train_seed": args.train_seed,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "n_episodes": args.n_episodes,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()

# Example:
# python scripts/05_evaluate_iq_learn.py \
#   --env-id CartPole-v1 \
#   --model-path models/iq_learn/CartPole-v1/K10_seed0.pt \
#   --K 10 --train-seed 0 \
#   --save-json results/raw/CartPole-v1/iq_learn_K10_seed0.json
