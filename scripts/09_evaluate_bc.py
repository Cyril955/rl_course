"""Evaluate a saved BC model (from 05_train_bc.py).

Results → results/evaluation/bc/{env}/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).parent))
from csil_agent import BCPolicyDiscrete


@torch.no_grad()
def evaluate_bc_policy(
    env,
    policy: BCPolicyDiscrete,
    n_episodes: int,
    device: str,
    eval_seeds: list[int],
) -> tuple[float, float, list[float]]:
    """Returns (mean_return, std_return, per_eval_seed_means)."""
    per_seed_means = []
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        seed_returns = []
        for _ in range(n_episodes):
            state, _ = env.reset(seed=int(rng.integers(1 << 31)))
            done, ep_r = False, 0.0
            while not done:
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy.probs(s).argmax(-1).item()
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_r += r
            seed_returns.append(ep_r)
        per_seed_means.append(float(np.mean(seed_returns)))
    return float(np.mean(per_seed_means)), float(np.std(per_seed_means)), per_seed_means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  type=str, required=True,
                        help="Path to BC model (.pt) saved by 05_train_bc.py.")
    parser.add_argument("--n-episodes",  type=int, default=20)
    parser.add_argument("--eval-seeds",  type=int, nargs="+", default=None)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--save-json",   type=str, default=None)
    args = parser.parse_args()

    eval_seeds = args.eval_seeds or [42]
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location=args.device, weights_only=False)
    env_id      = ckpt["env_id"]
    K           = ckpt["K"]
    train_seed  = ckpt["seed"]

    policy = BCPolicyDiscrete(ckpt["state_dim"], ckpt["action_dim"], ckpt["hidden_size"])
    policy.load_state_dict(ckpt["policy"])
    policy = policy.to(args.device).eval()

    eval_env = gym.make(env_id)
    mean_ret, std_ret, per_seed_means = evaluate_bc_policy(
        eval_env, policy, args.n_episodes, args.device, eval_seeds)
    eval_env.close()

    print(f"[{env_id}] BC  K={K}  train_seed={train_seed}")
    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  "
          f"({len(eval_seeds)} eval seeds × {args.n_episodes} eps)")

    if args.save_json:
        result = {
            "env":                env_id,
            "method":             "bc",
            "K":                  K,
            "train_seed":         train_seed,
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