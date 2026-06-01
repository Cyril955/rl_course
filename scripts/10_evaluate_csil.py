"""Evaluate a saved CSIL model (from 06_train_csil.py).

Results → results/evaluation/csil/{env}/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).parent))
from csil_agent import BCPolicyDiscrete, CSILAgent


@torch.no_grad()
def evaluate_actor(env, actor, n_episodes: int, device: str, eval_seeds: list[int]):
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
                probs, _ = actor(s)
                action = probs.argmax(-1).item()
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_r += r
            seed_returns.append(ep_r)
        per_seed_means.append(float(np.mean(seed_returns)))
    return float(np.mean(per_seed_means)), float(np.std(per_seed_means)), per_seed_means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to CSIL model (.pt) saved by 06_train_csil.py.")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--save-json",  type=str, default=None)
    args = parser.parse_args()

    eval_seeds = args.eval_seeds or [42]
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt   = torch.load(model_path, map_location=args.device, weights_only=False)
    env_id = ckpt["env_id"]
    K      = ckpt["K"]
    seed   = ckpt["seed"]
    config = ckpt["config"]

    env        = gym.make(env_id)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    bc_policy = BCPolicyDiscrete(state_dim, action_dim, config["bc_hidden_size"])
    bc_policy.load_state_dict(ckpt["bc_policy"])

    agent = CSILAgent(state_dim, action_dim, bc_policy,
                      hidden_size=config["sac_hidden_size"], device=args.device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()

    eval_env = gym.make(env_id)
    mean_ret, std_ret, per_seed_means = evaluate_actor(
        eval_env, agent.actor, args.n_episodes, args.device, eval_seeds)
    eval_env.close()

    print(f"[{env_id}] CSIL  K={K}  train_seed={seed}")
    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  "
          f"({len(eval_seeds)} eval seeds × {args.n_episodes} eps)")

    if args.save_json:
        result = {
            "env":                env_id,
            "method":             "csil",
            "K":                  K,
            "train_seed":         seed,
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