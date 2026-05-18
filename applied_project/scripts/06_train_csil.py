"""Train and evaluate CSIL for a single (env, K, seed) configuration.

Loads the same expert NPZ pool used by IQ-Learn, subsamples K trajectories,
runs BC + SAC with coherent reward, evaluates 20 episodes, writes a JSON.

Results land in results/raw/csil/{env}/.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).parent))
from csil_agent import BCPolicyDiscrete, CSILAgent, run_csil, evaluate_csil


def subsample_trajectories(
    npz_path: Path,
    n_demos: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select n_demos trajectories from the NPZ pool; return flat (states, actions)."""
    data = np.load(npz_path)
    states  = data["states"]
    actions = data["actions"]
    dones   = data["dones"].astype(bool)

    trajectories: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i, done in enumerate(dones):
        if done:
            trajectories.append((states[start:i + 1], actions[start:i + 1]))
            start = i + 1
    if start < len(states):
        trajectories.append((states[start:], actions[start:]))

    n = min(n_demos, len(trajectories))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(trajectories), size=n, replace=False))

    expert_states  = np.concatenate([trajectories[i][0] for i in chosen])
    expert_actions = np.concatenate([trajectories[i][1] for i in chosen])
    return expert_states, expert_actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate CSIL on one (env, K, seed) configuration."
    )
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--expert-npz", type=str, required=True,
                        help="Expert pool NPZ (e.g. data/expert/CartPole-v1/expert_K15_seed0.npz).")
    parser.add_argument("--n-demos", type=int, required=True,
                        help="Number of expert trajectories to use (K).")
    parser.add_argument("--seed", type=int, default=0)
    # BC
    parser.add_argument("--bc-epochs", type=int, default=200)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    # SAC / CSIL
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--alpha-csil", type=float, default=0.1,
                        help="Coherent reward temperature α.")
    parser.add_argument("--early-stop-reward", type=float, default=None,
                        help="Stop training once 100-ep average exceeds this.")
    # Evaluation
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    # Output
    parser.add_argument("--save-json", type=str, default=None,
                        help="Where to write the result JSON.")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_npz = Path(args.expert_npz).resolve()
    if not expert_npz.exists():
        raise FileNotFoundError(f"Expert NPZ not found: {expert_npz}")

    expert_states, expert_actions = subsample_trajectories(expert_npz, args.n_demos, args.seed)
    print(f"[CSIL] {args.env_id}  K={args.n_demos}  seed={args.seed}")
    print(f"       Expert data: {args.n_demos} trajectories → {len(expert_states)} transitions")

    env = gym.make(args.env_id)

    csil_dict = {
        "bc_hidden_size":    64,
        "bc_epochs":         args.bc_epochs,
        "bc_lr":             args.bc_lr,
        "bc_batch_size":     args.bc_batch_size,
        "sac_hidden_size":   64,
        "sac_lr":            3e-4,
        "gamma":             0.99,
        "tau":               0.005,
        "alpha_csil":        args.alpha_csil,
        "buffer_size":       200_000,
        "batch_size":        256,
        "start_steps":       args.start_steps,
        "n_episodes":        args.n_episodes,
        "early_stop_reward": args.early_stop_reward,
        "device":            "cpu",
        "verbose":           True,
    }

    agent, _, _ = run_csil(env, expert_states, expert_actions, csil_dict)
    env.close()

    eval_env = gym.make(args.env_id)
    mean_ret, std_ret = evaluate_csil(eval_env, agent, n_episodes=args.n_eval_episodes)
    eval_env.close()

    print(f"\n[{args.env_id}] CSIL  K={args.n_demos}  seed={args.seed}")
    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  (n={args.n_eval_episodes})")

    if args.save_json:
        result = {
            "env":         args.env_id,
            "method":      "csil",
            "K":           args.n_demos,
            "train_seed":  args.seed,
            "mean_return": mean_ret,
            "std_return":  std_ret,
            "n_episodes":  args.n_eval_episodes,
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()

# Example – single run:
# python scripts/06_train_csil.py \
#   --env-id CartPole-v1 \
#   --expert-npz data/expert/CartPole-v1/expert_K15_seed0.npz \
#   --n-demos 10 --seed 0 --n-episodes 1000 --early-stop-reward 495 \
#   --save-json results/raw/csil/CartPole-v1/csil_K10_seed0.json
#
# Use run_sweep.sh to run the full K × seed grid.
