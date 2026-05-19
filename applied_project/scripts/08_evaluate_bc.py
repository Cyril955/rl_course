"""Train and evaluate a BC-only policy for one (env, K, seed) configuration.

Reuses the same BC training code as CSIL Phase 1 but stops before SAC fine-tuning,
giving a pure behavioural cloning baseline to compare against IQ-Learn and CSIL.

Results land in results/raw/bc/{env}/.
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
from csil_agent import BCPolicyDiscrete, train_bc


@torch.no_grad()
def evaluate_bc(env, policy: BCPolicyDiscrete, n_episodes: int = 20, device: str = "cpu") -> tuple[float, float]:
    """Greedy (argmax) evaluation of BC policy."""
    returns = []
    for _ in range(n_episodes):
        try:
            state, _ = env.reset()
        except TypeError:
            state = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy.probs(s).argmax(-1).item()
            try:
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                state, r, done, _ = env.step(action)
            ep_r += r
        returns.append(ep_r)
    arr = np.array(returns)
    return float(arr.mean()), float(arr.std())


def subsample_trajectories(npz_path: Path, n_demos: int, seed: int, subsample_freq: int = 1):
    data = np.load(npz_path)
    states  = data["states"]
    actions = data["actions"]
    dones   = data["dones"].astype(bool)

    trajectories = []
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
    expert_states  = np.concatenate([trajectories[i][0][::subsample_freq] for i in chosen])
    expert_actions = np.concatenate([trajectories[i][1][::subsample_freq] for i in chosen])
    return expert_states, expert_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id",      type=str, required=True)
    parser.add_argument("--expert-npz",  type=str, required=True)
    parser.add_argument("--n-demos",     type=int, required=True)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--bc-epochs",   type=int, default=200)
    parser.add_argument("--bc-lr",       type=float, default=3e-4)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--save-json",   type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_npz = Path(args.expert_npz).resolve()
    expert_states, expert_actions = subsample_trajectories(
        expert_npz, args.n_demos, args.seed, args.subsample_freq)
    print(f"[BC] {args.env_id}  K={args.n_demos}  seed={args.seed}  "
          f"subsample_freq={args.subsample_freq}  ({len(expert_states)} transitions)")

    env = gym.make(args.env_id)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = BCPolicyDiscrete(state_dim, action_dim, hidden_size=64)
    train_bc(policy, expert_states, expert_actions,
             n_epochs=args.bc_epochs, lr=args.bc_lr,
             batch_size=args.bc_batch_size, verbose=False)
    env.close()

    eval_env = gym.make(args.env_id)
    mean_ret, std_ret = evaluate_bc(eval_env, policy, n_episodes=args.n_eval_episodes)
    eval_env.close()

    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  (n={args.n_eval_episodes})")

    if args.save_json:
        result = {
            "env":         args.env_id,
            "method":      "bc",
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
