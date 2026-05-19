"""Train and evaluate CSIL+SOAR for a single (env, K, seed) configuration.

Mirrors 06_train_csil.py exactly, but uses CSILSOARAgent instead of CSILAgent.
Two extra hyperparameters control the SOAR ensemble:
  --n-critics  L   number of independent critics  (default 4, paper uses 4)
  --sigma-clip σ   std-deviation clipping threshold (default 1.0)

Results land in results/raw/csil_soar/{env}/.
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
from csil_soar_agent import run_csil_soar, evaluate_csil_soar


def subsample_trajectories(
    npz_path: Path,
    n_demos: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select n_demos trajectories from the NPZ pool; return flat (states, actions)."""
    data    = np.load(npz_path)
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
        description="Train and evaluate CSIL+SOAR on one (env, K, seed) configuration."
    )
    parser.add_argument("--env-id",      type=str, required=True)
    parser.add_argument("--expert-npz",  type=str, required=True,
                        help="Expert pool NPZ (e.g. data/expert/CartPole-v1/expert_K15_seed0.npz).")
    parser.add_argument("--n-demos",     type=int, required=True,
                        help="Number of expert trajectories to use (K).")
    parser.add_argument("--seed",        type=int, default=0)
    # BC
    parser.add_argument("--bc-epochs",     type=int,   default=200)
    parser.add_argument("--bc-lr",         type=float, default=3e-4)
    parser.add_argument("--bc-batch-size", type=int,   default=256)
    # SAC / CSIL
    parser.add_argument("--n-episodes",       type=int,   default=1000)
    parser.add_argument("--start-steps",      type=int,   default=1000)
    parser.add_argument("--alpha-csil",       type=float, default=0.1,
                        help="Coherent reward temperature α.")
    parser.add_argument("--early-stop-reward", type=float, default=None,
                        help="Stop training once 100-ep average exceeds this.")
    # SOAR-specific
    parser.add_argument("--n-critics",  type=int,   default=4,
                        help="Number of critics in the SOAR ensemble (L).")
    parser.add_argument("--sigma-clip", type=float, default=1.0,
                        help="Std-deviation clipping threshold σ for OptimisticQ-NN.")
    # Evaluation
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    # Output
    parser.add_argument("--save-json", type=str, default=None,
                        help="Where to write the result JSON.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_npz = Path(args.expert_npz).resolve()
    if not expert_npz.exists():
        raise FileNotFoundError(f"Expert NPZ not found: {expert_npz}")

    expert_states, expert_actions = subsample_trajectories(expert_npz, args.n_demos, args.seed)
    print(f"[CSIL-SOAR] {args.env_id}  K={args.n_demos}  seed={args.seed}  "
          f"L={args.n_critics}  σ={args.sigma_clip}")
    print(f"            Expert data: {args.n_demos} trajectories → {len(expert_states)} transitions")

    env = gym.make(args.env_id)

    csil_soar_dict = {
        "bc_hidden_size":    64,
        "bc_epochs":         args.bc_epochs,
        "bc_lr":             args.bc_lr,
        "bc_batch_size":     args.bc_batch_size,
        "sac_hidden_size":   64,
        "sac_lr":            3e-4,
        "gamma":             0.99,
        "tau":               0.005,
        "alpha_csil":        args.alpha_csil,
        "n_critics":         args.n_critics,
        "sigma_clip":        args.sigma_clip,
        "buffer_size":       200_000,
        "batch_size":        256,
        "start_steps":       args.start_steps,
        "n_episodes":        args.n_episodes,
        "early_stop_reward": args.early_stop_reward,
        "device":            "cpu",
        "verbose":           True,
    }

    agent, _, _ = run_csil_soar(env, expert_states, expert_actions, csil_soar_dict)
    env.close()

    eval_env = gym.make(args.env_id)
    mean_ret, std_ret = evaluate_csil_soar(eval_env, agent, n_episodes=args.n_eval_episodes)
    eval_env.close()

    print(f"\n[{args.env_id}] CSIL-SOAR  K={args.n_demos}  seed={args.seed}  "
          f"L={args.n_critics}  σ={args.sigma_clip}")
    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  (n={args.n_eval_episodes})")

    if args.save_json:
        result = {
            "env":         args.env_id,
            "method":      "csil_soar",
            "K":           args.n_demos,
            "train_seed":  args.seed,
            "n_critics":   args.n_critics,
            "sigma_clip":  args.sigma_clip,
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

# ── Example — single run (PowerShell) ─────────────────────────────────────────
# python scripts/08_train_csil_soar.py `
#   --env-id CartPole-v1 `
#   --expert-npz data/expert/CartPole-v1/expert_K15_seed0.npz `
#   --n-demos 10 --seed 0 --n-episodes 1000 --early-stop-reward 495 `
#   --n-critics 4 --sigma-clip 1.0 `
#   --save-json results/raw/csil_soar/CartPole-v1/csil_soar_K10_seed0.json
#
# ── Full K × seed sweep (PowerShell) ──────────────────────────────────────────
# foreach ($env in @("CartPole-v1", "Acrobot-v1")) {
#   foreach ($K in @(1, 3, 7, 10, 15)) {
#     foreach ($seed in @(0, 1, 2, 3, 4)) {
#       python scripts/08_train_csil_soar.py `
#         --env-id $env `
#         --expert-npz "data/expert/$env/expert_K15_seed${seed}.npz" `
#         --n-demos $K --seed $seed `
#         --n-episodes 1000 --early-stop-reward 495 `
#         --n-critics 4 --sigma-clip 1.0 `
#         --save-json "results/raw/csil_soar/$env/csil_soar_K${K}_seed${seed}.json"
#     }
#   }
# }
