"""Train a BC policy for one (env, K, seed) configuration and save the model.

Evaluation is handled separately by 09_evaluate_bc.py.

Models  → models/bc/{env}/
Training curves → results/training/bc/{env}/
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
from config import BC_CONFIG, subsample_trajectories


def main() -> None:
    cfg = BC_CONFIG

    parser = argparse.ArgumentParser(
        description="Train a BC policy for one (env, K, seed) configuration."
    )
    parser.add_argument("--env-id",         type=str, required=True)
    parser.add_argument("--expert-npz",     type=str, required=True)
    parser.add_argument("--n-demos",        type=int, required=True)
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--bc-epochs",      type=int,   default=cfg["epochs"])
    parser.add_argument("--bc-lr",          type=float, default=cfg["lr"])
    parser.add_argument("--bc-batch-size",  type=int,   default=cfg["batch_size"])
    parser.add_argument("--bc-hidden-size", type=int,   default=cfg["hidden_size"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Defaults to models/bc/{env}/K{K}_seed{seed}.pt")
    parser.add_argument("--save-training", type=str, default=None,
                        help="Defaults to results/training/bc/{env}/bc_K{K}_seed{seed}.json")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] CUDA not available, falling back to cpu.")
        args.device = "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_npz = Path(args.expert_npz).resolve()
    if not expert_npz.exists():
        raise FileNotFoundError(f"Expert NPZ not found: {expert_npz}")

    expert_states, expert_actions = subsample_trajectories(
        expert_npz, args.n_demos, args.seed, args.subsample_freq)
    print(f"[BC] {args.env_id}  K={args.n_demos}  seed={args.seed}  "
          f"subsample_freq={args.subsample_freq}  → {len(expert_states)} transitions")

    env = gym.make(args.env_id)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    policy = BCPolicyDiscrete(state_dim, action_dim, hidden_size=args.bc_hidden_size)
    bc_losses = train_bc(
        policy, expert_states, expert_actions,
        n_epochs=args.bc_epochs, lr=args.bc_lr,
        batch_size=args.bc_batch_size, device=args.device, verbose=True,
    )
    print(f"  BC training done. Final NLL: {bc_losses[-1]:.4f}")

    # ── Save model ─────────────────────────────────────────────────────────────
    model_path = Path(
        args.save_model
        or f"models/bc/{args.env_id}/K{args.n_demos}_seed{args.seed}.pt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy":      policy.state_dict(),
        "env_id":      args.env_id,
        "K":           args.n_demos,
        "seed":        args.seed,
        "state_dim":   state_dim,
        "action_dim":  action_dim,
        "hidden_size": args.bc_hidden_size,
    }, model_path)
    print(f"  Model saved → {model_path}")

    # ── Save training curve ────────────────────────────────────────────────────
    training_path = Path(
        args.save_training
        or f"results/training/bc/{args.env_id}/bc_K{args.n_demos}_seed{args.seed}.json"
    )
    training_path.parent.mkdir(parents=True, exist_ok=True)
    with open(training_path, "w") as f:
        json.dump({
            "env":         args.env_id,
            "method":      "bc",
            "K":           args.n_demos,
            "train_seed":  args.seed,
            "epoch_losses": bc_losses,
            "n_epochs":    len(bc_losses),
        }, f)
    print(f"  Training curve saved → {training_path}")


if __name__ == "__main__":
    main()

# ── Example (PowerShell) ───────────────────────────────────────────────────────
# python scripts/05_train_bc.py `
#   --env-id CartPole-v1 `
#   --expert-npz data/expert_trajectories/CartPole-v1/expert_K15_seed0.npz `
#   --n-demos 10 --seed 0 --subsample-freq 20 --device cuda