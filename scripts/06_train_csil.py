"""Train CSIL for one (env, K, seed) configuration and save the model.

Loads a pre-trained BC model from models/bc/ if available (run 05_train_bc.py first).
Evaluation is handled separately by 10_evaluate_csil.py.

Models         → models/csil/{env}/
Training curves → results/training/csil/{env}/
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
from csil_agent import BCPolicyDiscrete, run_csil
from config import BC_CONFIG, CSIL_CONFIG, ENV_CONFIG, subsample_trajectories


def load_bc_policy(bc_model_path: Path, device: str) -> BCPolicyDiscrete | None:
    if not bc_model_path.exists():
        return None
    ckpt = torch.load(bc_model_path, map_location=device, weights_only=False)
    policy = BCPolicyDiscrete(ckpt["state_dim"], ckpt["action_dim"], ckpt["hidden_size"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


def main() -> None:
    cfg = CSIL_CONFIG

    parser = argparse.ArgumentParser(
        description="Train CSIL for one (env, K, seed) configuration."
    )
    parser.add_argument("--env-id",         type=str, required=True)
    parser.add_argument("--expert-npz",     type=str, required=True)
    parser.add_argument("--n-demos",        type=int, required=True)
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--bc-model",       type=str, default=None,
                        help="Path to pre-trained BC model. "
                             "Defaults to models/bc/{env}/K{K}_seed{seed}.pt")
    # BC (used only if BC model not found)
    parser.add_argument("--bc-epochs",     type=int,   default=None,
                        help="BC training epochs (fallback if no BC model found). "
                             "Defaults to ENV_CONFIG[env]['bc_epochs'] if not specified.")
    parser.add_argument("--bc-lr",         type=float, default=cfg["bc_lr"])
    parser.add_argument("--bc-batch-size", type=int,   default=cfg["bc_batch_size"])
    # SAC / CSIL
    parser.add_argument("--n-episodes",        type=int,   default=cfg["n_episodes"])
    parser.add_argument("--start-steps",       type=int,   default=cfg["start_steps"])
    parser.add_argument("--alpha-csil",        type=float, default=cfg["alpha_csil"])
    parser.add_argument("--early-stop-reward", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Defaults to models/csil/{env}/K{K}_seed{seed}.pt")
    parser.add_argument("--save-training", type=str, default=None,
                        help="Defaults to results/training/csil/{env}/csil_K{K}_seed{seed}.json")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] CUDA not available, falling back to cpu.")
        args.device = "cpu"

    if args.bc_epochs is None:
        args.bc_epochs = ENV_CONFIG.get(args.env_id, {}).get("bc_epochs", BC_CONFIG["epochs"])
    if args.early_stop_reward is None:
        args.early_stop_reward = ENV_CONFIG.get(args.env_id, {}).get("early_stop_reward", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_npz = Path(args.expert_npz).resolve()
    if not expert_npz.exists():
        raise FileNotFoundError(f"Expert NPZ not found: {expert_npz}")

    expert_states, expert_actions = subsample_trajectories(
        expert_npz, args.n_demos, args.seed, args.subsample_freq)
    print(f"[CSIL] {args.env_id}  K={args.n_demos}  seed={args.seed}  "
          f"subsample_freq={args.subsample_freq}  → {len(expert_states)} transitions")

    # ── Load pre-trained BC if available ──────────────────────────────────────
    bc_model_path = Path(
        args.bc_model or f"models/bc/{args.env_id}/K{args.n_demos}_seed{args.seed}.pt")
    bc_policy = load_bc_policy(bc_model_path, args.device)
    if bc_policy is not None:
        print(f"[CSIL] Loaded BC policy from {bc_model_path}")
    else:
        print(f"[CSIL] BC model not found at {bc_model_path}, training BC from scratch.")

    env = gym.make(args.env_id)

    csil_dict = {
        "bc_hidden_size":    cfg["bc_hidden_size"],
        "bc_epochs":         args.bc_epochs,
        "bc_lr":             args.bc_lr,
        "bc_batch_size":     args.bc_batch_size,
        "sac_hidden_size":   cfg["sac_hidden_size"],
        "sac_lr":            cfg["sac_lr"],
        "gamma":             cfg["gamma"],
        "tau":               cfg["tau"],
        "alpha_csil":        args.alpha_csil,
        "buffer_size":       cfg["buffer_size"],
        "batch_size":        cfg["batch_size"],
        "start_steps":       args.start_steps,
        "n_episodes":        args.n_episodes,
        "early_stop_reward": args.early_stop_reward,
        "device":            args.device,
        "verbose":           True,
    }

    agent, scores, bc_policy = run_csil(
        env, expert_states, expert_actions, csil_dict, bc_policy=bc_policy)
    env.close()

    # ── Save model ─────────────────────────────────────────────────────────────
    model_path = Path(
        args.save_model or f"models/csil/{args.env_id}/K{args.n_demos}_seed{args.seed}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor":     agent.actor.state_dict(),
        "bc_policy": bc_policy.state_dict(),
        "env_id":    args.env_id,
        "K":         args.n_demos,
        "seed":      args.seed,
        "state_dim": env.observation_space.shape[0] if hasattr(env, "observation_space")
                     else agent.actor.net[0].in_features,
        "action_dim": agent.action_dim,
        "config":    csil_dict,
    }, model_path)
    print(f"  Model saved → {model_path}")

    # ── Save training curve ────────────────────────────────────────────────────
    training_path = Path(
        args.save_training
        or f"results/training/csil/{args.env_id}/csil_K{args.n_demos}_seed{args.seed}.json")
    training_path.parent.mkdir(parents=True, exist_ok=True)
    with open(training_path, "w") as f:
        json.dump({
            "env":             args.env_id,
            "method":          "csil",
            "K":               args.n_demos,
            "train_seed":      args.seed,
            "episode_returns": scores,
            "n_episodes":      len(scores),
        }, f)
    print(f"  Training curve saved → {training_path}")


if __name__ == "__main__":
    main()

# ── Example (PowerShell) ───────────────────────────────────────────────────────
# python scripts/06_train_csil.py `
#   --env-id CartPole-v1 `
#   --expert-npz data/expert_trajectories/CartPole-v1/expert_K15_seed0.npz `
#   --n-demos 10 --seed 0 --subsample-freq 20 --device cuda