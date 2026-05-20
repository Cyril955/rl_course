"""Train and evaluate CSIL+SOAR for a single (env, K, seed) configuration.

Mirrors 06_train_csil.py exactly, but uses CSILSOARAgent instead of CSILAgent.
Two extra hyperparameters control the SOAR ensemble:
  --n-critics  L   number of independent critics  (default 4, paper uses 4)
  --sigma-clip σ   std-deviation clipping threshold (default 1.0)

Results  → results/raw/csil_soar/{env}/
Models   → models/csil_soar/{env}/
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
from config import CSIL_SOAR_CONFIG, ENV_CONFIG, subsample_trajectories


def main() -> None:
    cfg = CSIL_SOAR_CONFIG  # shorthand

    parser = argparse.ArgumentParser(
        description="Train and evaluate CSIL+SOAR on one (env, K, seed) configuration."
    )
    parser.add_argument("--env-id",     type=str, required=True)
    parser.add_argument("--expert-npz", type=str, required=True,
                        help="Expert pool NPZ (e.g. data/expert/CartPole-v1/expert_K15_seed0.npz).")
    parser.add_argument("--n-demos",    type=int, required=True,
                        help="Number of expert trajectories to use (K).")
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--subsample-freq", type=int, default=1,
                        help="Keep every Nth transition within each expert trajectory "
                             "(temporal decimation, same as IQ-Learn's subsample_freq).")
    # BC
    parser.add_argument("--bc-epochs",     type=int,   default=cfg["bc_epochs"])
    parser.add_argument("--bc-lr",         type=float, default=cfg["bc_lr"])
    parser.add_argument("--bc-batch-size", type=int,   default=cfg["bc_batch_size"])
    # SAC / CSIL
    parser.add_argument("--n-episodes",        type=int,   default=cfg["n_episodes"])
    parser.add_argument("--start-steps",       type=int,   default=cfg["start_steps"])
    parser.add_argument("--alpha-csil",        type=float, default=cfg["alpha_csil"],
                        help="Coherent reward temperature α.")
    parser.add_argument("--early-stop-reward", type=float, default=None,
                        help="Stop training once 100-ep average exceeds this.")
    # SOAR-specific
    parser.add_argument("--n-critics",  type=int,   default=cfg["n_critics"],
                        help="Number of critics in the SOAR ensemble (L).")
    parser.add_argument("--sigma-clip", type=float, default=cfg["sigma_clip"],
                        help="Std-deviation clipping threshold σ for OptimisticQ-NN.")
    # Evaluation
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="Seed for evaluation episodes. Keeps eval independent "
                             "of training randomness for fair cross-method comparison.")
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device: 'cpu', 'cuda', or 'cuda:0' etc.")
    # Output
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save the trained model (.pt). "
                             "Defaults to models/csil_soar/{env}/K{K}_seed{seed}.pt")
    parser.add_argument("--save-json",  type=str, default=None,
                        help="Path to write the evaluation result JSON.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] --device {args.device} requested but CUDA is not available. Falling back to cpu.")
        args.device = "cpu"

    # Fill early-stop and sigma-clip from per-env config if not provided on CLI
    env_cfg = ENV_CONFIG.get(args.env_id, {})
    if args.early_stop_reward is None:
        args.early_stop_reward = env_cfg.get("early_stop_reward", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_npz = Path(args.expert_npz).resolve()
    if not expert_npz.exists():
        raise FileNotFoundError(f"Expert NPZ not found: {expert_npz}")

    expert_states, expert_actions = subsample_trajectories(
        expert_npz, args.n_demos, args.seed, args.subsample_freq)
    print(f"[CSIL-SOAR] {args.env_id}  K={args.n_demos}  seed={args.seed}  subsample_freq={args.subsample_freq}  "
          f"L={args.n_critics}  σ={args.sigma_clip}")
    print(f"            Expert data: {args.n_demos} trajectories → {len(expert_states)} transitions")

    env = gym.make(args.env_id)

    csil_soar_dict = {
        "bc_hidden_size":    cfg["bc_hidden_size"],
        "bc_epochs":         args.bc_epochs,
        "bc_lr":             args.bc_lr,
        "bc_batch_size":     args.bc_batch_size,
        "sac_hidden_size":   cfg["sac_hidden_size"],
        "sac_lr":            cfg["sac_lr"],
        "gamma":             cfg["gamma"],
        "tau":               cfg["tau"],
        "alpha_csil":        args.alpha_csil,
        "n_critics":         args.n_critics,
        "sigma_clip":        args.sigma_clip,
        "buffer_size":       cfg["buffer_size"],
        "batch_size":        cfg["batch_size"],
        "start_steps":       args.start_steps,
        "n_episodes":        args.n_episodes,
        "early_stop_reward": args.early_stop_reward,
        "device":            args.device,
        "verbose":           True,
    }

    agent, _, bc_policy = run_csil_soar(env, expert_states, expert_actions, csil_soar_dict)
    env.close()

    eval_env = gym.make(args.env_id)
    mean_ret, std_ret = evaluate_csil_soar(eval_env, agent, n_episodes=args.n_eval_episodes,
                                            device=args.device, eval_seed=args.eval_seed)
    eval_env.close()

    print(f"\n[{args.env_id}] CSIL-SOAR  K={args.n_demos}  seed={args.seed}  "
          f"L={args.n_critics}  σ={args.sigma_clip}")
    print(f"  Mean return: {mean_ret:.2f} ± {std_ret:.2f}  (n={args.n_eval_episodes})")

    # ── Save model ─────────────────────────────────────────────────────────────
    model_path = Path(
        args.save_model
        or f"models/csil_soar/{args.env_id}/K{args.n_demos}_seed{args.seed}.pt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor":      agent.actor.state_dict(),
        "bc_policy":  bc_policy.state_dict(),
        "env_id":     args.env_id,
        "K":          args.n_demos,
        "seed":       args.seed,
        "n_critics":  args.n_critics,
        "sigma_clip": args.sigma_clip,
        "config":     csil_soar_dict,
    }, model_path)
    print(f"  Model saved → {model_path}")

    # ── Save JSON result ───────────────────────────────────────────────────────
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
        print(f"  Result saved → {save_path}")


if __name__ == "__main__":
    main()

# ── Example — single run (PowerShell) ─────────────────────────────────────────
# python scripts/08_train_csil_soar.py `
#   --env-id CartPole-v1 `
#   --expert-npz data/expert/CartPole-v1/expert_K15_seed0.npz `
#   --n-demos 10 --seed 0 --n-episodes 1000 `
#   --n-critics 4 --sigma-clip 1.0 `
#   --device cuda `
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
#         --n-critics 4 --sigma-clip 1.0 `
#         --device cuda `
#         --save-json "results/raw/csil_soar/$env/csil_soar_K${K}_seed${seed}.json"
#     }
#   }
# }