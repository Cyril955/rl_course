"""
Centralised configuration for the applied project.

All hyperparameters and per-environment settings live here.
Training scripts import from this module and use the values as
argparse defaults, so individual runs can still override via CLI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ── Directory layout ───────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

EXPERT_DATA_DIR   = DATA_DIR / "expert"
IQ_LEARN_DATA_DIR = DATA_DIR / "iq_learn"

# ── Sweep grid ─────────────────────────────────────────────────────────────────
K_VALUES        = [1, 3, 7, 10, 15]
SEEDS           = [0, 1, 2, 3, 4]
EVAL_SEEDS      = [10, 11, 12, 13, 14]
EXPERT_POOL_K   = 15
N_EVAL_EPISODES = 20

# ── Shell variable prefix per environment (for run_sweep.sh) ──────────────────
SHELL_PREFIX: dict[str, str] = {
    "CartPole-v1": "CARTPOLE",
    "Acrobot-v1":  "ACROBOT",
}

# ── Per-environment settings ───────────────────────────────────────────────────
# early_stop_reward : None means no early stopping (Acrobot reward is negative)
# sigma_clip        : std-deviation clipping threshold σ for CSIL-SOAR
ENV_CONFIG: dict[str, dict] = {
    "CartPole-v1": {
        "expert_timesteps":      100_000,
        "early_stop_reward":     495.0,
        "n_episodes_csil":       1_500,
        "n_episodes_csil_soar":  1_500,
        "learn_steps_iq":        100_000,
        "subsample_freq_iq":     20,
        "sigma_clip_csil_soar":  1.0,
    },
    "Acrobot-v1": {
        "expert_timesteps":      300_000,
        "early_stop_reward":     None,
        "n_episodes_csil":       3_000,
        "n_episodes_csil_soar":  3_000,
        "learn_steps_iq":        200_000,
        "subsample_freq_iq":     5,
        "sigma_clip_csil_soar":  1.0,
    },
}

# ── BC hyperparameters ─────────────────────────────────────────────────────────
BC_CONFIG: dict = {
    "hidden_size": 64,
    "epochs":      200,
    "lr":          3e-4,
    "batch_size":  256,
}

# ── CSIL hyperparameters ───────────────────────────────────────────────────────
CSIL_CONFIG: dict = {
    "bc_hidden_size":  BC_CONFIG["hidden_size"],
    "bc_epochs":       BC_CONFIG["epochs"],
    "bc_lr":           BC_CONFIG["lr"],
    "bc_batch_size":   BC_CONFIG["batch_size"],
    "sac_hidden_size": 64,
    "sac_lr":          3e-4,
    "gamma":           0.99,
    "tau":             0.005,
    "alpha_csil":      0.1,
    "buffer_size":     200_000,
    "batch_size":      256,
    "start_steps":     1_000,
    "n_episodes":      1_000,
}

# ── CSIL-SOAR hyperparameters ──────────────────────────────────────────────────
CSIL_SOAR_CONFIG: dict = {
    **CSIL_CONFIG,
    "n_critics":  4,
    "sigma_clip": 1.0,
}


# ── Shared data utility ────────────────────────────────────────────────────────
def subsample_trajectories(
    npz_path: Path,
    n_demos: int,
    seed: int,
    subsample_freq: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Select n_demos trajectories from an expert NPZ pool.

    subsample_freq: keep every Nth transition within each trajectory
                    (temporal decimation, same as IQ-Learn's expert.subsample_freq).
    Returns flat (states, actions) arrays ready for BC / CSIL training.
    """
    data    = np.load(npz_path)
    states  = data["states"]
    actions = data["actions"]
    dones   = data["dones"].astype(bool)

    trajectories: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i, done in enumerate(dones):
        if done:
            trajectories.append((states[start : i + 1], actions[start : i + 1]))
            start = i + 1
    if start < len(states):
        trajectories.append((states[start:], actions[start:]))

    n = min(n_demos, len(trajectories))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(trajectories), size=n, replace=False))

    expert_states  = np.concatenate([trajectories[i][0][::subsample_freq] for i in chosen])
    expert_actions = np.concatenate([trajectories[i][1][::subsample_freq] for i in chosen])
    return expert_states, expert_actions


# ── Shell export (called by run_sweep.sh) ──────────────────────────────────────
if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--shell", action="store_true",
                    help="Print bash variable assignments for sourcing in run_sweep.sh.")
    _a = _p.parse_args()

    if not _a.shell:
        _p.print_help()
    else:
        # ── Global sweep parameters ──────────────────────────────────────────
        envs_str = " ".join(f'"{e}"' for e in ENV_CONFIG)
        print(f'ENVS=({envs_str})')
        print(f'K_VALUES=({" ".join(str(k) for k in K_VALUES)})')
        print(f'SEEDS=({" ".join(str(s) for s in SEEDS)})')
        print(f'EVAL_SEEDS=({" ".join(str(s) for s in EVAL_SEEDS)})')
        print(f'EXPERT_POOL_K={EXPERT_POOL_K}')
        print(f'N_EVAL_EPISODES={N_EVAL_EPISODES}')
        print(f'N_CRITICS={CSIL_SOAR_CONFIG["n_critics"]}')
        print(f'DEVICE={DEVICE}')

        # ── Associative array for env → shell prefix lookup ──────────────────
        pairs = " ".join(f'["{env}"]="{pfx}"' for env, pfx in SHELL_PREFIX.items())
        print(f'declare -A SHELL_PREFIXES=({pairs})')

        # ── Per-environment variables ─────────────────────────────────────────
        for env_id, ecfg in ENV_CONFIG.items():
            pfx = SHELL_PREFIX[env_id]
            early = ecfg["early_stop_reward"]
            print(f'EXPERT_TIMESTEPS_{pfx}={ecfg["expert_timesteps"]}')
            print(f'LEARN_STEPS_{pfx}={ecfg["learn_steps_iq"]}')
            print(f'SUBSAMPLE_FREQ_{pfx}={ecfg["subsample_freq_iq"]}')
            print(f'N_EPISODES_CSIL_{pfx}={ecfg["n_episodes_csil"]}')
            print(f'N_EPISODES_CSIL_SOAR_{pfx}={ecfg["n_episodes_csil_soar"]}')
            print(f'SIGMA_CLIP_{pfx}={ecfg["sigma_clip_csil_soar"]}')
            # Empty string when None so the shell can test with [ -n "$VAR" ]
            print(f'EARLY_STOP_{pfx}={early if early is not None else ""}')