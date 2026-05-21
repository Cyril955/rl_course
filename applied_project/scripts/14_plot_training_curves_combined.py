"""
Training-curve figure: both environments, K=1 and K=15.
Layout: 2 rows (CartPole, Acrobot) × 2 cols (K=1, K=15).
Single legend centred below all subplots.

Usage:
  python scripts/14_plot_training_curves_combined.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# ── Config ───────────────────────────────────────────────────────────────────
K_VALUES  = [1, 15]
ENVS      = ["CartPole-v1", "Acrobot-v1"]
N_INTERP  = 300
SMOOTH_W  = 15
N_SEEDS   = 5   # number of training seeds; std is divided by sqrt(N_SEEDS)

COLOURS = {
    "iq_learn":  "#9f0fbf",
    "csil":      "#0fa7d1",
    "csil_soar": "#e05a00",
    "bc":        "#F0A500",
    "expert":    "#4CAF50",
    "random":    "#9E9E9E",
}
LABELS = {
    "iq_learn":  r"IQ-Learn ($\chi^2$)",
    "csil":      "CSIL",
    "csil_soar": "CSIL + SOAR",
    "bc":        "BC",
    "expert":    "Expert (PPO)",
    "random":    "Random",
}
LINESTYLES = {
    "iq_learn":  "-",
    "csil":      "-",
    "csil_soar": "-",
    "bc":        "--",
    "expert":    "--",
    "random":    "--",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_training_curves(method, env, k):
    pattern = os.path.join(RESULTS_DIR, "training", method, env,
                           f"{method}_K{k}_seed*.json")
    curves = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        if "episode_returns" in d:
            curves.append(np.array(d["episode_returns"], dtype=float))
    return curves


def interpolate_curves(curves):
    x_new = np.linspace(0, 1, N_INTERP)
    resampled = []
    for c in curves:
        smooth = uniform_filter1d(c, size=SMOOTH_W, mode="nearest")
        x_old  = np.linspace(0, 1, len(smooth))
        resampled.append(np.interp(x_new, x_old, smooth))
    arr = np.array(resampled)
    return x_new, arr.mean(axis=0), arr.std(axis=0)


def load_eval_mean(method, env, k):
    pattern = os.path.join(RESULTS_DIR, "evaluation", method, env,
                           f"{method}_K{k}_seed*.json")
    means = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        if "per_eval_seed_means" in d:
            means.extend(d["per_eval_seed_means"])
        elif "mean_return" in d:
            means.append(d["mean_return"])
    return float(np.mean(means)) if means else None


def load_baseline(name, env):
    base = os.path.join(RESULTS_DIR, "evaluation", "baselines", env)
    if name == "random":
        path = os.path.join(base, "random.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)["mean_return"]
    means = [json.load(open(p))["mean_return"]
             for p in glob.glob(os.path.join(base, "expert_seed*.json"))]
    return float(np.mean(means)) if means else None


# ── Build figure ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    len(ENVS), len(K_VALUES),
    figsize=(5 * len(K_VALUES), 4 * len(ENVS)),
    sharey="row",
)

handles_global = {}   # method → first artist encountered, for shared legend

for row, env in enumerate(ENVS):
    expert_val = load_baseline("expert", env)
    random_val = load_baseline("random", env)

    for col, K in enumerate(K_VALUES):
        ax = axes[row, col]

        # training curves
        for method in ("iq_learn", "csil", "csil_soar"):
            curves = load_training_curves(method, env, K)
            if not curves:
                continue
            x, mean, std = interpolate_curves(curves)
            se    = std / np.sqrt(N_SEEDS)
            color = COLOURS[method]
            line, = ax.plot(x, mean, color=color, lw=1.5,
                            linestyle=LINESTYLES[method])
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)
            handles_global.setdefault(method, line)

        # flat reference lines
        bc_val = load_eval_mean("bc", env, K)
        for name, val in [("bc", bc_val), ("expert", expert_val), ("random", random_val)]:
            if val is None:
                continue
            line = ax.axhline(val, color=COLOURS[name], lw=1.5,
                              linestyle=LINESTYLES[name])
            handles_global.setdefault(name, line)

        ax.set_title(f"{env} — K = {K}", fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

        if row == len(ENVS) - 1:
            ax.set_xlabel("Normalised training progress", fontsize=10)
        if col == 0:
            ax.set_ylabel("Episode return", fontsize=10)

# ── Shared legend (bottom, full width) ───────────────────────────────────────
order       = ["iq_learn", "csil", "csil_soar", "bc", "expert", "random"]
leg_handles = [handles_global[m] for m in order if m in handles_global]
leg_labels  = [LABELS[m]         for m in order if m in handles_global]

fig.legend(
    leg_handles, leg_labels,
    loc="lower center",
    ncol=len(leg_handles),
    fontsize=9,
    frameon=True,
    framealpha=0.9,
    bbox_to_anchor=(0.5, 0.0),
)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.06, 1, 1])   # leave room for the bottom legend

out_dir  = os.path.join(RESULTS_DIR, "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "training_curves_combined.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
