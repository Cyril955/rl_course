"""
Training-curve figure: 2×3 grid — top-left = legend, then K=1,3 / K=7,10,15.
Methods shown as curves: CSIL, IQ-Learn, CSIL-SOAR (skipped if data missing).
Reference flat lines: BC, Expert, Random (from evaluation results).
X-axis is normalised training progress [0, 1].

Usage:
  python scripts/13_plot_training_curves.py --env CartPole-v1
  python scripts/13_plot_training_curves.py --env Acrobot-v1
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="CartPole-v1",
                    choices=["CartPole-v1", "Acrobot-v1"])
args = parser.parse_args()
ENV = args.env

# ── Config ───────────────────────────────────────────────────────────────────
K_VALUES  = [1, 3, 7, 10, 15]
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

# subplot grid positions for K=1,3,7,10,15 (row, col); (0,0) reserved for legend
SUBPLOT_POS = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


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
        with open(os.path.join(base, "random.json")) as f:
            return json.load(f)["mean_return"]
    means = [json.load(open(p))["mean_return"]
             for p in glob.glob(os.path.join(base, "expert_seed*.json"))]
    return float(np.mean(means)) if means else None


# ── Build figure ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)

expert_val = load_baseline("expert", ENV)
random_val = load_baseline("random", ENV)

handles_added = {}

for (row, col), K in zip(SUBPLOT_POS, K_VALUES):
    ax = axes[row, col]

    # training curves
    for method in ("iq_learn", "csil", "csil_soar"):
        curves = load_training_curves(method, ENV, K)
        if not curves:
            continue
        x, mean, std = interpolate_curves(curves)
        color = COLOURS[method]
        line, = ax.plot(x, mean, color=color, lw=1.5,
                        linestyle=LINESTYLES[method])
        se = std / np.sqrt(N_SEEDS)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)
        handles_added.setdefault(method, line)

    # flat reference lines
    bc_val = load_eval_mean("bc", ENV, K)
    for name, val in [("bc", bc_val), ("expert", expert_val), ("random", random_val)]:
        if val is None:
            continue
        line = ax.axhline(val, color=COLOURS[name], lw=1.5,
                          linestyle=LINESTYLES[name])
        handles_added.setdefault(name, line)

    ax.set_title(f"K = {K}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # x-label only on bottom row
    if row == 1:
        ax.set_xlabel("Normalised training progress", fontsize=10)

    # y-label only on leftmost data column per row
    if col == 0 or (row == 0 and col == 1):
        ax.set_ylabel("Episode return", fontsize=10)

# ── Legend panel (top-left) ──────────────────────────────────────────────────
ax_leg = axes[0, 0]
ax_leg.set_axis_off()

order = ["iq_learn", "csil", "csil_soar", "bc", "expert", "random"]
leg_handles = [handles_added[m] for m in order if m in handles_added]
leg_labels  = [LABELS[m]        for m in order if m in handles_added]
ax_leg.legend(leg_handles, leg_labels,
              loc="center", fontsize=9,
              frameon=True, framealpha=0.9,
              title=ENV, title_fontsize=10,
              handlelength=2.2)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout()

out_dir  = os.path.join(RESULTS_DIR, "figures")
os.makedirs(out_dir, exist_ok=True)
env_slug = ENV.lower().replace("-", "").replace("v1", "")
out_path = os.path.join(out_dir, f"training_curves_{env_slug}.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
