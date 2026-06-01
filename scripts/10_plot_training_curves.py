"""
Training-curve figure: 5 subplots (one per K), for a given environment.
Methods shown as curves: CSIL, IQ-Learn, CSIL-SOAR (skipped if data missing).
Reference flat lines: BC, Expert, Random (from evaluation results).
X-axis is normalized training progress [0, 1] so different total lengths compare fairly.

Usage:
  python scripts/10_plot_training_curves.py --env CartPole-v1
  python scripts/10_plot_training_curves.py --env Acrobot-v1
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# ── Config ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="CartPole-v1",
                    choices=["CartPole-v1", "Acrobot-v1"])
args = parser.parse_args()
ENV = args.env
K_VALUES = [1, 3, 7, 10, 15]
N_INTERP = 300   # x-grid resolution for interpolation
SMOOTH_W = 15    # uniform smoothing window (episodes) applied before plotting

COLORS = {
    "iq_learn":  "#9f0fbf",
    "csil":      "#0fa7d1",
    "csil_soar": "#e05a00",
    "bc":        "#E8732A",
    "expert":    "#4CAF50",
    "random":    "#9E9E9E",
}
LABELS = {
    "iq_learn":  "IQ-Learn",
    "csil":      "CSIL",
    "csil_soar": "CSIL+SOAR",
    "bc":        "BC",
    "expert":    "Expert",
    "random":    "Random",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_training_curves(method, env, k):
    pattern = os.path.join(
        RESULTS_DIR, "training", method, env,
        f"{method}_K{k}_seed*.json",
    )
    curves = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        if "episode_returns" in d:
            curves.append(np.array(d["episode_returns"], dtype=float))
    return curves


def interpolate_curves(curves, n=N_INTERP):
    """Resample each curve to n points on [0,1], return (x, mean, std)."""
    x_new = np.linspace(0, 1, n)
    resampled = []
    for c in curves:
        smooth = uniform_filter1d(c, size=SMOOTH_W, mode="nearest")
        x_old = np.linspace(0, 1, len(smooth))
        resampled.append(np.interp(x_new, x_old, smooth))
    arr = np.array(resampled)
    return x_new, arr.mean(axis=0), arr.std(axis=0)


def load_eval_mean(method, env, k):
    pattern = os.path.join(
        RESULTS_DIR, "evaluation", method, env,
        f"{method}_K{k}_seed*.json",
    )
    means = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        if "mean_return" in d:
            means.append(d["mean_return"])
    return float(np.mean(means)) if means else None


def load_baseline(name, env):
    base = os.path.join(RESULTS_DIR, "evaluation", "baselines", env)
    if name == "random":
        with open(os.path.join(base, "random.json")) as f:
            return json.load(f)["mean_return"]
    means = []
    for path in glob.glob(os.path.join(base, "expert_seed*.json")):
        with open(path) as f:
            means.append(json.load(f)["mean_return"])
    return float(np.mean(means)) if means else None


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(K_VALUES), figsize=(15, 3.2), sharey=True)
fig.suptitle(f"Training curves — {ENV}  (x-axis: normalised training progress)", fontsize=10, y=1.01)

expert_val = load_baseline("expert", ENV)
random_val = load_baseline("random", ENV)

handles_added = {}  # track legend handles

for ax, K in zip(axes, K_VALUES):
    # ── training curves ──
    for method in ("iq_learn", "csil", "csil_soar"):
        curves = load_training_curves(method, ENV, K)
        if not curves:
            continue
        x, mean, std = interpolate_curves(curves)
        color = COLORS[method]
        label = LABELS[method] if method not in handles_added else "_nolegend_"
        line, = ax.plot(x, mean, color=color, lw=1.6, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        handles_added[method] = line

    # ── flat reference lines ──
    bc_val = load_eval_mean("bc", ENV, K)
    for name, val in [("bc", bc_val), ("expert", expert_val), ("random", random_val)]:
        if val is None:
            continue
        label = LABELS[name] if name not in handles_added else "_nolegend_"
        line = ax.axhline(val, color=COLORS[name], lw=1.2,
                          linestyle="--", label=label)
        handles_added[name] = line

    ax.set_title(f"K = {K}", fontsize=10)
    ax.set_xlabel("Training progress", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

axes[0].set_ylabel("Episode return", fontsize=9)

# single legend below the subplots
order = ["iq_learn", "csil", "csil_soar", "bc", "expert", "random"]
legend_handles = [handles_added[m] for m in order if m in handles_added]
legend_labels  = [LABELS[m] for m in order if m in handles_added]
fig.legend(legend_handles, legend_labels,
           loc="lower center", ncol=len(legend_handles),
           fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.12))

plt.tight_layout()

out_dir = os.path.join(RESULTS_DIR, "figures")
os.makedirs(out_dir, exist_ok=True)
env_slug = ENV.lower().replace("-", "").replace("v1", "")
out_path = os.path.join(out_dir, f"training_curves_{env_slug}.pdf")
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
