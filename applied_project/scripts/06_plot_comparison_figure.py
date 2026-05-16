"""Reproduce Figure 2 from the IQ-Learn paper for CartPole and Acrobot.

Reads JSON result files from results/raw/{env}/ and produces a matplotlib figure
saved to results/figures/comparison_figure.png.

Expected JSON layout
--------------------
results/raw/
  CartPole-v1/
    random.json                     {"method": "random",  "mean_return": ..., "std_return": ...}
    expert_seed0.json               {"method": "expert",  "mean_return": ..., "std_return": ...}
    expert_seed1.json
    expert_seed2.json
    iq_learn_K1_seed0.json          {"method": "iq_learn", "K": 1,  "mean_return": ..., ...}
    iq_learn_K1_seed1.json
    ...
    iq_learn_K10_seed2.json
  Acrobot-v1/
    (same structure)
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENV_IDS = ["CartPole-v1", "Acrobot-v1"]
BASELINES_DIR = Path("results/raw/baselines")
IQ_LEARN_DIR = Path("results/raw/iq_learn")
FIGURES_DIR = Path("results/figures")

# Minimum possible episode return per env (used to clip lower std band)
ENV_RETURN_FLOOR = {
    "CartPole-v1": 0.0,    # reward is +1/step, can't go negative
    "Acrobot-v1": -500.0,  # reward is -1/step, episode capped at 500 steps
}

# Colours match Figure 2 style roughly
COLOURS = {
    "iq_learn": "#9f0fbf",   # purple (solid)
    "expert":   "#4CAF50",   # green (dashed)
    "random":   "#9E9E9E",   # grey (dashed)
}
LABELS = {
    "iq_learn": r"IQ-Learn ($\chi^2$)",
    "expert":   "Expert (PPO)",
    "random":   "Random",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_iq_results(env_id: str) -> dict[int, list[float]]:
    """Return {K: [mean_return per seed]} for iq_learn runs."""
    env_dir = IQ_LEARN_DIR / env_id
    k_to_returns: dict[int, list[float]] = {}

    for p in sorted(env_dir.glob("iq_learn_K*_seed*.json")):
        d = load_json(p)
        k = d["K"]
        k_to_returns.setdefault(k, []).append(d["mean_return"])

    return k_to_returns


def collect_baseline(env_id: str, method: str) -> tuple[float, float]:
    """Return (mean, std) for expert or random baseline aggregated across seeds."""
    env_dir = BASELINES_DIR / env_id

    if method == "random":
        p = env_dir / "random.json"
        if not p.exists():
            return None, None
        d = load_json(p)
        return d["mean_return"], d.get("std_return", 0.0)

    if method == "expert":
        vals = []
        for p in sorted(env_dir.glob("expert_seed*.json")):
            vals.append(load_json(p)["mean_return"])
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    raise ValueError(f"Unknown method: {method}")


def plot_env(ax: plt.Axes, env_id: str) -> None:
    k_results = collect_iq_results(env_id)
    if not k_results:
        ax.set_title(f"{env_id}\n(no IQ-Learn results yet)")
        return

    floor = ENV_RETURN_FLOOR.get(env_id, -np.inf)
    k_sorted = sorted(k_results.keys())
    iq_means = [float(np.mean(k_results[k])) for k in k_sorted]
    iq_stds = [float(np.std(k_results[k])) for k in k_sorted]

    # IQ-Learn curve
    ax.plot(k_sorted, iq_means, "o-", color=COLOURS["iq_learn"], label=LABELS["iq_learn"], linewidth=1.5)
    ax.fill_between(
        k_sorted,
        [max(floor, m - s) for m, s in zip(iq_means, iq_stds)],
        [m + s for m, s in zip(iq_means, iq_stds)],
        alpha=0.2, color=COLOURS["iq_learn"],
    )

    # Horizontal baselines
    x_range = [k_sorted[0], k_sorted[-1]]
    for method in ("expert", "random"):
        mean, std = collect_baseline(env_id, method)
        if mean is None:
            continue
        ax.hlines(mean, x_range[0], x_range[-1],
                  colors=COLOURS[method], linestyles="--", linewidth=1.5,
                  label=LABELS[method])

    ax.set_title(env_id, fontsize=12)
    ax.set_xlabel("Number of Expert Trajectories", fontsize=10)
    ax.set_ylabel("Reward", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=ENV_IDS)
    parser.add_argument("--out", type=str, default=str(FIGURES_DIR / "comparison_figure.png"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    n = len(args.envs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, env_id in zip(axes, args.envs):
        plot_env(ax, env_id)

    fig.suptitle("Offline IL Results (IQ-Learn vs Expert vs Random)", fontsize=13)
    fig.tight_layout()
    fig.text(0.5, -0.02, "Shaded regions: ±1 std over 3 seeds",
             ha="center", fontsize=8, style="italic", color="gray")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

# python scripts/06_plot_comparison_figure.py
# python scripts/06_plot_comparison_figure.py --out results/figures/comparison_figure_cartpole.png --envs CartPole-v1
