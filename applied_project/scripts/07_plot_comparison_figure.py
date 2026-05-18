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
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENV_IDS = ["CartPole-v1", "Acrobot-v1"]
BASELINES_DIR = Path("results/raw/baselines")
IQ_LEARN_DIR  = Path("results/raw/iq_learn")
CSIL_DIR      = Path("results/raw/csil")
FIGURES_DIR   = Path("results/figures")

# Minimum possible episode return per env (used to clip lower std band)
ENV_RETURN_FLOOR = {
    "CartPole-v1": 0.0,    # reward is +1/step, can't go negative
    "Acrobot-v1": -500.0,  # reward is -1/step, episode capped at 500 steps
}

# Colours match Figure 2 style roughly
COLOURS = {
    "iq_learn": "#9f0fbf",   # purple
    "csil":     "#0fa7d1",   # blue
    "expert":   "#4CAF50",   # green (dashed)
    "random":   "#9E9E9E",   # grey (dashed)
}
LABELS = {
    "iq_learn": r"IQ-Learn ($\chi^2$)",
    "csil":     "CSIL",
    "expert":   "Expert (PPO)",
    "random":   "Random",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_results(results_dir: Path, env_id: str, glob: str) -> dict[int, list[float]]:
    """Return {K: [mean_return per seed]} by globbing JSON files in results_dir/env_id/."""
    env_dir = results_dir / env_id
    if not env_dir.exists():
        return {}
    k_to_returns: dict[int, list[float]] = {}
    for p in sorted(env_dir.glob(glob)):
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


MARKERS = {
    "iq_learn": "^",   # triangle
    "csil":     "o",   # circle
}


def _draw_curve(ax, k_results: dict, method: str, floor: float) -> None:
    """Plot mean line + shaded ±1 std band for one IL method."""
    if not k_results:
        return
    k_sorted = sorted(k_results.keys())
    means = [float(np.mean(k_results[k])) for k in k_sorted]
    stds  = [float(np.std(k_results[k]))  for k in k_sorted]
    ax.plot(k_sorted, means, f"{MARKERS[method]}-",
            color=COLOURS[method], label=LABELS[method], linewidth=1.5, markersize=6)
    ax.fill_between(
        k_sorted,
        [max(floor, m - s) for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2, color=COLOURS[method],
        linewidth=0.0, edgecolor=None,
    )


def plot_env(ax: plt.Axes, env_id: str) -> None:
    floor = ENV_RETURN_FLOOR.get(env_id, -np.inf)

    iq_results   = collect_results(IQ_LEARN_DIR, env_id, "iq_learn_K*_seed*.json")
    csil_results = collect_results(CSIL_DIR,     env_id, "csil_K*_seed*.json")

    if not iq_results and not csil_results:
        ax.set_title(f"{env_id}\n(no results yet)")
        return

    # Determine x range from whichever results exist
    all_k = sorted(set(list(iq_results) + list(csil_results)))
    x_range = [all_k[0], all_k[-1]]

    for method in ("expert", "random"):
        mean, std = collect_baseline(env_id, method)
        if mean is None:
            continue
        ax.hlines(mean, x_range[0], x_range[-1],
                  colors=COLOURS[method], linestyles="--", linewidth=1.5,
                  label=LABELS[method])
        if std and std > 0:
            ax.fill_between(x_range,
                            [max(floor, mean - std)] * 2,
                            [mean + std] * 2,
                            alpha=0.15, color=COLOURS[method])

    _draw_curve(ax, iq_results,   "iq_learn", floor)
    _draw_curve(ax, csil_results, "csil",     floor)

    ax.set_title(env_id, fontsize=12)
    ax.set_xlabel("Number of Expert Trajectories", fontsize=10)
    ax.set_ylabel("Reward", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)


FIGURES = [
    # (envs, out_path, show_title)
    (ENV_IDS,          FIGURES_DIR / "comparison_figure_5_seeds.png",  True),
    (["CartPole-v1"],  FIGURES_DIR / "cartpole_figure_5_seeds.png",    False),
    (["Acrobot-v1"],   FIGURES_DIR / "acrobot_figure_5_seeds.png",     False),
]


def save_figure(envs: list, out_path: Path, show_title: bool, dpi: int = 300) -> None:
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, env_id in zip(axes, envs):
        plot_env(ax, env_id)
        if not show_title:
            ax.set_title("")

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)


def main() -> None:
    for envs, out_path, show_title in FIGURES:
        save_figure(envs, out_path, show_title)


if __name__ == "__main__":
    main()

# python scripts/07_plot_comparison_figure.py
# python scripts/07_plot_comparison_figure.py --out results/figures/comparison_figure_cartpole.png --envs CartPole-v1
