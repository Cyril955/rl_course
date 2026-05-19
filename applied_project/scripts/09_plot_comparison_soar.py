"""Comparison figure: IQ-Learn vs CSIL vs CSIL+SOAR.

Extends 07_plot_comparison_figure.py by adding the CSIL+SOAR results.
Reads JSON files from results/raw/{method}/{env}/ and plots mean ± std band
over expert trajectory counts K.

Expected JSON layout
--------------------
results/raw/
  baselines/{env}/
    random.json                         {"method": "random",  ...}
    expert_seed0.json                   {"method": "expert",  ...}
  iq_learn/{env}/
    iq_learn_K1_seed0.json              {"method": "iq_learn", "K": 1, ...}
    ...
  csil/{env}/
    csil_K1_seed0.json                  {"method": "csil", "K": 1, ...}
    ...
  csil_soar/{env}/
    csil_soar_K1_seed0.json             {"method": "csil_soar", "K": 1, ...}
    ...
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ENV_IDS       = ["CartPole-v1", "Acrobot-v1"]
BASELINES_DIR = Path("results/raw/baselines")
IQ_LEARN_DIR  = Path("results/raw/iq_learn")
CSIL_DIR      = Path("results/raw/csil")
CSIL_SOAR_DIR = Path("results/raw/csil_soar")
FIGURES_DIR   = Path("results/figures")

ENV_RETURN_FLOOR = {
    "CartPole-v1": 0.0,
    "Acrobot-v1":  -500.0,
}

COLOURS = {
    "iq_learn":  "#9f0fbf",  # purple
    "csil":      "#0fa7d1",  # blue
    "csil_soar": "#e05a00",  # orange — SOAR variant
    "expert":    "#4CAF50",  # green  (dashed baseline)
    "random":    "#9E9E9E",  # grey   (dashed baseline)
}
LABELS = {
    "iq_learn":  r"IQ-Learn ($\chi^2$)",
    "csil":      "CSIL",
    "csil_soar": "CSIL + SOAR",
    "expert":    "Expert (PPO)",
    "random":    "Random",
}
MARKERS = {
    "iq_learn":  "^",
    "csil":      "o",
    "csil_soar": "s",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_results(results_dir: Path, env_id: str, glob: str) -> dict[int, list[float]]:
    """Return {K: [mean_return per seed]} by globbing JSON files."""
    env_dir = results_dir / env_id
    if not env_dir.exists():
        return {}
    k_to_returns: dict[int, list[float]] = {}
    for p in sorted(env_dir.glob(glob)):
        d = load_json(p)
        k = d["K"]
        k_to_returns.setdefault(k, []).append(d["mean_return"])
    return k_to_returns


def collect_baseline(env_id: str, method: str) -> tuple[float | None, float | None]:
    env_dir = BASELINES_DIR / env_id

    if method == "random":
        p = env_dir / "random.json"
        if not p.exists():
            return None, None
        d = load_json(p)
        return d["mean_return"], d.get("std_return", 0.0)

    if method == "expert":
        vals = [load_json(p)["mean_return"] for p in sorted(env_dir.glob("expert_seed*.json"))]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    raise ValueError(f"Unknown baseline method: {method}")


def _draw_curve(ax, k_results: dict, method: str, floor: float) -> None:
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
        alpha=0.2, color=COLOURS[method], linewidth=0.0, edgecolor=None,
    )


def plot_env(ax: plt.Axes, env_id: str) -> None:
    floor = ENV_RETURN_FLOOR.get(env_id, -np.inf)

    iq_results        = collect_results(IQ_LEARN_DIR,  env_id, "iq_learn_K*_seed*.json")
    csil_results      = collect_results(CSIL_DIR,      env_id, "csil_K*_seed*.json")
    csil_soar_results = collect_results(CSIL_SOAR_DIR, env_id, "csil_soar_K*_seed*.json")

    all_results = [iq_results, csil_results, csil_soar_results]
    if not any(all_results):
        ax.set_title(f"{env_id}\n(no results yet)")
        return

    all_k   = sorted({k for r in all_results for k in r})
    x_range = [all_k[0], all_k[-1]] if all_k else [0, 1]

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

    _draw_curve(ax, iq_results,        "iq_learn",  floor)
    _draw_curve(ax, csil_results,      "csil",      floor)
    _draw_curve(ax, csil_soar_results, "csil_soar", floor)

    ax.set_title(env_id, fontsize=12)
    ax.set_xlabel("Number of Expert Trajectories (K)", fontsize=10)
    ax.set_ylabel("Evaluation Return", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)


FIGURES = [
    (ENV_IDS,          FIGURES_DIR / "comparison_soar_all.png",         True),
    (["CartPole-v1"],  FIGURES_DIR / "comparison_soar_cartpole.png",    False),
    (["Acrobot-v1"],   FIGURES_DIR / "comparison_soar_acrobot.png",     False),
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
    parser = argparse.ArgumentParser(description="Plot IQ-Learn vs CSIL vs CSIL+SOAR comparison.")
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Env IDs to plot (default: CartPole-v1 Acrobot-v1).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for a single custom figure.")
    args = parser.parse_args()

    if args.out:
        envs = args.envs or ENV_IDS
        save_figure(envs, Path(args.out), show_title=True)
    else:
        for envs, out_path, show_title in FIGURES:
            save_figure(envs, out_path, show_title)


if __name__ == "__main__":
    main()

# ── Usage ──────────────────────────────────────────────────────────────────────
# python scripts/09_plot_comparison_soar.py
# python scripts/09_plot_comparison_soar.py --envs CartPole-v1 --out results/figures/soar_cartpole.png
