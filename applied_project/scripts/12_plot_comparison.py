"""Unified comparison figure: BC, IQ-Learn, CSIL, CSIL+SOAR.

Mean and std are computed over ALL training seeds × evaluation seeds.
Each evaluation JSON stores per_eval_seed_means (one mean per eval seed).
For each K, all per_eval_seed_means across training seeds are aggregated,
giving 5 train_seeds × 5 eval_seeds = 25 data points for mean/std.

Results → results/figures/
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR    = Path("results/evaluation")
FIGURES_DIR = Path("results/figures")

ENV_IDS = ["CartPole-v1", "Acrobot-v1"]

# Total number of seeds used to compute the standard error (train_seeds × eval_seeds)
N_SEEDS = 25

ENV_RETURN_FLOOR = {
    "CartPole-v1":  0.0,
    "Acrobot-v1": -500.0,
}

COLOURS = {
    "bc":        "#F0A500",   # orange
    "iq_learn":  "#9f0fbf",   # purple
    "csil":      "#0fa7d1",   # blue
    "csil_soar": "#e05a00",   # red-orange
    "expert":    "#4CAF50",   # green (dashed)
    "random":    "#9E9E9E",   # grey  (dashed)
}
LABELS = {
    "bc":        "BC",
    "iq_learn":  r"IQ-Learn ($\chi^2$)",
    "csil":      "CSIL",
    "csil_soar": "CSIL + SOAR",
    "expert":    "Expert (PPO)",
    "random":    "Random",
}
MARKERS = {"bc": "D", "iq_learn": "^", "csil": "o", "csil_soar": "s"}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_results(method: str, env_id: str, glob: str) -> dict[int, list[float]]:
    """Return {K: flat list of all per_eval_seed_means across training seeds}.

    If a JSON was produced by an old script (no per_eval_seed_means),
    falls back to mean_return as a single-point estimate.
    """
    env_dir = EVAL_DIR / method / env_id
    if not env_dir.exists():
        return {}
    k_to_values: dict[int, list[float]] = {}
    for p in sorted(env_dir.glob(glob)):
        d = load_json(p)
        k = d["K"]
        if "per_eval_seed_means" in d:
            k_to_values.setdefault(k, []).extend(d["per_eval_seed_means"])
        else:
            k_to_values.setdefault(k, []).append(d["mean_return"])
    return k_to_values


def collect_baseline(env_id: str, method: str) -> tuple[float | None, float | None]:
    env_dir = EVAL_DIR / "baselines" / env_id
    if method == "random":
        p = env_dir / "random.json"
        if not p.exists():
            return None, None
        d = load_json(p)
        return d["mean_return"], d.get("std_return", 0.0)
    if method == "expert":
        vals = [load_json(p)["mean_return"]
                for p in sorted(env_dir.glob("expert_seed*.json"))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
    raise ValueError(method)


def _draw_curve(ax, k_results: dict, method: str, lower: float, upper: float) -> None:
    if not k_results:
        return
    k_sorted = sorted(k_results)
    means = [float(np.mean(k_results[k])) for k in k_sorted]
    stds  = [float(np.std(k_results[k])) / np.sqrt(N_SEEDS) for k in k_sorted]
    print(f"  [{LABELS[method]}]")
    for k, m, s in zip(k_sorted, means, stds):
        print(f"    K={k:3d}  mean={m:8.3f}  se={s:.3f}")
    ax.plot(k_sorted, means, f"{MARKERS[method]}-",
            color=COLOURS[method], label=LABELS[method], linewidth=1.5, markersize=6)
    ax.fill_between(
        k_sorted,
        [max(lower, m - s) for m, s in zip(means, stds)],
        [min(upper, m + s) for m, s in zip(means, stds)],
        alpha=0.1, color=COLOURS[method], linewidth=0.0,
    )


def plot_env(ax: plt.Axes, env_id: str) -> None:
    floor = ENV_RETURN_FLOOR.get(env_id, -np.inf)

    method_results = {
        "bc":        collect_results("bc",        env_id, "bc_K*_seed*.json"),
        "iq_learn":  collect_results("iq_learn",  env_id, "iq_learn_K*_seed*.json"),
        "csil":      collect_results("csil",       env_id, "csil_K*_seed*.json"),
        "csil_soar": collect_results("csil_soar", env_id, "csil_soar_K*_seed*.json"),
    }

    if not any(method_results.values()):
        ax.set_title(f"{env_id}\n(no results yet)")
        return

    all_k   = sorted({k for r in method_results.values() for k in r})
    x_range = [all_k[0], all_k[-1]] if all_k else [0, 1]

    random_mean, _ = collect_baseline(env_id, "random")
    expert_mean, _ = collect_baseline(env_id, "expert")
    shade_lower = random_mean if random_mean is not None else floor
    shade_upper = expert_mean if expert_mean is not None else np.inf

    for baseline in ("expert", "random"):
        mean, std = collect_baseline(env_id, baseline)
        if mean is None:
            continue
        ax.hlines(mean, x_range[0], x_range[-1],
                  colors=COLOURS[baseline], linestyles="--", linewidth=1.5,
                  label=LABELS[baseline])
        if std and std > 0:
            se = std / np.sqrt(N_SEEDS)
            ax.fill_between(x_range,
                            [max(floor, mean - se)] * 2,
                            [mean + se] * 2,
                            alpha=0.10, color=COLOURS[baseline])

    print(f"\n=== {env_id} ===")
    for method in ("bc", "iq_learn", "csil", "csil_soar"):
        _draw_curve(ax, method_results[method], method, shade_lower, shade_upper)

    ax.set_title(env_id, fontsize=12)
    ax.set_xlabel("Number of Expert Trajectories (K)", fontsize=10)
    ax.set_ylabel("Evaluation Return", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)


def save_figure(envs: list[str], out_path: Path, show_title: bool, dpi: int = 300) -> None:
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


FIGURES = [
    (ENV_IDS, FIGURES_DIR / "comparison_all.png", True),
]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--out",  type=str,  default=None)
    args = parser.parse_args()

    if args.out:
        save_figure(args.envs or ENV_IDS, Path(args.out), show_title=True)
    else:
        for envs, out_path, show_title in FIGURES:
            save_figure(envs, out_path, show_title)


if __name__ == "__main__":
    main()

# ── Usage ──────────────────────────────────────────────────────────────────────
# python scripts/12_plot_comparison.py
# python scripts/12_plot_comparison.py --envs CartPole-v1 --out results/figures/cartpole.png