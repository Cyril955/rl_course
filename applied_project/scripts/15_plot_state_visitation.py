"""State visitation mechanism plot: BC vs CSIL vs CSIL+SOAR on Acrobot at K=1.

Rolls out each saved policy in the Acrobot environment, records every visited
state, and plots a 3-panel 2D histogram in (theta_1, theta_2) space. Overlay:
the K=1 expert trajectory and the goal-region curve. The plot evidences the
"SOAR explores beyond the expert footprint" claim.

Output -> results/figures/state_visitation_<env>_K<K>.png

Rollouts are cached to NPZ so plot iterations don't re-execute the env.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from csil_agent import BCPolicyDiscrete, CSILAgent
from csil_soar_agent import CSILSOARAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data" / "expert_trajectories"
FIGURES_DIR  = PROJECT_ROOT / "results" / "figures"


@torch.no_grad()
def rollout_states(env, action_fn, n_episodes: int, seed: int) -> np.ndarray:
    """Greedy rollouts. Returns array of all visited states, shape (N, state_dim)."""
    rng = np.random.default_rng(seed)
    all_states = []
    for _ in range(n_episodes):
        state, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False
        while not done:
            all_states.append(state)
            s = torch.FloatTensor(state).unsqueeze(0)
            action = action_fn(s)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.asarray(all_states, dtype=np.float32)


def load_bc_action_fn(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = BCPolicyDiscrete(ckpt["state_dim"], ckpt["action_dim"], ckpt["hidden_size"])
    policy.load_state_dict(ckpt["policy"])
    policy = policy.to(device).eval()
    def act(s):
        return policy.probs(s).argmax(-1).item()
    return act


def load_csil_action_fn(ckpt_path: Path, device: str):
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    env_id = ckpt["env_id"]
    env    = gym.make(env_id)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    env.close()
    bc_policy = BCPolicyDiscrete(state_dim, action_dim, config["bc_hidden_size"])
    bc_policy.load_state_dict(ckpt["bc_policy"])
    agent = CSILAgent(state_dim, action_dim, bc_policy,
                      hidden_size=config["sac_hidden_size"], device=device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()
    def act(s):
        probs, _ = agent.actor(s.to(device))
        return probs.argmax(-1).item()
    return act


def load_csil_soar_action_fn(ckpt_path: Path, device: str):
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    config     = ckpt["config"]
    env_id     = ckpt["env_id"]
    n_critics  = ckpt.get("n_critics", config.get("n_critics", 4))
    sigma_clip = ckpt.get("sigma_clip", config.get("sigma_clip", 1.0))
    env        = gym.make(env_id)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    env.close()
    bc_policy = BCPolicyDiscrete(state_dim, action_dim, config["bc_hidden_size"])
    bc_policy.load_state_dict(ckpt["bc_policy"])
    agent = CSILSOARAgent(state_dim, action_dim, bc_policy,
                          n_critics=n_critics, sigma_clip=sigma_clip,
                          hidden_size=config["sac_hidden_size"], device=device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor.eval()
    def act(s):
        probs, _ = agent.actor(s.to(device))
        return probs.argmax(-1).item()
    return act


METHOD_LOADERS = {
    "bc":        load_bc_action_fn,
    "csil":      load_csil_action_fn,
    "csil_soar": load_csil_soar_action_fn,
}
METHOD_LABELS = {"bc": "BC", "csil": "CSIL", "csil_soar": "CSIL+SOAR"}


def collect_method_states(method: str, env_id: str, K: int, seeds: list[int],
                          n_episodes: int, device: str) -> np.ndarray:
    """Pool states across all seeds for one method."""
    loader = METHOD_LOADERS[method]
    pooled = []
    for seed in seeds:
        ckpt_path = MODELS_DIR / method / env_id / f"K{K}_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        action_fn = loader(ckpt_path, device)
        env = gym.make(env_id)
        states = rollout_states(env, action_fn, n_episodes, seed=seed)
        env.close()
        pooled.append(states)
        print(f"  {method:10s} seed={seed}: {len(states):>6d} states")
    return np.concatenate(pooled, axis=0)


def acrobot_state_to_angles(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Acrobot state = [cos t1, sin t1, cos t2, sin t2, dt1, dt2]. Extract (t1, t2)."""
    t1 = np.arctan2(states[:, 1], states[:, 0])
    t2 = np.arctan2(states[:, 3], states[:, 2])
    return t1, t2


def load_expert_K1_angles(env_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the first expert trajectory only (what K=1 methods were trained on)."""
    npz_path = DATA_DIR / env_id / "expert_K15_seed0.npz"
    data = np.load(npz_path)
    n0 = int(data["episode_lengths"][0])
    return acrobot_state_to_angles(data["states"][:n0])


def goal_region_curve():
    """Acrobot goal: -cos(t1) - cos(t1+t2) > 1. Return contour as image grid."""
    t1 = np.linspace(-np.pi, np.pi, 400)
    t2 = np.linspace(-np.pi, np.pi, 400)
    T1, T2 = np.meshgrid(t1, t2, indexing="xy")
    Z = -np.cos(T1) - np.cos(T1 + T2)
    return t1, t2, Z


def plot_visitation(method_states: dict[str, np.ndarray], expert_angles: tuple,
                    out_path: Path, env_id: str, K: int) -> None:
    methods = ["bc", "csil", "csil_soar"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)

    t1_grid, t2_grid, Z = goal_region_curve()

    # Shared color scale across panels for fair visual comparison
    vmax = max(
        np.histogram2d(*acrobot_state_to_angles(method_states[m]),
                       bins=60, range=[[-np.pi, np.pi], [-np.pi, np.pi]])[0].max()
        for m in methods
    )

    expert_t1, expert_t2 = expert_angles

    for ax, method in zip(axes, methods):
        t1, t2 = acrobot_state_to_angles(method_states[method])
        h = ax.hist2d(
            t1, t2,
            bins=60,
            range=[[-np.pi, np.pi], [-np.pi, np.pi]],
            norm=LogNorm(vmin=1, vmax=vmax),
            cmap="Blues",
        )
        ax.contour(t1_grid, t2_grid, Z, levels=[1.0], colors="red",
                   linewidths=2.0, linestyles="--")
        ax.scatter(expert_t1, expert_t2,
                   s=26, facecolors="none", edgecolors="#4CAF50",
                   linewidths=0.8, alpha=0.9, zorder=5)
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$\theta_1$", fontsize=10)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal")

    axes[0].set_ylabel(r"$\theta_2$", fontsize=10)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="#4CAF50",
               markeredgewidth=1.0, markersize=7, label="Expert demos (K=1)"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2,
               label="Goal region boundary"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    cbar = fig.colorbar(h[3], ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Policy state-visit count (log scale)", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",         type=str, default="Acrobot-v1")
    parser.add_argument("--K",           type=int, default=1)
    parser.add_argument("--seeds",       type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-episodes",  type=int, default=20)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--cache",       type=str, default=None,
                        help="NPZ cache path. Defaults to results/figures/_cache_visitation_<env>_K<K>.npz")
    parser.add_argument("--force-rollout", action="store_true",
                        help="Ignore cache and re-roll out.")
    args = parser.parse_args()

    if args.env != "Acrobot-v1":
        print(f"[warn] axes/goal-curve are hard-coded for Acrobot-v1; you passed {args.env}")

    cache_path = Path(args.cache) if args.cache else (
        FIGURES_DIR / f"_cache_visitation_{args.env}_K{args.K}.npz"
    )

    if cache_path.exists() and not args.force_rollout:
        print(f"Loading cached rollouts -> {cache_path}")
        cached = np.load(cache_path)
        method_states = {m: cached[m] for m in ["bc", "csil", "csil_soar"]}
    else:
        print(f"Rolling out policies (env={args.env}, K={args.K}, "
              f"seeds={args.seeds}, n_eps={args.n_episodes})...")
        method_states = {}
        for method in ["bc", "csil", "csil_soar"]:
            print(f"[{METHOD_LABELS[method]}]")
            method_states[method] = collect_method_states(
                method, args.env, args.K, args.seeds, args.n_episodes, args.device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **method_states)
        print(f"Cached rollouts -> {cache_path}")

    expert_angles = load_expert_K1_angles(args.env)

    out_path = FIGURES_DIR / f"state_visitation_{args.env}_K{args.K}.png"
    plot_visitation(method_states, expert_angles, out_path, args.env, args.K)


if __name__ == "__main__":
    main()
