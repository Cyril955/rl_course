"""State visitation mechanism plot: PPO Expert, IQ-Learn, BC, CSIL, CSIL+SOAR
on Acrobot at K=1.

Rolls out each saved policy in the Acrobot environment, records every visited
state, and plots a 5-panel 2D histogram in (theta_1, theta_2) space with a
shared color scale. Overlay: the goal-region curve. The plot compares the
state-space footprint of each algorithm against the PPO expert reference.

Output -> results/figures/state_visitation_<env>_K<K>_5panel.png

Rollouts are cached to NPZ so plot iterations don't re-execute the env.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent))
from csil_agent import BCPolicyDiscrete, CSILAgent
from csil_soar_agent import CSILSOARAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "results" / "figures"


class IQLearnQNetwork(nn.Module):
    """Mirrors OfflineQNetwork from 08_evaluate_iq_learn.py (64-ELU-64-ELU -> action_dim)."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.elu(self.fc2(self.elu(self.fc1(x)))))


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


def load_iq_learn_action_fn(ckpt_path: Path, device: str):
    env_id = ckpt_path.parent.name  # e.g. "Acrobot-v1" or "CartPole-v1"
    env = gym.make(env_id)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    env.close()
    q_net = IQLearnQNetwork(obs_dim, action_dim)
    q_net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    q_net.to(device).eval()
    def act(s):
        q = q_net(s.to(device))
        return q.argmax(-1).item()  # alpha=0.001 softmax -> argmax in practice
    return act


def load_expert_ppo_action_fn(ckpt_path: Path, device: str):
    """PPO expert: stable-baselines3 model. Greedy (deterministic=True).

    The expert was saved when SB3 used the legacy `gym` package; passing
    gymnasium spaces via `custom_objects` skips cloudpickle deserialization
    of the gym-pickled fields. The env id is inferred from the path so that
    the substituted spaces match the actual checkpoint dimensions.
    """
    env_id = ckpt_path.parent.name  # e.g. "Acrobot-v1" or "CartPole-v1"
    env = gym.make(env_id)
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space":      env.action_space,
        "learning_rate":     0.0,
        "lr_schedule":       lambda _: 0.0,
        "clip_range":        lambda _: 0.0,
    }
    env.close()
    model = PPO.load(str(ckpt_path), device=device, custom_objects=custom_objects)
    def act(s):
        action, _ = model.predict(s.cpu().numpy().squeeze(0), deterministic=True)
        return int(action)
    return act


METHOD_LOADERS = {
    "expert":    load_expert_ppo_action_fn,
    "iq_learn":  load_iq_learn_action_fn,
    "bc":        load_bc_action_fn,
    "csil":      load_csil_action_fn,
    "csil_soar": load_csil_soar_action_fn,
}
METHOD_LABELS = {
    "expert":    "PPO Expert",
    "iq_learn":  r"IQ-Learn ($\chi^2$)",
    "bc":        "BC",
    "csil":      "CSIL",
    "csil_soar": "CSIL+SOAR",
}
METHOD_ORDER = ["expert", "iq_learn", "bc", "csil", "csil_soar"]


def ckpt_path_for(method: str, env_id: str, K: int, seed: int) -> Path:
    """PPO expert is K-independent; other methods follow the K<K>_seed<seed>.pt scheme."""
    if method == "expert":
        return MODELS_DIR / "experts" / env_id / f"ppo_expert_seed_{seed}.zip"
    return MODELS_DIR / method / env_id / f"K{K}_seed{seed}.pt"


def collect_method_states(method: str, env_id: str, K: int, seeds: list[int],
                          n_episodes: int, device: str) -> np.ndarray:
    """Pool states across all seeds for one method."""
    loader = METHOD_LOADERS[method]
    pooled = []
    for seed in seeds:
        ckpt_path = ckpt_path_for(method, env_id, K, seed)
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


def cartpole_state_to_xtheta(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CartPole state = [x, x_dot, theta, theta_dot]. Extract (x, theta)."""
    return states[:, 0], states[:, 2]


def acrobot_goal_overlay(ax) -> None:
    """Acrobot success condition: -cos(t1) - cos(t1+t2) > 1. Draw as dashed contour."""
    t1 = np.linspace(-np.pi, np.pi, 400)
    t2 = np.linspace(-np.pi, np.pi, 400)
    T1, T2 = np.meshgrid(t1, t2, indexing="xy")
    Z = -np.cos(T1) - np.cos(T1 + T2)
    ax.contour(t1, t2, Z, levels=[1.0], colors="red",
               linewidths=2.0, linestyles="--")


CARTPOLE_THETA_MAX = float(np.deg2rad(12.0))
CARTPOLE_X_MAX = 2.4


def cartpole_termination_overlay(ax) -> None:
    """CartPole "safe" box: |x| < 2.4 and |theta| < 12 deg. Crossing terminates the episode."""
    rect = Rectangle(
        (-CARTPOLE_X_MAX, -CARTPOLE_THETA_MAX),
        2 * CARTPOLE_X_MAX, 2 * CARTPOLE_THETA_MAX,
        fill=False, edgecolor="red", linewidth=2.0, linestyle="--",
    )
    ax.add_patch(rect)


ENV_SPECS = {
    "Acrobot-v1": {
        "project":       acrobot_state_to_angles,
        "x_label":       r"$\theta_1$",
        "y_label":       r"$\theta_2$",
        "x_range":       (-np.pi, np.pi),
        "y_range":       (-np.pi, np.pi),
        "aspect":        "equal",
        "overlay_fn":    acrobot_goal_overlay,
        "overlay_label": "Goal region boundary",
    },
    "CartPole-v1": {
        "project":       cartpole_state_to_xtheta,
        "x_label":       r"$x$ (m)",
        "y_label":       r"$\theta$ (rad)",
        # Extended ~25% beyond the termination thresholds so the safe box
        # is visible *inside* the plot rather than flush with the axes.
        "x_range":       (-1.25 * CARTPOLE_X_MAX,    1.25 * CARTPOLE_X_MAX),
        "y_range":       (-1.25 * CARTPOLE_THETA_MAX, 1.25 * CARTPOLE_THETA_MAX),
        "aspect":        "auto",
        "overlay_fn":    cartpole_termination_overlay,
        "overlay_label": "Termination boundary",
    },
}


def plot_visitation(method_states: dict[str, np.ndarray],
                    out_path: Path, env_id: str, K: int) -> None:
    spec = ENV_SPECS[env_id]
    project = spec["project"]
    x_range = list(spec["x_range"])
    y_range = list(spec["y_range"])

    methods = METHOD_ORDER
    fig, axes = plt.subplots(1, len(methods), figsize=(18, 4.2),
                             sharex=True, sharey=True)

    # Shared color scale across panels for fair visual comparison
    vmax = max(
        np.histogram2d(*project(method_states[m]),
                       bins=60, range=[x_range, y_range])[0].max()
        for m in methods
    )

    for ax, method in zip(axes, methods):
        x, y = project(method_states[method])
        h = ax.hist2d(
            x, y,
            bins=60,
            range=[x_range, y_range],
            norm=LogNorm(vmin=1, vmax=vmax),
            cmap="Blues",
        )
        spec["overlay_fn"](ax)
        ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight="bold")
        ax.set_xlabel(spec["x_label"], fontsize=10)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_aspect(spec["aspect"])
        ax.set_box_aspect(1)  # force square axes box regardless of data scales

    axes[0].set_ylabel(spec["y_label"], fontsize=10)

    swatch_color = cm.get_cmap("Blues")(0.6)  # mid-blue, ~10^2 on the log scale
    legend_handles = [
        Patch(facecolor=swatch_color, edgecolor="gray", linewidth=0.5,
              label="Visited state count"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2,
               label=spec["overlay_label"]),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.2, handleheight=1.2)

    cbar = fig.colorbar(h[3], ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Visited state count (log scale)", fontsize=10)

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

    if args.env not in ENV_SPECS:
        raise ValueError(
            f"Unsupported env: {args.env}. Supported: {list(ENV_SPECS.keys())}"
        )

    cache_path = Path(args.cache) if args.cache else (
        FIGURES_DIR / f"_cache_visitation_{args.env}_K{args.K}_5panel.npz"
    )

    cache_keys_present = (
        cache_path.exists()
        and not args.force_rollout
        and all(k in np.load(cache_path).files for k in METHOD_ORDER)
    )

    if cache_keys_present:
        print(f"Loading cached rollouts -> {cache_path}")
        cached = np.load(cache_path)
        method_states = {m: cached[m] for m in METHOD_ORDER}
    else:
        print(f"Rolling out policies (env={args.env}, K={args.K}, "
              f"seeds={args.seeds}, n_eps={args.n_episodes})...")
        method_states = {}
        for method in METHOD_ORDER:
            print(f"[{METHOD_LABELS[method]}]")
            method_states[method] = collect_method_states(
                method, args.env, args.K, args.seeds, args.n_episodes, args.device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **method_states)
        print(f"Cached rollouts -> {cache_path}")

    out_path = FIGURES_DIR / f"state_visitation_{args.env}_K{args.K}_5panel.png"
    plot_visitation(method_states, out_path, args.env, args.K)


if __name__ == "__main__":
    main()
