"""
Coherent Soft Imitation Learning (CSIL)
Watson, Huang & Heess — NeurIPS 2023
https://arxiv.org/abs/2305.16498

Standalone PyTorch implementation for discrete and continuous gym environments.

Algorithm overview
──────────────────
  Phase 1 — Behavioral Cloning (BC):
      Train π_bc on expert dataset D via MLE: max  E_{s,a~D}[log π_bc(a|s)]

  Phase 2 — Coherent Reward derivation:
      Invert the entropy-regularised policy update to obtain a shaped reward
      for which π_bc is optimal (Theorem 1 of the paper):
          r̃(s,a) = α · log( π_bc(a|s) / p(a|s) )
      where p is the policy prior (uniform for discrete, N(0,I) for continuous).

  Phase 3 — SAC Fine-tuning:
      Run SAC with r̃ as the reward signal, warm-starting the actor from π_bc.
      This overcomes the covariate-shift problem of plain BC and avoids the
      instabilities of adversarial IRL.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Optional

import numpy as np
# NumPy 2.0 removed np.bool8 (and other aliases like np.int0, np.uint0).
# The legacy `gym` library still references them internally, so we patch them
# back in before any gym code runs. Safe to leave in even on NumPy < 2.0.
for _alias, _dtype in [
    ("bool8",   np.bool_),
    ("int0",    np.intp),
    ("uint0",   np.uintp),
    ("object0", np.object_),
    ("str0",    np.str_),
    ("bytes0",  np.bytes_),
]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _dtype)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BEHAVIORAL CLONING
# ═══════════════════════════════════════════════════════════════════════════════

class BCPolicyDiscrete(nn.Module):
    """BC policy for environments with a Discrete action space."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.action_dim = action_dim

    def probs(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(state), dim=-1)

    def log_probs(self, state: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(state), dim=-1)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-probability of a specific action."""
        return self.log_probs(state).gather(1, action.view(-1, 1)).squeeze(1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        return Categorical(self.probs(s)).sample().item()


class BCPolicyContinuous(nn.Module):
    """
    BC policy for environments with a Box action space.
    Uses a Gaussian with learned mean and std (squashed to action bounds).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        action_scale: float = 1.0,
        log_std_min: float = -5,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    def _dist(self, state: torch.Tensor) -> Normal:
        h = self.shared(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of a (pre-squash) action.
        NOTE: If your expert actions are stored post-tanh, pass them through
              atanh before calling this method.
        """
        return self._dist(state).log_prob(action).sum(-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self._dist(s)
        action = torch.tanh(dist.sample()) * self.action_scale
        return action.squeeze(0).cpu().numpy()


def train_bc(
    policy: nn.Module,
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    is_discrete: bool,
    n_epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """
    Train a BC policy via maximum likelihood on expert demonstrations.

    Args:
        policy:          BCPolicyDiscrete or BCPolicyContinuous instance
        expert_states:   np.ndarray of shape (N, state_dim)
        expert_actions:  np.ndarray of shape (N,) [discrete] or (N, action_dim) [continuous]
        is_discrete:     True for Discrete action spaces
        n_epochs:        Number of full passes over the expert dataset
        lr:              Adam learning rate
        batch_size:      Mini-batch size
        device:          "cpu" or "cuda"
        verbose:         Print training progress every 20 epochs
    """
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    states = torch.FloatTensor(expert_states).to(device)
    if is_discrete:
        actions = torch.LongTensor(expert_actions.astype(int)).to(device)
    else:
        actions = torch.FloatTensor(expert_actions).to(device)

    dataset = torch.utils.data.TensorDataset(states, actions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for s_batch, a_batch in loader:
            nll = -policy.log_prob(s_batch, a_batch).mean()
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()
            epoch_loss += nll.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if verbose and epoch % 20 == 0:
            print(f"  [BC] Epoch {epoch:4d}/{n_epochs} | NLL: {avg_loss:.4f}")

    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COHERENT REWARD
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def coherent_reward_discrete(
    bc_policy: BCPolicyDiscrete,
    states: torch.Tensor,
    actions: torch.Tensor,
    alpha_csil: float = 1.0,
) -> torch.Tensor:
    """
    Coherent shaped reward for discrete actions:
        r̃(s,a) = α · (log π_bc(a|s) − log(1/|A|))
                = α · (log π_bc(a|s) + log|A|)

    Args:
        bc_policy:   Trained BCPolicyDiscrete
        states:      Tensor (B, state_dim)
        actions:     Tensor (B,) — long dtype
        alpha_csil:  Coherence temperature (α in the paper)

    Returns:
        Tensor (B, 1) — shaped reward for each transition
    """
    log_pi_bc = bc_policy.log_prob(states, actions)
    log_uniform = torch.log(torch.tensor(1.0 / bc_policy.action_dim, device=states.device))
    return (alpha_csil * (log_pi_bc - log_uniform)).unsqueeze(1)


@torch.no_grad()
def coherent_reward_continuous(
    bc_policy: BCPolicyContinuous,
    states: torch.Tensor,
    actions: torch.Tensor,
    alpha_csil: float = 1.0,
) -> torch.Tensor:
    """
    Coherent shaped reward for continuous actions using a N(0,I) prior:
        r̃(s,a) = α · (log π_bc(a|s) − log N(a; 0, I))

    Args:
        bc_policy:   Trained BCPolicyContinuous
        states:      Tensor (B, state_dim)
        actions:     Tensor (B, action_dim)
        alpha_csil:  Coherence temperature (α in the paper)

    Returns:
        Tensor (B, 1) — shaped reward for each transition
    """
    log_pi_bc = bc_policy.log_prob(states, actions)
    prior = Normal(torch.zeros_like(actions), torch.ones_like(actions))
    log_prior = prior.log_prob(actions).sum(-1)
    return (alpha_csil * (log_pi_bc - log_prior)).unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SAC BACKBONE (DISCRETE)
# ═══════════════════════════════════════════════════════════════════════════════

class SACActorDiscrete(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state: torch.Tensor):
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> int:
        probs, _ = self.forward(state)
        return Categorical(probs).sample().item()


class SACCriticDiscrete(nn.Module):
    """Twin Q-network (Clipped Double Q-learning) for discrete SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state: torch.Tensor):
        return self.q1(state), self.q2(state)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition (reward is NOT stored — computed later from BC policy)."""
        self.buffer.append((state, action, next_state, float(done)))

    def sample(self, batch_size: int, is_discrete: bool, device: str):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

        if is_discrete:
            actions = torch.LongTensor(np.array(actions)).to(device)
        else:
            actions = torch.FloatTensor(np.array(actions)).to(device)

        return states, actions, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CSIL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class CSILAgent:
    """
    CSIL agent for discrete action spaces (CartPole, MountainCar, …).

    Combines:
      • A pre-trained BC policy that provides the coherent reward
      • A SAC actor initialised from BC weights (coherent warm-start)
      • Twin Q-critics with automatic entropy tuning
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        bc_policy: BCPolicyDiscrete,
        hidden_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,          # soft target update coefficient
        alpha_csil: float = 1.0,     # coherent reward temperature
        device: str = "cpu",
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha_csil = alpha_csil
        self.bc_policy = bc_policy.to(device).eval()

        # SAC networks
        self.actor = SACActorDiscrete(state_dim, action_dim, hidden_size).to(device)
        self.critic = SACCriticDiscrete(state_dim, action_dim, hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Automatic entropy tuning (Haarnoja et al. 2018)
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        self.log_alpha_sac = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_sac_opt = optim.Adam([self.log_alpha_sac], lr=lr)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self._updates = 0

    # ── Warm-start ──────────────────────────────────────────────────────────

    def initialize_from_bc(self) -> None:
        """
        Copy the BC policy weights into the SAC actor.
        This is the 'coherence' of CSIL — the actor starts from the same
        policy that defines the reward, ensuring immediate reward signal.
        """
        with torch.no_grad():
            for p_a, p_bc in zip(self.actor.net.parameters(), self.bc_policy.net.parameters()):
                p_a.data.copy_(p_bc.data)
        print("[CSIL] SAC actor warm-started from BC policy weights.")

    # ── Action selection ─────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.act(s)

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, next_states, dones = replay_buffer.sample(
            batch_size, is_discrete=True, device=self.device
        )

        # ── Coherent reward (replaces environment reward) ──
        rewards = coherent_reward_discrete(
            self.bc_policy, states, actions, self.alpha_csil
        )  # (B, 1)

        alpha_sac = self.log_alpha_sac.exp().item()

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_states)
            q1_next, q2_next = self.critic_target(next_states)
            min_q_next = torch.min(q1_next, q2_next)
            # Soft value: V(s') = Σ_a π(a|s') [Q(s',a) - α log π(a|s')]
            v_next = (next_probs * (min_q_next - alpha_sac * next_log_probs)).sum(
                -1, keepdim=True
            )
            q_target = rewards + self.gamma * (1.0 - dones) * v_next

        q1_all, q2_all = self.critic(states)
        q1_pred = q1_all.gather(1, actions.unsqueeze(1))
        q2_pred = q2_all.gather(1, actions.unsqueeze(1))
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ───────────────────────────────────────────────────
        probs, log_probs = self.actor(states)
        q1_all, q2_all = self.critic(states)
        min_q = torch.min(q1_all, q2_all)
        # Policy loss: minimise E_π[ α log π - Q ]
        actor_loss = (probs * (alpha_sac * log_probs - min_q)).sum(-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Entropy temperature update ─────────────────────────────────────
        with torch.no_grad():
            probs_det, log_probs_det = self.actor(states)
        entropy = -(probs_det * log_probs_det).sum(-1).mean()
        alpha_loss = self.log_alpha_sac * (entropy - self.target_entropy).detach()

        self.alpha_sac_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_sac_opt.step()

        # ── Polyak update of target critic ─────────────────────────────────
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - self.tau)
                p_t.data.add_(self.tau * p.data)

        self._updates += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_sac": alpha_sac,
            "entropy": entropy.item(),
            "mean_reward": rewards.mean().item(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def load_expert_dataset(path: str):
    """
    Load a pickled expert dataset.

    Expected format: dict with keys 'states' and 'actions'
        or a list of (state, action) tuples.
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        return np.array(data["states"]), np.array(data["actions"])
    elif isinstance(data, (list, tuple)) and isinstance(data[0], (list, tuple)):
        states, actions = zip(*data)
        return np.array(states), np.array(actions)
    else:
        raise ValueError(f"Unrecognised expert dataset format: {type(data)}")

def _str_device(val, default="cpu"):
        """Accept a plain string or a torch.device object."""
        if val is None or (isinstance(val, str) and val.strip().lower() in ("none", "null", "")):
            return default
        return str(val)  # works for both "cpu" and torch.device("cpu")

def run_csil(
    env,
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    csil_dict: dict,
) -> tuple[CSILAgent, list[float], BCPolicyDiscrete]:
    """
    Full CSIL pipeline:
      1. Train BC on expert data
      2. Initialise SAC actor from BC weights
      3. Run SAC with coherent reward

    Args:
        env:                  OpenAI Gym / Gymnasium environment (Discrete action space)
        expert_states:        np.ndarray (N, state_dim) from PPO rollouts
        expert_actions:       np.ndarray (N,) from PPO rollouts
        alpha_csil:           Coherent reward temperature (α in the paper).
                              Higher → stronger shaping toward BC behaviour.
                              Lower  → more exploration freedom.
        start_steps:          Steps of random exploration before SAC updates begin.
        early_stop_reward:    Stop training once the 100-episode average exceeds this.

    Returns:
        (csil_agent, episode_scores, bc_policy)
    """

    def _nullable_float(val, default=None):
        """Return float(val), or None if val is None / 'None' / 'null' / ''."""
        if val is None:
            return default
        if isinstance(val, str) and val.strip().lower() in ("none", "null", ""):
            return default
        return float(val)
    
    # Extract dictionaries

    # BC params
    bc_hidden_size = int(csil_dict.get("bc_hidden_size", 64))
    bc_epochs = int(csil_dict.get("bc_epochs", 200))
    bc_lr = float(csil_dict.get("bc_lr", 3e-4))
    bc_batch_size = int(csil_dict.get("bc_batch_size", 256))
    
    # SAC / CSIL params
    sac_hidden_size = int(csil_dict.get("sac_hidden_size", 64))
    sac_lr = float(csil_dict.get("sac_lr", 3e-4))
    gamma = float(csil_dict.get("gamma", 0.99))
    tau = float(csil_dict.get("tau", 0.005))
    alpha_csil = float(csil_dict.get("alpha_csil", 1.0))
    buffer_size = int(csil_dict.get("buffer_size", 200_000))
    batch_size = int(csil_dict.get("batch_size", 256))
    start_steps = int(csil_dict.get("start_steps", 1000))      # random exploration before SAC kicks in
    n_episodes = int(csil_dict.get("n_episodes", 1000))
    early_stop_reward = _nullable_float(csil_dict.get("early_stop_reward", None))  # e.g. 195 for CartPole

    device = _str_device(csil_dict.get("device", "cpu"))
    verbose = bool(csil_dict.get("verbose", True))

    if verbose:
        print("Parameters loaded:")
        print(f"  BC: hidden_size={bc_hidden_size}, epochs={bc_epochs}, lr={bc_lr}, batch_size={bc_batch_size}")
        print(f"  SAC: hidden_size={sac_hidden_size}, lr={sac_lr}, gamma={gamma}, tau={tau}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ── Phase 1: Behavioural Cloning ─────────────────────────────────────────
    print("=" * 60)
    print("Phase 1 — Behavioural Cloning")
    print("=" * 60)
    bc_policy = BCPolicyDiscrete(state_dim, action_dim, bc_hidden_size).to(device)
    bc_losses = train_bc(
        bc_policy, expert_states, expert_actions,
        is_discrete=True, n_epochs=bc_epochs,
        lr=bc_lr, batch_size=bc_batch_size,
        device=device, verbose=verbose,
    )
    print(f"  BC training done. Final NLL: {bc_losses[-1]:.4f}")

    # ── Phase 2 & 3: CSIL (SAC + Coherent Reward) ────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 — CSIL Fine-tuning with SAC + Coherent Reward")
    print(f"  α_csil = {alpha_csil}")
    print("=" * 60)

    agent = CSILAgent(
        state_dim, action_dim, bc_policy,
        hidden_size=sac_hidden_size, lr=sac_lr,
        gamma=gamma, tau=tau,
        alpha_csil=alpha_csil, device=device,
    )
    agent.initialize_from_bc()

    replay_buffer = ReplayBuffer(buffer_size)
    scores: list[float] = []
    total_steps = 0

    for ep in range(1, n_episodes + 1):
        try:
            state, _ = env.reset()          # gymnasium API
        except TypeError:
            state = env.reset()             # legacy gym API

        ep_score = 0.0
        done = False

        while not done:
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            try:
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, _, done, _ = env.step(action)   # legacy gym

            replay_buffer.push(state, action, next_state, done)
            state = next_state
            ep_score += 1  # count steps (env reward not used — see note below)
            total_steps += 1

            if total_steps >= start_steps:
                agent.update(replay_buffer, batch_size)

        scores.append(ep_score)
        avg100 = np.mean(scores[-100:])

        if verbose and ep % 50 == 0:
            print(
                f"  Ep {ep:5d} | Score: {ep_score:7.1f} | "
                f"Avg(100): {avg100:7.1f} | Steps: {total_steps:7d}"
            )

        if early_stop_reward is not None and avg100 >= early_stop_reward:
            print(f"\n  ✓ Solved at episode {ep} (avg100={avg100:.1f})")
            break

    # NOTE on ep_score: we track episode length (or env score) for monitoring
    # only. The learning signal comes entirely from the coherent reward.
    # For final evaluation use agent.act() deterministically.

    return agent, scores, bc_policy


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_policy(env, agent: CSILAgent, csil_dict) -> float:
    """
    Evaluate the CSIL agent (greedy / mode action) over n_episodes.

    Returns average undiscounted return and standard deviation across episodes.
    """

    n_episodes = int(csil_dict.get("eval_n_episodes", 10))
    device = _str_device(csil_dict.get("device", "cpu"), default="cpu")

    rewards = []
    for _ in range(n_episodes):
        try:
            state, _ = env.reset()
        except TypeError:
            state = env.reset()

        done = False
        ep_r = 0.0
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs, _ = agent.actor(s)
            action = probs.argmax(-1).item()       # greedy action
            try:
                next_state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, r, done, _ = env.step(action)
            ep_r += r
            state = next_state
        rewards.append(ep_r)

    rewards = np.array(rewards)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    
    return mean_reward, std_reward