"""
Coherent Soft Imitation Learning (CSIL)
Watson, Huang & Heess — NeurIPS 2023
https://arxiv.org/abs/2305.16498

Standalone PyTorch implementation for discrete gym environments.

Algorithm overview
──────────────────
  Phase 1 — Behavioral Cloning (BC):
      Train π_bc on expert dataset D via MLE: max  E_{s,a~D}[log π_bc(a|s)]

  Phase 2 — Coherent Reward derivation:
      Invert the entropy-regularised policy update to obtain a shaped reward
      for which π_bc is optimal (Theorem 1 of the paper):
          r̃(s,a) = α · log( π_bc(a|s) / p(a|s) )
      where p is the policy prior (uniform for discrete actions).

  Phase 3 — SAC Fine-tuning:
      Run SAC with r̃ as the reward signal, warm-starting the actor from π_bc.
      This overcomes the covariate-shift problem of plain BC and avoids the
      instabilities of adversarial IRL.
"""

from __future__ import annotations

import copy
import random
from collections import deque

import numpy as np
# NumPy 2.0 removed np.bool8 (and other aliases). The legacy `gym` library
# still references them internally; patch them back before any gym code runs.
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
from torch.distributions import Categorical


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
        return self.log_probs(state).gather(1, action.view(-1, 1)).squeeze(1)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        return Categorical(self.probs(s)).sample().item()


def train_bc(
    policy: nn.Module,
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    n_epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    states = torch.FloatTensor(expert_states).to(device)
    actions = torch.LongTensor(expert_actions.astype(int)).to(device)

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
    r̃(s,a) = α · (log π_bc(a|s) − log(1/|A|))
            = α · (log π_bc(a|s) + log|A|)
    """
    log_pi_bc = bc_policy.log_prob(states, actions)
    log_uniform = torch.log(torch.tensor(1.0 / bc_policy.action_dim, device=states.device))
    return (alpha_csil * (log_pi_bc - log_uniform)).unsqueeze(1)


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

    def push(self, state: np.ndarray, action, next_state: np.ndarray, done: bool) -> None:
        """Reward is NOT stored — computed on-the-fly from the BC policy."""
        self.buffer.append((state, action, next_state, float(done)))

    def sample(self, batch_size: int, device: str):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CSIL AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class CSILAgent:
    """
    CSIL agent for discrete action spaces (CartPole, Acrobot, …).

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
        tau: float = 0.005,
        alpha_csil: float = 1.0,
        device: str = "cpu",
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha_csil = alpha_csil
        self.bc_policy = bc_policy.to(device).eval()

        self.actor = SACActorDiscrete(state_dim, action_dim, hidden_size).to(device)
        self.critic = SACCriticDiscrete(state_dim, action_dim, hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Paper (Algorithm 2, Eq. 13): SAC uses KL against BC policy, not uniform.
        # We target KL(q || q_bc) ≤ target_kl to allow improvement while
        # keeping coherence with the BC policy.
        self.target_kl = 0.3 * np.log(action_dim)
        self.log_alpha_sac = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_sac_opt = optim.Adam([self.log_alpha_sac], lr=lr)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def initialize_from_bc(self) -> None:
        """Copy BC weights into the SAC actor (coherent warm-start)."""
        with torch.no_grad():
            for p_a, p_bc in zip(self.actor.net.parameters(), self.bc_policy.net.parameters()):
                p_a.data.copy_(p_bc.data)
        print("[CSIL] SAC actor warm-started from BC policy weights.")

    def act(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.act(s)

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, next_states, dones = replay_buffer.sample(batch_size, self.device)
        rewards = coherent_reward_discrete(self.bc_policy, states, actions, self.alpha_csil)

        alpha_sac = self.log_alpha_sac.exp().item()

        # Critic update — Eq. 3 with BC policy as prior (paper Section 4 & I)
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_states)
            bc_next_log_probs = self.bc_policy.log_probs(next_states)
            q1_next, q2_next = self.critic_target(next_states)
            min_q_next = torch.min(q1_next, q2_next)
            kl_next = next_log_probs - bc_next_log_probs
            v_next = (next_probs * (min_q_next - alpha_sac * kl_next)).sum(-1, keepdim=True)
            q_target = rewards + self.gamma * (1.0 - dones) * v_next

        q1_all, q2_all = self.critic(states)
        q1_pred = q1_all.gather(1, actions.unsqueeze(1))
        q2_pred = q2_all.gather(1, actions.unsqueeze(1))
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update — Eq. 13: minimise KL(q || q_bc) rather than KL(q || uniform)
        probs, log_probs = self.actor(states)
        with torch.no_grad():
            bc_log_probs = self.bc_policy.log_probs(states)
        q1_all, q2_all = self.critic(states)
        min_q = torch.min(q1_all, q2_all)
        kl_from_bc = log_probs - bc_log_probs
        actor_loss = (probs * (alpha_sac * kl_from_bc - min_q)).sum(-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature update — target KL(q || q_bc), not entropy
        with torch.no_grad():
            probs_det, log_probs_det = self.actor(states)
            bc_log_probs_det = self.bc_policy.log_probs(states)
        kl_det = (probs_det * (log_probs_det - bc_log_probs_det)).sum(-1).mean()
        alpha_loss = self.log_alpha_sac * (self.target_kl - kl_det).detach()

        self.alpha_sac_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_sac_opt.step()

        # Polyak update of target critic
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - self.tau)
                p_t.data.add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_sac": alpha_sac,
            "kl_from_bc": kl_det.item(),
            "mean_reward": rewards.mean().item(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_csil(
    env,
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    csil_dict: dict,
    bc_policy: BCPolicyDiscrete | None = None,
) -> tuple[CSILAgent, list[float], BCPolicyDiscrete]:
    """
    Full CSIL pipeline:
      1. Use provided bc_policy, or train BC on expert data if none given
      2. Initialise SAC actor from BC weights
      3. Run SAC with coherent reward

    Returns (csil_agent, episode_returns, bc_policy).
    """
    bc_hidden_size = int(csil_dict.get("bc_hidden_size", 64))
    bc_epochs      = int(csil_dict.get("bc_epochs", 200))
    bc_lr          = float(csil_dict.get("bc_lr", 3e-4))
    bc_batch_size  = int(csil_dict.get("bc_batch_size", 256))

    sac_hidden_size = int(csil_dict.get("sac_hidden_size", 64))
    sac_lr          = float(csil_dict.get("sac_lr", 3e-4))
    gamma           = float(csil_dict.get("gamma", 0.99))
    tau             = float(csil_dict.get("tau", 0.005))
    alpha_csil      = float(csil_dict.get("alpha_csil", 1.0))
    buffer_size     = int(csil_dict.get("buffer_size", 200_000))
    batch_size      = int(csil_dict.get("batch_size", 256))
    start_steps     = int(csil_dict.get("start_steps", 1000))
    n_episodes      = int(csil_dict.get("n_episodes", 1000))
    device          = str(csil_dict.get("device", "cpu"))
    verbose         = bool(csil_dict.get("verbose", True))

    early_stop_reward = csil_dict.get("early_stop_reward", None)
    if isinstance(early_stop_reward, str) and early_stop_reward.lower() in ("none", "null", ""):
        early_stop_reward = None
    elif early_stop_reward is not None:
        early_stop_reward = float(early_stop_reward)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ── Phase 1: Behavioural Cloning ─────────────────────────────────────────
    print("=" * 60)
    if bc_policy is not None:
        print("Phase 1 — BC policy provided (skipping training)")
        bc_policy = bc_policy.to(device).eval()
    else:
        print("Phase 1 — Behavioural Cloning")
        print("=" * 60)
        bc_policy = BCPolicyDiscrete(state_dim, action_dim, bc_hidden_size).to(device)
        bc_losses = train_bc(
            bc_policy, expert_states, expert_actions,
            n_epochs=bc_epochs, lr=bc_lr, batch_size=bc_batch_size,
            device=device, verbose=verbose,
        )
        print(f"  BC training done. Final NLL: {bc_losses[-1]:.4f}")

    # ── Phase 2 & 3: CSIL (SAC + Coherent Reward) ────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 — CSIL Fine-tuning (SAC + Coherent Reward)")
    print(f"  α_csil={alpha_csil}  episodes={n_episodes}  start_steps={start_steps}")
    print("=" * 60)

    agent = CSILAgent(
        state_dim, action_dim, bc_policy,
        hidden_size=sac_hidden_size, lr=sac_lr,
        gamma=gamma, tau=tau, alpha_csil=alpha_csil, device=device,
    )
    agent.initialize_from_bc()

    replay_buffer = ReplayBuffer(buffer_size)
    scores: list[float] = []
    total_steps = 0

    for ep in range(1, n_episodes + 1):
        try:
            state, _ = env.reset()
        except TypeError:
            state = env.reset()

        ep_return = 0.0
        done = False

        while not done:
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, next_state, done)
            state = next_state
            ep_return += reward
            total_steps += 1

            if total_steps >= start_steps:
                agent.update(replay_buffer, batch_size)

        scores.append(ep_return)
        avg100 = float(np.mean(scores[-100:]))

        if verbose and ep % 50 == 0:
            print(
                f"  Ep {ep:5d} | Return: {ep_return:7.1f} | "
                f"Avg(100): {avg100:7.1f} | Steps: {total_steps:7d}"
            )

        if early_stop_reward is not None and avg100 >= early_stop_reward:
            print(f"\n  Early stop at episode {ep} (avg100={avg100:.1f})")
            break

    return agent, scores, bc_policy


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_csil(
    env,
    agent: CSILAgent,
    n_episodes: int = 20,
    device: str = "cpu",
    eval_seeds: list[int] | None = None,
) -> tuple[float, float]:
    """Evaluate with greedy (argmax) policy. Returns (mean_return, std_return).

    eval_seeds: list of RNG seeds. For each seed, n_episodes episodes are run,
                giving len(eval_seeds) * n_episodes total episodes. This separates
                evaluation randomness from training randomness and allows computing
                a reliable mean/std across a larger episode pool.
    """
    seeds = eval_seeds if eval_seeds is not None else [None]
    returns = []
    for seed in seeds:
        rng = np.random.default_rng(seed) if seed is not None else None
        for _ in range(n_episodes):
            seed_i = int(rng.integers(1 << 31)) if rng is not None else None
            try:
                state, _ = env.reset(seed=seed_i)
            except TypeError:
                state = env.reset()

        done = False
        ep_r = 0.0
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs, _ = agent.actor(s)
            action = probs.argmax(-1).item()
            try:
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                state, r, done, _ = env.step(action)
            ep_r += r
        returns.append(ep_r)

    arr = np.array(returns)
    return float(arr.mean()), float(arr.std())
