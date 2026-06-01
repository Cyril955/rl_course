"""
CSIL + SOAR: Coherent Soft Imitation Learning with Soft Optimistic Actor cRitic
Viel, Viano & Cevher — IL-SOAR (arXiv:2502.19859, 2025)

SOAR wraps CSIL by replacing the standard twin-critic with an ensemble of L critics.
The policy update uses an optimistic Q estimate (Algorithm 5 — OptimisticQ-NN):

    Q_opt(s, a) = mean_ℓ Q_ℓ(s, a) + clip(std_ℓ Q_ℓ(s, a), 0, σ)

This is the "Mean-Std" aggregation rule of Algorithm 4, converted from the paper's
cost-form (mean − clip(std)) to the reward-form used here. Adding the clipped std
overestimates reward (= underestimates cost) for state-action pairs the ensemble
disagrees about, which drives optimistic exploration toward uncertain regions.

All BC and coherent-reward components are identical to vanilla CSIL.
"""

from __future__ import annotations

import copy
import random
from collections import deque

import numpy as np

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
# 1. BEHAVIORAL CLONING  (unchanged from CSIL)
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

    states  = torch.FloatTensor(expert_states).to(device)
    actions = torch.LongTensor(expert_actions.astype(int)).to(device)

    dataset = torch.utils.data.TensorDataset(states, actions)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
# 2. COHERENT REWARD  (unchanged from CSIL)
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
    log_pi_bc   = bc_policy.log_prob(states, actions)
    log_uniform = torch.log(torch.tensor(1.0 / bc_policy.action_dim, device=states.device))
    return (alpha_csil * (log_pi_bc - log_uniform)).unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SAC ACTOR  (unchanged from CSIL)
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
        logits   = self.net(state)
        probs    = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> int:
        probs, _ = self.forward(state)
        return Categorical(probs).sample().item()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SOAR CRITIC ENSEMBLE  (new — replaces the single twin-critic)
# ═══════════════════════════════════════════════════════════════════════════════

class SOARCriticEnsemble(nn.Module):
    """
    Ensemble of L independent Q-networks for discrete actions.

    Each Q_ℓ : S → R^|A| is a separate network trained independently.
    The optimistic estimate follows Algorithm 5 (OptimisticQ-NN) of the paper:
        Q_opt(s, a) = mean_ℓ Q_ℓ(s, a) − clip(std_ℓ Q_ℓ(s, a), 0, σ)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_critics: int = 4,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.n_critics  = n_critics
        self.action_dim = action_dim
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )
            for _ in range(n_critics)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns shape (n_critics, batch, action_dim)."""
        return torch.stack([c(state) for c in self.critics], dim=0)

    def optimistic_q(self, state: torch.Tensor, sigma_clip: float) -> torch.Tensor:
        """
        OptimisticQ-NN (Algorithm 5) in reward form: Q_opt = mean + clip(std, 0, σ).
        Returns shape (batch, action_dim).

        The SOAR paper writes the rule in cost form (mean − clip(std)); converting
        to reward form (Q = expected discounted return, higher is better) flips the
        sign so the bonus is *added*. Inflating the Q-estimate where the ensemble
        disagrees makes those actions look more attractive to the policy, driving
        exploration toward uncertain state-action pairs.
        """
        q_all  = self.forward(state)            # (L, B, A)
        q_mean = q_all.mean(dim=0)              # (B, A)
        # unbiased=False matches the paper's 1/L formula
        q_std  = q_all.std(dim=0, unbiased=False)
        q_std_clipped = q_std.clamp(0.0, sigma_clip)
        return q_mean + q_std_clipped           # (B, A)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. REPLAY BUFFER  (unchanged from CSIL)
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action, next_state: np.ndarray, done: bool) -> None:
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
# 6. CSIL-SOAR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class CSILSOARAgent:
    """
    CSIL + SOAR agent for discrete action spaces.

    Combines:
      • Pre-trained BC policy → coherent reward r̃(s,a)
      • SAC actor warm-started from BC (coherent warm-start)
      • Ensemble of L critics with optimistic Q estimate (SOAR)

    The only structural difference vs. vanilla CSIL is the critic:
      - CSIL  : one twin-critic, uses min(Q1, Q2) in actor/target
      - SOAR  : L independent critics, uses mean − clip(std, 0, σ)
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
        n_critics: int = 4,
        sigma_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.device     = device
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.alpha_csil = alpha_csil
        self.sigma_clip = sigma_clip
        self.bc_policy  = bc_policy.to(device).eval()

        self.actor = SACActorDiscrete(state_dim, action_dim, hidden_size).to(device)

        self.ensemble        = SOARCriticEnsemble(state_dim, action_dim, n_critics, hidden_size).to(device)
        self.ensemble_target = copy.deepcopy(self.ensemble).to(device)
        for p in self.ensemble_target.parameters():
            p.requires_grad = False

        self.target_kl    = 0.3 * np.log(action_dim)
        self.log_alpha_sac = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_sac_opt = optim.Adam([self.log_alpha_sac], lr=lr)

        self.actor_opt    = optim.Adam(self.actor.parameters(), lr=lr)
        self.ensemble_opt = optim.Adam(self.ensemble.parameters(), lr=lr)

    def initialize_from_bc(self) -> None:
        """Copy BC weights into the SAC actor (coherent warm-start)."""
        with torch.no_grad():
            for p_a, p_bc in zip(self.actor.net.parameters(), self.bc_policy.net.parameters()):
                p_a.data.copy_(p_bc.data)
        print("[CSIL-SOAR] SAC actor warm-started from BC policy weights.")

    def act(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.act(s)

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, next_states, dones = replay_buffer.sample(batch_size, self.device)
        rewards = coherent_reward_discrete(self.bc_policy, states, actions, self.alpha_csil)

        alpha_sac = self.log_alpha_sac.exp().item()

        # ── Critic update ────────────────────────────────────────────────────
        # Bootstrap target uses the optimistic Q from the TARGET ensemble.
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_states)
            bc_next_log_probs = self.bc_policy.log_probs(next_states)
            # Optimistic Q over next states (target network)
            q_opt_next = self.ensemble_target.optimistic_q(next_states, self.sigma_clip)
            kl_next    = next_log_probs - bc_next_log_probs
            v_next     = (next_probs * (q_opt_next - alpha_sac * kl_next)).sum(-1, keepdim=True)
            q_target   = rewards + self.gamma * (1.0 - dones) * v_next

        # Each critic in the ensemble is trained on the same batch.
        # Independence comes from different initializations + SGD trajectory.
        q_all = self.ensemble(states)  # (L, B, A)
        q_preds = q_all[:, torch.arange(batch_size), actions]  # (L, B)
        critic_loss = sum(
            F.mse_loss(q_preds[ell].unsqueeze(1), q_target)
            for ell in range(self.ensemble.n_critics)
        )

        self.ensemble_opt.zero_grad()
        critic_loss.backward()
        self.ensemble_opt.step()

        # ── Actor update ─────────────────────────────────────────────────────
        # Uses optimistic Q from the ONLINE ensemble (not target).
        probs, log_probs = self.actor(states)
        with torch.no_grad():
            bc_log_probs = self.bc_policy.log_probs(states)
        q_opt = self.ensemble.optimistic_q(states, self.sigma_clip).detach()
        kl_from_bc = log_probs - bc_log_probs
        actor_loss  = (probs * (alpha_sac * kl_from_bc - q_opt)).sum(-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Temperature update (target KL vs. BC, same as CSIL) ──────────────
        # Reuse probs/log_probs from the actor step (detached) — bc_log_probs
        # was already computed under no_grad above.
        kl_det = (probs.detach() * (log_probs.detach() - bc_log_probs)).sum(-1).mean()
        alpha_loss = self.log_alpha_sac * (self.target_kl - kl_det).detach()

        self.alpha_sac_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_sac_opt.step()

        # ── Polyak update of target ensemble ─────────────────────────────────
        with torch.no_grad():
            for p, p_t in zip(self.ensemble.parameters(), self.ensemble_target.parameters()):
                p_t.data.mul_(1 - self.tau)
                p_t.data.add_(self.tau * p.data)

        return {
            "critic_loss":  critic_loss.item(),
            "actor_loss":   actor_loss.item(),
            "alpha_sac":    alpha_sac,
            "kl_from_bc":   kl_det.item(),
            "mean_reward":  rewards.mean().item(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_csil_soar(
    env,
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    csil_soar_dict: dict,
    bc_policy: BCPolicyDiscrete | None = None,
) -> tuple[CSILSOARAgent, list[float], BCPolicyDiscrete]:
    """
    Full CSIL-SOAR pipeline:
      1. Use provided bc_policy, or train BC on expert data if none given
      2. Initialise SAC actor from BC weights
      3. Run SAC with coherent reward + optimistic ensemble critic (SOAR)

    Returns (agent, episode_returns, bc_policy).
    """
    bc_hidden_size = int(csil_soar_dict.get("bc_hidden_size", 64))
    bc_epochs      = int(csil_soar_dict.get("bc_epochs", 200))
    bc_lr          = float(csil_soar_dict.get("bc_lr", 3e-4))
    bc_batch_size  = int(csil_soar_dict.get("bc_batch_size", 256))

    sac_hidden_size = int(csil_soar_dict.get("sac_hidden_size", 64))
    sac_lr          = float(csil_soar_dict.get("sac_lr", 3e-4))
    gamma           = float(csil_soar_dict.get("gamma", 0.99))
    tau             = float(csil_soar_dict.get("tau", 0.005))
    alpha_csil      = float(csil_soar_dict.get("alpha_csil", 1.0))
    n_critics       = int(csil_soar_dict.get("n_critics", 4))
    sigma_clip      = float(csil_soar_dict.get("sigma_clip", 1.0))
    buffer_size     = int(csil_soar_dict.get("buffer_size", 200_000))
    batch_size      = int(csil_soar_dict.get("batch_size", 256))
    start_steps     = int(csil_soar_dict.get("start_steps", 1000))
    n_episodes      = int(csil_soar_dict.get("n_episodes", 1000))
    device          = str(csil_soar_dict.get("device", "cpu"))
    verbose         = bool(csil_soar_dict.get("verbose", True))

    early_stop_reward = csil_soar_dict.get("early_stop_reward", None)
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

    # ── Phase 2 & 3: CSIL-SOAR (SAC + Coherent Reward + Optimistic Ensemble) ─
    print("\n" + "=" * 60)
    print("Phase 2 — CSIL-SOAR Fine-tuning")
    print(f"  α_csil={alpha_csil}  L={n_critics}  σ={sigma_clip}  "
          f"episodes={n_episodes}  start_steps={start_steps}")
    print("=" * 60)

    agent = CSILSOARAgent(
        state_dim, action_dim, bc_policy,
        hidden_size=sac_hidden_size, lr=sac_lr,
        gamma=gamma, tau=tau,
        alpha_csil=alpha_csil,
        n_critics=n_critics,
        sigma_clip=sigma_clip,
        device=device,
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
                terminated = done  # legacy gym API: no truncation signal available

            # Store `terminated` (true env termination), not `done` — bootstrapping
            # must continue past time-limit truncation, otherwise long episodes
            # (e.g. CartPole hitting the 500-step cap) are systematically undervalued.
            replay_buffer.push(state, action, next_state, terminated)
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
# 8. EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_csil_soar(
    env,
    agent: CSILSOARAgent,
    n_episodes: int = 20,
    device: str = "cpu",
    eval_seeds: list[int] | None = None,
) -> tuple[float, float]:
    """Greedy evaluation (argmax policy). Returns (mean_return, std_return).

    eval_seeds: list of RNG seeds. For each seed, n_episodes episodes are run,
                giving len(eval_seeds) * n_episodes total episodes.
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

        done  = False
        ep_r  = 0.0
        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs, _ = agent.actor(s)
            action   = probs.argmax(-1).item()
            try:
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                state, r, done, _ = env.step(action)
            ep_r += r
        returns.append(ep_r)

    arr = np.array(returns)
    return float(arr.mean()), float(arr.std())
