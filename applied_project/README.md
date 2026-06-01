# Applied Project 3 – Imitation Learning: IQ-Learn vs CSIL

Comparison of [IQ-Learn (Garg et al., NeurIPS 2021)](https://arxiv.org/abs/2106.12142) and [CSIL (Watson et al., NeurIPS 2023)](https://arxiv.org/abs/2305.16498): average environment return vs. number of expert trajectories K, for CartPole-v1 and Acrobot-v1.

## Result

![Comparison Figure](results/figures/comparison_figure.png)

---

## Repository layout

```
applied_project/
├── scripts/
│   ├── 01_train_expert_PPO.py        # Train PPO expert (stable-baselines3)
│   ├── 02_generate_expert_dataset.py # Roll out expert → .npz transition pool
│   ├── 03_collect_baselines.py       # Expert-PPO and random baselines → JSON
│   ├── 04_train_iq_learn.py          # Convert dataset, call IQ-Learn, copy model
│   ├── 05_evaluate_iq_learn.py       # Evaluate saved IQ-Learn Q-net, write JSON
│   ├── 06_train_csil.py              # Train + evaluate CSIL, write JSON
│   ├── 07_plot_comparison_figure.py  # Aggregate JSONs, produce comparison figure
│   ├── csil_agent.py                 # CSIL implementation module (BC + SAC)
│   └── run_sweep.sh                  # Full pipeline (idempotent orchestrator)
├── data/
│   ├── expert/{env}/expert_K15_seed{s}.npz    # Expert transition pools (shared)
│   └── iq_learn/{env}/*.pkl                   # IQ-Learn PKL format (auto-generated)
├── models/
│   ├── experts/{env}/ppo_expert_seed_{s}.zip  # Trained PPO weights (shared)
│   └── iq_learn/{env}/K{k}_seed{s}.pt         # Trained IQ-Learn Q-nets
├── results/
│   ├── raw/
│   │   ├── baselines/{env}/random.json              # Shared baselines (all algorithms)
│   │   ├── baselines/{env}/expert_seed{s}.json
│   │   ├── iq_learn/{env}/iq_learn_K{k}_seed{s}.json
│   │   └── csil/{env}/csil_K{k}_seed{s}.json
│   └── figures/comparison_figure.png               # Final comparison plot
└── external/IQ-Learn/                         # IQ-Learn repo (submodule)
```

---

## Quick start

```bash
cd applied_project/
PYTHON=/path/to/venv/bin/python bash scripts/run_sweep.sh
```

The sweep is **idempotent**: every step checks whether its output file already exists and skips if so. Re-running after a partial failure resumes from where it stopped.

---

## Pipeline steps

### Step 1 – PPO expert (`01_train_expert_PPO.py`)
Trains a PPO agent with stable-baselines3 for each `(env, seed)` pair.

| Env | Timesteps | Typical return |
|-----|-----------|---------------|
| CartPole-v1 | 100 000 | 500 (max) |
| Acrobot-v1 | 300 000 | ~−85 (optimal ≈ −63) |

**Seeds**: 0, 1, 2 — each produces an independent expert. The seed controls weight initialisation and environment sampling during training.

### Step 2 – Expert dataset (`02_generate_expert_dataset.py`)
Rolls out the expert deterministically for **K = 15 trajectories per seed** (the pool size). Saves flat transition arrays `(states, actions, rewards, next_states, dones)` as `.npz`.

- CartPole episodes are exactly **500 steps** (the `TimeLimit` cap).
- Acrobot episodes end early when the goal is reached; typical length is 70–200 steps.

### Step 3 – Baselines (`03_collect_baselines.py`)
Evaluates the PPO expert (20 episodes × 3 seeds) and a random policy (60 episodes). Results land in `results/raw/baselines/{env}/`. **These files are reused across all K values and both algorithms**; re-running is only needed if expert models change.

### Step 4 – IQ-Learn training (`04_train_iq_learn.py`)
Converts the `.npz` pool to IQ-Learn's PKL format (once per seed), then calls `external/IQ-Learn/iq_learn/train_iq.py` via subprocess. Key hyperparameters:

| Parameter | CartPole | Acrobot | Why |
|-----------|----------|---------|-----|
| `expert.demos` | K | K | trajectories subsampled from the pool |
| `expert.subsample_freq` | **20** | **5** | matches paper — limits transitions per trajectory |
| `env.learn_steps` | 100 000 | 200 000 | gradient update budget |
| `agent.init_temp` | 0.001 | 0.001 | temperature α for soft value V = α log Σ exp(Q/α) |
| `method.loss` | `value_expert` | `value_expert` | uses only expert states → offline-compatible |
| `method.chi` | True | True | χ² divergence regularisation |

**Why subsample_freq matters**: with `subsample_freq=20` and K=1, IQ-Learn sees only **25 transitions** from a CartPole trajectory (500 steps ÷ 20). This sparse-data regime creates the K-variation in the figure; without it every K gives perfect performance.

**Seed usage**: the same seed controls (a) which K trajectories are subsampled, (b) IQ-Learn's Q-network initialisation, and (c) the online rollout environment during training.

After training, the Q-net state-dict is copied from IQ-Learn's Hydra output directory to `models/iq_learn/{env}/K{k}_seed{s}.pt`.

### Step 5 – IQ-Learn evaluation (`05_evaluate_iq_learn.py`)
Loads the saved Q-net, runs 20 episodes with a softmax policy (temperature α = 0.001), and writes `results/raw/iq_learn/{env}/iq_learn_K{k}_seed{s}.json`.

### Step 6 – CSIL training + evaluation (`06_train_csil.py`)

**Algorithm overview** (Watson, Huang & Heess, NeurIPS 2023):

1. **Phase 1 – Behavioural Cloning**: train `π_bc` on the K expert trajectories by MLE (cross-entropy) for 200 epochs.
2. **Phase 2 – Coherent reward derivation**: invert the entropy-regularised Bellman optimality to obtain a shaped reward for which `π_bc` is the optimal policy (Theorem 1):
   ```
   r̃(s,a) = α_csil · (log π_bc(a|s) − log(1/|A|))
           = α_csil · (log π_bc(a|s) + log|A|)
   ```
3. **Phase 3 – SAC fine-tuning with BC prior** (Algorithm 2, Eq. 13): run SAC with `r̃` as the reward signal, with the SAC actor warm-started from `π_bc` weights. Crucially, the SAC regulariser penalises deviation from the **BC policy** (KL divergence from `π_bc`), not from the uniform distribution as in standard max-entropy SAC:
   - **Critic target**: `V(s') = E_{a~q}[Q(s',a) − β · KL(q(·|s') ‖ π_bc(·|s'))]`
   - **Actor loss**: `min_q E[β · KL(q ‖ π_bc) − Q(s,a)]`
   - **Temperature**: automatic tuning targeting `KL(q ‖ π_bc) = 0.3 · log|A|`

This "coherence" property keeps the fine-tuned policy anchored to the BC prior, preventing it from drifting toward random (which would happen if the SAC entropy target were set against the uniform distribution).

**Key hyperparameters:**

| Parameter | CartPole | Acrobot | Note |
|-----------|----------|---------|------|
| `n_episodes` | 1 000 (early-stop at avg ≥ 495) | 3 000 | |
| `alpha_csil` (α) | 0.1 | 0.1 | coherent reward temperature; kept small so r̃ guides but does not dominate Q-learning |
| `start_steps` | 1 000 | 1 000 | random exploration before SAC updates begin |
| BC hidden size | 64 | 64 | two-layer MLP |
| BC epochs | 200 | 200 | MLE pre-training |
| SAC hidden size | 64 | 64 | twin Q-network |
| `target_kl` | 0.3 · log 2 | 0.3 · log 3 | KL target for temperature tuning |
| `gamma` | 0.99 | 0.99 | discount factor |
| `tau` | 0.005 | 0.005 | Polyak update coefficient |

Evaluation uses 20 greedy (argmax) episodes and writes `results/raw/csil/{env}/csil_K{k}_seed{s}.json`.

### Step 7 – Plot (`07_plot_comparison_figure.py`)
Aggregates all JSONs, computes per-K mean ± std across the 3 seeds for each algorithm, and saves `results/figures/comparison_figure.png`. Baselines are shown as dashed horizontal lines (no std band). The lower bound of the IQ-Learn and CSIL std bands is clipped to the environment's minimum possible return.

---

## Environment notes

- **Python**: 3.11, venv at `../rl_venv/`
- **Key packages**: `gymnasium==1.2.3`, `gym==0.17.1` (required by IQ-Learn), `stable-baselines3==1.0`, `torch`, `hydra-core==1.0.6`
- IQ-Learn's `train_iq.py` uses the **old `gym` API** (4-value `step`, 1-value `reset`). All other scripts use the **new `gymnasium` API** (5-value `step`, 2-value `reset`). Both coexist in the same venv.
