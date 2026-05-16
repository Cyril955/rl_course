# Applied Project 3 – Imitation Learning with IQ-Learn

Reproduction of Figure 2 from [IQ-Learn (Garg et al., NeurIPS 2021)](https://arxiv.org/abs/2106.12142): average environment return vs. number of expert trajectories K, for CartPole-v1 and Acrobot-v1.

## Result

![Figure 2](results/figures/figure2.png)

---

## Repository layout

```
applied_project/
├── scripts/
│   ├── 01_train_expert_PPO.py        # Train PPO expert (stable-baselines3)
│   ├── 02_generate_expert_dataset.py # Roll out expert → .npz transition file
│   ├── 03_train_iq_learn.py          # Convert dataset, call IQ-Learn, copy model
│   ├── 04_evaluate_iq_learn.py       # Evaluate saved Q-net, write JSON
│   ├── 05_collect_baselines.py       # Expert-PPO and random baselines → JSON
│   ├── 06_plot_figure2.py            # Read JSONs, produce figure2.png
│   └── run_sweep.sh                  # Full pipeline (idempotent orchestrator)
├── data/
│   ├── expert/{env}/expert_K15_seed{s}.npz    # Expert transition pools (shared)
│   └── iq_learn/{env}/*.pkl                   # IQ-Learn PKL format (auto-generated)
├── models/
│   ├── experts/{env}/ppo_expert_seed_{s}.zip  # Trained PPO weights (shared)
│   └── iq_learn/{env}/K{k}_seed{s}.pt         # Trained IQ-Learn Q-nets
├── results/
│   ├── raw/
│   │   ├── baselines/{env}/random.json         # Shared baselines (all algorithms)
│   │   ├── baselines/{env}/expert_seed{s}.json
│   │   └── iq_learn/{env}/iq_learn_K{k}_seed{s}.json
│   └── figures/figure2.png                    # Final plot
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

**Seeds**: 0, 1, 2 — each produces an independent expert. The seed is passed to SB3 `set_random_seed` and controls weight initialisation and environment sampling during training.

### Step 2 – Expert dataset (`02_generate_expert_dataset.py`)
Rolls out the expert deterministically for **K = 15 trajectories per seed** (the pool size). Saves flat transition arrays `(states, actions, rewards, next_states, dones)` as `.npz`.

- CartPole episodes are exactly **500 steps** (the `TimeLimit` cap).
- Acrobot episodes end early when the goal is reached; typical length is 70–200 steps.

### Step 3 – Baselines (`05_collect_baselines.py`)
Evaluates the PPO expert (20 episodes × 3 seeds) and a random policy (60 episodes). Results are saved to `results/raw/{env}/expert_seed{s}.json` and `random.json`. **These files are reused across all K values**; re-running this step is only needed if expert models change.

### Step 4 – IQ-Learn training (`03_train_iq_learn.py`)
Converts the `.npz` pool to IQ-Learn's PKL format (once per seed), then calls `external/IQ-Learn/iq_learn/train_iq.py` via subprocess. Key hyperparameters passed at call time:

| Parameter | CartPole | Acrobot | Why |
|-----------|----------|---------|-----|
| `expert.demos` | K | K | number of trajectories subsampled from the pool |
| `expert.subsample_freq` | **20** | **5** | matches paper — limits transitions per trajectory |
| `env.learn_steps` | 100 000 | 200 000 | gradient update budget |
| `agent.init_temp` | 0.001 | 0.001 | temperature α for soft value V = α log Σ exp(Q/α) |
| `method.loss` | `value_expert` | `value_expert` | uses only expert states → works offline |
| `method.chi` | True | True | χ² divergence regularisation |

**Why subsample_freq matters**: with `subsample_freq=20` and K=1, IQ-Learn sees only **25 transitions** from CartPole (500 steps ÷ 20). With K=15 it sees 375. This sparse-data regime is what creates the interesting K-variation in the figure; without it every K gives perfect performance.

**Seed usage**: the same seed controls (a) which K trajectories are subsampled from the pool, (b) IQ-Learn's Q-network initialisation, and (c) the online environment used for rollouts during training.

After training, the Q-net state-dict is copied from IQ-Learn's Hydra output directory (`external/IQ-Learn/iq_learn/outputs/{date}/{time}/results/`) to `models/iq_learn/{env}/K{k}_seed{s}.pt` before the next run overwrites it.

### Step 5 – Evaluation (`04_evaluate_iq_learn.py`)
Loads the saved Q-net, runs 20 episodes with a softmax policy (temperature α = 0.001), and writes a JSON with `mean_return` and `std_return`.

### Step 6 – Plot (`06_plot_figure2.py`)
Aggregates all JSONs, computes per-K mean ± std across the 3 seeds, and saves `results/figures/figure2.png`.

---

## Environment notes

- **Python**: 3.11, venv at `../rl_venv/`
- **Key packages**: `gymnasium==1.2.3`, `gym==0.17.1` (required by IQ-Learn), `stable-baselines3==1.0`, `torch`, `hydra-core==1.0.6`
- IQ-Learn's `train_iq.py` uses the **old `gym` API** (4-value `step`, 1-value `reset`). Our scripts use the **new `gymnasium` API** (5-value `step`, 2-value `reset`). Both coexist in the same venv.
