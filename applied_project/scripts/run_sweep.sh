#!/usr/bin/env bash
# Reproduce Figure 2 (offline IL, CartPole + Acrobot) end-to-end.
#
# Run from the applied_project/ directory:
#   PYTHON=/path/to/venv/bin/python bash scripts/run_sweep.sh
#
# The script is idempotent: each step is skipped if its output already exists.

set -euo pipefail
cd "$(dirname "$0")/.."   # ensure CWD = applied_project/
PYTHON="${PYTHON:-$(which python3)}"

# ── Configuration ─────────────────────────────────────────────────────────────
ENVS=("CartPole-v1" "Acrobot-v1")
K_VALUES=(1 3 7 10 15)        # Expert trajectory counts to sweep
SEEDS=(0 1 2 3 4)             # 5 seeds → mean ± std in the plot
EXPERT_POOL_K=15              # Trajectories in the expert pool (must be ≥ max(K_VALUES))
LEARN_STEPS_CARTPOLE=100000
LEARN_STEPS_ACROBOT=200000
EXPERT_TIMESTEPS_CARTPOLE=100000
EXPERT_TIMESTEPS_ACROBOT=300000
N_EVAL_EPISODES=20

# ── Step 1: Train missing PPO expert models ────────────────────────────────────
echo "============================================================"
echo " STEP 1: PPO experts"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    MODEL="models/experts/${ENV}/ppo_expert_seed_${SEED}.zip"
    if [ -f "$MODEL" ]; then
      echo "[SKIP] $MODEL"
      continue
    fi
    STEPS=$EXPERT_TIMESTEPS_CARTPOLE
    if [ "$ENV" = "Acrobot-v1" ]; then STEPS=$EXPERT_TIMESTEPS_ACROBOT; fi
    echo "[RUN ] Training PPO expert: $ENV  seed=$SEED  steps=$STEPS"
    $PYTHON scripts/01_train_expert_PPO.py \
      --env-id "$ENV" --seed "$SEED" --timesteps "$STEPS"
  done
done

# ── Step 2: Generate expert datasets (K=EXPERT_POOL_K per seed) ───────────────
echo ""
echo "============================================================"
echo " STEP 2: Expert datasets (pool K=${EXPERT_POOL_K})"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    DATASET="data/expert/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"
    if [ -f "$DATASET" ]; then
      echo "[SKIP] $DATASET"
      continue
    fi
    MODEL="models/experts/${ENV}/ppo_expert_seed_${SEED}.zip"
    echo "[RUN ] Generating dataset: $ENV  seed=$SEED  K=${EXPERT_POOL_K}"
    $PYTHON scripts/02_generate_expert_dataset.py \
      --env-id "$ENV" \
      --model-path "$MODEL" \
      --n-trajectories "$EXPERT_POOL_K" \
      --save-path "$DATASET"
  done
done

# ── Step 3: Baselines (expert PPO + random) ───────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 3: Baselines"
echo "============================================================"
$PYTHON scripts/03_collect_baselines.py --n-episodes "$N_EVAL_EPISODES" --seeds "${SEEDS[@]}"

# ── Step 4: IQ-Learn sweep (env × K × seed) ───────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 4: IQ-Learn training sweep"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  LEARN_STEPS=$LEARN_STEPS_CARTPOLE
  if [ "$ENV" = "Acrobot-v1" ]; then LEARN_STEPS=$LEARN_STEPS_ACROBOT; fi

  for SEED in "${SEEDS[@]}"; do
    EXPERT_NPZ="data/expert/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"

    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/raw/iq_learn/${ENV}/iq_learn_K${K}_seed${SEED}.json"

      if [ -f "$RESULT_JSON" ]; then
        echo "[SKIP] $RESULT_JSON"
        continue
      fi

      MODEL_OUT="models/iq_learn/${ENV}/K${K}_seed${SEED}.pt"
      echo ""
      echo "[RUN ] IQ-Learn: $ENV  K=$K  seed=$SEED  steps=$LEARN_STEPS"

      # Subsample frequency matches IQ-Learn paper (CartPole=20, Acrobot=5)
      SUBSAMPLE_FREQ=20
      if [ "$ENV" = "Acrobot-v1" ]; then SUBSAMPLE_FREQ=5; fi

      # Train
      $PYTHON scripts/04_train_iq_learn.py \
        --env-id "$ENV" \
        --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" \
        --seed "$SEED" \
        --learn-steps "$LEARN_STEPS" \
        --subsample-freq "$SUBSAMPLE_FREQ" \
        --output-model "$MODEL_OUT"

      # Evaluate
      $PYTHON scripts/05_evaluate_iq_learn.py \
        --env-id "$ENV" \
        --model-path "$MODEL_OUT" \
        --n-episodes "$N_EVAL_EPISODES" \
        --K "$K" \
        --train-seed "$SEED" \
        --save-json "$RESULT_JSON"
    done
  done
done

# ── Step 5: CSIL sweep (env × K × seed) ──────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 5: CSIL training sweep"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  N_EPISODES_CSIL=1000
  if [ "$ENV" = "Acrobot-v1" ]; then N_EPISODES_CSIL=3000; fi

  # Early-stop only for CartPole (Acrobot reward is negative, target unclear)
  CSIL_EXTRA_ARGS=()
  if [ "$ENV" = "CartPole-v1" ]; then
    CSIL_EXTRA_ARGS+=("--early-stop-reward" "495")
  fi

  for SEED in "${SEEDS[@]}"; do
    EXPERT_NPZ="data/expert/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"

    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/raw/csil/${ENV}/csil_K${K}_seed${SEED}.json"

      if [ -f "$RESULT_JSON" ]; then
        echo "[SKIP] $RESULT_JSON"
        continue
      fi

      echo ""
      echo "[RUN ] CSIL: $ENV  K=$K  seed=$SEED  episodes=$N_EPISODES_CSIL"

      $PYTHON scripts/06_train_csil.py \
        --env-id "$ENV" \
        --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" \
        --seed "$SEED" \
        --n-episodes "$N_EPISODES_CSIL" \
        --save-json "$RESULT_JSON" \
        ${CSIL_EXTRA_ARGS[@]+"${CSIL_EXTRA_ARGS[@]}"}
    done
  done
done

# ── Step 6: Plot ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 6: Plotting Comparison Figure"
echo "============================================================"
$PYTHON scripts/07_plot_comparison_figure.py

echo ""
echo "Done!  Figure saved to results/figures/comparison_figure.png"
