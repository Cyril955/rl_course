#!/usr/bin/env bash
# Full end-to-end sweep: train all methods, evaluate all, plot.
# All hyperparameters come from scripts/config.py.
#
# Run from applied_project/:
#   PYTHON=/path/to/python bash scripts/run_sweep.sh
#
# Idempotent: training steps skip if the model file exists;
#             evaluation steps skip if the result JSON exists.

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(which python3)}"

eval "$($PYTHON scripts/config.py --shell)"
env_var() { local _v="${2}_$(env_prefix "$1")"; echo "${!_v}"; }

# ── Step 1: Train PPO experts ──────────────────────────────────────────────────
echo "============================================================"
echo " STEP 1: PPO experts"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    MODEL="models/experts/${ENV}/ppo_expert_seed_${SEED}.zip"
    if [ -f "$MODEL" ]; then echo "[SKIP] $MODEL"; continue; fi
    STEPS=$(env_var "$ENV" EXPERT_TIMESTEPS)
    echo "[RUN ] PPO: $ENV  seed=$SEED  steps=$STEPS"
    $PYTHON scripts/01_train_expert_PPO.py --env-id "$ENV" --seed "$SEED" --timesteps "$STEPS"
  done
done

# ── Step 2: Generate expert datasets ──────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Expert datasets (pool K=${EXPERT_POOL_K})"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    DATASET="data/expert_trajectories/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"
    if [ -f "$DATASET" ]; then echo "[SKIP] $DATASET"; continue; fi
    echo "[RUN ] Dataset: $ENV  seed=$SEED"
    $PYTHON scripts/02_generate_expert_dataset.py \
      --env-id "$ENV" \
      --model-path "models/experts/${ENV}/ppo_expert_seed_${SEED}.zip" \
      --n-trajectories "$EXPERT_POOL_K" \
      --save-path "$DATASET"
  done
done

# ── Step 3: Baselines ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 3: Baselines"
echo "============================================================"
$PYTHON scripts/03_collect_baselines.py --n-episodes "$N_EVAL_EPISODES" --seeds "${SEEDS[@]}"

# ── Step 4: Train IQ-Learn ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 4: Train IQ-Learn"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  LEARN_STEPS=$(env_var "$ENV" LEARN_STEPS)
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)
  for SEED in "${SEEDS[@]}"; do
    EXPERT_NPZ="data/expert_trajectories/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"
    for K in "${K_VALUES[@]}"; do
      MODEL_OUT="models/iq_learn/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$MODEL_OUT" ]; then echo "[SKIP] $MODEL_OUT"; continue; fi
      echo "[RUN ] IQ-Learn: $ENV  K=$K  seed=$SEED  steps=$LEARN_STEPS  subsample_freq=$SUBSAMPLE_FREQ"
      $PYTHON scripts/04_train_iq_learn.py \
        --env-id "$ENV" --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" --seed "$SEED" \
        --learn-steps "$LEARN_STEPS" --subsample-freq "$SUBSAMPLE_FREQ" \
        --output-model "$MODEL_OUT"
    done
  done
done

# ── Step 5: Train BC ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 5: Train BC"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)
  for SEED in "${SEEDS[@]}"; do
    EXPERT_NPZ="data/expert_trajectories/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"
    for K in "${K_VALUES[@]}"; do
      MODEL_OUT="models/bc/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$MODEL_OUT" ]; then echo "[SKIP] $MODEL_OUT"; continue; fi
      echo "[RUN ] BC: $ENV  K=$K  seed=$SEED  subsample_freq=$SUBSAMPLE_FREQ"
      $PYTHON scripts/05_train_bc.py \
        --env-id "$ENV" --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" --seed "$SEED" \
        --subsample-freq "$SUBSAMPLE_FREQ" --device "$DEVICE" \
        --save-model "$MODEL_OUT" \
        --save-training "results/training/bc/${ENV}/bc_K${K}_seed${SEED}.json"
    done
  done
done

# ── Step 6: Train CSIL ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 6: Train CSIL"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  N_EPISODES=$(env_var "$ENV" N_EPISODES_CSIL)
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)
  EARLY_STOP=$(env_var "$ENV" EARLY_STOP)
  EXTRA_ARGS=()
  if [ -n "$EARLY_STOP" ]; then EXTRA_ARGS+=("--early-stop-reward" "$EARLY_STOP"); fi
  for SEED in "${SEEDS[@]}"; do
    EXPERT_NPZ="data/expert_trajectories/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"
    for K in "${K_VALUES[@]}"; do
      MODEL_OUT="models/csil/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$MODEL_OUT" ]; then echo "[SKIP] $MODEL_OUT"; continue; fi
      echo "[RUN ] CSIL: $ENV  K=$K  seed=$SEED  episodes=$N_EPISODES  subsample_freq=$SUBSAMPLE_FREQ"
      $PYTHON scripts/06_train_csil.py \
        --env-id "$ENV" --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" --seed "$SEED" \
        --n-episodes "$N_EPISODES" --subsample-freq "$SUBSAMPLE_FREQ" \
        --bc-model "models/bc/${ENV}/K${K}_seed${SEED}.pt" \
        --device "$DEVICE" --save-model "$MODEL_OUT" \
        --save-training "results/training/csil/${ENV}/csil_K${K}_seed${SEED}.json" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
  done
done

# ── Step 7: Train CSIL-SOAR ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 7: Train CSIL-SOAR (L=${N_CRITICS})"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  N_EPISODES=$(env_var "$ENV" N_EPISODES_CSIL_SOAR)
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)
  SIGMA_CLIP=$(env_var "$ENV" SIGMA_CLIP)
  EARLY_STOP=$(env_var "$ENV" EARLY_STOP)
  EXTRA_ARGS=()
  if [ -n "$EARLY_STOP" ]; then EXTRA_ARGS+=("--early-stop-reward" "$EARLY_STOP"); fi
  for SEED in "${SEEDS[@]}"; do
    EXPERT_NPZ="data/expert_trajectories/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"
    for K in "${K_VALUES[@]}"; do
      MODEL_OUT="models/csil_soar/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$MODEL_OUT" ]; then echo "[SKIP] $MODEL_OUT"; continue; fi
      echo "[RUN ] CSIL-SOAR: $ENV  K=$K  seed=$SEED  episodes=$N_EPISODES  subsample_freq=$SUBSAMPLE_FREQ  L=${N_CRITICS}"
      $PYTHON scripts/07_train_csil_soar.py \
        --env-id "$ENV" --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" --seed "$SEED" \
        --n-episodes "$N_EPISODES" --subsample-freq "$SUBSAMPLE_FREQ" \
        --bc-model "models/bc/${ENV}/K${K}_seed${SEED}.pt" \
        --n-critics "$N_CRITICS" --sigma-clip "$SIGMA_CLIP" \
        --device "$DEVICE" --save-model "$MODEL_OUT" \
        --save-training "results/training/csil_soar/${ENV}/csil_soar_K${K}_seed${SEED}.json" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
  done
done

# ── Step 8: Evaluate IQ-Learn ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 8: Evaluate IQ-Learn"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/evaluation/iq_learn/${ENV}/iq_learn_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi
      MODEL="models/iq_learn/${ENV}/K${K}_seed${SEED}.pt"
      if [ ! -f "$MODEL" ]; then echo "[SKIP] model missing: $MODEL"; continue; fi
      $PYTHON scripts/08_evaluate_iq_learn.py \
        --env-id "$ENV" --model-path "$MODEL" \
        --n-episodes "$N_EVAL_EPISODES" --K "$K" --train-seed "$SEED" \
        --eval-seeds "${EVAL_SEEDS[@]}" --save-json "$RESULT_JSON"
    done
  done
done

# ── Step 9: Evaluate BC ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 9: Evaluate BC"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/evaluation/bc/${ENV}/bc_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi
      MODEL="models/bc/${ENV}/K${K}_seed${SEED}.pt"
      if [ ! -f "$MODEL" ]; then echo "[SKIP] model missing: $MODEL"; continue; fi
      $PYTHON scripts/09_evaluate_bc.py \
        --model-path "$MODEL" --n-episodes "$N_EVAL_EPISODES" \
        --eval-seeds "${EVAL_SEEDS[@]}" --device "$DEVICE" \
        --save-json "$RESULT_JSON"
    done
  done
done

# ── Step 10: Evaluate CSIL ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 10: Evaluate CSIL"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/evaluation/csil/${ENV}/csil_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi
      MODEL="models/csil/${ENV}/K${K}_seed${SEED}.pt"
      if [ ! -f "$MODEL" ]; then echo "[SKIP] model missing: $MODEL"; continue; fi
      $PYTHON scripts/10_evaluate_csil.py \
        --model-path "$MODEL" --n-episodes "$N_EVAL_EPISODES" \
        --eval-seeds "${EVAL_SEEDS[@]}" --device "$DEVICE" \
        --save-json "$RESULT_JSON"
    done
  done
done

# ── Step 11: Evaluate CSIL-SOAR ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 11: Evaluate CSIL-SOAR"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/evaluation/csil_soar/${ENV}/csil_soar_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi
      MODEL="models/csil_soar/${ENV}/K${K}_seed${SEED}.pt"
      if [ ! -f "$MODEL" ]; then echo "[SKIP] model missing: $MODEL"; continue; fi
      $PYTHON scripts/11_evaluate_csil_soar.py \
        --model-path "$MODEL" --n-episodes "$N_EVAL_EPISODES" \
        --eval-seeds "${EVAL_SEEDS[@]}" --device "$DEVICE" \
        --save-json "$RESULT_JSON"
    done
  done
done

# ── Step 12: Plot ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 12: Plotting"
echo "============================================================"
$PYTHON scripts/12_plot_comparison.py

echo ""
echo "Done!  Figures saved to results/figures/"