#!/usr/bin/env bash
# Reproduce the full comparison (CartPole + Acrobot) end-to-end.
#
# Run from the applied_project/ directory:
#   PYTHON=/path/to/venv/bin/python bash scripts/run_sweep.sh
#
# The script is idempotent: each step is skipped if its output already exists.
# All hyperparameters come from scripts/config.py — edit that file to change them.

set -euo pipefail
cd "$(dirname "$0")/.."   # ensure CWD = applied_project/
PYTHON="${PYTHON:-$(which python3)}"

# ── Load all configuration from config.py ─────────────────────────────────────
eval "$($PYTHON scripts/config.py --shell)"

# ── Helper: look up a per-environment variable ────────────────────────────────
# Usage: env_var <env_id> <VAR_BASE>
# e.g.   env_var "CartPole-v1" LEARN_STEPS  →  value of LEARN_STEPS_CARTPOLE
env_var() { local _v="${2}_${SHELL_PREFIXES[$1]}"; echo "${!_v}"; }

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
    STEPS=$(env_var "$ENV" EXPERT_TIMESTEPS)
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
    echo "[RUN ] Generating dataset: $ENV  seed=$SEED  K=${EXPERT_POOL_K}"
    $PYTHON scripts/02_generate_expert_dataset.py \
      --env-id "$ENV" \
      --model-path "models/experts/${ENV}/ppo_expert_seed_${SEED}.zip" \
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
  LEARN_STEPS=$(env_var "$ENV" LEARN_STEPS)
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)

  for SEED_IDX in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$SEED_IDX]}"
    EVAL_SEED="${EVAL_SEEDS[$SEED_IDX]}"
    EXPERT_NPZ="data/expert/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"

    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/raw/iq_learn/${ENV}/iq_learn_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi

      MODEL_OUT="models/iq_learn/${ENV}/K${K}_seed${SEED}.pt"
      echo ""
      echo "[RUN ] IQ-Learn: $ENV  K=$K  seed=$SEED  eval_seed=$EVAL_SEED  steps=$LEARN_STEPS"

      $PYTHON scripts/04_train_iq_learn.py \
        --env-id "$ENV" \
        --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" \
        --seed "$SEED" \
        --learn-steps "$LEARN_STEPS" \
        --subsample-freq "$SUBSAMPLE_FREQ" \
        --output-model "$MODEL_OUT"

      $PYTHON scripts/05_evaluate_iq_learn.py \
        --env-id "$ENV" \
        --model-path "$MODEL_OUT" \
        --n-episodes "$N_EVAL_EPISODES" \
        --K "$K" \
        --train-seed "$SEED" \
        --eval-seed "$EVAL_SEED" \
        --save-json "$RESULT_JSON"
    done
  done
done

# ── Step 5: CSIL sweep (env × K × seed) ───────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 5: CSIL training sweep"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  N_EPISODES=$(env_var "$ENV" N_EPISODES_CSIL)
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)
  EARLY_STOP=$(env_var "$ENV" EARLY_STOP)
  EXTRA_ARGS=()
  if [ -n "$EARLY_STOP" ]; then EXTRA_ARGS+=("--early-stop-reward" "$EARLY_STOP"); fi

  for SEED_IDX in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$SEED_IDX]}"
    EVAL_SEED="${EVAL_SEEDS[$SEED_IDX]}"
    EXPERT_NPZ="data/expert/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"

    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/raw/csil/${ENV}/csil_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi

      MODEL_OUT="models/csil/${ENV}/K${K}_seed${SEED}.pt"
      echo ""
      echo "[RUN ] CSIL: $ENV  K=$K  seed=$SEED  eval_seed=$EVAL_SEED  episodes=$N_EPISODES  subsample_freq=$SUBSAMPLE_FREQ"

      $PYTHON scripts/06_train_csil.py \
        --env-id "$ENV" \
        --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" \
        --seed "$SEED" \
        --n-episodes "$N_EPISODES" \
        --subsample-freq "$SUBSAMPLE_FREQ" \
        --eval-seed "$EVAL_SEED" \
        --save-model "$MODEL_OUT" \
        --save-json "$RESULT_JSON" \
        --device "$DEVICE" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
  done
done

# ── Step 6: BC baseline sweep (env × K × seed) ────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 6: BC baseline sweep"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)

  for SEED_IDX in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$SEED_IDX]}"
    EVAL_SEED="${EVAL_SEEDS[$SEED_IDX]}"
    EXPERT_NPZ="data/expert/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"

    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/raw/bc/${ENV}/bc_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi

      echo "[RUN ] BC: $ENV  K=$K  seed=$SEED  eval_seed=$EVAL_SEED  subsample_freq=$SUBSAMPLE_FREQ"
      $PYTHON scripts/08_evaluate_bc.py \
        --env-id "$ENV" \
        --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" \
        --seed "$SEED" \
        --subsample-freq "$SUBSAMPLE_FREQ" \
        --eval-seed "$EVAL_SEED" \
        --save-json "$RESULT_JSON"
    done
  done
done

# ── Step 7: CSIL-SOAR sweep (env × K × seed) ──────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 7: CSIL-SOAR training sweep (L=${N_CRITICS})"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  N_EPISODES=$(env_var "$ENV" N_EPISODES_CSIL_SOAR)
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)
  SIGMA_CLIP=$(env_var "$ENV" SIGMA_CLIP)
  EARLY_STOP=$(env_var "$ENV" EARLY_STOP)
  EXTRA_ARGS=()
  if [ -n "$EARLY_STOP" ]; then EXTRA_ARGS+=("--early-stop-reward" "$EARLY_STOP"); fi

  for SEED_IDX in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$SEED_IDX]}"
    EVAL_SEED="${EVAL_SEEDS[$SEED_IDX]}"
    EXPERT_NPZ="data/expert/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"

    for K in "${K_VALUES[@]}"; do
      RESULT_JSON="results/raw/csil_soar/${ENV}/csil_soar_K${K}_seed${SEED}.json"
      if [ -f "$RESULT_JSON" ]; then echo "[SKIP] $RESULT_JSON"; continue; fi

      MODEL_OUT="models/csil_soar/${ENV}/K${K}_seed${SEED}.pt"
      echo ""
      echo "[RUN ] CSIL-SOAR: $ENV  K=$K  seed=$SEED  eval_seed=$EVAL_SEED  episodes=$N_EPISODES  subsample_freq=$SUBSAMPLE_FREQ  L=${N_CRITICS}  σ=${SIGMA_CLIP}"

      $PYTHON scripts/08_train_csil_soar.py \
        --env-id "$ENV" \
        --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" \
        --seed "$SEED" \
        --n-episodes "$N_EPISODES" \
        --subsample-freq "$SUBSAMPLE_FREQ" \
        --eval-seed "$EVAL_SEED" \
        --n-critics "$N_CRITICS" \
        --sigma-clip "$SIGMA_CLIP" \
        --save-model "$MODEL_OUT" \
        --save-json "$RESULT_JSON" \
        --device "$DEVICE" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
  done
done

# ── Step 8: Plot ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 8: Plotting Comparison Figures"
echo "============================================================"
$PYTHON scripts/07_plot_comparison_figure.py
$PYTHON scripts/09_plot_comparison_soar.py

echo ""
echo "Done!  Figures saved to results/figures/"