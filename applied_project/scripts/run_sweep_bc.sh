#!/usr/bin/env bash
# Train all BC models (all envs × K values × seeds). Skips existing models.
# Run from applied_project/: PYTHON=/path/to/python bash scripts/run_sweep_bc.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(which python3)}"

eval "$($PYTHON scripts/config.py --shell)"
env_var() { local _v="${2}_${SHELL_PREFIXES[$1]}"; echo "${!_v}"; }

echo "============================================================"
echo " BC Training Sweep"
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

echo "Done!"