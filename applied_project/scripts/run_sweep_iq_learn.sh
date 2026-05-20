#!/usr/bin/env bash
# Train all IQ-Learn models (all envs × K values × seeds). Skips existing models.
# Run from applied_project/: PYTHON=/path/to/python bash scripts/run_sweep_iq_learn.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(which python3)}"

eval "$($PYTHON scripts/config.py --shell)"
env_var() { local _v="${2}_${SHELL_PREFIXES[$1]}"; echo "${!_v}"; }

echo "============================================================"
echo " IQ-Learn Training Sweep"
echo "============================================================"

for ENV in "${ENVS[@]}"; do
  LEARN_STEPS=$(env_var "$ENV" LEARN_STEPS)
  SUBSAMPLE_FREQ=$(env_var "$ENV" SUBSAMPLE_FREQ)
  for SEED in "${SEEDS[@]}"; do
    EXPERT_NPZ="data/expert_trajectories/${ENV}/expert_K${EXPERT_POOL_K}_seed${SEED}.npz"
    for K in "${K_VALUES[@]}"; do
      MODEL_OUT="models/iq_learn/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$MODEL_OUT" ]; then echo "[SKIP] $MODEL_OUT"; continue; fi
      echo "[RUN ] IQ-Learn: $ENV  K=$K  seed=$SEED  steps=$LEARN_STEPS"
      $PYTHON scripts/04_train_iq_learn.py \
        --env-id "$ENV" --expert-npz "$EXPERT_NPZ" \
        --n-demos "$K" --seed "$SEED" \
        --learn-steps "$LEARN_STEPS" --subsample-freq "$SUBSAMPLE_FREQ" \
        --output-model "$MODEL_OUT"
    done
  done
done

echo "Done!"