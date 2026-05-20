#!/usr/bin/env bash
# Train all CSIL models (all envs × K values × seeds). Skips existing models.
# Loads pre-trained BC from models/bc/ if available (run run_sweep_bc.sh first).
# Run from applied_project/: PYTHON=/path/to/python bash scripts/run_sweep_csil.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(which python3)}"

eval "$($PYTHON scripts/config.py --shell)"
env_var() { local _v="${2}_${SHELL_PREFIXES[$1]}"; echo "${!_v}"; }

echo "============================================================"
echo " CSIL Training Sweep"
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
      echo "[RUN ] CSIL: $ENV  K=$K  seed=$SEED  episodes=$N_EPISODES"
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

echo "Done!"