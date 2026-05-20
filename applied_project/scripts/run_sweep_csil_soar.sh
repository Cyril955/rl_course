#!/usr/bin/env bash
# Train all CSIL+SOAR models (all envs × K values × seeds). Skips existing models.
# Loads pre-trained BC from models/bc/ if available (run run_sweep_bc.sh first).
# Run from applied_project/: PYTHON=/path/to/python bash scripts/run_sweep_csil_soar.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(which python3)}"

eval "$($PYTHON scripts/config.py --shell)"
env_var() { local _v="${2}_${SHELL_PREFIXES[$1]}"; echo "${!_v}"; }

echo "============================================================"
echo " CSIL-SOAR Training Sweep (L=${N_CRITICS})"
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
      echo "[RUN ] CSIL-SOAR: $ENV  K=$K  seed=$SEED  L=${N_CRITICS}  σ=${SIGMA_CLIP}"
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

echo "Done!"