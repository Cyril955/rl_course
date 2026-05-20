#!/usr/bin/env bash
# Evaluate all trained models (IQ-Learn, BC, CSIL, CSIL-SOAR) and plot.
# Skips if the result JSON already exists or the model file is missing.
#
# Run from applied_project/:
#   PYTHON=/path/to/python bash scripts/run_sweep_eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$(which python3)}"

eval "$($PYTHON scripts/config.py --shell)"

# ── IQ-Learn ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Evaluate IQ-Learn"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT="results/evaluation/iq_learn/${ENV}/iq_learn_K${K}_seed${SEED}.json"
      MODEL="models/iq_learn/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$RESULT" ]; then echo "[SKIP] $RESULT"; continue; fi
      if [ ! -f "$MODEL" ]; then echo "[SKIP] no model: $MODEL"; continue; fi
      echo "[RUN ] IQ-Learn: $ENV  K=$K  seed=$SEED"
      $PYTHON scripts/08_evaluate_iq_learn.py \
        --env-id "$ENV" --model-path "$MODEL" \
        --n-episodes "$N_EVAL_EPISODES" --K "$K" --train-seed "$SEED" \
        --eval-seeds "${EVAL_SEEDS[@]}" --save-json "$RESULT"
    done
  done
done

# ── BC ────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Evaluate BC"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT="results/evaluation/bc/${ENV}/bc_K${K}_seed${SEED}.json"
      MODEL="models/bc/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$RESULT" ]; then echo "[SKIP] $RESULT"; continue; fi
      if [ ! -f "$MODEL" ]; then echo "[SKIP] no model: $MODEL"; continue; fi
      echo "[RUN ] BC: $ENV  K=$K  seed=$SEED"
      $PYTHON scripts/09_evaluate_bc.py \
        --model-path "$MODEL" --n-episodes "$N_EVAL_EPISODES" \
        --eval-seeds "${EVAL_SEEDS[@]}" --device "$DEVICE" \
        --save-json "$RESULT"
    done
  done
done

# ── CSIL ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Evaluate CSIL"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT="results/evaluation/csil/${ENV}/csil_K${K}_seed${SEED}.json"
      MODEL="models/csil/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$RESULT" ]; then echo "[SKIP] $RESULT"; continue; fi
      if [ ! -f "$MODEL" ]; then echo "[SKIP] no model: $MODEL"; continue; fi
      echo "[RUN ] CSIL: $ENV  K=$K  seed=$SEED"
      $PYTHON scripts/10_evaluate_csil.py \
        --model-path "$MODEL" --n-episodes "$N_EVAL_EPISODES" \
        --eval-seeds "${EVAL_SEEDS[@]}" --device "$DEVICE" \
        --save-json "$RESULT"
    done
  done
done

# ── CSIL-SOAR ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Evaluate CSIL-SOAR"
echo "============================================================"
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for K in "${K_VALUES[@]}"; do
      RESULT="results/evaluation/csil_soar/${ENV}/csil_soar_K${K}_seed${SEED}.json"
      MODEL="models/csil_soar/${ENV}/K${K}_seed${SEED}.pt"
      if [ -f "$RESULT" ]; then echo "[SKIP] $RESULT"; continue; fi
      if [ ! -f "$MODEL" ]; then echo "[SKIP] no model: $MODEL"; continue; fi
      echo "[RUN ] CSIL-SOAR: $ENV  K=$K  seed=$SEED"
      $PYTHON scripts/11_evaluate_csil_soar.py \
        --model-path "$MODEL" --n-episodes "$N_EVAL_EPISODES" \
        --eval-seeds "${EVAL_SEEDS[@]}" --device "$DEVICE" \
        --save-json "$RESULT"
    done
  done
done

# ── Plot ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Plot"
echo "============================================================"
$PYTHON scripts/12_plot_comparison.py

echo ""
echo "Done!  Figures saved to results/figures/"
