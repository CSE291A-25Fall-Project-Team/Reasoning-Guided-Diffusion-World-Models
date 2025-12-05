#!/usr/bin/env bash

CONFIG_DIR="scripts/sampling/configs/svd_xt_tasks"
COMPLETED_FILE="completed.cfgs"

for cfg in "$CONFIG_DIR"/svd_xt_*.yaml; do
  if grep -q "$cfg" "$COMPLETED_FILE"; then
    echo "Skipping already completed config: $cfg"
    continue
  fi
  echo "===================================================="
  echo "Running Robocasa eval with config: $cfg"
  echo "===================================================="

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
    python scripts/sampling/robocasa_experiment.py \
      --config="$cfg"

  echo "$cfg" >> "$COMPLETED_FILE"

  echo "Finished: $cfg"
  echo
done
