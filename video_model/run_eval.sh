#!/bin/bash

# Loop 10 times
for i in {1..10}
do
    echo "Starting iteration $i..."
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py --config=scripts/sampling/configs/svd_xt.yaml
    echo "Iteration $i finished."
done

echo "All 10 iterations completed."
