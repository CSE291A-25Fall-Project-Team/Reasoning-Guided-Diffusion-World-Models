# #!/bin/bash
# # running_experiments.sh - Simple fixed version

# BASE_DIR="/home/hanan/dev/videopolicy/video_model"
# cd "$BASE_DIR"

# echo "Starting experiment runner from: $(pwd)"
# echo "This will run until all experiments are completed."
# echo "Press Ctrl+C to stop at any time."
# echo ""

# for i in {1..1000}; do
#     echo "Run #$i - $(date)"
#     echo "================================"
    
#     # Run the experiment command
#     CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/sampling/run_with_planner.py --config=scripts/sampling/configs/svd_xt.yaml
    
#     EXIT_CODE=$?
    
#     if [ $EXIT_CODE -eq 0 ]; then
#         echo "Run #$i completed successfully"
#     else
#         echo "Run #$i failed with exit code $EXIT_CODE"
#         echo "Will retry after 10 seconds..."
#         sleep 10
#     fi
    
#     echo "Waiting 5 seconds before next run..."
#     echo ""
#     sleep 5
# done

# echo "Reached maximum runs (1000). Stopping."

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py --config=scripts/sampling/configs/svd_xt.yaml
 # > ./run.log 2>&1 &