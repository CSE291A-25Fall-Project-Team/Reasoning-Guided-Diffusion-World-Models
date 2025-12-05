import yaml
from copy import deepcopy
from pathlib import Path

base_cfg_path = Path("/home/hanan/dev/videopolicy_baseline_eval/videopolicy/video_model/scripts/sampling/configs/svd_xt.yaml")  # your main config
out_dir = base_cfg_path.parent / "svd_xt_tasks"
out_dir.mkdir(exist_ok=True, parents=True)

with base_cfg_path.open("r") as f:
    base_cfg = yaml.safe_load(f)

tasks = base_cfg["data"]["params"]["tasks"]

for task_name, task_cfg in tasks.items():
    cfg = deepcopy(base_cfg)

    # Keep only this one task
    cfg["data"]["params"]["tasks"] = {task_name: task_cfg}

    # Optionally set number_of_experiments to this task's num_experiments
    if "num_experiments" in task_cfg:
        cfg["number_dev/videopolicy_baseline_eval/videopolicy/video_model/scripts/sampling/configs/svd_xt_tasksof_experiments"] = task_cfg["num_experiments"]

    cfg["log_folder"] = f'example_inferences/svd_xt_{task_name}'

    out_path = out_dir / f"svd_xt_{task_name}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote {out_path}")
