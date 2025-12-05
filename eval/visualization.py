import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import textwrap


b40 =  "/home/hanan/dev/videopolicy_baseline_eval/videopolicy_for_eval_40K/video_model/experiments/example_inferences"
b30 =  "/home/hanan/dev/videopolicy_baseline_eval/videopolicy_for_eval_30K/video_model/experiments/example_inferences"
pb40 =  "/home/hanan/dev/videopolicy/video_model/experiments/example_inference_planner_40k_step"
pbr30 =  "/home/hanan/dev/videopolicy/video_model/experiments/example_inference_planner_plus_ddpo_30k_step"
pbr40  = "/home/hanan/dev/videopolicy/video_model/experiments/example_inference_planner_plus_ddpo_40k_step"
#count the success rate with dictionary {model:{task:success_rate}}

success_rates = defaultdict(dict)

def generate_table(filtered_data):
    """
    Print a clean, aligned table of success counts, totals, and success rates.
    """
    if not filtered_data:
        print("No data to print.")
        return

    models = list(filtered_data.keys())
    tasks = sorted(list(next(iter(filtered_data.values())).keys()))

    # Build table rows
    rows = []
    for task in tasks:
        row = [task]
        for model in models:
            s, t = filtered_data[model][task]
            rate = s / t if t > 0 else 0
            row.append(f"{s}/{t} ({rate:.2f})")
        rows.append(row)

    # Column headers
    headers = ["Task"] + models

    # Compute column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Format helper
    def fmt(row):
        return " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))

    # Print table
    print("\n" + fmt(headers))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

    for row in rows:
        print(fmt(row))

    print()



# Iterate over each folder in the root path
def get_baseline_data(model_name,root_path,file_name = "multi_environment_experiment_record.json"):
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, file_name)
            if os.path.isfile(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                
                # Assume folder_name is the model name
                for env in data["environments"]:
                    task_name = env
                    exp = data["environments"][env]["experiments"]
                    succ_count = 0
                    total = 0

                    for demo in exp:
                        # Both must be finished
                        if exp[demo]["status"] != "done":
                            continue
                        succ = exp[demo]["success"]
                        if succ == 1:
                            succ_count += 1
                        total += 1        
                    if total != 0:
                        success_rates[model_name][task_name] = (succ_count,total)
    # Print the result

def get_ourapproach_data(model_name,root_path,file_name = "multi_environment_experiment_record.json"):
    json_file = os.path.join(root_path,file_name)
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Assume folder_name is the model name
        for env in data["environments"]:
            task_name = env
            exp = data["environments"][env]["experiments"]
            succ_count = 0
            total = 0

            for demo in exp:
                # Both must be finished
                if exp[demo]["status"] != "done":
                    continue
                succ = exp[demo]["success"]
                if succ == 1:
                    succ_count += 1
                total += 1        
            if total != 0:
                success_rates[model_name][task_name] = (succ_count,total)



def filter_common_tasks(data, models):
    """
    Keep only tasks that appear in all models from the list.

    Args:
        data (dict): Nested dict of form {model: {task: (success_count, total_count)}}
        models (list): List of model names to consider

    Returns:
        dict: Filtered dict with only common tasks
    """
    if not models:
        return {}

    # Start with tasks from the first model
    common_tasks = set(data[models[0]].keys())

    # Intersect with tasks from the remaining models
    for model in models[1:]:
        common_tasks &= set(data[model].keys())

    # Build filtered dict
    filtered_data = {model: {task: data[model][task] for task in common_tasks} for model in models}
    
    return filtered_data


def plot_success_rates(filtered_data, title=None, label_wrap_width=15, bar_width_ratio=0.8):
    """
    Visualize success rates of tasks across models side by side with custom colors.

    Args:
        filtered_data (dict): Output from `filter_common_tasks`
            Format: {model: {task: (success_count, total_count)}}
        title (str, optional): Custom title for the plot.
        label_wrap_width (int): Character limit for wrapping x-axis labels.
        bar_width_ratio (float): Ratio of available space for bars (0.8 means 80% is used).
    """
    if not filtered_data:
        print("No data to plot.")
        return

    models = list(filtered_data.keys())
    if not models or not filtered_data[models[0]]:
        print("Data structure is invalid or empty.")
        return
        
    tasks = sorted(list(next(iter(filtered_data.values())).keys()))
    num_models = len(models)
    num_tasks = len(tasks)

    # --- 🌟 Improvement: Define Custom Color Map ---
    # Using specific hex codes for precise colors, but Matplotlib names work too.
    # The dictionary keys must exactly match the model names in your 'filtered_data'.
    color_map = {
        "baseline": "darkblue",
        # "baseline30": "lightblue",
        "planner+baseline": "orange", # I've changed 'orange' to 'darkorange' for distinction
        # "planner+baseline+reward30": "lightcoral", # Using lightcoral for light red
        "planner+baseline+reward": "darkred",
    }
    # -----------------------------------------------

    # Prepare success rate data (Ensures tasks are accessed in sorted order)
    success_rates = []
    for model in models:
        rates = []
        for task in tasks:
            success, total = filtered_data[model].get(task, (0, 0)) 
            rate = success / total if total > 0 else 0
            rates.append(rate)
        success_rates.append(rates)

    success_rates = np.array(success_rates)  # shape: (num_models, num_tasks)

    # Plot Setup
    x = np.arange(num_tasks) 
    bar_group_width = bar_width_ratio
    width = bar_group_width / num_models

    fig_width = max(8, num_tasks * 0.7) 
    fig_height = max(6, num_tasks * 0.5) 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Bar Plotting
    x_offset = - (bar_group_width / 2) + (width / 2)
    
    for i, model in enumerate(models):
        # --- 🌟 Improvement: Use the color_map for the 'color' argument ---
        bar_color = color_map.get(model, 'gray') # Default to gray if model name is missing
        ax.bar(x + x_offset + i * width, success_rates[i], width, label=model, color=bar_color)
        
    # X-Axis Labels 
    wrapped_labels = ['\n'.join(textwrap.wrap(task, label_wrap_width)) for task in tasks]
    
    ax.set_xlabel('Tasks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title(title if title else 'Task Success Rates Across Models', fontsize=14, pad=15)
    
    # Set tick positions and labels
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right')
    
    # Place legend outside the plot area
    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0), title='Model', fancybox=True, shadow=True) 
    ax.set_ylim(0, 1.05) 
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for external legend
    
    # Save the figure
    name = "v.s.".join(models)
    save_path = f"graphs/{name}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()



def build_graph(models):
    filtered_data = filter_common_tasks(success_rates,models)
    # print(models,"(success count, total count): ",filtered_data)
    print()
    plot_success_rates(filtered_data)
    generate_table(filtered_data)


get_baseline_data("baseline",b40)
get_ourapproach_data("planner+baseline",pb40)
get_ourapproach_data("planner+baseline+reward",pbr40,"machine2_ours_40k_experiment_record.json")
print(success_rates)

build_graph(["baseline","planner+baseline"])
build_graph(["baseline","planner+baseline+reward"])
build_graph(["planner+baseline","planner+baseline+reward"])
build_graph(["baseline","planner+baseline","planner+baseline+reward"])
print("$$Graphs updated in the graphs folder")
