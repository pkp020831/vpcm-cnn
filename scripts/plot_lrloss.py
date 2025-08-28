import pandas as pd
import seaborn as sns
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from pathlib import Path
import math

# ==============================================================================
# Configurations
# ==============================================================================

WANDB_ENTITY = "hongroot-seoul-national-university"
# Add the project names you want to compare
WANDB_PROJECTS = ["lrloss_512x2_goemup", "lrloss_512x3_goemup", "lrloss_512x4_goemup", "lrloss_512x5_goemup"]
RUN_NAME_PREFIX = "lossgraph-"


OUTPUT_DIR = Path("plotting/lr_vs_loss")


# ==============================================================================
# Helper Functions
# ==============================================================================

def fetch_wandb_data(entity, project, run_prefix):
    """Fetches run data from WandB and returns it as a pandas DataFrame."""
    print(f"\n--- Fetching runs from {entity}/{project} ---")
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"Error connecting to WandB: {e}")
        return None

    data = []
    for run in runs:
        if run.name.startswith(run_prefix) and run.state == "finished":
            if "train/loss" in run.summary and "train/acc" in run.summary:
                data.append({
                    "lr": run.config["model"]["optimizer"]["lr"],
                    "seed": run.config["seed"],
                    "train_loss": run.summary["train/loss"],
                    "train_acc": run.summary["train/acc"]
                })

    if not data:
        print(f"No completed runs with prefix '{run_prefix}' found.")
        return None

    print(f"Found {len(data)} completed runs to plot.")
    return pd.DataFrame(data)

def generate_line_plot(df, value_column, title, ylabel, filename, log_y):
    """
    Generates and saves a line plot of a given value vs. learning rate for multiple projects.
    """
    print(f"--- Generating plot: {title} ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / filename

    plot_data = df.sort_values(by='lr')

    if plot_data.empty:
        print("No data found. Skipping plot generation.")
        return
    
    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=plot_data, x='lr', y=value_column, hue='project', marker='o', errorbar=('ci', 95))
    
    ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(ScalarFormatter())
    
    ax.set_title(title)
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel(ylabel)
    
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Project')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved plot to {output_file}")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main function to run the script."""
    all_data = []
    for project in WANDB_PROJECTS:
        print(f"\n{'='*20} Processing project: {project} {'='*20}")
        df = fetch_wandb_data(WANDB_ENTITY, project, RUN_NAME_PREFIX)
        if df is not None and not df.empty:
            df['project'] = project
            all_data.append(df)

    if not all_data:
        print("\nNo data found for any project. Exiting.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Generate Loss Plot
    generate_line_plot(
        df=combined_df,
        value_column='train_loss',
        title='Training Loss vs. Learning Rate Comparison',
        ylabel='Final Training Loss (log scale, 95% CI)',
        filename='lr_vs_loss_comparison.png',
        log_y=True
    )

    # Generate Accuracy Plot
    generate_line_plot(
        df=combined_df,
        value_column='train_acc',
        title='Training Accuracy vs. Learning Rate Comparison',
        ylabel='Final Training Accuracy (95% CI)',
        filename='lr_vs_acc_comparison.png',
        log_y=False
    )


if __name__ == "__main__":
    main()
