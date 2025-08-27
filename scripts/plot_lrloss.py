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
WANDB_PROJECT = "lrloss_512x3_goemup"  # Make sure this is the correct project
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


def generate_line_plot(df):
    """
    Generates and saves a line plot of loss vs. learning rate.
    If multiple seeds are found for the same LR, it plots the mean and a confidence interval.
    """
    print(f"--- Generating line plot ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"lr_vs_loss_multi_seed.png"

    plot_data = df.sort_values(by='lr')

    if plot_data.empty:
        print(f"No data found. Skipping plot generation.")
        return

    num_seeds = plot_data.groupby('lr')['seed'].nunique().min()

    plt.figure(figsize=(12, 7))
    # seaborn's lineplot will automatically aggregate data, plotting the mean
    # and a 95% confidence interval by default when it sees multiple y-values for the same x.
    ax = sns.lineplot(data=plot_data, x='lr', y='train_loss', marker='o', errorbar=('ci', 95))
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Format y-axis to display regular numbers instead of scientific notation
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(ScalarFormatter())
    
    ax.set_title(f"Training Loss vs. Learning Rate (Mean over {num_seeds} seeds)")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Final Training Loss (log scale, 95% CI)")
    
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Saved line plot to {output_file}")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main function to run the script."""
    raw_data_df = fetch_wandb_data(WANDB_ENTITY, WANDB_PROJECT, RUN_NAME_PREFIX)

    if raw_data_df is not None and not raw_data_df.empty:
        generate_line_plot(raw_data_df)


if __name__ == "__main__":
    main()
