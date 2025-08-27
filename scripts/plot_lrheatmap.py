import pandas as pd
import seaborn as sns
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

# ==============================================================================
# Configurations
# ==============================================================================

WANDB_ENTITY = "hongroot-seoul-national-university"
WANDB_PROJECT = "heatmap_512x3_mnist"
RUN_NAME_PREFIX = "heatmap-test-"

OUTPUT_DIR = Path("plotting/loss_heatmap")


# ==============================================================================
# Helper Functions
# ==============================================================================

def format_tick_label(value):
    """Formats a number into a TeX string for matplotlib labels for a log scale."""
    # Use round to handle potential float inaccuracies for powers of 10
    exponent = int(round(np.log10(abs(value))))
    return f"$10^{{{exponent}}}$"

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
                    "beta": run.config["model"]["net"]["beta"],
                    "train_loss": run.summary["train/loss"],
                    "train_acc": run.summary["train/acc"]
                })

    if not data:
        print("No completed runs with 'train/loss' and 'train/acc' found yet.")
        return None

    print(f"Found {len(data)} completed runs to plot.")
    return pd.DataFrame(data)

def prepare_heatmap_data(df, value_column):
    """Takes a DataFrame and returns a pivot table for the specified value column."""
    # 1. Define the desired plotting grid with clean log-scale intervals
    betas_grid = np.logspace(-3, 6, num=10)
    lrs_grid = np.logspace(-6, 1, num=8)

    # 2. Sort lrs for plot
    lrs_grid = np.sort(lrs_grid)[::-1] # Sort descending for plot

    # 3. Create a new, empty pivot table with this clean, unscaled grid
    pivot_df = pd.DataFrame(np.nan, index=lrs_grid, columns=betas_grid)

    # 4. Map the fetched data to the clean pivot table
    for _, row in df.iterrows():
        lr_val, beta_val, value = row['lr'], row['beta'], row[value_column]
        
        # Find the closest index and column in our clean grid
        lr_idx = (np.abs(pivot_df.index - lr_val)).argmin()
        beta_idx = (np.abs(pivot_df.columns - beta_val)).argmin()
        
        # Place the value in the corresponding cell
        pivot_df.iat[lr_idx, beta_idx] = value

    # 5. Create labels from the clean, unscaled grid
    beta_labels = [format_tick_label(b) for b in betas_grid]
    lr_labels = [format_tick_label(l) for l in lrs_grid]

    return pivot_df, beta_labels, lr_labels

def generate_heatmaps(pivot_df, beta_labels, lr_labels, value_type):
    """Generates and saves heatmaps for a given metric (loss or acc)."""
    print(f"--- Generating new heatmaps for {value_type} ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_original = OUTPUT_DIR / f"loss_heatmap_original_{value_type}.png"
    output_smoothed = OUTPUT_DIR / f"loss_heatmap_smoothed_{value_type}.png"

    if value_type == 'acc':
        cmap_orig = "viridis"
        cmap_smooth = "coolwarm_r"  # Higher is better (blue)
        title_orig = "Original Training Accuracy after 5 Epochs"
        cbar_label = "Training Accuracy"
    else: # loss
        cmap_orig = "viridis_r"
        cmap_smooth = "coolwarm"  # Lower is better (blue)
        title_orig = "Original Training Loss after 5 Epochs"
        cbar_label = "Training Loss"

    # --- Generate Original Heatmap ---
    plt.figure(figsize=(10, 8))
    ax_orig = sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap=cmap_orig, xticklabels=beta_labels, yticklabels=lr_labels)
    ax_orig.set_title(title_orig)
    ax_orig.set_xlabel("Beta")
    ax_orig.set_ylabel("Learning Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_original)
    plt.close()
    print(f"Saved updated original heatmap to {output_original}")

    # --- Generate Smoothed Heatmap ---
    plt.figure(figsize=(12, 8))
    img = plt.imshow(pivot_df.values, interpolation='bilinear', cmap=cmap_smooth, aspect='auto')

    plt.xticks(ticks=np.arange(len(beta_labels)), labels=beta_labels, rotation=45, ha="right", fontsize=16)
    plt.yticks(ticks=np.arange(len(lr_labels)), labels=lr_labels, fontsize=16)

    plt.xlabel("Beta", fontsize=18)
    plt.ylabel("Learning Rate", fontsize=18)
    plt.title("(512,3)", fontsize=20)
    cbar = plt.colorbar(img)
    cbar.set_label(cbar_label, size=18)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(output_smoothed)
    plt.close()
    print(f"Saved updated smoothed heatmap to {output_smoothed}")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main function to run the script."""
    raw_data_df = fetch_wandb_data(WANDB_ENTITY, WANDB_PROJECT, RUN_NAME_PREFIX)

    if raw_data_df is not None and not raw_data_df.empty:
        # Generate Loss Heatmaps
        pivot_df_loss, beta_labels, lr_labels = prepare_heatmap_data(raw_data_df, 'train_loss')
        if not pivot_df_loss.isnull().all().all():
            generate_heatmaps(pivot_df_loss, beta_labels, lr_labels, 'loss')
        else:
            print("Pivot table for loss is empty. Skipping heatmap generation.")

        # Generate Accuracy Heatmaps
        pivot_df_acc, beta_labels, lr_labels = prepare_heatmap_data(raw_data_df, 'train_acc')
        if not pivot_df_acc.isnull().all().all():
            generate_heatmaps(pivot_df_acc, beta_labels, lr_labels, 'acc')
        else:
            print("Pivot table for accuracy is empty. Skipping heatmap generation.")


if __name__ == "__main__":
    main()
