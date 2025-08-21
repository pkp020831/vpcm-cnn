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
WANDB_PROJECT = "heatmap_1024x3"
RUN_NAME_PREFIX = "heatmap-test-"

# This should match the values in the training script
NETWORK_DEPTH_L = 4

OUTPUT_DIR = Path("plotting/loss_heatmap")
OUTPUT_FILE_ORIGINAL = OUTPUT_DIR / "loss_heatmap_original.png"
OUTPUT_FILE_SMOOTHED = OUTPUT_DIR / "loss_heatmap_smoothed.png"


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
            if "train/loss" in run.summary:
                data.append({
                    "lr": run.config["model"]["optimizer"]["lr"],
                    "beta": run.config["model"]["net"]["beta"],
                    "train_loss": run.summary["train/loss"]
                })

    if not data:
        print("No completed runs with 'train/loss' found yet.")
        return None

    print(f"Found {len(data)} completed runs to plot.")
    return pd.DataFrame(data)

def prepare_heatmap_data(df, scale_factor):
    """Takes a DataFrame and returns a pivot table with a perfect log-scale grid."""
    # 1. Define the desired plotting grid with clean log-scale intervals

    unscaled_betas_grid = np.logspace(-3, 3, num=7)  
    unscaled_lrs_grid = np.logspace(-6, 1, num=8)

    # 2. Scale this 'true' grid to match the value ranges stored in wandb
    scaled_betas_grid = unscaled_betas_grid / scale_factor
    scaled_lrs_grid = np.sort(unscaled_lrs_grid / scale_factor)[::-1] # Sort descending for plot

    # 3. Create a new, empty pivot table with this clean, scaled grid
    pivot_df = pd.DataFrame(np.nan, index=scaled_lrs_grid, columns=scaled_betas_grid)

    # 4. Map the fetched data to the clean pivot table
    for _, row in df.iterrows():
        lr_val, beta_val, loss_val = row['lr'], row['beta'], row['train_loss']
        
        # Find the closest index and column in our clean grid
        lr_idx = (np.abs(pivot_df.index - lr_val)).argmin()
        beta_idx = (np.abs(pivot_df.columns - beta_val)).argmin()
        
        # Place the loss value in the corresponding cell
        pivot_df.iat[lr_idx, beta_idx] = loss_val

    # 5. Create labels from the clean, unscaled grid
    beta_labels = [format_tick_label(b) for b in unscaled_betas_grid]
    lr_labels = [format_tick_label(l) for l in np.sort(unscaled_lrs_grid)[::-1]]

    return pivot_df, beta_labels, lr_labels

def generate_heatmaps(pivot_df, beta_labels, lr_labels, output_original, output_smoothed):
    """Generates and saves the original and smoothed heatmaps from a pivot table."""
    print("--- Generating new heatmaps ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generate Original Heatmap ---
    plt.figure(figsize=(10, 8))
    ax_orig = sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis_r", xticklabels=beta_labels, yticklabels=lr_labels)
    ax_orig.set_title("Original Training Loss after 5 Epochs")
    ax_orig.set_xlabel("Unscaled Beta")
    ax_orig.set_ylabel("Unscaled Learning Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_original)
    plt.close()
    print(f"Saved updated original heatmap to {output_original}")

    # --- Generate Smoothed Heatmap ---
    plt.figure(figsize=(12, 8))
    # Use the numeric data from pivot_df for imshow
    img = plt.imshow(pivot_df.values, interpolation='bilinear', cmap='coolwarm', aspect='auto')

    # Annotations are commented out as per user's file
    # for i in range(len(pivot_df.index)):
    #     for j in range(len(pivot_df.columns)):
    #         plt.text(j, i, f"{pivot_df.values[i, j]:.4f}", ha="center", va="center", color="black")

    # Set ticks and labels based on the grid dimensions
    plt.xticks(ticks=np.arange(len(beta_labels)), labels=beta_labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(lr_labels)), labels=lr_labels)

    plt.xlabel("Unscaled Beta", fontsize=14)
    plt.ylabel("Unscaled Learning Rate", fontsize=14)
    plt.title("Smoothed Training Loss", fontsize=16)
    plt.colorbar(img, label="Training Loss")
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
        scale_factor = math.sqrt(NETWORK_DEPTH_L)
        pivot_df, beta_labels, lr_labels = prepare_heatmap_data(raw_data_df, scale_factor)
        generate_heatmaps(pivot_df, beta_labels, lr_labels, OUTPUT_FILE_ORIGINAL, OUTPUT_FILE_SMOOTHED)

if __name__ == "__main__":
    main()
