import subprocess
import sys
import pandas as pd
import seaborn as sns
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import math

# ==============================================================================
# PART 1: PLOTTING FUNCTION (to be called after each run)
# ==============================================================================

def update_and_plot_heatmap():
    """Fetches all available data from WandB and regenerates the heatmaps."""
    # --- Read WandB project from config file ---
    wandb_config_path = Path("configs/logger/wandb.yaml")
    project_name = "default_project"
    try:
        with open(wandb_config_path, 'r') as f:
            wandb_config = yaml.safe_load(f)
            if "wandb" in wandb_config and "project" in wandb_config["wandb"]:
                project_name = wandb_config["wandb"]["project"]
            else:
                print(f"Warning: Could not find key 'wandb.project' in {wandb_config_path}.")
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not read wandb project name from {wandb_config_path}. Error: {e}")

    # --- Plotting Configuration ---
    WANDB_ENTITY = "hongroot-seoul-national-university"
    WANDB_PROJECT = project_name
    run_name_prefix = "heatmap-test-"
    OUTPUT_DIR = Path("plotting/loss_heatmap")
    OUTPUT_FILE_ORIGINAL = OUTPUT_DIR / "loss_heatmap_original.png"
    OUTPUT_FILE_SMOOTHED = OUTPUT_DIR / "loss_heatmap_smoothed.png"

    print(f"\n--- Updating heatmaps, fetching runs from {WANDB_ENTITY}/{WANDB_PROJECT} ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    data = []
    for run in runs:
        if run.name.startswith(run_name_prefix):
            if "train/loss" in run.summary:
                final_train_loss = run.summary["train/loss"]
                data.append({
                    "lr": run.config["model"]["optimizer"]["lr"],
                    "beta": run.config["model"]["net"]["beta"],
                    "train_loss": final_train_loss
                })

    if not data:
        print("No data found yet. Skipping plot generation.")
        return

    print(f"Found {len(data)} completed runs to plot.")

    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(index="lr", columns="beta", values="train_loss", aggfunc="mean")
    pivot_df.sort_index(ascending=False, inplace=True)
    pivot_df.sort_index(axis=1, ascending=True, inplace=True)

    # --- Unscale beta and lr for plotting ---
    NETWORK_DEPTH_L = 4  # As defined in the training script part
    SCALE_FACTOR = math.sqrt(NETWORK_DEPTH_L)

    beta_labels = pivot_df.columns
    lr_labels = pivot_df.index
    try:
        unscaled_betas = pivot_df.columns.to_numpy(dtype=float) * SCALE_FACTOR
        unscaled_lrs = pivot_df.index.to_numpy(dtype=float) * SCALE_FACTOR

        # Format labels as powers of 10 for better readability
        beta_labels = [f"$10^{{{int(round(np.log10(b)))}}}" for b in unscaled_betas]
        lr_labels = [f"$10^{{{int(round(np.log10(l)))}}}" for l in unscaled_lrs]
    except Exception as e:
        print(f"Warning: Could not generate new labels, falling back to default. Error: {e}")


    # --- Generate Original Heatmap ---
    plt.figure(figsize=(10, 8))
    ax_orig = sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis_r", xticklabels=beta_labels, yticklabels=lr_labels)
    ax_orig.set_title("Original Training Loss after 5 Epochs")
    ax_orig.set_xlabel("Beta")
    ax_orig.set_ylabel("Learning Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_ORIGINAL)
    plt.close()
    print(f"Saved updated original heatmap to {OUTPUT_FILE_ORIGINAL}")

    # --- Generate Smoothed Heatmap ---
    plt.figure(figsize=(12, 8))
    img = plt.imshow(pivot_df.values, interpolation='bilinear', cmap='coolwarm', aspect='auto')

    # Annotations are commented out as per user's file
    # for i in range(len(pivot_df.index)):
    #     for j in range(len(pivot_df.columns)):
    #         plt.text(j, i, f"{pivot_df.values[i, j]:.4f}", ha="center", va="center", color="black")

    plt.xticks(ticks=np.arange(len(pivot_df.columns)), labels=beta_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(pivot_df.index)), labels=lr_labels)

    plt.xlabel("Beta", fontsize=14)
    plt.ylabel("Learning Rate", fontsize=14)
    plt.title("Training Loss", fontsize=16)
    plt.colorbar(img, label="Training Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_SMOOTHED)
    plt.close()
    print(f"Saved updated smoothed heatmap to {OUTPUT_FILE_SMOOTHED}")


# ==============================================================================
# PART 2: RUN TRAINING & LIVE PLOTTING
# ==============================================================================

# --- Training Configuration ---
betas = np.logspace(-3, 3, num=8)
lrs = np.logspace(-6, 1, num=8)
sconda_env_name = "myenv"
NETWORK_DEPTH_L = 4

print(f"--- Starting grid search with live heatmap updates ---")
print(f"--- Using network depth L={NETWORK_DEPTH_L} for scaling beta and lr ---")

for beta in betas:
    for lr in lrs:
        scaled_beta = beta / math.sqrt(NETWORK_DEPTH_L)
        scaled_lr = lr / math.sqrt(NETWORK_DEPTH_L)
        run_name = f"heatmap-test-beta{beta}-lr{lr}"

        print(f"\n---> Running training for unscaled beta: {beta}, lr: {lr} (SCALED to beta: {scaled_beta:.4f}, lr: {scaled_lr:.6f})")

        command = [
            "conda", "run", "-n", sconda_env_name, "--no-capture-output",
            "python", "src/train.py",
            "model/net=ident_ep_mnist", "model/optimizer=adamw", "trainer=gpu",
            "model.net.solver.amp_factor=1.35", "data.batch_size=64", "trainer.max_epochs=5",
            f"model.net.beta={scaled_beta}", f"model.optimizer.lr={scaled_lr}",
            f"+logger.wandb.name={run_name}"
        ]

        try:
            process = subprocess.Popen(command)
            process.wait()
            
            if process.returncode == 0:
                print(f"---> Run SUCCEEDED. Updating heatmap...")
                update_and_plot_heatmap()
            else:
                print(f"---> Run FAILED with exit code {process.returncode}. Check logs above.", file=sys.stderr)

        except Exception as e:
            print(f"--- An exception occurred: {e} ---", file=sys.stderr)

print("\n--- All hyperparameter combinations have been run. ---")
