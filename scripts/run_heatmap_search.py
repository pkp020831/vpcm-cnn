import subprocess
import sys
import pandas as pd
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

# ==============================================================================
# PART 1: RUN TRAINING
# ==============================================================================

# --- Training Configuration ---
betas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
lrs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
sconda_env_name = "myenv"
run_name_prefix = "heatmap-test-"

print(f"--- Starting 3x3 grid search using base command from user ---")

# Loop through all combinations
for beta in betas:
    for lr in lrs:
        print(f"\n---> Running training for beta: {beta}, lr: {lr}")
        run_name = f"{run_name_prefix}beta{beta}-lr{lr}"

        command = [
            "conda", "run", "-n", sconda_env_name, "--no-capture-output",
            "python", "src/train.py",
            "model/net=ident_ep_mnist",
            "model/optimizer=adamw",
            "trainer=gpu",
            "model.net.solver.amp_factor=1.35",
            "data.batch_size=64",
            "trainer.max_epochs=5",
            f"model.net.beta={beta}",
            f"model.optimizer.lr={lr}",
            f"+logger.wandb.name={run_name}"
        ]

        try:
            # Execute the command and let it print directly to the terminal
            # This allows rich UI elements like progress bars to render correctly.
            process = subprocess.Popen(command)
            process.wait()  # Wait for the process to complete
            
            if process.returncode == 0:
                print(f"---> Run SUCCEEDED for beta: {beta}, lr: {lr}")
            else:
                print(f"---> Run FAILED with exit code {process.returncode} for beta: {beta}, lr: {lr}", file=sys.stderr)

        except Exception as e:
            print(f"--- An exception occurred while running command for beta={beta}, lr={lr} ---", file=sys.stderr)
            print(e, file=sys.stderr)

print("\n--- All hyperparameter combinations have been run. ---")

# ==============================================================================
# PART 2: PLOT HEATMAPS
# ==============================================================================

# --- Read WandB project from config file ---
wandb_config_path = Path("configs/logger/wandb.yaml")
project_name = "default_project"  # Fallback project name
try:
    with open(wandb_config_path, 'r') as f:
        wandb_config = yaml.safe_load(f)
        # The project name is nested under the top-level 'wandb' key
        if "wandb" in wandb_config and "project" in wandb_config["wandb"]:
            project_name = wandb_config["wandb"]["project"]
        else:
            print(f"Warning: Could not find key 'wandb.project' in {wandb_config_path}. Using fallback '{project_name}'.")
except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Warning: Could not read wandb project name from {wandb_config_path}. Using fallback '{project_name}'. Error: {e}")


# --- Plotting Configuration ---
WANDB_ENTITY = "hongroot-seoul-national-university"
WANDB_PROJECT = project_name  # Use the loaded name
OUTPUT_DIR = Path("plotting/loss_heatmap")
OUTPUT_FILE_ORIGINAL = OUTPUT_DIR / "loss_heatmap_original.png"
OUTPUT_FILE_SMOOTHED = OUTPUT_DIR / "loss_heatmap_smoothed.png"

print(f"\n--- Fetching runs from WandB project: {WANDB_ENTITY}/{WANDB_PROJECT} to generate heatmaps ---")
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
    print(f"No data found for runs with prefix '{run_name_prefix}'. Exiting.")
    sys.exit()

print(f"Found {len(data)} completed runs to plot.")

df = pd.DataFrame(data)
pivot_df = df.pivot_table(index="lr", columns="beta", values="train_loss", aggfunc="mean")
pivot_df.sort_index(ascending=False, inplace=True)
pivot_df.sort_index(axis=1, ascending=True, inplace=True)

# --- Generate Original Heatmap ---
print("--- Generating Original Heatmap ---")
plt.figure(figsize=(10, 8))
ax_orig = sns.heatmap(
    pivot_df,
    annot=True,
    fmt=".4f",
    cmap="viridis_r"
)
ax_orig.set_title("Original Training Loss after 5 Epochs")
ax_orig.set_xlabel("Beta")
ax_orig.set_ylabel("Learning Rate")
plt.savefig(OUTPUT_FILE_ORIGINAL)
plt.close()
print(f"Saved original heatmap to {OUTPUT_FILE_ORIGINAL}")

# --- Generate Smoothed Heatmap ---
print("--- Generating Smoothed Heatmap with Blue-White-Red Colormap ---")
plt.figure(figsize=(12, 8))
img = plt.imshow(
    pivot_df.values,
    interpolation='bilinear',
    cmap='coolwarm',
    aspect='auto'
)

#for i in range(len(pivot_df.index)):
#    for j in range(len(pivot_df.columns)):
#        plt.text(j, i, f"{pivot_df.values[i, j]:.4f}", ha="center", va="center", color="black")

plt.xticks(ticks=np.arange(len(pivot_df.columns)), labels=pivot_df.columns)
plt.yticks(ticks=np.arange(len(pivot_df.index)), labels=pivot_df.index)

plt.xlabel("Beta", fontsize=14)
plt.ylabel("Learning Rate", fontsize=14)
plt.title("Training Loss", fontsize=16)
plt.colorbar(img, label="Training Loss")
plt.savefig(OUTPUT_FILE_SMOOTHED)
plt.close()
print(f"Saved smoothed heatmap to {OUTPUT_FILE_SMOOTHED}")

print(f"\n--- All heatmaps saved in {OUTPUT_DIR} ---")
