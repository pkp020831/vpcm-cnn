import subprocess
import sys
import numpy as np

# ==============================================================================
# Configurations
# ==============================================================================

WANDB_PROJECT = "lrloss_512x3_goemup" # Make sure this is the correct project

# --- Training Configuration ---
lrs = np.logspace(-5, -1, num=8)  # The range of learning rates to test
seeds = [42, 43, 44] # The seeds to run the experiment on
CONDA_ENV_NAME = "myenv" # your conda environment name


# ==============================================================================
# RUN TRAINING
# ==============================================================================

print(f"--- Starting runs for loss vs. LR plot generation ---")
print(f"--- Using wandb project: {WANDB_PROJECT} ---")

for lr in lrs:
    for seed in seeds:
        run_name = f"lossgraph-lr{lr}-seed{seed}"

        print(f"\n---> Running training for lr: {lr}, seed: {seed}")

        command = [
            "conda", "run", "-n", CONDA_ENV_NAME, "--no-capture-output",
            "python", "src/train.py", "model.net.solver.amp_factor=2.2",
            "model/net=ident_ep_mnist", "model/optimizer=adamw", "trainer=gpu",
            "data.batch_size=64", "trainer.max_epochs=5", # Using 5 epochs like the heatmap
            f"seed={seed}",
            f"model.net.beta=10",
            f"model.optimizer.lr={lr}",
            f"logger.wandb.project={WANDB_PROJECT}",
            f"+logger.wandb.name={run_name}"
        ]

        try:
            process = subprocess.Popen(command)
            process.wait()
            
            if process.returncode == 0:
                print(f"---> Run SUCCEEDED.")
            else:
                print(f"---> Run FAILED with exit code {process.returncode}. Check logs above.", file=sys.stderr)

        except Exception as e:
            print(f"--- An exception occurred: {e} ---", file=sys.stderr)

print("\n--- All learning rates and seeds have been run. ---")