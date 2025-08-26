import subprocess
import sys
import numpy as np
import math

# ==============================================================================
# Configurations
# ==============================================================================

WANDB_PROJECT = "heatmap_512x3_default_SGD"

# --- Training Configuration ---
betas = np.logspace(-1, 3, num=12)
lrs = np.logspace(-5, -2, num=40)
sconda_env_name = "myenv"


# ============================================================================== 
# RUN TRAINING
# ============================================================================== 

print(f"--- Starting grid search for heatmap data generation ---")
print(f"--- Using wandb project: {WANDB_PROJECT} ---")

for beta in betas:
    for lr in lrs:
        run_name = f"heatmap-test-beta{beta}-lr{lr}"

        print(f"\n---> Running training for beta: {beta}, lr: {lr}")

        command = [
            "conda", "run", "-n", sconda_env_name, "--no-capture-output",
            "python", "src/train.py",
            "model/net=ident_ep_mnist", "model/optimizer=sgd", "trainer=gpu",
            "model.net.solver.amp_factor=1.35", "data.batch_size=64", "trainer.max_epochs=5",
            f"model.net.beta={beta}", f"model.optimizer.lr={lr}",
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

print("\n--- All hyperparameter combinations have been run. ---")