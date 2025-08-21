import subprocess
import sys
import numpy as np
import math

# ==============================================================================
# Configurations
# ==============================================================================

WANDB_PROJECT = "heatmap_64x3"

# --- Training Configuration ---
betas = np.logspace(-3, 3, num=8)
lrs = np.logspace(-6, 1, num=8)
sconda_env_name = "myenv"
NETWORK_DEPTH_L = 4


# ============================================================================== 
# RUN TRAINING
# ============================================================================== 

print(f"--- Starting grid search for heatmap data generation ---")
print(f"--- Using wandb project: {WANDB_PROJECT} ---")
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