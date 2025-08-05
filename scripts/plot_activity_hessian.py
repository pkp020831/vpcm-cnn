import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from omegaconf import DictConfig
from torch.autograd.functional import hessian
from copy import deepcopy
from scipy.ndimage import gaussian_filter  # ADDED: For smoothing

from lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from src.core.eqprop.solvers import EqPropSolver

@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/plot_activity_hessian.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to run the activity hessian analysis across different network widths and depths."""
    # Set seed for reproducibility
    seed_everything(cfg.seed)

    # --- Define ranges for width and number of hidden layers ---
    # CHANGED: Use a logarithmic range for widths and depths
    log_range = [1, 2, 4, 8, 16, 32, 64]
    widths = log_range
    n_hiddens = log_range

    # Initialize a 2D array to store condition numbers
    condition_numbers = np.zeros((len(widths), len(n_hiddens)))

    # --- Instantiate Data (once) ---
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    x, y_true = next(iter(datamodule.train_dataloader()))
    x = x[0].unsqueeze(0).view(x[0].size(0), -1)
    y_true = y_true[0].unsqueeze(0)

    beta_val = cfg.analysis.solver.beta

    for i, width in enumerate(widths):
        for j, n_hidden in enumerate(n_hiddens):
            print(f"\n--- Processing N={width}, H={n_hidden} ---")

            # Dynamically create network architecture parameters
            net_params = {
                "cfg": [784] + [width] * n_hidden + [10],
                "initialization": cfg.model.net.initialization,
            }
            
            # Instantiate model with dynamic architecture
            net = hydra.utils.instantiate(
                DictConfig({
                    "_target_": cfg.model.net._target_,
                    "cfg": net_params["cfg"],
                    "initialization": cfg.model.net.initialization,
                    "bias": cfg.model.net.bias
                })
            )
            model: LightningModule = hydra.utils.instantiate(cfg.model, net=net)

            # --- Instantiate Solver ---
            strategy_partial = hydra.utils.instantiate(cfg.analysis.solver_strategy)
            strategy = strategy_partial()
            solver: EqPropSolver = hydra.utils.instantiate(cfg.analysis.solver, strategy=strategy)
            solver.set_model(model.net)

            # --- Construct Hessian based on Eq. (28) ---
            W_matrices = solver.strategy.W
            layer_dims = [W_matrices[0].shape[1]] + [w.shape[0] for w in W_matrices]
            total_dim = sum(layer_dims)
            hessian = torch.zeros((total_dim, total_dim), device=x.device)

            current_dim_i = 0
            for k, dim_i in enumerate(layer_dims):
                # Diagonal blocks
                if k == len(layer_dims) - 1:
                    hessian[current_dim_i:current_dim_i+dim_i, current_dim_i:current_dim_i+dim_i] = torch.eye(dim_i, device=x.device) * (1 + beta_val)
                else:
                    hessian[current_dim_i:current_dim_i+dim_i, current_dim_i:current_dim_i+dim_i] = torch.eye(dim_i, device=x.device)

                # Off-diagonal blocks
                if k < len(W_matrices):
                    W = W_matrices[k]
                    dim_j = layer_dims[k+1]
                    current_dim_j = current_dim_i + dim_i
                    hessian[current_dim_i:current_dim_i+dim_i, current_dim_j:current_dim_j+dim_j] = -W.t()
                    hessian[current_dim_j:current_dim_j+dim_j, current_dim_i:current_dim_i+dim_i] = -W
                current_dim_i += dim_i

            # --- Calculate Condition Number ---
            try:
                cond_num = torch.linalg.cond(hessian).item()
                condition_numbers[i, j] = cond_num
                print(f"Calculated Condition Number: {cond_num}")
            except Exception as e:
                print(f"Error calculating condition number: {e}")
                condition_numbers[i, j] = np.nan

    # --- Fill NaN values by interpolating from neighbors ---
    # (This part remains the same)
    nan_indices = np.argwhere(np.isnan(condition_numbers))
    for i, j in nan_indices:
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0: continue
                ni, nj = i + di, j + dj
                if 0 <= ni < condition_numbers.shape[0] and 0 <= nj < condition_numbers.shape[1]:
                    neighbor_val = condition_numbers[ni, nj]
                    if not np.isnan(neighbor_val):
                        neighbors.append(neighbor_val)
        if neighbors:
            mean_val = np.mean(neighbors)
            condition_numbers[i, j] = mean_val
            print(f"Filled NaN at ({i}, {j}) with interpolated value: {mean_val}")

    # --- ADDED: Apply Gaussian smoothing ---
    sigma = 1.0  # Adjust this value to control the amount of smoothing
    smoothed_conditions = gaussian_filter(condition_numbers, sigma=sigma)
    print(f"Applied Gaussian smoothing with sigma={sigma}")

    # --- Plotting ---
    # CHANGED: Switched to pcolormesh for true log-scale axes
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create meshgrid for pcolormesh
    H_mesh, W_mesh = np.meshgrid(n_hiddens, widths)

    # Use pcolormesh for non-uniform coordinates
    mesh = ax.pcolormesh(
    H_mesh, 
    W_mesh, 
    smoothed_conditions,
    cmap="viridis",
    norm=mcolors.LogNorm(),
    shading='gouraud'  # CHANGED: 'gouraud' 셰이딩으로 부드러운 효과 적용
)
    
    # Set log scale on both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label=r"$\kappa(H_z)$", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel("Depth (H)", fontsize=20)
    ax.set_ylabel("Width (N)", fontsize=20)
    
    # Set ticks to be the exact values used
    ax.set_xticks(n_hiddens)
    ax.set_yticks(widths)
    # Use default log formatter for labels (e.g., 1, 10, 100 or 1, 2, 4, 8)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.minorticks_off() # Turn off minor ticks for clarity
    plt.tight_layout()
    
    plt.savefig(f"plotting/activity_hessian/ah_logscale_smoothed_{cfg.model.net.initialization}.png")
    print(f"Plot saved to plotting/activity_hessian/ah_logscale_smoothed_{cfg.model.net.initialization}.png")

if __name__ == "__main__":
    main()