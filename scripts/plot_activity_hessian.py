import hydra
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from omegaconf import DictConfig, OmegaConf

from lightning import LightningDataModule, seed_everything
from src.core.eqprop.solvers import AnalogEqPropSolver


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/plot_activity_hessian.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to run the activity hessian analysis."""
    seed_everything(cfg.seed)

    log_range = [1, 2, 4, 8, 16, 32, 64]
    widths = np.array(log_range)
    n_hiddens = np.array(log_range)

    initialization_methods = ["default", "orthogonal", "mup"]

    for init_name in initialization_methods:
        print(f"\n===== Processing Initialization: {init_name} =====")
        condition_numbers = np.zeros((len(widths), len(n_hiddens)))

        # --- Instantiate Solver (once for each init type if needed, but it's independent) ---
        strategy = hydra.utils.instantiate(cfg.analysis.solver_strategy)
        solver = AnalogEqPropSolver(
            strategy=strategy,
            amp_factor=cfg.analysis.solver.amp_factor,
            beta=cfg.analysis.solver.beta
        )

        for i, width in enumerate(widths):
            for j, n_hidden in enumerate(n_hiddens):
                print(f"--- N={width}, H={n_hidden} ---")
                
                init_config = {"name": init_name}
                if init_name == "mup":
                    init_config["width"] = width
                    init_config["depth"] = n_hidden + 1
                    init_config["variance"] = cfg.model.net.mup_variance

                net_params = {
                    "_target_": "src._eqprop.backbone.EqPropSequentialBackbone",
                    "cfg": [784] + [width] * n_hidden + [10],
                    "initialization": init_config,
                    "bias": cfg.model.net.bias,
                    "solver": solver
                }
                model_net = hydra.utils.instantiate(net_params)

                # --- Compute Hessian ---
                solver.set_model(model_net)
                W_matrices = solver.strategy.W
                layer_dims = [W_matrices[0].shape[1]] + [w.shape[0] for w in W_matrices]
                total_dim = sum(layer_dims)
                hessian = torch.zeros((total_dim, total_dim))

                current_dim_i = 0
                for k, dim_i in enumerate(layer_dims):
                    if k == len(layer_dims) - 1:
                        hessian[current_dim_i:current_dim_i + dim_i, current_dim_i:current_dim_i + dim_i] = torch.eye(dim_i) * (1 + solver.beta)
                    else:
                        hessian[current_dim_i:current_dim_i + dim_i, current_dim_i:current_dim_i + dim_i] = torch.eye(dim_i)

                    if k < len(W_matrices):
                        W = W_matrices[k]
                        current_dim_j = current_dim_i + dim_i
                        hessian[current_dim_i:current_dim_i + dim_i, current_dim_j:current_dim_j + W.shape[0]] = -W.t()
                        hessian[current_dim_j:current_dim_j + W.shape[0], current_dim_i:current_dim_i + dim_i] = -W
                    current_dim_i += dim_i

                try:
                    cond_num = torch.linalg.cond(hessian).item()
                except Exception as e:
                    print(f"Error calculating condition number: {e}")
                    cond_num = np.nan
                
                print(f"    Condition Number: {cond_num}") # DEBUGGING OUTPUT
                condition_numbers[i, j] = cond_num

        # --- Fill NaN values by interpolating from neighbors ---
        nan_indices = np.argwhere(np.isnan(condition_numbers))
        if nan_indices.size > 0:
            print("\nFilling NaN values...")
            for i_nan, j_nan in nan_indices:
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0: continue
                        ni, nj = i_nan + di, j_nan + dj
                        if 0 <= ni < condition_numbers.shape[0] and 0 <= nj < condition_numbers.shape[1]:
                            neighbor_val = condition_numbers[ni, nj]
                            if not np.isnan(neighbor_val):
                                neighbors.append(neighbor_val)
                if neighbors:
                    mean_val = np.mean(neighbors)
                    condition_numbers[i_nan, j_nan] = mean_val
                    print(f"  - Filled NaN at (N={widths[i_nan]}, H={n_hiddens[j_nan]}) with interpolated value: {mean_val:.2f}")

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # To handle log scale with pcolormesh, we define the edges of the pixels.
        H_edges = np.sqrt(n_hiddens[:-1] * n_hiddens[1:])
        H_edges = np.insert(H_edges, 0, n_hiddens[0] / np.sqrt(n_hiddens[1]/n_hiddens[0]))
        H_edges = np.append(H_edges, n_hiddens[-1] * np.sqrt(n_hiddens[-1]/n_hiddens[-2]))

        W_edges = np.sqrt(widths[:-1] * widths[1:])
        W_edges = np.insert(W_edges, 0, widths[0] / np.sqrt(widths[1]/widths[0]))
        W_edges = np.append(W_edges, widths[-1] * np.sqrt(widths[-1]/widths[-2]))

        H_mesh, W_mesh = np.meshgrid(H_edges, W_edges)

        mesh = ax.pcolormesh(
            H_mesh, 
            W_mesh, 
            condition_numbers,
            cmap="viridis",
            norm=mcolors.LogNorm()
        )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(label=r"$\kappa(H_z)$", fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        ax.set_xlabel("Depth (H)", fontsize=20)
        ax.set_ylabel("Width (N)", fontsize=20)
        ax.set_title(f"Hessian Condition Number (init={init_name})", fontsize=22)
        
        ax.set_xticks(n_hiddens)
        ax.set_yticks(widths)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.minorticks_off()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()

        save_path = f"plotting/activity_hessian/ah_{init_name}.png"
        plt.savefig(save_path)
        print(f"\nPlot for {init_name} saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    main()
