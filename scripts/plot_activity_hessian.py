import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from omegaconf import DictConfig

from lightning import LightningDataModule, LightningModule, seed_everything
from src.core.eqprop.solvers import EqPropSolver


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/plot_activity_hessian.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to run the activity hessian analysis across different network widths and depths."""
    # Set seed for reproducibility
    seed_everything(cfg.seed)

    # Define ranges for width and number of hidden layers
    widths = list(range(1, 33))
    n_hiddens = list(range(1, 33))

    # Initialize a 2D array to store condition numbers
    condition_numbers = np.zeros((len(n_hiddens), len(widths)))

    # --- Instantiate Data (once) ---
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    x, y_true = next(iter(datamodule.train_dataloader())) # Get x and y_true
    # Use a single data point and flatten it
    x = x[0].unsqueeze(0).view(x[0].size(0), -1)
    y_true = y_true[0].unsqueeze(0) # Use corresponding y_true

    # Get beta from config
    beta_val = cfg.analysis.solver.beta

    for i, n_hidden in enumerate(n_hiddens):
        for j, width in enumerate(widths):

            print(f"\n--- Processing H={n_hidden}, N={width} ---")

            # Dynamically create network architecture parameters
            # Now pass hidden_sizes as a list
            net_params = {
                "input_size": cfg.model.net.input_size,
                "output_size": cfg.model.net.output_size,
                "hidden_sizes": [width] * n_hidden, # Create a list of hidden layer sizes
            }
            
            # Instantiate model with dynamic architecture
            # Create a new DictConfig for the network parameters
            net_config = DictConfig({"_target_": cfg.model.net._target_})
            for key, value in net_params.items():
                net_config[key] = value
            
            # Instantiate the network directly, then wrap it in MNISTLitModule
            net = hydra.utils.instantiate(net_config)
            model: LightningModule = hydra.utils.instantiate(cfg.model, net=net)

            # --- Instantiate Solver ---
            strategy_partial = hydra.utils.instantiate(cfg.analysis.solver_strategy)
            strategy = strategy_partial()
            solver_partial = hydra.utils.instantiate(cfg.analysis.solver)
            solver: EqPropSolver = solver_partial(strategy=strategy)
            solver.set_model(model.net)

            # --- Construct Hessian based on Eq. (28) ---
            W_matrices = solver.strategy.W
            for k_w, W_mat in enumerate(W_matrices):
                print(f"  Weight matrix W_{k_w+1} norm: {torch.linalg.norm(W_mat).item()}")
            
            # The dimensions of the activity vectors z_0, z_1, ..., z_L
            # z_0 = x (input), z_1...z_{L-1} = hidden, z_L = output
            layer_dims = [W_matrices[0].shape[1]] + [w.shape[0] for w in W_matrices]
            total_dim = sum(layer_dims)
            hessian = torch.zeros((total_dim, total_dim), device=x.device)

            current_dim_i = 0
            # Iterate over activity layers z_0, z_1, ...
            for k, dim_i in enumerate(layer_dims):
                # Diagonal blocks: I or I + beta
                if k == len(layer_dims) - 1:  # Last layer z_L
                    hessian[current_dim_i:current_dim_i+dim_i, current_dim_i:current_dim_i+dim_i] = torch.eye(dim_i, device=x.device) * (1 + beta_val)
                else:
                    hessian[current_dim_i:current_dim_i+dim_i, current_dim_i:current_dim_i+dim_i] = torch.eye(dim_i, device=x.device)

                # Off-diagonal blocks: -W_{k+1} and -W_{k+1}^T
                # This connects z_k and z_{k+1}
                if k < len(W_matrices):
                    W = W_matrices[k]  # This is W_{k+1}, shape (out, in) = (dim_{k+1}, dim_k)
                    dim_j = layer_dims[k+1]
                    current_dim_j = current_dim_i + dim_i
                    
                    # Block (k, k+1) should be -W_{k+1}^T
                    hessian[current_dim_i:current_dim_i+dim_i, current_dim_j:current_dim_j+dim_j] = -W.t()
                    # Block (k+1, k) should be -W_{k+1}
                    hessian[current_dim_j:current_dim_j+dim_j, current_dim_i:current_dim_i+dim_i] = -W

                current_dim_i += dim_i

            # --- Calculate Condition Number ---
            try:
                cond_num = torch.linalg.cond(hessian).item()
                condition_numbers[i, j] = cond_num
                print(f"Calculated Condition Number: {cond_num}")
            except Exception as e:
                print(f"Error calculating condition number: {e}")
                condition_numbers[i, j] = np.nan # Mark as NaN if calculation fails

    # --- Fill NaN values by interpolating from neighbors ---
    nan_indices = np.argwhere(np.isnan(condition_numbers))
    for i, j in nan_indices:
        neighbors = []
        # Check 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                # Check bounds
                if 0 <= ni < condition_numbers.shape[0] and 0 <= nj < condition_numbers.shape[1]:
                    neighbor_val = condition_numbers[ni, nj]
                    if not np.isnan(neighbor_val):
                        neighbors.append(neighbor_val)
        
        if neighbors:
            mean_val = np.mean(neighbors)
            condition_numbers[i, j] = mean_val
            print(f"Filled NaN at ({i}, {j}) with interpolated value: {mean_val}")

    # --- Plotting ---
    plt.figure(figsize=(12, 12))
    plt.imshow(
        condition_numbers,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        norm=mcolors.LogNorm(),
        extent=[widths[0] - 0.5, widths[-1] + 0.5, n_hiddens[0] - 0.5, n_hiddens[-1] + 0.5]
    )
    cbar = plt.colorbar(label=r"$\kappa(H_z)$")
    cbar.set_label(label=r"$\kappa(H_z)$", fontsize=20)
    plt.xlabel("Width (N)", fontsize=20)
    plt.ylabel("Number of Hidden Layers (H)", fontsize=20)
    log_ticks = [2**i for i in range(1, 6)] # 2, 4, 8, 16, 32
    plt.xticks(log_ticks, labels=[f"$2^{{{int(np.log2(w))}}}$" for w in log_ticks])
    plt.yticks(log_ticks, labels=[f"$2^{{{int(np.log2(h))}}}$" for h in log_ticks])
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.savefig("activity_hessian_hn_heatmap.png")
    print("\nPlot saved to activity_hessian_hn_heatmap.png")

if __name__ == "__main__":
    main()