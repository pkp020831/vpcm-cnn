import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from omegaconf import DictConfig
from torch.autograd.functional import hessian
from copy import deepcopy
from scipy.ndimage import gaussian_filter

from lightning import LightningDataModule, LightningModule, seed_everything
from src.core.eqprop.solvers import EqPropSolver

#This Code computes the Hessian by using the energy function defined in the EqPropSolver.
def compute_hessian_metrics(solver: EqPropSolver, x: torch.Tensor, y_true: torch.Tensor, beta_val: float):
    """Computes the Hessian of the total energy function and its metrics."""

    def _energy_for_analysis(nodes, x_input, activation_fn):
        """A clearer, more robust implementation of the energy function for analysis."""
        all_zs = [x_input] + nodes
        weights = solver.strategy.W

        self_energy = sum(0.5 * torch.sum(torch.pow(z, 2)) for z in all_zs)

        interaction_energy = 0.0
        for l_idx_formula in range(1, len(all_zs)): # l_idx_formula corresponds to 'l' in the formula
            z_l = all_zs[l_idx_formula] # This is z_l
            z_l_minus_1 = all_zs[l_idx_formula - 1] # This is z_{l-1}
            W_l = weights[l_idx_formula - 1] # This is W_l (W_1 for l=1, W_2 for l=2, etc.)

            # Apply activation function to activities before calculating interaction
            # This is crucial for non-linear networks to match the energy function definition.
            # Check if activation_fn is a callable (e.g., a torch.nn.Module instance)
            if callable(activation_fn) and not isinstance(activation_fn, torch.nn.Identity):
                activated_z_l = activation_fn(z_l)
                activated_z_l_minus_1 = activation_fn(z_l_minus_1)
            else:
                activated_z_l = z_l
                activated_z_l_minus_1 = z_l_minus_1

            # Calculate z_l^T W_l z_{l-1} for each batch element
            term_per_batch = torch.sum(torch.matmul(activated_z_l, W_l) * activated_z_l_minus_1, dim=1)
            interaction_energy -= term_per_batch.sum() # Sum over batch dimension too

        return self_energy + interaction_energy

    def energy_func_for_hessian(activities_tensor: torch.Tensor) -> torch.Tensor:
        """A wrapper that handles tensor shapes for Hessian calculation."""
        # FIX: Add a batch dimension since internal functions expect it.
        activities_tensor_batched = activities_tensor.unsqueeze(0)
        nodes = list(torch.split(activities_tensor_batched, solver.strategy.dims, dim=-1))
        
        # Pass the activation function to _energy_for_analysis
        total_e = _energy_for_analysis(nodes, x, solver.strategy.activation)
        
        if beta_val != 0 and y_true is not None:
            # One-hot encode the target label and ensure correct shape
            num_classes = nodes[-1].shape[1]
            y_one_hot = torch.nn.functional.one_hot(y_true.squeeze(), num_classes=num_classes).float()
            # Ensure y_one_hot has a batch dimension if needed for mse_loss
            if y_one_hot.dim() == 1: # If it became 1D after squeeze, add batch dim
                y_one_hot = y_one_hot.unsqueeze(0)
            loss = torch.nn.functional.mse_loss(nodes[-1], y_one_hot)
            total_e += beta_val * loss
            
        return total_e.sum()

    layer_dims = solver.strategy.dims
    # FIX: Squeeze batch dimension to create a 1D tensor for hessian calculation.
    initial_activities = torch.cat([torch.zeros(x.shape[0], dim) for dim in layer_dims], dim=-1).squeeze(0)
    initial_activities.requires_grad_(True)

    hessian_matrix = hessian(energy_func_for_hessian, initial_activities)

    eigenvalues = torch.linalg.eigvalsh(hessian_matrix) # Calculate eigenvalues from the Hessian matrix
    condition_number = torch.linalg.cond(hessian_matrix).item() # Use original matrix for cond_num

    return hessian_matrix, eigenvalues, condition_number

@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/plot_activity_hessian.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to run the activity hessian analysis across different network widths and depths."""
    seed_everything(cfg.seed)

    widths = [1,2,4,8,16,32,64,128]
    n_hiddens = [1,2,4,8,16,32,64,128]

    condition_numbers = np.zeros((len(widths), len(n_hiddens)))
    all_eigenvalues = {}

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    x, y_true = next(iter(datamodule.train_dataloader()))
    x = x[0].unsqueeze(0).view(x[0].size(0), -1)
    y_true = y_true[0].unsqueeze(0)

    beta_val = cfg.analysis.solver.beta

    for i, width in enumerate(widths):
        for j, n_hidden in enumerate(n_hiddens):
            print(f"\n--- Processing N={width}, H={n_hidden} ---")

            net_params = {
                "cfg": [784] + [width] * n_hidden + [10], # EqPropBackbone uses 'cfg' for layer sizes
                "initialization": cfg.model.net.initialization,
            }
            
            net = hydra.utils.instantiate(
                DictConfig({
                    "_target_": cfg.model.net._target_,
                    "cfg": net_params["cfg"],
                    "initialization": cfg.model.net.initialization,
                    "bias": cfg.model.net.bias
                })
            )
            model: LightningModule = hydra.utils.instantiate(cfg.model, net=net)

            strategy_partial = hydra.utils.instantiate(cfg.analysis.solver_strategy)
            strategy = strategy_partial()
            solver: EqPropSolver = hydra.utils.instantiate(cfg.analysis.solver, strategy=strategy)
            solver.set_model(model.net)

            try:
                _, eigenvalues, cond_num = compute_hessian_metrics(solver, x, y_true, beta_val)
                condition_numbers[i, j] = cond_num
                all_eigenvalues[(width, n_hidden)] = eigenvalues.detach().numpy()
                print(f"Calculated Condition Number: {cond_num}")
            except Exception as e:
                print(f"Error calculating Hessian or metrics for N={width}, H={n_hidden}: {e}")
                condition_numbers[i, j] = np.nan
                all_eigenvalues[(width, n_hidden)] = np.array([np.nan])

    # Apply Gaussian filter for smoothing
    sigma = 1.0 # Adjust sigma for desired smoothing amount
    smoothed_condition_numbers = gaussian_filter(condition_numbers, sigma=sigma)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create meshgrid for pcolormesh
    H_mesh, W_mesh = np.meshgrid(n_hiddens, widths)

    # Use pcolormesh for non-uniform coordinates
    mesh = ax.pcolormesh(
        H_mesh, 
        W_mesh, 
        smoothed_condition_numbers,
        cmap="viridis",
        norm=mcolors.LogNorm(),
        shading='gouraud'
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
    
    serializable_eigenvalues = {str(k): v for k, v in all_eigenvalues.items() if not np.isnan(v).any()}
    np.savez("plotting/activity_hessian/activity_hessian_eigenvalues.npz", **serializable_eigenvalues)
    print("Eigenvalues saved to plotting/activity_hessian/activity_hessian_eigenvalues.npz")

if __name__ == "__main__":
    main()
