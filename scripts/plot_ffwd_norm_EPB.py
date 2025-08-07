import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import gc
import os

from lightning import LightningDataModule, LightningModule, seed_everything

def run_forward_pass(model: LightningModule, datamodule: LightningDataModule):
    """Runs a single forward pass and returns the activities of all layers."""
    model.eval()
    datamodule.setup(stage='test')
    x, _ = next(iter(datamodule.test_dataloader()))
    x = x.view(x.size(0), -1)
    
    with torch.no_grad():
        activities = model.net(x, return_all_activities=True)
    return activities

@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/ffwd_norm_epb.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to plot the mean L1 norm of feedforward pass activities vs. network depth for different strategies and initializations."""
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Define the strategies and initializations to test
    strategies_to_test = [
        {
            "name": "ProxQPStrategy",
            "config": {
                "_target_": "src.core.eqprop.strategy.ProxQPStrategy",
                "activation": {"_target_": "src.core.eqprop.activation.IdealRectifier"},
                "amp_factor": cfg.model.net.solver.strategy.get("amp_factor", 1.0),
            }
        },
        {
            "name": "NewtonStrategy",
            "config": {
                "_target_": "src.core.eqprop.strategy.NewtonStrategy",
                "activation": {"_target_": "src.core.eqprop.activation.P3OTS"} # Use SymReLU as per its default config
            }
        },
        {
            "name": "ResistiveNetworkStrategy",
            "config": {
                "_target_": "src.core.eqprop.strategy.ResistiveNetworkStrategy",
                "activation": {"_target_": "src.core.eqprop.activation.IdealRectifier"},
            }
        }
    ]
    initializations_to_test = ["orthogonal", "default"]

    # --- Loop through every combination of strategy and initialization ---
    for strategy_info in strategies_to_test:
        for init_method in initializations_to_test:
            
            # Create a new figure for each combination
            plt.figure(figsize=(12, 8))
            print(f"\n--- Processing Combination: strategy={strategy_info['name']}, init={init_method} ---")

            l1_norms_at_l1 = []
            l1_norms_at_l_quarter = []
            l1_norms_at_l_half = []
            l1_norms_at_l_three_quarter = []
            l1_norms_at_l_full = []

            for depth in cfg.analysis.depths:
                print(f"    Testing Depth (L): {depth}")
                
                layer_idx_l1 = 2
                layer_idx_l_quarter = min(2 * depth, max(2, int(2 * (depth / 4))))
                layer_idx_l_half = min(2 * depth, max(2, int(2 * (depth / 2))))
                layer_idx_l_three_quarter = min(2 * depth, max(2, int(2 * (3 * depth / 4))))
                layer_idx_l_full = 2 * depth

                norms_for_current_depth = {"l1": [], "l_quarter": [], "l_half": [], "l_three_quarter": [], "l_full": []}

                for seed in range(cfg.analysis.n_seeds):
                    seed_everything(cfg.seed + seed)
                    
                    # --- Instantiate Model with current strategy and initialization ---
                    # Start with the base solver config from the main YAML
                    solver_config = OmegaConf.create(cfg.model.net.solver)
                    
                    # Create a config for the specific strategy we are testing
                    strategy_cfg = OmegaConf.create(strategy_info["config"])
                    
                    # Add debugging parameters for ProxQPStrategy with default init
                    if strategy_info['name'] == "ProxQPStrategy" and init_method == "default":
                        strategy_cfg.verbose = True
                        strategy_cfg.max_iter = 5000 # Increase max_iter significantly
                    
                    # Override the default strategy with our specific one
                    solver_config.strategy = strategy_cfg

                    net_params = {
                        "cfg": [cfg.analysis.input_size * 2] + [cfg.analysis.width] * depth + [cfg.analysis.output_size * 2],
                        "initialization": init_method,
                        "solver": solver_config,
                        "bias": [True] * (depth + 1),
                        "layer_scale": 1 # Set layer_scale to 1.0 to prevent explosion
                    }
                    
                    net_config = OmegaConf.create(cfg.model.net)
                    net_config.update(net_params)
                    
                    net = hydra.utils.instantiate(net_config)
                    model: LightningModule = hydra.utils.instantiate(cfg.model, net=net)

                    all_layer_activities = run_forward_pass(model, datamodule)

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    if len(all_layer_activities) > layer_idx_l1:
                        activity = all_layer_activities[layer_idx_l1]
                        norms_for_current_depth["l1"].append(torch.mean(torch.abs(activity)).item())
                    if len(all_layer_activities) > layer_idx_l_quarter:
                        activity = all_layer_activities[layer_idx_l_quarter]
                        norms_for_current_depth["l_quarter"].append(torch.mean(torch.abs(activity)).item())
                    if len(all_layer_activities) > layer_idx_l_half:
                        activity = all_layer_activities[layer_idx_l_half]
                        norms_for_current_depth["l_half"].append(torch.mean(torch.abs(activity)).item())
                    if len(all_layer_activities) > layer_idx_l_three_quarter:
                        activity = all_layer_activities[layer_idx_l_three_quarter]
                        norms_for_current_depth["l_three_quarter"].append(torch.mean(torch.abs(activity)).item())
                    if len(all_layer_activities) > layer_idx_l_full:
                        activity = all_layer_activities[layer_idx_l_full]
                        norms_for_current_depth["l_full"].append(torch.mean(torch.abs(activity)).item())
                
                mean_l1 = np.mean(norms_for_current_depth["l1"]) if norms_for_current_depth["l1"] else np.nan
                mean_l_quarter = np.mean(norms_for_current_depth["l_quarter"]) if norms_for_current_depth["l_quarter"] else np.nan
                mean_l_half = np.mean(norms_for_current_depth["l_half"]) if norms_for_current_depth["l_half"] else np.nan
                mean_l_three_quarter = np.mean(norms_for_current_depth["l_three_quarter"]) if norms_for_current_depth["l_three_quarter"] else np.nan
                mean_l_full = np.mean(norms_for_current_depth["l_full"]) if norms_for_current_depth["l_full"] else np.nan

                l1_norms_at_l1.append(mean_l1)
                l1_norms_at_l_quarter.append(mean_l_quarter)
                l1_norms_at_l_half.append(mean_l_half)
                l1_norms_at_l_three_quarter.append(mean_l_three_quarter)
                l1_norms_at_l_full.append(mean_l_full)

                print(f"        Mean L1 Norms for Depth {depth}:\n" \
                      f"            l=1: {mean_l1:.4f}\n" \
                      f"            l=L/4: {mean_l_quarter:.4f}\n" \
                      f"            l=L/2: {mean_l_half:.4f}\n" \
                      f"            l=3L/4: {mean_l_three_quarter:.4f}\n" \
                      f"            l=L: {mean_l_full:.4f}")
            
            # Plot results for the current combination
            plt.plot(cfg.analysis.depths, l1_norms_at_l1, marker='o', linestyle='-', label='l=1')
            plt.plot(cfg.analysis.depths, l1_norms_at_l_quarter, marker='x', linestyle='--', label='l=L/4')
            plt.plot(cfg.analysis.depths, l1_norms_at_l_half, marker='s', linestyle='-', label='l=L/2')
            plt.plot(cfg.analysis.depths, l1_norms_at_l_three_quarter, marker='D', linestyle='--', label='l=3L/4')
            plt.plot(cfg.analysis.depths, l1_norms_at_l_full, marker='^', linestyle='-', label='l=L')

            # --- Finalize Plot for the current combination ---
            plt.xlabel("Network Depth (L)", fontsize=14)
            plt.ylabel("Mean L1 Norm of Activity", fontsize=14)
            plt.title(f"Feedforward Pass Stability (strategy={strategy_info['name']}, init={init_method}, res_scale={cfg.model.net.res_scale})", fontsize=16)
            plt.xscale('log')
            plt.yscale('log')
            plt.xticks(cfg.analysis.depths, labels=cfg.analysis.depths)
            plt.grid(True, which="both", ls="--")
            plt.legend()
            
            save_dir = f"plotting/ffwd_activity_norm/EPB_strategies/"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}{strategy_info['name']}_{init_method}.png"
            plt.savefig(save_path)
            print(f"\nPlot for {strategy_info['name']} with {init_method} init saved to {save_path}")
            plt.close()

if __name__ == "__main__":
    main()
