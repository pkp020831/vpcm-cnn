import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import gc # Import garbage collector

from lightning import LightningDataModule, LightningModule, seed_everything


def run_forward_pass(model: LightningModule, datamodule: LightningDataModule):
    """Runs a single forward pass and returns the activities of the last layer."""
    model.eval() # Set model to evaluation mode
    datamodule.setup(stage='test')
    x, _ = next(iter(datamodule.test_dataloader()))
    
    with torch.no_grad():
        # The SimpleDenseNet returns the final layer's output directly
        activities = model.net(x, return_all_activities=True)
        for i, act in enumerate(activities):
            print(f"Layer {i} shape: {act.shape}")
    return activities

@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment/ffwd_activity_norm.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to plot the mean L1 norm of feedforward pass activities vs. network depth."""
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    for params in cfg.analysis.param_combinations:
        plt.figure(figsize=(10, 7)) # New figure for each combination
        print(f"\n--- Processing Combination: init={params.initialization}, act={params.act_fn} ---")
        
        # Store L1 norms for different relative depths
        l1_norms_at_l1 = []
        l1_norms_at_l_quarter = []
        l1_norms_at_l_half = []
        l1_norms_at_l_three_quarter = []
        l1_norms_at_l_full = []
        
        for depth in cfg.analysis.depths:
            print(f"  Testing Depth (L): {depth}")
            
            # Calculate layer indices for this depth
            # activities list: [input, linear1_out, act1_out, linear2_out, act2_out, ..., linearL_out, actL_out]
            # So, activity after l-th activation is at index 2*l
            
            # l=1 (first hidden layer after activation)
            layer_idx_l1 = 2 # Index for the activity after the first activation
            
            # l=1/4L
            layer_idx_l_quarter = int(2 * (depth / 4))
            if layer_idx_l_quarter == 0: # Ensure it's at least the first hidden layer
                layer_idx_l_quarter = 2
            
            # l=1/2L
            layer_idx_l_half = int(2 * (depth / 2))
            if layer_idx_l_half == 0:
                layer_idx_l_half = 2
            
            # l=3/4L
            layer_idx_l_three_quarter = int(2 * (3 * depth / 4))
            if layer_idx_l_three_quarter == 0:
                layer_idx_l_three_quarter = 2
            
            # l=L (last hidden layer after activation)
            layer_idx_l_full = 2 * depth
            
            # Ensure indices do not exceed the maximum possible index for this depth
            # Max index for activities list is 2*depth (for actL_out)
            max_activity_idx = 2 * depth
            layer_idx_l1 = min(layer_idx_l1, max_activity_idx)
            layer_idx_l_quarter = min(layer_idx_l_quarter, max_activity_idx)
            layer_idx_l_half = min(layer_idx_l_half, max_activity_idx)
            layer_idx_l_three_quarter = min(layer_idx_l_three_quarter, max_activity_idx)
            layer_idx_l_full = min(layer_idx_l_full, max_activity_idx)

            # Collect norms for current depth across seeds
            norms_for_current_depth = {
                "l1": [], "l_quarter": [], "l_half": [], "l_three_quarter": [], "l_full": []
            }

            for seed in range(cfg.analysis.n_seeds):
                seed_everything(cfg.seed + seed)
                
                # --- Instantiate Model ---
                net_params = {
                    "input_size": cfg.analysis.input_size,
                    "hidden_sizes": [cfg.analysis.width] * depth,
                    "output_size": cfg.analysis.output_size,
                    "initialization": params.initialization,
                    "act_fn": params.act_fn,
                }
                
                net_config = OmegaConf.create(cfg.model.net)
                net_config.update(net_params)
                
                net = hydra.utils.instantiate(net_config)
                model: LightningModule = hydra.utils.instantiate(cfg.model, net=net)

                # --- Run Forward Pass and Calculate Norm ---
                all_layer_activities = run_forward_pass(model, datamodule)

                # Explicitly delete model and clear cache to free up memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Ensure we have enough layers for the calculated indices
                if len(all_layer_activities) > layer_idx_l1:
                    norms_for_current_depth["l1"].append(torch.linalg.norm(all_layer_activities[layer_idx_l1].view(-1), ord=1).item())
                if len(all_layer_activities) > layer_idx_l_quarter:
                    norms_for_current_depth["l_quarter"].append(torch.linalg.norm(all_layer_activities[layer_idx_l_quarter].view(-1), ord=1).item())
                if len(all_layer_activities) > layer_idx_l_half:
                    norms_for_current_depth["l_half"].append(torch.linalg.norm(all_layer_activities[layer_idx_l_half].view(-1), ord=1).item())
                if len(all_layer_activities) > layer_idx_l_three_quarter:
                    norms_for_current_depth["l_three_quarter"].append(torch.linalg.norm(all_layer_activities[layer_idx_l_three_quarter].view(-1), ord=1).item())
                if len(all_layer_activities) > layer_idx_l_full:
                    norms_for_current_depth["l_full"].append(torch.linalg.norm(all_layer_activities[layer_idx_l_full].view(-1), ord=1).item())
            
            # Append mean norms for this depth
            l1_norms_at_l1.append(np.mean(norms_for_current_depth["l1"]) if norms_for_current_depth["l1"] else np.nan)
            l1_norms_at_l_quarter.append(np.mean(norms_for_current_depth["l_quarter"]) if norms_for_current_depth["l_quarter"] else np.nan)
            l1_norms_at_l_half.append(np.mean(norms_for_current_depth["l_half"]) if norms_for_current_depth["l_half"] else np.nan)
            l1_norms_at_l_three_quarter.append(np.mean(norms_for_current_depth["l_three_quarter"]) if norms_for_current_depth["l_three_quarter"] else np.nan)
            l1_norms_at_l_full.append(np.mean(norms_for_current_depth["l_full"]) if norms_for_current_depth["l_full"] else np.nan)
            
        # Plot results for the current combination
        plt.plot(cfg.analysis.depths, l1_norms_at_l1, marker='o', linestyle='-', label='l=1')
        plt.plot(cfg.analysis.depths, l1_norms_at_l_quarter, marker='x', linestyle='--', label='l=L/4')
        plt.plot(cfg.analysis.depths, l1_norms_at_l_half, marker='s', linestyle='-', label='l=L/2')
        plt.plot(cfg.analysis.depths, l1_norms_at_l_three_quarter, marker='D', linestyle='--', label='l=3L/4')
        plt.plot(cfg.analysis.depths, l1_norms_at_l_full, marker='^', linestyle='-', label='l=L')

        # --- Finalize Plot ---
        plt.xlabel("Network Depth (L)", fontsize=14)
        plt.ylabel("Mean L1 Norm of Activity", fontsize=14)
        plt.title(f"Feedforward Pass Stability (init={params.initialization}, act={params.act_fn})", fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(cfg.analysis.depths, labels=cfg.analysis.depths)
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        save_path = f"plotting/ffwd_activity_norm/{params.initialization}_{params.act_fn}.png"
        plt.savefig(save_path)
        print(f"\nPlot saved to {save_path}")

    plt.close('all') # Close all figures after saving them

if __name__ == "__main__":
    main()
