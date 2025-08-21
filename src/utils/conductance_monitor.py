from lightning import Trainer
from lightning import LightningModule
import torch


from src.core.eqprop import nn as enn
from src.utils.pycallbacks import WandbLoggerCallback


class ConductanceMonitor(WandbLoggerCallback):
    """Monitor and log the ratio of currents between normal and shortcut paths."""

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        net = pl_module.net
        if not hasattr(net, 'model'):
            return

        # Try to get res_scale from the network attribute
        res_scale = getattr(net, 'res_scale', 1.0)

        eqprop_layers = [m for m in net.model.modules() if isinstance(m, enn.EqPropLinear)]
        if not eqprop_layers:
            return

        layer_stats = {}
        for i, layer in enumerate(eqprop_layers):
            # The ratio of currents is equivalent to the ratio of conductances,
            # which is the scaled weight of the normal path, since the shortcut
            # conductance is 1.0.
            scaled_conductance = layer.weight * res_scale
            current_ratio = scaled_conductance.abs().mean().item()
            layer_stats[f"layer_{i}/current_ratio"] = current_ratio

            # For completeness, also log the unscaled conductance
            conductance = layer.weight.abs().mean().item()
            layer_stats[f"layer_{i}/conductance"] = conductance

        if layer_stats:
            if self.logger and hasattr(self.logger, 'log_metrics'):
                self.logger.log_metrics(layer_stats, step=trainer.current_epoch)