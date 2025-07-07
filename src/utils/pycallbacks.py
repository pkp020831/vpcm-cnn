from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping

import lightning as L
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger

from src._eqprop.direct_backbone import AnalogEP2
from src.utils.eqprop_utils import deltaV

LOGGER_TYPE = {"tb": TensorBoardLogger, "wandb": WandbLogger}


class AdvLoggerCallback(Callback):
    def __init__(
        self,
        on_step: bool = False,
        on_epoch: bool = True,
        log_options: dict[str, bool] = {
            "histogram": True,
            "scalar": True,
            "Vdrops": True,
            "minimize": True,
        },
    ):
        super().__init__()
        self.on_step = on_step
        self.on_epoch = on_epoch if not on_step else False
        assert self.on_step or self.on_epoch is True, UserWarning(
            "both on_step and on_epoch are False. Nothing to log"
        )
        self.log_optn = log_options
        self.logger = None

    # def on_init_start(self, trainer: Trainer) -> None:
    #     # return super().on_init_start(trainer)
    #     # further setup required if multiple loggers are used
    #     trainer.logger = self.logger

    def on_batch_end(self, trainer: Trainer, pl_module: L.LightningModule):
        """Log at the batch end."""
        (
            self.log_everything(pl_module.net, step=trainer.global_step, key_suffix="_batch")
            if self.on_step
            else None
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        """Log at the end of the training epoch."""
        self.log_everything(pl_module.net, step=trainer.current_epoch, key_suffix="_epoch")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        """Log at the end of the validation epoch."""
        self.log_weights(pl_module.named_parameters(), step=trainer.current_epoch)

    # logging functions for debugging
    @abstractmethod
    def log_histogram(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        """Log the histogram of the data."""
        pass

    @abstractmethod
    def log_scalars(
        self,
        scalars: Mapping,
        layer_name: str = None,
        step: int = None,
        key_suffix: str = "",
        **kwargs,
    ):
        """Log the scalars."""
        pass

    @abstractmethod
    def log_plot(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        """Log the plot."""
        pass

    @abstractmethod
    def log_image(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        """Log the image."""
        pass

    # logger independent functions

    def log_Vdrops(
        self,
        Vdrops: Iterable,
        step: int = None,
        key_prefix: str = "",
        key_suffix: str = None,
        **kwargs,
    ):
        """Log the Voltage drops.

        Args:
            Vdrops (Iterable): _description_
            step (int, optional): _description_. Defaults to None.
            key_prefix (str, optional): _description_. Defaults to "".
            key_suffix (str, optional): _description_. Defaults to None.
        """
        if Vdrops is not None:
            for idx, value in enumerate(Vdrops):
                self.log_histogram(f"Vdrop_{key_prefix}_{idx}", value, step, key_suffix, **kwargs)

    # TODO: add log metrics
    def log_everything(self, net: AnalogEP2, key_suffix: str, step: int = None, **kwargs):
        if self.log_optn["Vdrops"]:
            voltages: Iterator = net.named_buffers()
            for key, value in voltages:
                if "free" in key:
                    value_prev = value
                    continue
                imdata = deltaV(value, value_prev)
                self.log_image(key, imdata, step)
        # layerwise
        params: Iterator = net.named_parameters()
        for key, value in params:
            if self.log_optn["histogram"]:
                self.log_histogram(key, value, step, key_suffix, **kwargs)
                self.log_histogram(key + ".grad", value.grad, step, key_suffix, **kwargs)
            if self.log_optn["scalar"]:
                var, mean = torch.var_mean(value)
                var_grad, mean_grad = (
                    torch.var_mean(value.grad) if value.grad is not None else (0, 0)
                )
                layer_name = key
                metrics = {
                    layer_name + "var": var,
                    layer_name + "mean": mean,
                    layer_name + "var_grad": var_grad,
                    layer_name + "mean_grad": mean_grad,
                }
            # if len(value.shape) == 2:
            #     metrics[layer_name + "condition_number"] = torch.linalg.cond(value)
            # self.log_scalars(metrics, layer_name, step, key_suffix, **kwargs)
        # phasewise
        if self.log_optn["minimize"]:
            handler = net.metric_handler
            # for _ in self.num_phases:
            for key, val in handler.metrics.items():
                self.log_plot(key, val, step, key_suffix, **kwargs)
            handler.clear()

    def log_weights(self, Weights: Iterable[torch.Tensor], step: int = None, **kwargs):
        """Log the weights of the model.

        Args:
            Weights (Iterable[torch.Tensor]): _description_
            step (int, optional): _description_. Defaults to None.
        """
        for key, value in Weights:
            self.log_histogram(key, value, step, **kwargs)

    def log_weights_norm(self, Weights: Iterable[torch.Tensor], step: int = None, **kwargs):
        """Log the norm of the weights."""
        norms = dict()
        norms.update({key: value.norm() for key, value in Weights})
        # [norms.update({key:value.norm()}) for key, value in Weights]
        key = "Weights_norm"
        self.log_scalars(norms, key, step, **kwargs)

    def log_weights_grad(self, Weights: Iterable[torch.Tensor], step: int = None, **kwargs):
        """Log the gradients of the weights."""
        for key, value in Weights:
            key += ".grad"
            self.log_histogram(key, value.grad, step, **kwargs)

    def log_weights_grad_norm(self, Weights: Iterable[torch.Tensor], step: int = None, **kwargs):
        """Log the norm of the gradients of the weights."""
        norms = dict()
        norms.update({key: value.grad.norm() for key, value in Weights})
        # [norms.update({key:value.grad.norm()}) for key, value in Weights]
        key = "Weights_grad_norm"
        self.log_scalars(norms, key, step, **kwargs)

    def log_weights_condition_number(
        self, Weights: Iterable[torch.Tensor], step: int = None, **kwargs
    ):
        """Log the condition number of the weights.

        Args:
            Weights (Iterable[torch.Tensor]): model weights
            step (int, optional): _description_. Defaults to None.
        """
        conds = dict()
        conds.update({key: torch.linalg.cond(value) for key, value in Weights})
        # [conds.update({key:value.grad.norm()}) for key, value in Weights]
        key = "Weights_cond"
        self.log_scalars(conds, key, step, **kwargs)


class TBLoggerCallback(AdvLoggerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_sanity_check_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        """Find logger at the start of the sanity check."""
        if type(pl_module.logger) is TensorBoardLogger:
            self.logger = pl_module.logger.experiment
        else:
            raise ValueError("no tensorboard logger found")

    def on_train_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        """Find logger at the start of the training.

        This method is required for fast_dev_run.
        """
        self.logger = pl_module.logger.experiment if self.logger is None else self.logger
        self.log_weights(pl_module.named_parameters(), step=-1)

    def log_histogram(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        """Log the histogram of the data."""
        key += key_suffix
        self.logger.add_histogram(key, data, step, **kwargs)

    def log_scalars(
        self,
        scalars: Mapping,
        layer_name: str = None,
        step: int = None,
        key_suffix: str = "",
        **kwargs,
    ):
        """Log the scalars."""
        for key, value in scalars.items():  # l
            self.logger.add_scalars(layer_name + key_suffix, {key: value}, step, **kwargs)

    def log_plot(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        """Log the plot."""
        raise NotImplementedError


# import matplotlib.pyplot as plt
import wandb


class WandbLoggerCallback(AdvLoggerCallback):
    # log directly from wandb

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_optn["histogram"] = False
        project = kwargs.get("project", None)
        # wandb.init(project=project)

    def on_sanity_check_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        """Find wandb logger at the start of the sanity check."""
        if type(pl_module.logger) is WandbLogger:
            self.logger = pl_module.logger
        else:
            raise ValueError("W&B logger not found")

    def on_train_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        """Find wandb logger at the start of the training.

        This method is required for fast_dev_run.
        """
        # assert type(pl_module.logger) is WandbLogger, "W&B logger not found"
        self.logger = WandbLogger() if self.logger is None else self.logger
        self.logger.watch(pl_module, log="all", log_graph=False)

    def log_histogram(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        """Log the histogram of the data."""
        key += key_suffix
        wandb.log({key: wandb.Histogram(data.cpu())}, **kwargs)

    def log_scalars(
        self,
        scalars: Mapping,
        layer_name: str = None,
        step: int = None,
        key_suffix: str = "",
        **kwargs,
    ):
        """Log the scalars."""
        if key_suffix != "":
            stepsize = key_suffix.strip("_")
            scalars[stepsize] = step
        wandb.log(scalars, **kwargs)

    def log_image(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        """Log the image."""
        key += key_suffix
        wandb.log({key: wandb.Image(data)})

    # def log_plot(
    #     self,
    #     key,
    #     data: Iterable[torch.Tensor],
    #     step: int = None,
    #     key_suffix: str = "",
    #     **kwargs,
    # ):
    #     plt.plot(data)
    #     plt.ylabel(key)
    #     wandb.log({key: plt}, **kwargs)
    # if kwargs.get('norm', False):
    #     data = torch.tensor(data)
    #     max = data.max().item()
    #     min = data.min().item()
    #     data = [(d.item() - min)/max for d in data]
    # zippeddata = [(idx, datum) for idx, datum in enumerate(data)]
    # table = wandb.Table(data = zippeddata, columns = ["x", "y"])
    # wandb.log({key: wandb.plot.line(table, x='step', y=f'{key}_{step}')}, **kwargs)

    # def log_everything(self, pl_module: L.LightningModule, key_suffix: str, step: int = None, **kwargs):
    #     #log vdrops additionally
    #     # if self.log_optn['Vdrops']:
    #     #     self.log_Vdrops(pl_module.fdV, step, key_prefix='free',  key_suffix=key_suffix, **kwargs)
    #     #     self.log_Vdrops(pl_module.ndV, step, key_prefix='nudge', key_suffix=key_suffix, **kwargs)
