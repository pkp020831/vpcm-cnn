from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.eqprop.strategy import AbstractStrategy
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class EqPropSolver:
    """Solve for the equilibrium point of the network.

    Args:
        strategy (AbstractStrategy|str): strategy to solve for the equilibrium point of the network.
        activation (Callable): activation function.
        amp_factor (float, optional): inter-layer potential amplifying factor. Defaults to 1.0.
        beta (float, optional): nudging factor. Defaults to 0.0.
    """

    # # singletons
    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, "_instance"):
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(
        self,
        amp_factor: float,
        beta: float,
        strategy: AbstractStrategy,
    ) -> None:
        self.amp_factor = amp_factor
        self.beta = beta
        self.strategy = strategy

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value: float | int):
        self._beta = value

    def flip_beta(self) -> None:
        """Flip the sign of beta."""
        self.beta = -self.beta

    def set_model(self, model: nn.Module) -> None:
        """Set strategy parameters from nn.Module.
        The strategy will store references to the parameters of the model.
        This method CLEARS existing parameters in the strategy before adding new ones.
        If `model` is an `EqPropSequential` (or similar container), it will iterate
        through its children that are `_EqPropMixin` instances. Otherwise, it iterates
        `model.named_parameters()`.
        """
        st = self.strategy
        st.W = []
        st.B = []
        st.dims = []

        is_eqprop_mixin_container = (
            hasattr(model, "IS_CONTAINER")
            and model.IS_CONTAINER
            and hasattr(model, "_eq_layers")
            and callable(getattr(model, "_eq_layers", None))
        )

        if is_eqprop_mixin_container:
            log.debug(f"Setting model for EqPropSequential type: {type(model)}")
            for i, layer in enumerate(model._eq_layers):  # type: ignore
                log.debug(f"  Processing child layer {i}: {type(layer)}")
                # Ensure child layer is an nn.Module to access named_parameters or specific attributes
                if isinstance(layer, nn.Module):
                    # Prefer specific weight/bias attributes if known, else iterate named_parameters
                    # This assumes _eq_layers returns the actual nn.Module instances
                    found_weight_for_layer = False
                    if hasattr(layer, "weight") and isinstance(layer.weight, nn.Parameter):
                        st.W.append(layer.weight)
                        st.dims.append(layer.weight.shape[0])
                        found_weight_for_layer = True
                    if (
                        hasattr(layer, "bias")
                        and layer.bias is not None
                        and isinstance(layer.bias, nn.Parameter)
                    ):
                        st.B.append(layer.bias)

                    # Fallback or additional parameters if the layer isn't a simple Linear/Conv
                    if not found_weight_for_layer:
                        # This part might need refinement based on how complex layers are structured
                        # For now, we assume primary weights are captured by 'layer.weight'
                        pass
                else:
                    log.warning(
                        f"Child {layer} in _eq_layers is not an nn.Module, skipping for set_model."
                    )

        else:  # Standalone _EqPropMixin layer or other nn.Module
            log.debug(f"Setting model for standalone EqProp layer type: {type(model)}")
            for name, param in model.named_parameters():
                if name.endswith("weight"):  # Be more specific if layers have multiple 'weights'
                    st.W.append(param)
                    st.dims.append(param.shape[0])
                elif name.endswith("bias"):
                    st.B.append(param)
        log.debug(f"Solver strategy configured with {len(st.W)} weights and {len(st.B)} biases.")

    def __call__(
        self,
        x: torch.Tensor,
        grad: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Call the solver.

        Args:
            x (torch.Tensor): input of the network.
            nudge_phase (bool, optional): Defaults to False.
            return_energy (bool, optional): Defaults to False.

        Returns:
            tuple[torch.Tensor, ...]: Tuple of tensors split by layer dimensions
        """
        i_ext = None

        if grad is not None:
            i_ext = self.beta * grad
            log.debug(f"i_ext: {i_ext.abs().mean():.3e}")
        else:
            self.strategy.reset()
        nodes = self.strategy.solve(x, i_ext, **kwargs)
        split_tensors = torch.split(nodes, self.strategy.dims, dim=-1)
        return split_tensors

    def energy(self, Nodes, x) -> torch.Tensor:
        """Energy function."""
        it = len(Nodes)
        act = self.strategy.activation

        def layer_energy(n: torch.Tensor, w: nn.Module, m: torch.Tensor):
            """Energy function for a layer.

            Args:
                n (torch.Tensor):B x I
                w (torch.nn.Linear): O x I
                m (torch.Tensor): B x O

            Returns:
                E_layer = E_nodes - E_weights - E_biases
            """
            nodes_energy = 0.5 * torch.sum(torch.pow(n, 2), dim=1)
            weights_energy = 0.5 * (torch.matmul(act(m), w.weight) * act(n)).sum(dim=1)
            biases_energy = torch.matmul(act(m), w.bias) if getattr(w, "bias") is not None else 0.0
            return nodes_energy - weights_energy - biases_energy

        for idx in range(it):
            if idx == 0:
                E = layer_energy(x, self.W[idx], Nodes[idx])
            else:
                E += layer_energy(Nodes[idx - 1], self.W[idx], Nodes[idx])
        E += 0.5 * torch.sum(torch.pow(Nodes[-1], 2), dim=1)  # add E_nodes of output layer
        return E

    def total_energy(self, Nodes, x, y, beta) -> torch.Tensor:
        """Compute Total Free Energy: Wsum rho(u_i)W_{ij}rho(u_j)"""
        E = self.energy(Nodes, x)
        L = None
        if beta != 0:
            assert y is not None, ValueError("y must be provided if beta != 0")
            L = self.criterion(Nodes[-1], y)
            E += beta * L


class AnalogEqPropSolver(EqPropSolver):
    """Solver for analog resistive network with EqProp."""

    def __init__(
        self,
        amp_factor: float,
        beta: float,
        strategy: AbstractStrategy,
    ) -> None:
        super().__init__(amp_factor, beta, strategy)

    # TODO: Check validity when amp_factor is not 1
    def energy(self, Nodes, x) -> torch.Tensor:
        """Energy function."""
        if self.amp_factor != 1:
            raise NotImplementedError(
                "energy function for analog EqProp is not implemented when amp_factor != 1"
            )
        num_layers = len(Nodes)
        assert num_layers == len(self.dims) - 1, ValueError(
            "number of nodes must match the number of layers"
        )

        # TODO: Add bias
        def layer_power(submodule: nn.Module, in_V: torch.Tensor, out_V: torch.Tensor):
            r"""Energy function for a layer.

            Args:
                in_V (torch.Tensor):B x I
                submodule (torch.nn.Linear): O x I
                out_V (torch.Tensor): B x O

            Returns:
                E_layer = 0.5 * \Sum{G * (n_i - m_i)^2} : B

            """
            W = submodule.weight
            in_V = in_V.unsqueeze(1)
            out_V = out_V.unsqueeze(2)
            return (
                0.5 * torch.bmm(in_V.pow(2), W).sum(dim=(1, 2))
                + torch.bmm(W, out_V.pow(2)).sum(dim=(1, 2))
                - 2 * (in_V @ W @ out_V).squeeze()
            )

        for idx in range(num_layers):
            if idx == 0:
                E = layer_power(x, self.W[idx], Nodes[idx]) + self.activation.p(Nodes[idx])
            elif idx != num_layers - 1:
                E += layer_power(
                    self.amp_factor * (Nodes[idx - 1]), self.W[idx], Nodes[idx]
                ) + self.activation.p(Nodes[idx])
            else:
                E += layer_power(self.amp_factor(Nodes[idx - 1]), self.W[idx], Nodes[idx])
        return E
