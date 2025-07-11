from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

from src.core.eqprop.solvers import EqPropSolver
from src.core.eqprop.functions import PositiveEqPropFunc, AlteredEqPropFunc, CenteredEqPropFunc
from src.utils import eqprop_utils, RankedLogger

__all__ = [
    "EqPropSolverManager",
    "EqPropLinear",
    "EqPropConv2d",
    "EqPropSequential",
    "PositiveEqPropFunc",
    "AlteredEqPropFunc",
    "CenteredEqPropFunc",
    "_EqPropMixin",
]

log = RankedLogger(__name__)


class EqPropSolverManager:
    """Manager for EqProp solver and layer registration.

    This class manages the registration of EqProp layers and automatically
    configures the solver when needed. Uses the Adapter pattern to present
    registered layers to the solver as a virtual container.
    """

    def __init__(self, solver: EqPropSolver):
        """Initialize with a solver instance.

        Args:
            solver: EqProp solver to manage
        """
        self._solver = solver
        self._registered_layers: list[_EqPropMixin] = []
        self._is_configured = False

    def register_layer(self, layer: _EqPropMixin) -> None:
        """Register a layer with the manager.

        Args:
            layer: EqProp layer to register
        """
        if layer not in self._registered_layers:
            self._registered_layers.append(layer)
            self._is_configured = False
            log.debug(f"Registered layer {type(layer).__name__} with SolverManager")

    def unregister_layer(self, layer: _EqPropMixin) -> None:
        """Unregister a layer from the manager.

        Args:
            layer: EqProp layer to unregister
        """
        if layer in self._registered_layers:
            self._registered_layers.remove(layer)
            self._is_configured = False
            log.debug(f"Unregistered layer {type(layer).__name__} from SolverManager")

    def configure_solver(self) -> None:
        """Configure the solver with all registered layers.

        Creates a virtual container that adapts the registered layers
        to the interface expected by solver.set_model().
        """
        if not self._is_configured and self._registered_layers:
            virtual_container = self._create_virtual_container()
            self._solver.set_model(virtual_container)
            self._is_configured = True
            log.debug(f"Configured solver with {len(self._registered_layers)} layers")

    def get_solver(self) -> EqPropSolver:
        """Get the configured solver.

        Automatically configures the solver if not already done (lazy configuration).

        Returns:
            Configured EqProp solver
        """
        if not self._is_configured:
            self.configure_solver()
        return self._solver

    def _create_virtual_container(self):
        """Create a virtual container for solver.set_model().

        This adapter allows the SolverManager to present its registered
        layers as a container that matches the interface expected by
        the solver's set_model method.

        Returns:
            Virtual container object with IS_CONTAINER and _eq_layers
        """
        registered_layers = self._registered_layers

        class VirtualContainer:
            IS_CONTAINER = True

            @property
            def _eq_layers(self):
                return registered_layers

            def named_parameters(self):
                """Yield parameters from all registered layers."""
                for layer in registered_layers:
                    yield from layer.named_parameters()

        return VirtualContainer()


class _EqPropMixin(ABC):
    """EqProp mixin class.

    Wraps EqProp function and solver for nn.Module. All EqProp layers
    should inherit this class and implement calc_n_set_param_grad_() and
    calc_x_grad().
    """

    IS_CONTAINER: bool = False

    def __init__(
        self,
        eqprop_fn_class: type[PositiveEqPropFunc],
        solver_manager: EqPropSolverManager | None = None,
        solver: EqPropSolver | None = None,
    ) -> None:
        """Initialize EqPropMixin.

        Args:
            eqprop_fn_class: Specific EqProp algorithm class
            solver_manager: Optional solver manager (can be set later)
            solver: Optional direct solver (alternative to solver_manager)
        """
        self.eqprop_fn = eqprop_fn_class.apply
        self._solver_manager = solver_manager
        self._direct_solver = solver

        # Auto-register with solver manager if provided
        if self._solver_manager is not None:
            self._solver_manager.register_layer(self)
        elif self._direct_solver is not None:
            # Configure direct solver with this single layer
            self._direct_solver.set_model(self)

    def _set_solver_manager(self, solver_manager: EqPropSolverManager) -> None:
        """Set or change the solver manager.

        Args:
            solver_manager: New solver manager to use
        """
        # Unregister from old manager if exists
        if self._solver_manager is not None:
            self._solver_manager.unregister_layer(self)

        # Clear direct solver if switching to manager
        self._direct_solver = None

        # Register with new manager
        self._solver_manager = solver_manager
        if solver_manager is not None:
            solver_manager.register_layer(self)
            log.debug(f"Set SolverManager for {type(self).__name__}")

    def _set_direct_solver(self, solver: EqPropSolver) -> None:
        """Set or change the direct solver.

        Args:
            solver: New direct solver to use
        """
        # Unregister from manager if exists
        if self._solver_manager is not None:
            self._solver_manager.unregister_layer(self)
            self._solver_manager = None

        # Set direct solver and configure it
        self._direct_solver = solver
        if solver is not None:
            solver.set_model(self)
            log.debug(f"Set direct solver for {type(self).__name__}")

    @property
    def solver(self) -> EqPropSolver:
        """Get the solver from the manager or direct assignment.

        Returns:
            Configured EqProp solver

        Raises:
            RuntimeError: If no SolverManager or direct solver is set
        """
        if self._solver_manager is not None:
            return self._solver_manager.get_solver()
        elif self._direct_solver is not None:
            return self._direct_solver
        else:
            raise RuntimeError(f"No SolverManager or direct solver set for {type(self).__name__}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for EqProp."""
        x.requires_grad_()
        return self.eqprop_fn(self, x)

    @abstractmethod
    def calc_n_set_param_grad_(
        self,
        prev_nodes: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        nodes: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Calculate & set gradient in-place for params.

        This function is called by EqPropFunc.backward.

        Args:
            prev_nodes: Input to this layer - either:
                - torch.Tensor: Original input x (for first layer)
                - tuple[torch.Tensor, torch.Tensor]: (positive_prev, negative_prev) from previous layer
            nodes: (positive_node, negative_node) output from this layer
        """

    @abstractmethod
    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate & return gradient for input dy/dx.

        This function is called by EqPropFunc.backward.
        """


class EqPropLinear(_EqPropMixin, nn.Linear):
    """nn.Linear equivalent that uses Equilibrium Propagation instead of
    backpropagation.

    Utilizes EqPropMixin to wrap EqProp function and solver.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        eqprop_fn: type[PositiveEqPropFunc] = CenteredEqPropFunc,
        solver_manager: EqPropSolverManager | None = None,
        solver: EqPropSolver | None = None,
        param_init_args: dict = {"min_w": 1e-6, "max_w": None, "max_w_gain": 0.28},
    ):
        """Initialize EqPropLinear with SolverManager or direct solver.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If False, layer will not learn additive bias
            device: Device to place parameters on
            dtype: Data type for parameters
            eqprop_fn: EqProp algorithm class to use
            solver_manager: Optional solver manager (can be set later)
            solver: Optional direct solver (alternative to solver_manager)
            param_init_args: Arguments for positive parameter initialization
        """
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        _EqPropMixin.__init__(self, eqprop_fn, solver_manager, solver)
        self.apply(eqprop_utils.positive_param_init(**param_init_args))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        solver_manager: EqPropSolverManager | None = None,
        solver: EqPropSolver | None = None,
        eqprop_fn: type[PositiveEqPropFunc] = CenteredEqPropFunc,
    ) -> EqPropLinear:
        """Create EqPropLinear from existing nn.Linear.

        Args:
            linear: Source nn.Linear layer
            solver_manager: Optional solver manager
            solver: Optional direct solver
            eqprop_fn: EqProp function to use

        Returns:
            New EqPropLinear with copied parameters
        """
        mapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            eqprop_fn=eqprop_fn,
            solver_manager=solver_manager,
            solver=solver,
        )
        # Copy parameters
        mapped.weight.data.copy_(linear.weight.detach())
        if linear.bias is not None and mapped.bias is not None:
            mapped.bias.data.copy_(linear.bias.detach())
        return mapped

    def forward(self, x):
        """Forward pass for EqPropLinear."""
        return _EqPropMixin.forward(self, x)

    @torch.no_grad()
    def calc_n_set_param_grad_(
        self,
        prev_nodes: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        nodes: tuple[torch.Tensor, torch.Tensor],
    ):
        """Calculate & set gradients of parameters manually.

        dy/dw = (nudge_dV^2 - free_dV^2)/beta \n
        = [prev_negative^2 + n_node^2 \n
        \\- (prev_positive^2 + p_node^2) \n
        \\- 2(prev_negative.T@n_node - prev_positive@p_node)]/beta \n
        = [2(p_node@prev_pos - n_node@prev_neg) \n
        \\+ prev_neg^2 - prev_pos^2 \n
        \\+ n_node^2 - p_node^2]/beta

        Args:
            prev_nodes: Input to this layer - either:
                - torch.Tensor: Original input x (for first layer)
                - tuple[torch.Tensor, torch.Tensor]: (positive_prev, negative_prev) from previous layer
            nodes: (positive_node, negative_node) output from this layer
        """
        beta = self.solver.beta
        positive_node, negative_node = nodes

        if isinstance(prev_nodes, torch.Tensor):
            # First layer: optimized path for x (prev_positive = prev_negative = x)
            x = prev_nodes
            # Weight gradient
            dw = 2 * (
                torch.bmm(positive_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
                - torch.bmm(negative_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
            )
            # x.pow(2) - x.pow(2) = 0, so skip prev_nodes^2 terms
            dw += (negative_node.pow(2).mean(0) - positive_node.pow(2).mean(0)).unsqueeze(1)
        else:
            # Subsequent layers: full computation with different prev_positive and prev_negative
            prev_positive, prev_negative = prev_nodes
            # Weight gradient
            dw = 2 * (
                torch.bmm(positive_node.unsqueeze(2), prev_positive.unsqueeze(1)).mean(0)
                - torch.bmm(negative_node.unsqueeze(2), prev_negative.unsqueeze(1)).mean(0)
            )
            # Add prev_nodes^2 terms (non-zero when prev_positive != prev_negative)
            dw += (prev_negative.pow(2).mean(0) - prev_positive.pow(2).mean(0)).unsqueeze(0)
            # Add current nodes^2 terms
            dw += (negative_node.pow(2).mean(0) - positive_node.pow(2).mean(0)).unsqueeze(1)

        dw /= beta
        self.weight.grad = dw if self.weight.grad is None else self.weight.grad + dw

        # Bias gradient (same for both cases)
        if self.bias is not None:
            db = ((negative_node - positive_node) * (negative_node + positive_node - 2)).mean(
                0
            ) / beta
            self.bias.grad = db if self.bias.grad is None else self.bias.grad + db

    @torch.no_grad()
    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate & return gradient for input dy/dx. dy/dx = g*(dV_nudge -dV_free)/beta.

        Args:
            nodes (tuple[torch.Tensor, torch.Tensor]): positive and negative nodes
        """
        positive_node, negative_node = nodes
        return ((positive_node - negative_node) / self.solver.beta) @ self.weight


class EqPropConv2d(_EqPropMixin, nn.LazyConv2d):
    def __init__(
        self,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] = (1, 1),
        padding: tuple[int, ...] | str = (0, 0),
        dilation: tuple[int, ...] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        eqprop_fn: type[PositiveEqPropFunc] = PositiveEqPropFunc,
        device=None,
        dtype=None,
        solver_manager: EqPropSolverManager | None = None,
        solver: EqPropSolver | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.LazyConv2d.__init__(
            self,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )
        _EqPropMixin.__init__(self, eqprop_fn, solver_manager, solver)
        self.unfold = nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
        )

    @classmethod
    def from_conv2d(
        cls,
        conv2d: nn.Conv2d,
        solver_manager: EqPropSolverManager | None = None,
        eqprop_fn: type[PositiveEqPropFunc] = PositiveEqPropFunc,
    ) -> EqPropConv2d:
        """Create EqPropConv2d from existing nn.Conv2d.

        Args:
            conv2d: Source nn.Conv2d layer
            solver_manager: Optional solver manager
            eqprop_fn: EqProp function to use

        Returns:
            New EqPropConv2d with copied parameters
        """
        mapped = cls(
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
            padding_mode=conv2d.padding_mode,
            device=conv2d.weight.device,
            dtype=conv2d.weight.dtype,
            eqprop_fn=eqprop_fn,
            solver_manager=solver_manager,
        )
        # Copy parameters if possible
        if hasattr(conv2d, "weight") and conv2d.weight is not None:
            if hasattr(mapped, "weight") and mapped.weight is not None:
                try:
                    mapped.weight.data.copy_(conv2d.weight.detach())
                except Exception as e:
                    log.warning(f"Could not copy weight to EqPropConv2d: {e}")
        if hasattr(conv2d, "bias") and conv2d.bias is not None:
            if hasattr(mapped, "bias") and mapped.bias is not None:
                try:
                    mapped.bias.data.copy_(conv2d.bias.detach())
                except Exception as e:
                    log.warning(f"Could not copy bias to EqPropConv2d: {e}")
        return mapped

    def forward(self, x):
        """Forward pass for EqPropConv2d.

        Equivalent to Unfold input tensor and apply to EqPropLinear.
        """
        self.initialize_parameters(x)
        x = self.unfold(x).transpose(1, 2)
        x = _EqPropMixin.forward(self, x)
        batch, out_channels = x.shape
        x = x.view(batch, out_channels, *self.output_shape)
        return x

    # TODO: verify that this is correct
    @torch.no_grad()
    def calc_n_set_param_grad_(
        self,
        prev_nodes: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        nodes: tuple[torch.Tensor, torch.Tensor],
    ):
        """Calculate & set gradients of parameters manually for Conv2d.

        Similar to EqPropLinear but handles the unfolded structure.
        dy/dw = (nudge_dV^2 - free_dV^2)/beta

        Args:
            prev_nodes: Input to this layer - either:
                - torch.Tensor: Original input x (for first layer)
                - tuple[torch.Tensor, torch.Tensor]: (positive_prev, negative_prev) from previous layer
            nodes: (positive_node, negative_node) output from this layer
        """
        beta = self.solver.beta
        positive_node, negative_node = nodes

        if isinstance(prev_nodes, torch.Tensor):
            # First layer: optimized path for x (prev_positive = prev_negative = x)
            x = prev_nodes
            # Weight gradient
            dw = 2 * (
                torch.bmm(positive_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
                - torch.bmm(negative_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
            )
            # x.pow(2) - x.pow(2) = 0, so skip prev_nodes^2 terms
            dw += (negative_node.pow(2).mean(0) - positive_node.pow(2).mean(0)).unsqueeze(1)
        else:
            # Subsequent layers: full computation with different prev_positive and prev_negative
            prev_positive, prev_negative = prev_nodes
            # Weight gradient
            dw = 2 * (
                torch.bmm(positive_node.unsqueeze(2), prev_positive.unsqueeze(1)).mean(0)
                - torch.bmm(negative_node.unsqueeze(2), prev_negative.unsqueeze(1)).mean(0)
            )
            # Add prev_nodes^2 terms (non-zero when prev_positive != prev_negative)
            dw += (prev_negative.pow(2).mean(0) - prev_positive.pow(2).mean(0)).unsqueeze(1)
            # Add current nodes^2 terms
            dw += (negative_node.pow(2).mean(0) - positive_node.pow(2).mean(0)).unsqueeze(1)

        dw /= beta

        # Reshape to match conv weight shape
        dw = dw.view(self.weight.shape)
        self.weight.grad = dw if self.weight.grad is None else self.weight.grad + dw

        # Bias gradient (same for both cases)
        if self.bias is not None:
            db = ((negative_node - positive_node) * (negative_node + positive_node - 2)).mean(
                0
            ) / beta
            self.bias.grad = db if self.bias.grad is None else self.bias.grad + db

    @torch.no_grad()
    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("calculate_x_grad not implemented")


class EqPropSequential(nn.Sequential):
    """PyTorch-like Sequential container for EqProp modules.

    Uses SolverManager to handle solver sharing and registration.
    Expects modules to already be converted to EqProp format.
    """

    IS_CONTAINER = True  # For solver.set_model() interface

    def __init__(
        self,
        *modules: nn.Module,
        solver: EqPropSolver | None = None,
        eqprop_fn: Literal["positive", "altered", "centered"] = "centered",
    ) -> None:
        """Initialize EqPropSequential.

        Args:
            modules: EqProp modules to register in the container
            solver: Optional solver (creates default if None)
            eqprop_fn: EqProp function type to use
        """
        # 1. Create solver if not provided
        if solver is None:
            solver = self._create_default_solver()

        # 2. Create SolverManager
        self._solver_manager = EqPropSolverManager(solver)
        # 3. Set SolverManager for all EqProp modules
        eqprop_modules = []
        for module in modules:
            if isinstance(module, _EqPropMixin):
                # Set SolverManager for EqProp modules
                module._set_solver_manager(self._solver_manager)
                eqprop_modules.append(module)
            else:
                # Keep non-EqProp modules as-is (activations, norm layers, etc.)
                eqprop_modules.append(module)

        # 4. Initialize parent Sequential
        super().__init__(*eqprop_modules)

        # 5. Configure solver with all registered layers
        self._solver_manager.configure_solver()
        self.solver = self._solver_manager.get_solver()
        # 6. Store eqprop_fn for forward pass
        self._eqprop_fn_class = _EQPROP_FN_REG[eqprop_fn.lower()]
        self.eqprop_fn = self._eqprop_fn_class.apply

        log.debug(f"Created EqPropSequential with {len(self._eq_layers)} EqProp layers")

    def _create_default_solver(self) -> EqPropSolver:
        """Create default solver.

        Returns:
            Default AnalogEqPropSolver instance
        """
        from src.core.eqprop.solvers import AnalogEqPropSolver
        from src.core.eqprop.strategy import ProxQPStrategy
        from src.core.eqprop.activation import IdealRectifier

        activation = IdealRectifier(Vl=0.1, Vr=0.9)
        strategy = ProxQPStrategy(amp_factor=1.0, activation=activation)
        return AnalogEqPropSolver(beta=0.1, strategy=strategy)

    @property
    def _eq_layers(self) -> list[_EqPropMixin]:
        """Return list of EqProp layers for solver interface."""
        return [m for m in self if isinstance(m, _EqPropMixin)]

    def _distribute_param_grads(
        self,
        x: torch.Tensor,
        pos_nodes: tuple[torch.Tensor],
        neg_nodes: tuple[torch.Tensor],
    ) -> None:
        """Distribute parameter gradients across all EqProp layers.

        Called by EqPropFunc.backward when the container is used.
        For the first layer, passes the original input x.
        For subsequent layers, passes tuple of (positive_prev, negative_prev) nodes.
        """
        prev_nodes: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = x
        for layer, p, n in zip(self._eq_layers, pos_nodes, neg_nodes):
            layer.calc_n_set_param_grad_(prev_nodes, (p, n))
            a = self.solver.amp_factor
            prev_nodes = (p * a, n * a)  # Update for next layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass using EqProp.

        Note that children layers doesn't call forward() directly"""
        x.requires_grad_()
        return self.eqprop_fn(self, x)

    # No-op methods for EqProp interface compatibility
    def calc_n_set_param_grad_(
        self,
        prev_nodes: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        nodes: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """No-op: parameter gradients handled by children."""
        pass

    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Delegate to first EqProp layer."""
        if self._eq_layers:
            return self._eq_layers[0].calc_x_grad(nodes)
        else:
            raise RuntimeError("No EqProp layers found for calc_x_grad")


_EQPROP_FN_REG: dict[str, type[PositiveEqPropFunc]] = {
    "positive": PositiveEqPropFunc,
    "altered": AlteredEqPropFunc,
    "centered": CenteredEqPropFunc,
}
# Conversion utilities have been moved to conversion.py
