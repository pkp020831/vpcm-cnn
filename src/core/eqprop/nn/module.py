from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from hydra_zen import instantiate

from src.core.eqprop.solvers import EqPropSolver
from src.utils import eqprop_utils, RankedLogger

__all__ = ["EqPropLinear", "EqPropConv2d", "EqPropSequential", "to_eqprop"]

log = RankedLogger(__name__)


class PositiveEqPropFunc(torch.autograd.Function):
    """EqProp function class.

    This class behaves similar to activation functions in
    torch.nn.functionals. Determines specific EqProp implementation.
    e.g. 3rd order, 2nd order, etc. Used internally with EqPropMixin.
    """

    @staticmethod
    def forward(ctx, eqprop_layer: _EqPropMixin, input) -> torch.Tensor:
        """Free phase for EqProp."""
        ctx.eqprop_layer = eqprop_layer
        positive_nodes = eqprop_layer.solver(input)  # Now returns a tuple of tensors
        ctx.save_for_backward(input, *positive_nodes)
        return positive_nodes[-1]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output) -> torch.Tensor:
        """Backward pass for EqProp."""
        tensors = ctx.saved_tensors
        input = tensors[0]  # First tensor is always the input
        positive_nodes = tensors[1:]  # The rest are positive nodes from each layer

        eqprop_layer: _EqPropMixin = ctx.eqprop_layer
        negative_nodes = eqprop_layer.solver(input, grad=grad_output)

        if eqprop_layer.IS_CONTAINER:
            # Distribute manual gradients layer‑wise
            eqprop_layer: EqPropSequential
            eqprop_layer._distribute_param_grads(input, positive_nodes, negative_nodes)
            # dL/dx ‑ comes from first EqProp layer
            first_eq = eqprop_layer._eq_layers[0]
            grad_input = first_eq.calc_x_grad((positive_nodes[0], negative_nodes[0]))
        else:
            nodes = (positive_nodes[0], negative_nodes[0])
            eqprop_layer.calc_n_set_param_grad_(input, nodes)
            grad_input = eqprop_layer.calc_x_grad(nodes)  # dL/dx = g*(dV_nudge -dV_free)/beta
        return None, grad_input


class AlteredEqPropFunc(PositiveEqPropFunc):
    """Flip beta for every nudge phase."""

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output) -> torch.Tensor:
        """Backward pass for AlteredEqProp."""
        tensors = ctx.saved_tensors
        input = tensors[0]  # First tensor is always the input
        positive_nodes = tensors[1:]  # The rest are positive nodes from each layer

        eqprop_layer: _EqPropMixin = ctx.eqprop_layer
        eqprop_layer.solver.flip_beta()
        negative_nodes = eqprop_layer.solver(input, grad=grad_output)

        if eqprop_layer.IS_CONTAINER:
            # Distribute manual gradients layer‑wise
            eqprop_layer: EqPropSequential
            eqprop_layer._distribute_param_grads(input, positive_nodes, negative_nodes)
            # dL/dx ‑ comes from first EqProp layer
            first_eq = eqprop_layer._eq_layers[0]
            grad_input = first_eq.calc_x_grad((positive_nodes[0], negative_nodes[0]))
        else:
            nodes = (positive_nodes[0], negative_nodes[0])
            eqprop_layer.calc_n_set_param_grad_(input, nodes)
            grad_input = eqprop_layer.calc_x_grad(nodes)

        return None, grad_input


class CenteredEqPropFunc(PositiveEqPropFunc):
    """Centered EqProp.

    Use 2 opposite nudged phases (+\beta/2) and (-\beta/2) to calculate
    gradient.
    """

    @staticmethod
    def forward(ctx, eqprop_layer: _EqPropMixin, input):
        """Forward pass for centered EqProp."""
        ctx.eqprop_layer = eqprop_layer
        free_node = eqprop_layer.solver(input)  # Now returns tuple of tensors
        # ctx.mark_non_differentiable(free_node)
        ctx.save_for_backward(input)
        return free_node[-1]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """Backward pass for centered EqProp."""
        (input,) = ctx.saved_tensors
        eqprop_layer: _EqPropMixin = ctx.eqprop_layer
        grad_output /= (
            2  # we need to divide by 2 to get the equivalent perturbation with the same magnitude
        )
        eqprop_layer.solver.flip_beta()
        positive_nodes = eqprop_layer.solver(input, grad=grad_output)
        eqprop_layer.solver.flip_beta()
        negative_nodes = eqprop_layer.solver(input, grad=grad_output)

        if eqprop_layer.IS_CONTAINER:
            # Distribute manual gradients layer‑wise
            eqprop_layer: EqPropSequential
            eqprop_layer._distribute_param_grads(input, positive_nodes, negative_nodes)
            # dL/dx ‑ comes from first EqProp layer
            first_eq = eqprop_layer._eq_layers[0]
            grad_input = first_eq.calc_x_grad((positive_nodes[0], negative_nodes[0]))
        else:
            nodes = (positive_nodes[0], negative_nodes[0])
            eqprop_layer.calc_n_set_param_grad_(input, nodes)
            grad_input = eqprop_layer.calc_x_grad(nodes)

        return None, grad_input


class _EqPropMixin(ABC):
    """EqProp mixin class.

    Wraps EqProp function and solver for nn.Module. All EqProp layers
    should inherit this class and implement calc_n_set_param_grad_() and
    calc_x_grad().
    """

    IS_CONTAINER: bool = False

    def __init__(
        self,
        solver: EqPropSolver | None,
        eqprop_fn: type[PositiveEqPropFunc],  # Expecting the class
        _suppress_set_model_call: bool = False,
    ) -> None:
        """Initialize EqPropMixin.

        Args:
            solver (EqPropSolver | None): EqProp solver.
            eqprop_fn (type[PositiveEqPropFunc]): Specific Eqprop algorithm class.
            _suppress_set_model_call (bool, optional): If True, solver.set_model(self)
                will not be called. Defaults to False.
        """
        self.eqprop_fn = eqprop_fn.apply
        if solver is None:
            # move import under here to avoid circular import
            from configs.eqprop.solver import (
                AnalogEqPropSolverConfig,
                IdealRectifierConfig,
                ProxQPStrategyConfig,
            )

            rectifier_cfg = IdealRectifierConfig(Vl=0.1, Vr=0.9)
            strategy_cfg = ProxQPStrategyConfig(amp_factor=1.0, activation=rectifier_cfg)
            solver_cfg = AnalogEqPropSolverConfig(beta=0.1, strategy=strategy_cfg)
            solver = instantiate(solver_cfg)
            log.info(
                f"Solver not provided for {self.__class__.__name__}. Using default solver: {solver}"
            )
        self.solver = solver

        if not _suppress_set_model_call:
            log.debug(f"Calling set_model from _EqPropMixin.__init__ for {self.__class__.__name__}")
            self.solver.set_model(self)
        else:
            log.debug(
                f"Suppressed set_model from _EqPropMixin.__init__ for {self.__class__.__name__}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for EqProp."""
        x.requires_grad_()
        return self.eqprop_fn(self, x)

    @abstractmethod
    def calc_n_set_param_grad_(
        self, x: torch.Tensor, nodes: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Calculate & set gradient in-place for params.

        This function is called by EqPropFunc.backward.
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
        solver: EqPropSolver | None = None,
        param_init_args: dict = {"min_w": 1e-6, "max_w": None, "max_w_gain": 0.28},
        _suppress_set_model_call: bool = False,
    ):
        """Initialize EqPropLinear. If solver is not provided, default solver
        is AnalogEqPropSolver with ProxQPStrategy.

        Args:
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            bias (bool, optional): If set to ``False``, the layer will not learn an additive bias. Defaults to ``True``.
            eqprop_fn (type[PositiveEqPropFunc], optional): specific Eqprop algorithm class. Defaults to CenteredEqPropFunc.
            solver (Optional[EqPropSolver], optional): EqProp solver. Defaults to None.
            param_init_args (dict, optional): args for set positive param init.
                Defaults to {"min_w":1e-6, "max_w":None, "max_w_gain":0.28}.
            _suppress_set_model_call (bool, optional): If True, solver.set_model(self)
                will not be called. Defaults to False.
        """
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        _EqPropMixin.__init__(
            self, solver, eqprop_fn, _suppress_set_model_call=_suppress_set_model_call
        )
        self.apply(eqprop_utils.positive_param_init(**param_init_args))

    def forward(self, x):
        """Forward pass for EqPropLinear."""
        return _EqPropMixin.forward(self, x)

    @torch.no_grad()
    def calc_n_set_param_grad_(self, x: torch.Tensor, nodes: tuple[torch.Tensor, torch.Tensor]):
        """Calculate & set gradients of parameters manually.

        dy/dw = (nudge_dV^2 - free_dV^2)/beta \n
        = [prev_negative^2 + n_node^2 \n
        \\- (prev_positive^2 + p_node^2) \n
        \\- 2(prev_negative.T@n_node - prev_positive@p_node)]/beta \n
        = [2(p_node@x - n_node@x) \n
        \\+ 0 \n
        \\+ n_node^2 - p_node^2]/beta

        Args:
            x (torch.Tensor): input tensor
            nodes (tuple[torch.Tensor, torch.Tensor]): positive and negative nodes
        """
        beta = self.solver.beta
        positive_node, negative_node = nodes
        # Weight gradient
        dw = 2 * (
            torch.bmm(positive_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
            - torch.bmm(negative_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
        )
        # x.pow(2) - x.pow(2) = 0
        dw += (negative_node.pow(2).mean(0) - positive_node.pow(2).mean(0)).unsqueeze(1)
        dw /= beta
        self.weight.grad = dw if self.weight.grad is None else self.weight.grad + dw

        # Bias gradient
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
            dy (torch.Tensor): gradient of output
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
        solver: EqPropSolver | None = None,
        _suppress_set_model_call: bool = False,
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
        _EqPropMixin.__init__(
            self, solver, eqprop_fn, _suppress_set_model_call=_suppress_set_model_call
        )
        self.unfold = nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
        )

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

    @torch.no_grad()
    def calc_x_grad(self, nodes, dy):
        raise NotImplementedError("calculate_x_grad not implemented")


class EqPropSequential(_EqPropMixin, nn.Sequential):
    """Sequential container solved by one shared `EqPropSolver`."""

    IS_CONTAINER = True  # gradients handled per‑child

    # ------------------------------------------------------------------
    def __init__(
        self,
        *modules: nn.Module,
        eqprop_fn: Literal["positive", "altered", "centered"] = "positive",
        solver: EqPropSolver | None = None,
    ) -> None:
        """Initialize EqPropSequential.
        Args:
            modules (nn.Module): list of modules to register in the container.
            eqprop_fn (str, optional): specific EqProp function to use. Defaults to "positive".
            solver (EqPropSolver | None, optional): EqProp solver. Defaults to None.

        Raises:
            ValueError: If eqprop_fn is not a valid string.
            TypeError: If eqprop_fn is not a string.

        """
        # 1) Resolve EqProp autograd Function class ------------------------------
        eqprop_fn_class: type[PositiveEqPropFunc]
        if isinstance(eqprop_fn, str):
            try:
                eqprop_fn_class = _EQPROP_FN_REG[eqprop_fn.lower()]
            except KeyError as e:
                raise ValueError(
                    f"Unknown eqprop_fn '{eqprop_fn}'. Options: {list(_EQPROP_FN_REG)}"
                ) from e
        else:
            raise TypeError(f"eqprop_fn must be a string, got {type(eqprop_fn)}")

        # 2) Ensure solver instance
        if solver is None:
            # Create a basic solver manually since config is already handled by hydra
            from src.core.eqprop.solvers import AnalogEqPropSolver
            from src.core.eqprop.strategy import ProxQPStrategy
            from src.core.eqprop.activation import IdealRectifier

            # Create default components
            activation = IdealRectifier(Vl=0.1, Vr=0.9)
            strategy = ProxQPStrategy(amp_factor=1.0, activation=activation)
            solver = AnalogEqPropSolver(beta=0.1, strategy=strategy)

            log.info(f"Solver not provided for EqPropSequential. Using default solver: {solver}")

        # 3) Convert vanilla modules to EqProp, suppressing their individual set_model calls
        #    If a module is already _EqPropMixin, its set_model might have run.
        #    The container's final set_model call will ensure correctness.
        converted = [
            to_eqprop(m, eq_fn=eqprop_fn_class, solver=solver, _suppress_set_model_call=True)
            for m in modules
        ]

        # 4) Register converted children via nn.Sequential
        nn.Sequential.__init__(self, *converted)

        # 5) Initialise mixin for the container itself.
        #    Suppress set_model here as the container will call it explicitly after all children are set.
        _EqPropMixin.__init__(self, solver, eqprop_fn_class, _suppress_set_model_call=True)

        # 6) *Finally*, call set_model on the container. This allows the solver to see all
        #    children in sequence via its _eq_layers and correctly populate its strategy.
        log.debug(f"Calling set_model from EqPropSequential.__init__ for {self.__class__.__name__}")
        self.solver.set_model(self)

        # 7) Ensure all EqProp children share *exactly* this solver instance
        #    (and the same eqprop_fn.apply, though to_eqprop should handle fn)
        for m_child in self:
            if isinstance(m_child, _EqPropMixin):
                m_child.solver = self.solver

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def _eq_layers(self) -> list[_EqPropMixin]:
        """Return live view of child EqProp layers (no extra storage)."""
        return [m for m in self if isinstance(m, _EqPropMixin)]

    def _distribute_param_grads(
        self,
        x: torch.Tensor,
        pos_nodes: tuple[torch.Tensor],
        neg_nodes: tuple[torch.Tensor],
    ) -> None:
        """Distribute parameter gradients across all EqProp layers.
        This is internally called by EqPropFunc.backward when the container is used.

        Args:
            x (torch.Tensor): _description_
            pos_nodes (tuple[torch.Tensor]): _description_
            neg_nodes (tuple[torch.Tensor]): _description_
        """
        inputs = [x, *pos_nodes[:-1]]  # input to each layer
        for inp, layer, p, n in zip(inputs, self._eq_layers, pos_nodes, neg_nodes):
            layer.calc_n_set_param_grad_(inp, (p, n))

    # ------------------------------------------------------------------
    # forward – single solve
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):  # type: ignore[override]
        x.requires_grad_()
        return self.eqprop_fn(self, x)

    # ------------------------------------------------------------------
    # These are **no‑ops** for the container itself – SRP delegates to children
    # ------------------------------------------------------------------
    def calc_n_set_param_grad_(self, x, nodes):  # noqa: D401 – intentional pass
        pass

    def calc_x_grad(self, nodes):  # noqa: D401 – container never asked for dL/dx
        raise RuntimeError(
            "EqPropSequential doesn't own calc_x_grad – use first EqProp layer instead."
        )


_EQPROP_FN_REG: dict[str, type[PositiveEqPropFunc]] = {
    "positive": PositiveEqPropFunc,
    "altered": AlteredEqPropFunc,
    "centered": CenteredEqPropFunc,
}


# =============================================================================
# Helper – map vanilla → EqProp layer without retaining vanilla ref
# =============================================================================


def to_eqprop(
    module: nn.Module,
    *,
    eq_fn: type[PositiveEqPropFunc],
    solver: EqPropSolver,
    _suppress_set_model_call: bool = False,
) -> nn.Module:  # noqa: N802 – util
    """Convert *module* to its EqProp equivalent where possible."""
    if isinstance(module, _EqPropMixin):
        # If already an EqProp layer, just update its solver and eqprop_fn.
        # Its own set_model call would have happened during its __init__.
        # The _suppress_set_model_call flag is not directly used here for existing instances,
        # as we are not calling its __init__ again.
        # The container (EqPropSequential) will call set_model on the solver, which clears
        # and re-adds parameters, ensuring the correct final state for the shared solver.
        module.solver = solver
        module.eqprop_fn = eq_fn.apply
        log.debug(
            f"Module {module.__class__.__name__} is already _EqPropMixin. Updated solver and eq_fn."
        )
        return module

    if isinstance(module, nn.Linear):
        log.debug(
            f"Converting nn.Linear to EqPropLinear. Suppress set_model: {_suppress_set_model_call}"
        )
        mapped = EqPropLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            eqprop_fn=eq_fn,
            solver=solver,
            _suppress_set_model_call=_suppress_set_model_call,  # Pass the flag
        )
        mapped.weight.data.copy_(module.weight.detach())
        if module.bias is not None and mapped.bias is not None:
            mapped.bias.data.copy_(module.bias.detach())
        return mapped

    elif isinstance(module, nn.Conv2d):
        log.debug(
            f"Converting nn.Conv2d to EqPropConv2d. Suppress set_model: {_suppress_set_model_call}"
        )
        mapped = EqPropConv2d(
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,  # Assuming weight exists for device/dtype
            dtype=module.weight.dtype,
            eqprop_fn=eq_fn,
            solver=solver,
            _suppress_set_model_call=_suppress_set_model_call,  # Pass the flag
        )
        # For LazyConv2d, direct weight copy before first forward needs careful handling.
        # If module.weight is guaranteed to exist and mapped can accept it:
        if hasattr(module, "weight") and module.weight is not None:
            # This copy might be tricky if 'mapped' hasn't initialized its parameters yet (due to being lazy).
            # For now, assuming it's handled or EqPropConv2d has a mechanism for this.
            # If EqPropConv2d is truly lazy, it might ignore this or error.
            # A robust way would be to pass initial weight/bias to EqPropConv2d constructor if supported.
            if (
                hasattr(mapped, "weight") and mapped.weight is not None
            ):  # Check if mapped has weight attr (might be UninitializedParameter)
                try:
                    mapped.weight.data.copy_(module.weight.detach())
                except Exception as e:
                    log.warning(
                        f"Could not copy weight to EqPropConv2d: {e}. May be due to lazy initialization."
                    )
            else:  # mapped.weight might be an UninitializedParameter
                # For Lazy modules, parameters are created on first forward.
                # We might need to defer this copy or have EqPropConv2d handle it.
                # For now, this reflects the original intent to copy if possible.
                pass

        if hasattr(module, "bias") and module.bias is not None:
            if hasattr(mapped, "bias") and mapped.bias is not None:
                try:
                    mapped.bias.data.copy_(module.bias.detach())
                except Exception as e:
                    log.warning(
                        f"Could not copy bias to EqPropConv2d: {e}. May be due to lazy initialization."
                    )
            else:
                pass
        return mapped

    else:
        # If module is not an _EqPropMixin and not convertible, it might be a non-EqProp layer (e.g., activation).
        # In that case, it should be returned as is, to be part of the nn.Sequential.
        # The original code raised TypeError, which implies only convertible layers are allowed directly in *modules.
        # If non-convertible nn.Modules are allowed, they should be returned.
        # log.warning(f"Module {module.__class__.__name__} is not an EqProp layer and not directly convertible. Passing through.")
        # return module # If passthrough is desired for non-convertible/non-EqProp layers.
        # The original code raises an error:
        raise TypeError(
            f"Cannot convert {module.__class__.__name__} to EqProp layer. "
            "Please use a supported layer type or ensure it's already an EqProp layer."
        )
