from __future__ import annotations

import torch.nn as nn
from typing import TYPE_CHECKING

from src.utils import RankedLogger

if TYPE_CHECKING:
    from src.core.eqprop.solvers import EqPropSolver
    from src.core.eqprop.nn.module import (
        EqPropSolverManager,
        _EqPropMixin,
        PositiveEqPropFunc,
        CenteredEqPropFunc,
    )

log = RankedLogger(__name__)

__all__ = [
    "to_eqprop",
    "to_eqprop_with_manager",
    "_create_default_solver",
]


def to_eqprop(
    module: nn.Module,
    *,
    eq_fn: type[PositiveEqPropFunc],
    solver: EqPropSolver,
    _suppress_set_model_call: bool = False,
) -> nn.Module:  # noqa: N802 – util
    """Convert *module* to its EqProp equivalent where possible."""
    from src.core.eqprop.nn.module import _EqPropMixin, EqPropSequential, EqPropLinear, EqPropConv2d

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

    if isinstance(module, nn.Sequential):
        log.debug(
            f"Converting nn.Sequential to EqPropSequential. Suppress set_model: {_suppress_set_model_call}"
        )
        return EqPropSequential(
            *module,
            eqprop_fn="centered",
            solver=solver,
        )

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
            solver_manager=None,  # Will be set when used in container
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
            solver_manager=None,  # Will be set when used in container
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

    # Pass through non-convertible layers like BatchNorm, activations, etc.
    elif isinstance(
        module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d | nn.LayerNorm | nn.GroupNorm
    ):
        log.debug(f"Passing through normalization layer: {module.__class__.__name__}")
        return module

    elif isinstance(
        module, nn.ReLU | nn.Sigmoid | nn.Tanh | nn.ELU | nn.LeakyReLU | nn.GELU | nn.SiLU
    ):
        log.debug(f"Passing through activation layer: {module.__class__.__name__}")
        return module

    elif isinstance(module, nn.Dropout | nn.Dropout2d | nn.Dropout3d):
        log.debug(f"Passing through dropout layer: {module.__class__.__name__}")
        return module

    else:
        # If module is not an _EqPropMixin and not convertible, it might be a non-EqProp layer (e.g., activation).
        # In that case, it should be returned as is, to be part of the nn.Sequential.
        # The original code raised TypeError, which implies only convertible layers are allowed directly in *modules.
        # If non-convertible nn.Modules are allowed, they should be returned.
        log.warning(
            f"Module {module.__class__.__name__} is not an EqProp layer and not directly convertible. Passing through."
        )
        return module  # If passthrough is desired for non-convertible/non-EqProp layers.
        # The original code raises an error:
        # raise TypeError(
        #     f"Cannot convert {module.__class__.__name__} to EqProp layer. "
        #     "Please use a supported layer type or ensure it's already an EqProp layer."
        # )


def to_eqprop_with_manager(
    module: nn.Module,
    solver_manager: EqPropSolverManager | None = None,
    eq_fn: type[PositiveEqPropFunc] = None,
) -> nn.Module:
    """Convert module to EqProp recursively using SolverManager.

    This function handles recursive conversion of complex models including
    nested Sequential containers.

    Args:
        module: Module to convert
        solver_manager: Optional solver manager (creates new one if None)
        eq_fn: EqProp function class to use

    Returns:
        Converted EqProp module
    """
    from src.core.eqprop.nn.module import (
        _EqPropMixin,
        EqPropSequential,
        EqPropLinear,
        EqPropConv2d,
        CenteredEqPropFunc,
    )

    # Default eq_fn if not provided
    if eq_fn is None:
        eq_fn = CenteredEqPropFunc

    # If already EqProp, just set SolverManager if provided
    if isinstance(module, _EqPropMixin):
        if solver_manager is not None:
            module._set_solver_manager(solver_manager)
        log.debug(f"Module {type(module).__name__} is already EqProp, set SolverManager")
        return module

    # Handle nn.Sequential recursively
    if isinstance(module, nn.Sequential):
        log.debug(f"Converting nn.Sequential with {len(module)} children")

        # Create new solver and manager for this Sequential
        solver = _create_default_solver()

        # Convert each child module recursively (without solver_manager to avoid conflicts)
        converted_children = []
        for child in module:
            converted_child = to_eqprop_with_manager(
                child,
                solver_manager=None,  # Each child converts independently
                eq_fn=eq_fn,
            )
            converted_children.append(converted_child)

        # Create EqPropSequential with converted children
        return EqPropSequential(*converted_children, solver=solver)

    # Convert individual layers
    if isinstance(module, nn.Linear):
        log.debug("Converting nn.Linear to EqPropLinear")
        return EqPropLinear.from_linear(module, solver_manager, eq_fn)

    elif isinstance(module, nn.Conv2d):
        log.debug("Converting nn.Conv2d to EqPropConv2d")
        return EqPropConv2d.from_conv2d(module, solver_manager, eq_fn)

    # Pass through non-convertible layers
    else:
        log.debug(f"Passing through {type(module).__name__}")
        return module


def _create_default_solver() -> EqPropSolver:
    """Helper function to create default solver.

    Returns:
        Default AnalogEqPropSolver instance
    """
    from src.core.eqprop.solvers import AnalogEqPropSolver
    from src.core.eqprop.strategy import ProxQPStrategy
    from src.core.eqprop.activation import IdealRectifier

    activation = IdealRectifier(Vl=0.1, Vr=0.9)
    strategy = ProxQPStrategy(amp_factor=1.0, activation=activation)
    return AnalogEqPropSolver(beta=0.1, strategy=strategy)
