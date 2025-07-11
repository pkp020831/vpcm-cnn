"""EqProp autograd functions for different algorithm variants."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.core.eqprop.nn.module import EqPropSequential, _EqPropMixin


class PositiveEqPropFunc(torch.autograd.Function):
    """EqProp function class.

    This class behaves similar to activation functions in
    torch.nn.functionals. Determines specific EqProp implementation.
    e.g. 3rd order, 2nd order, etc. Used internally with EqPropMixin.
    """

    @staticmethod
    def forward(ctx, eqprop_layer, input) -> torch.Tensor:
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

        eqprop_layer = ctx.eqprop_layer
        negative_nodes = eqprop_layer.solver(input, grad=grad_output)

        if eqprop_layer.IS_CONTAINER:
            eqprop_layer: EqPropSequential
            # Distribute manual gradients layer‑wise
            eqprop_layer._distribute_param_grads(input, positive_nodes, negative_nodes)
            # dL/dx ‑ comes from first EqProp layer
            first_eq = eqprop_layer._eq_layers[0]
            grad_input = first_eq.calc_x_grad((positive_nodes[0], negative_nodes[0]))
        else:
            nodes = (positive_nodes[0], negative_nodes[0])
            eqprop_layer.calc_n_set_param_grad_(input, nodes)
            grad_input = eqprop_layer.calc_x_grad(
                nodes
            )  # dL/dx = g*(dV_nudge -dV_free)/beta
        return None, grad_input


class AlteredEqPropFunc(PositiveEqPropFunc):
    """Flip beta for every nudge phase."""

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output) -> torch.Tensor:
        """Backward pass for AlteredEqProp."""
        eqprop_layer = ctx.eqprop_layer
        eqprop_layer.solver.flip_beta()
        return super().backward(ctx, grad_output)


class CenteredEqPropFunc(PositiveEqPropFunc):
    """Centered EqProp.

    Use 2 opposite nudged phases (+\beta/2) and (-\beta/2) to calculate
    gradient.
    """

    @staticmethod
    def forward(ctx, eqprop_layer, input):
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
        eqprop_layer = ctx.eqprop_layer
        grad_output /= 2  # we need to divide by 2 to get the equivalent perturbation with the same magnitude
        eqprop_layer.solver.flip_beta()
        positive_nodes = eqprop_layer.solver(input, grad=grad_output)
        eqprop_layer.solver.flip_beta()
        negative_nodes = eqprop_layer.solver(input, grad=grad_output)

        if eqprop_layer.IS_CONTAINER:
            # Distribute manual gradients layer‑wise
            eqprop_layer._distribute_param_grads(input, positive_nodes, negative_nodes)
            # dL/dx ‑ comes from first EqProp layer
            first_eq = eqprop_layer._eq_layers[0]
            grad_input = first_eq.calc_x_grad((positive_nodes[0], negative_nodes[0]))
        else:
            nodes = (positive_nodes[0], negative_nodes[0])
            eqprop_layer.calc_n_set_param_grad_(input, nodes)
            grad_input = eqprop_layer.calc_x_grad(nodes)

        return None, grad_input
