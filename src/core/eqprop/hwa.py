"""Hardware-aware (HWA) features for PyTorch models.

This module provides hardware-aware features that can be applied to PyTorch models
to simulate hardware constraints and characteristics:
1. Gradient noise injection: Adds Gaussian noise to gradients during backpropagation
2. Weight value limiting: Constrains weights to be within specified minimum and maximum values

These features can be applied to all submodules of a model using the `apply()` method.
"""

from typing import Optional
import torch
import torch.nn as nn


class GradientNoiseInjector:
    """Adds Gaussian noise to gradients during backpropagation.

    This simulates the noise that might be present in hardware implementations
    of neural networks, particularly in analog or mixed-signal circuits.
    """

    def __init__(
        self,
        std: float = 0.01,
        mean: float = 0.0,
        scale_with_grad_norm: bool = True,
        noise_scale_factor: float = 0.1,
    ):
        """Initialize the gradient noise injector.

        Args:
            std: Standard deviation of the Gaussian noise
            mean: Mean of the Gaussian noise
            scale_with_grad_norm: If True, scales the noise with the norm of the gradient
            noise_scale_factor: Factor to scale the noise when scale_with_grad_norm is True
        """
        self.std = std
        self.mean = mean
        self.scale_with_grad_norm = scale_with_grad_norm
        self.noise_scale_factor = noise_scale_factor

    def __call__(self, module: nn.Module) -> None:
        """Register the backward hook to the module.

        Args:
            module: The module to register the hook to
        """
        module.register_full_backward_hook(self._backward_hook)

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Add Gaussian noise to gradients.

        Args:
            module: The module that the hook is registered to
            grad_input: Gradient of the loss with respect to the input of the module
            grad_output: Gradient of the loss with respect to the output of the module

        Returns:
            Modified grad_input with added noise
        """
        if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
            return grad_input

        # Create a copy of grad_input to modify
        noisy_grad_input = list(grad_input)

        for i, grad in enumerate(noisy_grad_input):
            if grad is not None:
                # Determine noise scale
                if self.scale_with_grad_norm and grad.norm() > 0:
                    noise_std = self.std * grad.norm() * self.noise_scale_factor
                else:
                    noise_std = self.std

                # Generate noise with the same shape as the gradient
                noise = torch.randn_like(grad) * noise_std + self.mean

                # Add noise to the gradient
                noisy_grad_input[i] = grad + noise

        return tuple(noisy_grad_input)


class WeightLimiter:
    """Limits the weights of a module to be within specified minimum and maximum values.

    This simulates the limited weight range that might be present in hardware
    implementations of neural networks, particularly in analog or mixed-signal circuits.

    Supports different min/max values for different layers by providing dictionaries
    mapping module names to their respective min/max values.
    """

    def __init__(
        self,
        min_value: float | dict[str, float] | None = None,
        max_value: float | dict[str, float] | None = None,
        apply_to_bias: bool = True,
        default_min: float | None = None,
        default_max: float | None = None,
    ):
        """Initialize the weight limiter.

        Args:
            min_value: Minimum allowed value for weights (None for no minimum)
                       Can be a float (same for all modules) or a dict mapping module names to min values
            max_value: Maximum allowed value for weights (None for no maximum)
                       Can be a float (same for all modules) or a dict mapping module names to max values
            apply_to_bias: Whether to apply limits to bias parameters as well
            default_min: Default minimum value to use when a module name is not found in min_value dict
            default_max: Default maximum value to use when a module name is not found in max_value dict
        """
        self.min_value = min_value
        self.max_value = max_value
        self.apply_to_bias = apply_to_bias
        self.default_min = default_min
        self.default_max = default_max

    def __call__(self, module: nn.Module) -> None:
        """Register the backward hook to the module.

        Args:
            module: The module to register the hook to
        """
        module.register_full_backward_hook(self._backward_hook)

    def _get_min_value(self, module_name: str) -> float | None:
        """Get the minimum value for a specific module.

        Args:
            module_name: The name of the module

        Returns:
            The minimum value for the module or None
        """
        if self.min_value is None:
            return None
        if isinstance(self.min_value, dict):
            return self.min_value.get(module_name, self.default_min)
        return self.min_value

    def _get_max_value(self, module_name: str) -> float | None:
        """Get the maximum value for a specific module.

        Args:
            module_name: The name of the module

        Returns:
            The maximum value for the module or None
        """
        if self.max_value is None:
            return None
        if isinstance(self.max_value, dict):
            return self.max_value.get(module_name, self.default_max)
        return self.max_value

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Apply weight limits after gradient update.

        This hook doesn't modify the gradients but applies limits to the weights
        after the optimizer has updated them.

        Args:
            module: The module that the hook is registered to
            grad_input: Gradient of the loss with respect to the input of the module
            grad_output: Gradient of the loss with respect to the output of the module
        """
        # Get module name for layer-specific limits
        module_name = module.__class__.__name__
        if hasattr(module, "name") and module.name:
            module_name = module.name

        # Get min/max values for this specific module
        min_value = self._get_min_value(module_name)
        max_value = self._get_max_value(module_name)

        # Apply limits to weights
        if hasattr(module, "weight") and module.weight is not None:
            with torch.no_grad():
                if min_value is not None:
                    module.weight.data.clamp_(min=min_value)
                if max_value is not None:
                    module.weight.data.clamp_(max=max_value)

        # Apply limits to bias if specified
        if self.apply_to_bias and hasattr(module, "bias") and module.bias is not None:
            with torch.no_grad():
                if min_value is not None:
                    module.bias.data.clamp_(min=min_value)
                if max_value is not None:
                    module.bias.data.clamp_(max=max_value)


class HardwareAwareWrapper:
    """Wrapper class that combines multiple hardware-aware features.

    This class provides a convenient way to apply multiple hardware-aware
    features to a model at once.
    """

    def __init__(
        self,
        noise_std: float | None = None,
        noise_mean: float = 0.0,
        scale_with_grad_norm: bool = True,
        noise_scale_factor: float = 0.1,
        min_weight: float | dict[str, float] | None = None,
        max_weight: float | dict[str, float] | None = None,
        apply_to_bias: bool = True,
        default_min: float | None = None,
        default_max: float | None = None,
    ):
        """Initialize the hardware-aware wrapper.

        Args:
            noise_std: Standard deviation of the Gaussian noise (None to disable)
            noise_mean: Mean of the Gaussian noise
            scale_with_grad_norm: If True, scales the noise with the norm of the gradient
            noise_scale_factor: Factor to scale the noise when scale_with_grad_norm is True
            min_weight: Minimum allowed value for weights (None for no minimum)
                        Can be a float (same for all modules) or a dict mapping module names to min values
            max_weight: Maximum allowed value for weights (None for no maximum)
                        Can be a float (same for all modules) or a dict mapping module names to max values
            apply_to_bias: Whether to apply limits to bias parameters as well
            default_min: Default minimum value to use when a module name is not found in min_weight dict
            default_max: Default maximum value to use when a module name is not found in max_weight dict
        """
        self.features = []

        # Add gradient noise injector if std is provided
        if noise_std is not None:
            self.features.append(
                GradientNoiseInjector(
                    std=noise_std,
                    mean=noise_mean,
                    scale_with_grad_norm=scale_with_grad_norm,
                    noise_scale_factor=noise_scale_factor,
                )
            )

        # Add weight limiter if min_weight or max_weight is provided
        if min_weight is not None or max_weight is not None:
            self.features.append(
                WeightLimiter(
                    min_value=min_weight,
                    max_value=max_weight,
                    apply_to_bias=apply_to_bias,
                    default_min=default_min,
                    default_max=default_max,
                )
            )

    def __call__(self, module: nn.Module) -> None:
        """Apply all hardware-aware features to the module.

        Args:
            module: The module to apply the features to
        """
        for feature in self.features:
            feature(module)


def apply_gradient_noise(
    model: nn.Module,
    std: float = 0.01,
    mean: float = 0.0,
    scale_with_grad_norm: bool = True,
    noise_scale_factor: float = 0.1,
) -> nn.Module:
    """Apply gradient noise to all submodules of a model.

    Args:
        model: The model to apply gradient noise to
        std: Standard deviation of the Gaussian noise
        mean: Mean of the Gaussian noise
        scale_with_grad_norm: If True, scales the noise with the norm of the gradient
        noise_scale_factor: Factor to scale the noise when scale_with_grad_norm is True

    Returns:
        The model with gradient noise applied
    """
    noise_injector = GradientNoiseInjector(
        std=std,
        mean=mean,
        scale_with_grad_norm=scale_with_grad_norm,
        noise_scale_factor=noise_scale_factor,
    )
    model.apply(noise_injector)
    return model


def apply_weight_limits(
    model: nn.Module,
    min_value: float | dict[str, float] | None = None,
    max_value: float | dict[str, float] | None = None,
    apply_to_bias: bool = True,
    default_min: float | None = None,
    default_max: float | None = None,
) -> nn.Module:
    """Apply weight limits to all submodules of a model.

    Args:
        model: The model to apply weight limits to
        min_value: Minimum allowed value for weights (None for no minimum)
                   Can be a float (same for all modules) or a dict mapping module names to min values
        max_value: Maximum allowed value for weights (None for no maximum)
                   Can be a float (same for all modules) or a dict mapping module names to max values
        apply_to_bias: Whether to apply limits to bias parameters as well
        default_min: Default minimum value to use when a module name is not found in min_value dict
        default_max: Default maximum value to use when a module name is not found in max_value dict

    Returns:
        The model with weight limits applied
    """
    weight_limiter = WeightLimiter(
        min_value=min_value,
        max_value=max_value,
        apply_to_bias=apply_to_bias,
        default_min=default_min,
        default_max=default_max,
    )
    model.apply(weight_limiter)
    return model


def apply_hardware_aware_features(
    model: nn.Module,
    noise_std: float | None = None,
    noise_mean: float = 0.0,
    scale_with_grad_norm: bool = True,
    noise_scale_factor: float = 0.1,
    min_weight: float | dict[str, float] | None = None,
    max_weight: float | dict[str, float] | None = None,
    apply_to_bias: bool = True,
    default_min: float | None = None,
    default_max: float | None = None,
) -> nn.Module:
    """Apply multiple hardware-aware features to all submodules of a model.

    Args:
        model: The model to apply hardware-aware features to
        noise_std: Standard deviation of the Gaussian noise (None to disable)
        noise_mean: Mean of the Gaussian noise
        scale_with_grad_norm: If True, scales the noise with the norm of the gradient
        noise_scale_factor: Factor to scale the noise when scale_with_grad_norm is True
        min_weight: Minimum allowed value for weights (None for no minimum)
                    Can be a float (same for all modules) or a dict mapping module names to min values
        max_weight: Maximum allowed value for weights (None for no maximum)
                    Can be a float (same for all modules) or a dict mapping module names to max values
        apply_to_bias: Whether to apply limits to bias parameters as well
        default_min: Default minimum value to use when a module name is not found in min_weight dict
        default_max: Default maximum value to use when a module name is not found in max_weight dict

    Returns:
        The model with hardware-aware features applied
    """
    hwa_wrapper = HardwareAwareWrapper(
        noise_std=noise_std,
        noise_mean=noise_mean,
        scale_with_grad_norm=scale_with_grad_norm,
        noise_scale_factor=noise_scale_factor,
        min_weight=min_weight,
        max_weight=max_weight,
        apply_to_bias=apply_to_bias,
        default_min=default_min,
        default_max=default_max,
    )
    model.apply(hwa_wrapper)
    return model
