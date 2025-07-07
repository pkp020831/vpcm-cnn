import math
from abc import ABC, abstractmethod

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class AbstractRectifier(ABC):
    """Base class for rectifiers."""

    def __init__(self, Is: float, Vth: float, Vl: float, Vr: float):
        self.Is = Is
        self.Vth = Vth
        self.Vl = Vl
        self.Vr = Vr

    @abstractmethod
    def i(self, V: torch.Tensor):
        """Compute current."""
        pass

    @abstractmethod
    def a(self, V: torch.Tensor):
        """Compute admittance."""
        pass

    @abstractmethod
    def p(self, V: torch.Tensor):
        """Compute power."""
        pass


class IdealRectifier(AbstractRectifier):
    def __init__(self, Vl=-1.0, Vr=1.0):
        super().__init__(None, None, Vl, Vr)

    def i(self, V: torch.Tensor):
        return torch.zeros_like(V)

    def a(self, V: torch.Tensor):
        return torch.zeros_like(V)

    @classmethod
    def p(cls, V: torch.Tensor):
        """Compute power."""
        pass


class OTS(AbstractRectifier):
    """Ovonic Threshold Switch rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9):
        super().__init__(Is, Vth, Vl, Vr)

    def a(self, V: torch.Tensor):
        """Compute admittance using hyperbolic functions for numerical stability."""
        mid_point = (self.Vl + self.Vr) / 2
        delta = (self.Vr - self.Vl) / 2

        # Use exponential trick: exp(x) + exp(-x) = 2*cosh(x)
        shifted_V = V - mid_point
        return 2 * self.Is / self.Vth * torch.cosh((shifted_V + delta) / self.Vth)

    def i(self, V: torch.Tensor):
        """Compute current using hyperbolic functions for numerical stability."""
        mid_point = (self.Vl + self.Vr) / 2
        delta = (self.Vr - self.Vl) / 2

        # Use exponential trick: exp(x) - exp(-x) = 2*sinh(x)
        shifted_V = V - mid_point
        return 2 * self.Is * torch.sinh((shifted_V + delta) / self.Vth)

    @classmethod
    def p(cls, V: torch.Tensor):
        """Compute power."""
        pass


class SymOTS(OTS):
    """Symmetric OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0, Vr=0):
        assert Vl == Vr == 0, "Vl and Vr must be 0"
        super().__init__(Is, Vth, Vl, Vr)

    def i_div_a(self, V: torch.Tensor):
        """Compute current/admittance.

        Use exponential approximation.
        """
        return self.Vth * torch.tanh(V / self.Vth)

    def inv_a(self, V: torch.Tensor):
        """Compute inverse of admittance."""
        x = (V - self.Vr) / self.Vth
        abs_x = torch.abs(x)
        return (
            self.Vth / self.Is * torch.exp(abs_x) / (torch.exp(x - abs_x) + torch.exp(-x - abs_x))
        )


class PolyOTS(AbstractRectifier):
    """Polynomial Taylor expansion of OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9, power=2):
        super().__init__(Is, Vth, Vl, Vr)
        self.power = power

    def a(
        self,
        V: torch.Tensor,
    ):
        """Compute admittance.

        Use polynomial approximation.
        """
        x1 = (V - self.Vr) / self.Vth
        x2 = (V - self.Vl) / self.Vth
        res = 2
        for i in range(1, self.power + 1):
            res += i * (x1.pow(i) - (-x2).pow(i)) / math.factorial(i)
        return self.Is / (self.Vth**2) * res

    def i(
        self,
        V: torch.Tensor,
    ):
        """Compute current.

        Use polynomial approximation.
        """
        x1 = (V - self.Vr) / self.Vth
        x2 = (V - self.Vl) / self.Vth
        res = 0
        for i in range(1, self.power + 1):
            res += ((x1 / self.Vth).pow(i) - (-x2 / self.Vth).pow(i)) / math.factorial(i)
        return self.Is * res


class P3OTS(AbstractRectifier):
    """3rd order polynomial OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        """Compute current."""
        x = V - (self.Vl + self.Vr) / 2
        return 2 * self.Is / self.Vth * (x.pow(3))

    def a(self, V: torch.Tensor):
        """Compute admittance."""
        x = V - (self.Vl + self.Vr) / 2
        return 2 * self.Is / self.Vth * (3 * x.pow(2))

    def p(self, V: torch.Tensor):
        """Compute power."""
        pass


class SymReLU(AbstractRectifier):
    """Symmetric ReLU rectifier."""

    def __init__(self, Is=1, Vth=1, Vl=-0.6, Vr=0.6):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        """Compute current."""
        x = V * self.Is
        return ((x - self.Vl) / self.Vth).clamp(max=0) + ((x - self.Vr) / self.Vth).clamp(min=0)

    def a(self, V: torch.Tensor):
        x = V * self.Is
        left_active = x < self.Vl
        right_active = x > self.Vr
        return (left_active | right_active).float()

    def p(self, V: torch.Tensor):
        """Compute power."""
        pass


class BidDiode(AbstractRectifier):
    def __init__(self, Is=1, Vth=1, Vl=-0.6, Vr=0.6):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        """Compute current."""
        pass

    def a(self, V: torch.Tensor):
        """Compute admittance."""
        pass

    def p(self, V: torch.Tensor):
        """Compute power."""
        pass


class DifferentialPairRectifier(AbstractRectifier):
    """Differential pair rectifier for resistive networks.

    This rectifier implements the energy-based-learning approach where:
    - First half of nodes (excitatory): clamped to [0, +∞)
    - Second half of nodes (inhibitory): clamped to (-∞, 0]

    This is used in resistive networks where each logical unit is represented
    by a differential pair to handle both positive and negative values with
    non-negative weights.
    """

    def __init__(self, Is: float = 1.0, Vth: float = 1.0, **kwargs):
        """Initialize differential pair rectifier.

        Args:
            Is: Current scale parameter.
            Vth: Voltage threshold parameter.
            **kwargs: Additional parameters (for compatibility, ignored).
        """
        # Use large positive/negative bounds to represent ±∞
        super().__init__(Is, Vth, Vl=float("-inf"), Vr=float("inf"))

    def i(self, V: torch.Tensor) -> torch.Tensor:
        """Compute current with differential pair clamping.

        Args:
            V: Input voltages with shape (..., 2N) where N is number of logical units.

        Returns:
            Clamped voltages where first half ≥ 0 and second half ≤ 0.
        """
        if V.shape[-1] % 2 != 0:
            raise ValueError(f"Differential pair requires even number of nodes, got {V.shape[-1]}")

        dimension = V.shape[-1] // 2
        excitatory = V[..., :dimension].clamp(min=0.0)  # First half: ≥ 0
        inhibitory = V[..., dimension:].clamp(max=0.0)  # Second half: ≤ 0
        return torch.cat([excitatory, inhibitory], dim=-1)

    def a(self, V: torch.Tensor) -> torch.Tensor:
        """Compute admittance (derivative of current).

        Args:
            V: Input voltages with shape (..., 2N).

        Returns:
            Binary mask indicating active regions.
        """
        if V.shape[-1] % 2 != 0:
            raise ValueError(f"Differential pair requires even number of nodes, got {V.shape[-1]}")

        dimension = V.shape[-1] // 2
        excitatory_active = (V[..., :dimension] > 0.0).float()  # Active when > 0
        inhibitory_active = (V[..., dimension:] < 0.0).float()  # Active when < 0
        return torch.cat([excitatory_active, inhibitory_active], dim=-1) * self.Is

    def get_box_constraints(self, dims: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get layer-wise box constraints for ProxQP.

        Args:
            dims: List of layer dimensions [hidden_dim, output_dim, ...]

        Returns:
            Tuple of (lower_bounds, upper_bounds) tensors.
        """
        lb_parts = []
        ub_parts = []

        for dim in dims:
            if dim % 2 != 0:
                raise ValueError(
                    f"Each layer must have even nodes for differential pairs, got {dim}"
                )

            half_dim = dim // 2
            # For each layer: first half excitatory [0, +∞), second half inhibitory (-∞, 0]
            lb_parts.append(torch.zeros(half_dim))  # Excitatory lower bound
            lb_parts.append(torch.full((half_dim,), float("-inf")))  # Inhibitory lower bound
            ub_parts.append(torch.full((half_dim,), float("inf")))  # Excitatory upper bound
            ub_parts.append(torch.zeros(half_dim))  # Inhibitory upper bound

        return torch.cat(lb_parts), torch.cat(ub_parts)

    def i_layerwise(self, V: torch.Tensor, dims: list[int]) -> torch.Tensor:
        """Apply differential pair clamping layer by layer.

        Args:
            V: Input voltages with shape (batch, total_nodes)
            dims: List of layer dimensions

        Returns:
            Layer-wise clamped voltages
        """
        result_parts = []
        start_idx = 0

        for dim in dims:
            if dim % 2 != 0:
                raise ValueError(f"Layer must have even nodes for differential pairs, got {dim}")

            end_idx = start_idx + dim
            layer_v = V[:, start_idx:end_idx]

            half_dim = dim // 2
            excitatory = layer_v[:, :half_dim].clamp(min=0.0)  # First half ≥ 0
            inhibitory = layer_v[:, half_dim:].clamp(max=0.0)  # Second half ≤ 0

            result_parts.append(torch.cat([excitatory, inhibitory], dim=-1))
            start_idx = end_idx

        return torch.cat(result_parts, dim=-1)

    def p(self, V: torch.Tensor) -> torch.Tensor:
        """Compute power."""
        return V * self.i(V)
