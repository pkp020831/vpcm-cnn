import functools
import math
from abc import ABC, abstractmethod
from typing import Any, Literal, Union
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class interleave:
    """Decorator class for interleaving in/out nodes.

    If type is "in", doubles and flips adjacent element for first input
    tensor. \n If type is "out", sum every num_output elems to one for
    all output tensor. \n If type is "both", does both.
    """

    def __init__(self, type: Literal["in", "out", "both"]):
        self.type = type
        # self._num_output = 0
        # self.num_output = interleave._num_output

    def __call__(self, func: Callable[..., torch.Tensor]) -> Callable:
        """Decorator for interleaving in/out nodes."""

        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs) -> torch.Tensor:
            if self.type in ["in", "both"]:
                if len(args) == 1:
                    args = (self.interleave_input(*args),)
                elif len(args) == 2:
                    ins, *others = args
                    args = self.interleave_input(ins), *others
                else:
                    raise ValueError("No input argument found")
            outs = func(obj, *args, **kwargs)
            if self.type in ["out", "both"]:
                outs = self.interleave_output(outs)
            return outs

        return wrapper

    def interleave_output(self, t: torch.Tensor) -> torch.Tensor:
        """Interleave 2D tensor if num output set to even."""
        assert t.dim() == 2, "interleave only works on 2D tensors"
        if self._num_output == 1:
            return t

        elif self._num_output == 2:
            return t[:, ::2] - t[:, 1::2]
        else:
            t_reshaped = t.reshape(t.shape[0], t.shape[1] // self._num_output, self._num_output)
            result = t_reshaped.sum(-1)

            return result

    @torch.no_grad()
    def interleave_input(self, x: torch.Tensor) -> torch.Tensor:
        """Interleave input tensor."""
        if self._num_input == 1:
            return x.view(x.size(0), -1)
        elif self._num_input == 2:
            x = x.view(x.size(0), -1)  # == x.view(-1,x.size(-1)**2)
            x = x.repeat_interleave(2, dim=1)
            x[:, 1::2] = -x[:, ::2]
            return x
        else:
            raise ValueError("num_input must be 1 or 2")

    @classmethod
    def set_num_input(cls, num_input):
        assert num_input == 2 or num_input == 1, "num_input must be 2 or 1"
        cls._num_input = num_input

    @classmethod
    def set_num_output(cls, num_output):
        assert num_output % 2 == 0 or num_output == 1, "num_output must be even or 1"
        cls._num_output = num_output


class type_as:
    """Decorator class for match output tensor type as input tensor type."""

    def __init__(self, func: Callable[..., Sequence[torch.Tensor] | torch.Tensor]):
        self.func = func
        raise DeprecationWarning("Use torch.Tensor.to(self.device) instead.")

    def __call__(self, obj, *args, **kwargs):
        """Decorator for matching output tensor type as input tensor type."""
        out = self.func(obj, *args, **kwargs)
        if isinstance(out, list):
            return [t.type_as(args[0]) for t in out]
        else:
            return out.type_as(args[0])

    def __get__(self, instance, owner=None):
        return functools.partial(self.__call__, instance)


class AbstractRectifier(ABC):
    """Base class for rectifiers.

    Args:
        Is: Saturation current
        Vth: Threshold voltage
        Vl: Left voltage limit
        Vr: Right voltage limit
    """

    def __init__(self, Is, Vth, Vl, Vr):
        self.Is = Is
        self.Vth = Vth
        self.Vl = Vl
        self.Vr = Vr

    @abstractmethod
    def i(self, V: torch.Tensor):
        """Compute current from voltage.

        Args:
            V: Input voltage tensor

        Returns:
            Current tensor
        """
        pass

    @abstractmethod
    def a(self, V: torch.Tensor):
        """Compute admittance from voltage.

        Args:
            V: Input voltage tensor

        Returns:
            Admittance tensor
        """
        pass

    @abstractmethod
    def p(self, V: torch.Tensor):
        """Compute power from voltage.

        Args:
            V: Input voltage tensor

        Returns:
            Power tensor
        """
        pass


class IdealRectifier(AbstractRectifier):
    """Ideal rectifier with zero current and admittance.

    Args:
        Vl: Left voltage limit (default: -1)
        Vr: Right voltage limit (default: 1)
    """

    def __init__(self, Vl=-1, Vr=1):
        super().__init__(None, None, Vl, Vr)

    def i(self, V: torch.Tensor):
        """Compute current (always zero for ideal rectifier).

        Args:
            V: Input voltage tensor

        Returns:
            Zero tensor with same shape as V
        """
        return torch.zeros_like(V)

    def a(self, V: torch.Tensor):
        """Compute admittance (always zero for ideal rectifier).

        Args:
            V: Input voltage tensor

        Returns:
            Zero tensor with same shape as V
        """
        return torch.zeros_like(V)

    @classmethod
    def p(cls, V: torch.Tensor):
        """Compute power."""
        pass


class OTS(AbstractRectifier):
    """Ovonic Threshold Switch rectifier.

    Args:
        Is: Saturation current (default: 1e-8)
        Vth: Threshold voltage (default: 0.026)
        Vl: Left voltage limit (default: 0.1)
        Vr: Right voltage limit (default: 0.9)
    """

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9):
        super().__init__(Is, Vth, Vl, Vr)

    def a(self, V: torch.Tensor):
        """Compute admittance.

        Use exponential approximation.
        """
        admittance = (
            self.Is
            / self.Vth
            * (torch.exp((V - self.Vr) / self.Vth) + torch.exp((-V + self.Vl) / self.Vth))
        )
        return admittance

    def i(self, V: torch.Tensor):
        """Compute current."""
        return self.Is * (
            torch.exp((V - self.Vr) / self.Vth) - torch.exp((-V + self.Vl) / self.Vth)
        )

    @classmethod
    def p(cls, V: torch.Tensor):
        """Compute power."""
        pass


class SymOTS(OTS):
    """Symmetric OTS rectifier with Vl=Vr=0.

    Args:
        Is: Saturation current (default: 1e-8)
        Vth: Threshold voltage (default: 0.026)
        Vl: Left voltage limit (must be 0)
        Vr: Right voltage limit (must be 0)
    """

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
    """Polynomial Taylor expansion of OTS rectifier.

    Args:
        Is: Saturation current (default: 1e-8)
        Vth: Threshold voltage (default: 0.026)
        Vl: Left voltage limit (default: 0.1)
        Vr: Right voltage limit (default: 0.9)
        power: Order of polynomial expansion (default: 2)
    """

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
    """3rd order polynomial OTS rectifier.

    Args:
        Is: Saturation current (default: 1e-8)
        Vth: Threshold voltage (default: 0.026)
        Vl: Left voltage limit (default: 0.1)
        Vr: Right voltage limit (default: 0.9)
    """

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
    """Symmetric ReLU rectifier.

    Args:
        Is: Saturation current (default: 1)
        Vth: Threshold voltage (default: 1)
        Vl: Left voltage limit (default: -0.5)
        Vr: Right voltage limit (default: 0.5)
    """

    def __init__(self, Is=1, Vth=1, Vl=-0.5, Vr=0.5):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        """Compute current."""
        x = V * self.Is
        return ((x - self.Vl) / self.Vth).clamp(max=0) + ((x - self.Vr) / self.Vth).clamp(min=0)

    def a(self, V: torch.Tensor):
        """Compute admittance."""
        x = V * self.Is
        return -((x - self.Vl) / self.Vth < 0).float() + ((x - self.Vr) / self.Vth > 0).float()

    def p(self, V: torch.Tensor):
        """Compute power."""
        pass


@torch.jit.script
def deltaV(n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Compute batch-wise deltaV matrix from 2 node voltages.

    Where deltaV[i, j] = n[i] - m[j]

    Args:
        n (torch.Tensor): (B x) I
        m (torch.Tensor): (B x) O

    Returns:
        torch.Tensor: (B x) O x I
    """
    assert len(n.shape) in [1, 2], "n must be 1D or 2D"
    if len(n.shape) == 2:
        assert n.shape[0] == m.shape[0], "n and m must have the same batch size"
        N = n.clone().unsqueeze(dim=-1).repeat(1, 1, m.shape[-1]).transpose(1, 2)
        M = m.clone().unsqueeze(dim=-1).repeat(1, 1, n.shape[-1])
    else:
        N = n.clone().unsqueeze(dim=-1).repeat(1, m.shape[-1]).T
        M = m.clone().unsqueeze(dim=-1).repeat(1, n.shape[-1])

    return N - M


# Use .apply() for below classes


# TODO: bias 추가
class AdjustParams:
    """Parameter adjustment utility for neural network modules.

    Args:
        L: Lower bound for parameter clamping (default: 1e-7)
        U: Upper bound for parameter clamping (default: None)
        clamp: Whether to clamp parameters to bounds (default: True)
        normalize: Whether to normalize parameters (default: False)
    """

    def __init__(
        self,
        L: float | None = 1e-7,
        U: float | None = None,
        clamp: bool = True,
        normalize: bool = False,
    ) -> None:
        self.min = L
        self.max = U
        self.normalize = normalize
        self.clamp = clamp

    @torch.no_grad()
    def __call__(self, submodule: nn.Module):
        """Adjust parameters of a neural network module.

        Args:
            submodule: PyTorch module whose parameters to adjust
        """
        for name, param in submodule.named_parameters():
            if name in ["weight", "bias"]:
                if self.clamp:
                    (log.debug(f"Clamping {name}...") if torch.any(param.min() < self.min) else ...)
                    param.clamp_(self.min, self.max)
                if self.normalize:
                    nn.functional.normalize(param, dim=1, p=2)


def positive_param_init(min_w: float = 1e-6, max_w: float | None = None, max_w_gain: float = 0.08):
    """Initialize weights with positive values.

    Args:
        min_w: Minimum weight value (default: 1e-6)
        max_w: Maximum weight value (default: None)
        max_w_gain: Maximum weight gain factor (default: 0.08)

    Returns:
        Function that initializes module parameters

    Raises:
        ValueError: If both max_w and max_w_gain are None
    """
    if max_w is None and max_w_gain is None:
        raise ValueError("Either max_w or max_w_gain must be provided")

    def _init_params(submodule: nn.Module):
        nonlocal min_w, max_w, max_w_gain
        if hasattr(submodule, "weight") and getattr(submodule, "weight") is not None:
            param = submodule.get_parameter("weight")
            # positive xaiver_uniform
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
            upper_thres = max_w_gain / math.sqrt(fan_in + fan_out) if max_w is None else max_w
            nn.init.uniform_(param, min_w, upper_thres)
        if hasattr(submodule, "bias") and getattr(submodule, "bias") is not None:
            nn.init.zeros_(submodule.get_parameter("bias"))

    return _init_params


def gaussian_noise(
    std: float, noise_type: Literal["additive", "multiplicative"] = "additive"
) -> Callable[[nn.Module], None]:
    """Add Gaussian noise to named parameters.

    Args:
        std: Standard deviation of Gaussian noise
        noise_type: Type of noise to add - "additive" or "multiplicative" (default: "additive")

    Returns:
        Function that adds noise to module parameters
    """

    def _additive_noise(submodule, std: float):
        """Add Gaussian noise to all parameters of a module.

        Args:
            submodule: PyTorch module
            std: Standard deviation of noise
        """
        for _, param in submodule.named_parameters():
            param.data += torch.randn(param.shape, device=param.device) * std

    def _multiplicative_noise(submodule, std: float):
        """Add multiplicative Gaussian noise to all parameters of a module.

        Args:
            submodule: PyTorch module
            std: Standard deviation of noise
        """
        for _, param in submodule.named_parameters():
            param.data *= 1 + torch.randn(param.shape, device=param.device) * std

    assert std > 0, "std must be positive"
    if noise_type == "additive":
        return functools.partial(_additive_noise, std=std)
    elif noise_type == "multiplicative":
        return functools.partial(_multiplicative_noise, std=std)
    else:
        raise ValueError("Invalid noise_type. Must be 'additive' or 'multiplicative'.")
