"""
Strategy implementations for solving equilibrium points in neural networks.

This module provides various strategies for finding equilibrium points in neural networks,
including gradient descent, quadratic programming, and SPICE-based approaches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from collections.abc import Callable
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from src.core.eqprop import activation
from src.utils import _PROXSUITE_AVAILABLE, _QPSOLVERS_AVAILABLE, _XYCE_AVAILABLE, RankedLogger

# Conditional imports based on available packages
if _PROXSUITE_AVAILABLE:
    import proxsuite  # type: ignore

if _QPSOLVERS_AVAILABLE:
    from qpsolvers import solve_qp  # type: ignore

if _XYCE_AVAILABLE:
    from src.core.spice import circuits, xyce, spice_utils  # type: ignore

# Configure logger
log = RankedLogger(__name__, rank_zero_only=True)

# Type aliases
NpOrTensor = np.ndarray | torch.Tensor

__all__ = [
    "AbstractStrategy",
    "XyceStrategy",
    "GradientDescentStrategy",
    "QPStrategy",
    "ProxQPStrategy",
    "FirstOrderStrategy",
    "SecondOrderStrategy",
    "ResistiveNetworkStrategy",
]


def cache_free_solution(func: Callable) -> Callable:
    """Cache the free-phase solution of the network.

    This decorator caches the output of the solve method when i_ext is None (free phase),
    and reuses it when i_ext is not None (nudged phase).

    Args:
        func: The solve method to be decorated.

    Returns:
        The wrapped function that implements caching.
    """

    @wraps(func)
    def wrapper(self, x: torch.Tensor, i_ext: torch.Tensor | None, **kwargs: Any) -> torch.Tensor:
        if i_ext is None:
            self.free_solution = None
        vout = func(self, x, i_ext, **kwargs)
        if i_ext is None:
            self.free_solution = vout
        return vout

    return wrapper


class AbstractStrategy(ABC):
    """Abstract class for different strategies to solve for the equilibrium
    point of the network.

    Args:
        activation (Callable | str): activation function of the network.
        max_iter (int): maximum number of iterations.
        atol (float): absolute tolerance.
    """

    def __init__(
        self,
        activation: activation.AbstractRectifier,
        max_iter: int = 30,
        atol: float = 1e-6,
        reltol: float = 1e-6,
        vntol: float = 1e-6,
        **kwargs,
    ):
        self.activation = activation
        self.max_iter = max_iter
        self.atol = atol
        self.reltol = reltol
        self.vntol = vntol

        # hidden layer dimensions, set by the solver.set_model() method
        self.dims = []
        self.W = []
        self.B = []
        self.shortcuts: list[tuple[int, int, float]] | None = None

    def set_shortcuts(self, shortcuts: list[tuple[int, int, float]] | None) -> None:
        """Set shortcut information for the strategy."""
        self.shortcuts = shortcuts

    @abstractmethod
    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network.

        Args:
            x (_type_): input of the network.
            i_ext (_type_): external current.

        Returns:
            torch.Tensor: layer node potentials.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset the internal states at the beginning of 1 iteration."""
        ...

    def set_bias_type(self, bias_type: str) -> None:
        """Set the bias type of the model."""
        self.bias_type = bias_type
        ...


class AbstractSPICEStrategy(AbstractStrategy):
    """Calculate Node potentials with SPICE."""

    """
    def __new__(cls):
        # check if ngspice is installed
        cls._check_spice()
        return super().__new__(cls)
    """

    def __init__(self, SPICE_params: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.SPICE_params = SPICE_params

    @classmethod
    def _check_spice(cls):
        """Check if spice is installed."""
        raise NotImplementedError()


class XyceStrategy(AbstractSPICEStrategy):
    """Get Node potentials with Xyce."""

    def __init__(self, **kwargs) -> None:
        if not _XYCE_AVAILABLE:
            raise ImportError("Xyce is not available.")
        super().__init__(**kwargs)
        self.mpi_commands = kwargs.get("mpi_commands") or [
            "mpirun",
            "-use-hwthread-cpus",
        ]
        self.circuit = None

        if self.mpi_commands[-1] == "-cpu-set":
            # self.mpi_commands.append(str(id + 1))
            pass
        self.sim = xyce.XyceSim(mpi_commands=self.mpi_commands)

    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network with Xyce."""

        if i_ext is None:
            self.create_netlist(x)
            spice_utils.SPICENNParser.updateWeight(self.circuit, self.W)
        nodes_list = []
        # not multiprocessing yet
        batch_size = x.size(0)
        dims = [len(x[0])] + self.dims
        for i in range(batch_size):
            if i_ext is None:
                spice_utils.SPICENNParser.clampLayer(self.circuit, x[i])
            else:
                if batch_size != 1:
                    spice_utils.SPICENNParser.clampLayer(self.circuit, x[i])
                    spice_utils.SPICENNParser.releaseLayer(self.circuit, -i_ext[i])
                else:
                    spice_utils.SPICENNParser.releaseLayer(self.circuit, -i_ext[i])

            raw_file = self.sim(spice_input=self.circuit)
            voltages = spice_utils.SPICENNParser.fastRawfileParser(
                raw_file, nodenames=self.circuit.nodes, dimensions=dims
            )

            combined_voltages = np.concatenate([voltages[1][0], voltages[1][1]])
            nodes_list.append(combined_voltages)

        nodes_array = np.stack(nodes_list, axis=0)
        nodes = torch.from_numpy(nodes_array).type_as(x)
        return nodes

    def create_netlist(self, x):
        """Convert input to netlist."""
        """self.W, self.B, self.dims / diode model name in self.SPICE_params."""

        # assert x.size(0) == 1, "XyceStrategy does not support batched operation."

        if self.circuit is None:
            self.Pycircuit = circuits.create_circuit(
                input=x[0],
                bias=self.B,
                W=self.W,
                dimensions=self.dims,
                **self.SPICE_params,
            )
            self.circuit = circuits.ShallowCircuit.copyFromCircuit(self.Pycircuit)
            del self.Pycircuit

    def reset(self):
        """Reset the internal states after 1 iteration."""
        return NotImplementedError()


class PythonStrategy(AbstractStrategy):
    """Calculate Node potentials with Python."""

    def __init__(
        self,
        amp_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        """Initialize the solver."""
        self.OTS = self.activation
        self.amp_factor = amp_factor
        self.attrchecked: bool = False
        self.free_solution: torch.Tensor | None = None
        # self.sparse = (
        #     True if sum([dim**2 for dim in self.dims]) / sum(self.dims) ** 2 < 0.1 else False
        # )

    @property
    def free_solution(self):
        return self._free_solution.detach().clone() if self._free_solution is not None else None

    @free_solution.setter
    def free_solution(self, value: torch.Tensor | None):
        self._free_solution = value.detach().clone() if value is not None else None

    def check_and_set_attrs(self, kwargs: dict):
        """Check if all attributes are set and set them if not."""
        if kwargs is None and self.attrchecked:
            pass
        else:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    log.warning(f"key {key} not found in {self.__class__.__name__}")
            for attr in ["OTS", "dims", "W"]:
                if getattr(self, attr) is None:
                    raise ValueError(f"{attr} must be set before calling")


class FirstOrderStrategy(PythonStrategy):
    """Strategy for solving equilibrium points using first-order approximation.

    This strategy computes the equilibrium point by solving a linear system with
    first-order approximation of the network dynamics.

    Args:
        add_nonlin_last: Whether to add nonlinearity to the last layer.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, add_nonlin_last: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._L: torch.Tensor | None = None  # Laplacian matrix cache
        self._R: torch.Tensor | None = None  # RHS vector cache
        self._B: torch.Tensor | None = None  # Bias vector cache
        self.add_nonlin_last = add_nonlin_last
        self._set: bool = False  # Flag to track if caches are set

    @torch.no_grad()
    def bias(self) -> torch.Tensor:
        """Compute and cache the 1D bias row vector.

        Returns:
            The bias vector as a 1D tensor.
        """
        if self._B is None:
            dims = self.dims
            size = sum(dims)
            B = torch.zeros(size).type_as(self.W[0]) if not self.B else torch.cat(self.B, dim=-1)
            assert len(B.shape) == 1, "Bias must be a 1D tensor"
            self._B = B
        return self._B.detach().clone()

    @torch.no_grad()
    def build_lower_triangular(self) -> torch.Tensor:
        """Construct and return the lower triangular weight matrix.

        This method builds the sparse lower triangular matrix representing the
        connections between network layers, which is more memory-efficient than
        storing the full adjacency matrix.

        Returns:
            A tensor of shape (sum(dims), sum(dims)) representing the lower
            triangular part of the Laplacian matrix.
        """
        dims = self.dims
        size = sum(dims)
        total_network_depth = len(self.dims)

        to_layer_base_scales = {to_idx: scale for from_idx, to_idx, scale in self.shortcuts} if self.shortcuts else {}

        # 1. Build the sequential connections, applying depth-based scaling
        paddedG = [torch.zeros(dims[0], size).type_as(self.W[0])]
        for i, g in enumerate(self.W[1:]):
            layer_idx = i + 1
            
            final_scale = 1.0
            if layer_idx in to_layer_base_scales:
                base_scale = to_layer_base_scales[layer_idx]
                if total_network_depth > 0:
                    final_scale = base_scale / (total_network_depth**0.5)

            # Calculate padding sizes
            left_pad = sum(dims[:i])
            right_pad = sum(dims[1 + i :])
            # Pad each weight matrix to fit in the full-size matrix
            padded = F.pad(-g * final_scale, (left_pad, right_pad))
            paddedG.append(padded)

        # Construct lower triangular part
        Ll = torch.cat(paddedG, dim=-2)

        # 2. Add unscaled identity connections for the shortcut path `x`
        if self.shortcuts is not None:
            for from_idx, to_idx, scale in self.shortcuts:
                if from_idx >= to_idx:
                    raise ValueError(f"Shortcut 'from' index ({from_idx}) must be smaller than 'to' index ({to_idx}).")
                if dims[from_idx] != dims[to_idx]:
                    raise ValueError(f"Shortcut dimensions must match: from_layer {from_idx} (dim={dims[from_idx]}) -> to_layer {to_idx} (dim={dims[to_idx]})")

                # Get the start and end indices for the rows (to_layer)
                row_start = sum(dims[:to_idx])
                row_end = row_start + dims[to_idx]

                # Get the start and end indices for the columns (from_layer)
                col_start = sum(dims[:from_idx])
                col_end = col_start + dims[from_idx]
                
                # The shortcut connection is unscaled (identity)
                identity_block = -torch.eye(dims[from_idx], device=Ll.device, dtype=Ll.dtype)
                
                # Add the block to the Ll matrix
                Ll[row_start:row_end, col_start:col_end] += identity_block

        return Ll

    @torch.no_grad()
    def laplacian(self) -> torch.Tensor:
        """Compute and cache the 2D Laplacian + bias matrix.

        This method constructs the Laplacian matrix of the network, which represents
        the connectivity and weights between nodes.

        Returns:
            A square tensor of shape (size, size) representing the Laplacian matrix.

        Raises:
            ValueError: If the strategy needs to be reset before calling.
        """
        if self._L is None:
            dims = self.dims
            size = sum(dims)

            # Get lower triangular part using the new method
            Ll = self.build_lower_triangular()

            # Construct full Laplacian
            L = Ll * self.amp_factor + Ll.mT / self.amp_factor

            # Add diagonal terms
            D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(self.W[0].sum(-1), (0, size - dims[0]))
            L += D0.diag()

            # Add bias if present
            if self.B:
                L += self.bias().diag()

            self._L = L
            self._set = True
        elif self._set is False:
            raise ValueError("Reset the strategy before calling laplacian()")

        return self._L.detach().clone()

    @torch.no_grad()
    def rhs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute and cache the batched 2D right-hand side vector.

        This method constructs the right-hand side vector for the linear system,
        which includes the bias and input terms.

        Args:
            x: Input tensor of the network with shape (batch_size, input_dim).

        Returns:
            Batched 2D RHS vector with shape (batch_size, total_dim).

        Raises:
            ValueError: If the input shape is unsupported or if the strategy needs to be reset.
        """
        dims = self.dims
        if self._R is None:
            # Start with negative bias
            B = -self.bias()  # 1D row vector

            # Handle different input shapes
            if len(x.shape) == 2:  # batched
                B = B.expand(x.size(0), *B.shape).clone()
            elif len(x.shape) == 1:
                B = B.unsqueeze(0)
                x = x.unsqueeze(0)
            else:
                raise ValueError(f"Unsupported shape {x.shape} while constructing RHS")

            # Add input term to first layer
            B[:, : dims[0]] -= x @ self.W[0].T

            self._R = B
        elif self._set is False:
            raise ValueError("Reset the strategy before calling rhs()")

        return self._R.detach().clone()

    @torch.no_grad()
    def residual(
        self,
        v: NpOrTensor,
        x: torch.Tensor,
        i_ext: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the residual of the network equations.

        This method computes the residual v(L+b) + R + i_r(v) where v, R, i_r(v) are
        batched vectors. The residual should be zero at the equilibrium point.

        Args:
            v: Node potentials with shape (batch_size, total_dim).
            x: Input tensor of the network with shape (batch_size, input_dim).
            i_ext: External current tensor (None for free phase, tensor for nudged phase).

        Returns:
            Residual vector with shape (batch_size, total_dim).

        Raises:
            TypeError: If the input type is not supported.
        """
        # Get Laplacian and RHS
        L = self.laplacian()
        R = self.rhs(x)

        # Add external current if present
        if i_ext is not None:
            R[:, -self.dims[-1] :] += i_ext * self.amp_factor

        # Convert numpy array to tensor if needed
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type_as(x)

        # Add batch dimension if needed
        if len(v.shape) == 1:
            v = v.unsqueeze(0)

        # Compute linear part of residual: v*L + R
        f = torch.einsum("bi, oi->bo", v, L) + R

        # Add nonlinear part based on configuration
        if self.add_nonlin_last:
            if hasattr(self.OTS, "i_layerwise"):
                f += self.OTS.i_layerwise(v, self.dims)
            else:
                f += self.OTS.i(v)
        else:
            # Apply nonlinearity to all layers except the last one
            if hasattr(self.OTS, "i_layerwise"):
                hidden_dims = self.dims[:-1]
                hidden_nodes = sum(hidden_dims)
                f[:, :hidden_nodes] += self.OTS.i_layerwise(v[:, :hidden_nodes], hidden_dims)
            else:
                f[:, : -self.dims[-1]] += self.OTS.i(v[:, : -self.dims[-1]])

        # Return result in appropriate format
        return f

    @torch.no_grad()
    def lin_solve(self, x: torch.Tensor, i_ext: torch.Tensor | None) -> torch.Tensor:
        """Solve the linear system (L+b)v = -(R + i_ext).

        This method solves the linear part of the network equations, which provides
        a good initial guess for iterative solvers.

        Args:
            x: Input tensor of the network.
            i_ext: External current tensor (None for free phase, tensor for nudged phase).

        Returns:
            Solution of the linear system.
        """
        # Reuse free solution if available for nudged phase
        if self._free_solution is not None and i_ext is not None:
            return self.free_solution

        # Solve linear system using Cholesky decomposition
        lo = torch.linalg.cholesky(self.laplacian())
        R = self.rhs(x)

        # Add external current if present
        if i_ext is not None:
            R[:, -self.dims[-1] :] += i_ext * self.amp_factor

        # Solve and return
        v = torch.cholesky_solve(-R.unsqueeze(-1), lo).squeeze(-1)
        return v

    def reset(self) -> None:
        """Reset cached computations.

        This method clears all cached matrices and vectors, which should be called
        at the beginning of each free phase to ensure fresh computation.
        """
        self._L = None
        self._R = None
        self._B = None
        self._set = False


class SecondOrderStrategy(FirstOrderStrategy):
    """Strategy for solving equilibrium points using second-order approximation.

    This strategy extends the first-order strategy by computing the Jacobian matrix,
    which can be used for more advanced solvers like Newton's method.

    Args:
        eps: Small value to add to the diagonal of the Laplacian matrix for numerical stability.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, eps: float = 1e-8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    @torch.no_grad()
    def jacobian(self, v: NpOrTensor) -> torch.Tensor:
        """Compute the Jacobian matrix of the residual function.

        This method computes the Jacobian matrix of the residual function with respect
        to the node potentials. The Jacobian is used in Newton-type methods.

        Args:
            v: Node potentials with shape (batch_size, total_dim) or (total_dim,).

        Returns:
            Jacobian matrix with shape (batch_size, total_dim, total_dim).

        Raises:
            ValueError: If the input shape is unsupported.
            TypeError: If the input type is not supported.
        """
        # Get Laplacian and add small diagonal term for stability
        v_torch = v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
        L = self.laplacian() + self.eps * torch.eye(v.shape[-1]).type_as(v_torch)

        # Handle different input shapes
        if len(v.shape) == 2:
            batchsize = v.shape[0]
        elif len(v.shape) == 1:
            batchsize = 1
            v_torch.unsqueeze_(0)
        else:
            raise ValueError(f"Unsupported shape {v.shape} while constructing Jacobian")

        # Create batched Jacobian matrix
        J = L.expand(batchsize, *L.shape).clone()

        # Add nonlinear derivative terms to diagonal
        if self.add_nonlin_last:
            J.diagonal(dim1=1, dim2=2)[:] += self.OTS.a(v_torch)
        else:
            J.diagonal(dim1=1, dim2=2)[:, : -self.dims[-1]] += self.OTS.a(
                v_torch[:, : -self.dims[-1]]
            )

        # Return in appropriate format
        if isinstance(v, torch.Tensor):
            return J
        elif isinstance(v, np.ndarray):
            return J.numpy()
        else:
            raise TypeError(f"Unsupported type {type(v)} while constructing Jacobian")


class GradientDescentStrategy(FirstOrderStrategy):
    """Strategy for solving equilibrium points using gradient descent.

    This strategy uses gradient descent to iteratively minimize the residual
    and find the equilibrium point of the network.

    Args:
        alpha: Learning rate for gradient descent steps.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    @cache_free_solution
    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext: torch.Tensor | None, **kwargs: Any) -> torch.Tensor:
        """Solve for the equilibrium point of the network using gradient descent.

        This method iteratively updates the node potentials in the direction of
        the negative residual until convergence or maximum iterations.

        Args:
            x: Input tensor of the network.
            i_ext: External current tensor (None for free phase, tensor for nudged phase).
            **kwargs: Additional keyword arguments for attribute setting.

        Returns:
            Equilibrium node potentials.
        """
        # Check and set attributes
        self.check_and_set_attrs(kwargs)

        # Initialize with linear solution
        v = self.lin_solve(x, i_ext)

        # Gradient descent iterations
        for idx in range(self.max_iter):
            # Compute update direction
            dv = -self.residual(v, x, i_ext)

            # Update node potentials
            v += self.alpha * dv

            # Check convergence
            if dv.abs().max() < self.atol:
                log.debug(f"Gradient descent converged in {idx} iterations")
                return v

        # Warning if not converged
        log.warning(
            f"Gradient descent did not converge in {self.max_iter} iterations, "
            f"residual={dv.abs().max():.3e}"
        )
        return v


class QPStrategy(FirstOrderStrategy):
    """Strategy for solving equilibrium points using quadratic programming.

    This strategy formulates the equilibrium point finding problem as a quadratic program
    and uses solvers from the qpsolvers library to find the solution.

    Args:
        add_nonlin_last: Whether to add nonlinearity to the last layer.
        solver_type: Type of QP solver to use (default: "proxqp").
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self, add_nonlin_last: bool = True, solver_type: str = "proxqp", **kwargs: Any
    ) -> None:
        if not _QPSOLVERS_AVAILABLE:
            raise ImportError("QPSolvers is not available. Please install the qpsolvers package.")
        super().__init__(add_nonlin_last, **kwargs)
        self.solver_type = solver_type

    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext: torch.Tensor | None, **kwargs: Any) -> torch.Tensor:
        """Solve for the equilibrium point of the network using quadratic programming.

        This method formulates the problem as a quadratic program with box constraints
        and uses a QP solver to find the solution.

        Args:
            x: Input tensor of the network.
            i_ext: External current tensor (None for free phase, tensor for nudged phase).
            **kwargs: Additional keyword arguments for attribute setting and QP solver.

        Returns:
            Equilibrium node potentials.
        """
        # Check and set attributes
        self.check_and_set_attrs(kwargs)

        # Get Laplacian matrix (P) and right-hand side vector (q)
        P = self.laplacian().cpu().numpy()
        R = self.rhs(x)

        # Add external current if present
        if i_ext is not None:
            R[:, -self.dims[-1] :] += i_ext * self.amp_factor

        # Convert to numpy for QP solver
        q = R.squeeze().cpu().numpy()

        # Set box constraints based on activation function limits
        lb = self.OTS.Vl * np.ones_like(q)  # Lower bounds
        ub = self.OTS.Vr * np.ones_like(q)  # Upper bounds

        # Solve QP problem
        v = solve_qp(P, q, lb=lb, ub=ub, solver=self.solver_type, **kwargs)

        # Convert back to tensor and add batch dimension if needed
        v = torch.from_numpy(v).type_as(x).unsqueeze(0)

        return v

    def sparse_laplacian(self) -> None:
        """Compute the 2D Laplacian + bias matrix in sparse format.

        This method is a placeholder for a potential optimization using sparse matrices.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("Sparse Laplacian computation not implemented")


class ProxQPStrategy(FirstOrderStrategy):
    """Strategy for solving equilibrium points using the ProxQP solver.

    This strategy uses the ProxQP solver from the proxsuite package, which is
    optimized for large-scale quadratic programming problems and supports
    parallel solving for batched inputs.

    Args:
        num_threads: Number of threads to use for parallel solving (None for auto).
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, num_threads: int | None = None, **kwargs: Any) -> None:
        if not _PROXSUITE_AVAILABLE:
            raise ImportError("ProxSuite is not available. Please install the proxsuite package.")
        super().__init__(**kwargs)
        self.num_threads = (
            proxsuite.proxqp.omp_get_max_threads() - 1 if num_threads is None else num_threads
        )
        self.qps: proxsuite.proxqp.dense.VectorQP | None = None
        self._qp_initialized = False

    def __del__(self):
        """Destructor to ensure proper cleanup of QP problems."""
        self._cleanup_qps()
    
    # Gemini에 의해 추가
    def __getstate__(self):
        # Copy the object's state
        state = self.__dict__.copy()
        # Remove the unpicklable 'qps' attribute
        if 'qps' in state:
            del state['qps']
        return state

    # Gemini에 의해 추가
    def __setstate__(self, state):
        # Restore the object's state
        self.__dict__.update(state)
        # Re-initialize 'qps' to None, it will be created when needed
        self.qps = None
        self._qp_initialized = False

    def _cleanup_qps(self) -> None:
        """Internal method to properly clean up QP problems."""
        if hasattr(self, "qps") and self.qps is not None:
            try:
                # Clear each QP problem individually to prevent nanobind leaks
                for qp in self.qps:
                    if qp is not None and hasattr(qp, "clear"):
                        qp.clear()
                # Clear the vector container
                self.qps.clear()
                del self.qps
                self.qps = None
                self._qp_initialized = False
            except Exception as e:
                log.warning(f"Error during QP cleanup: {e}")
                # Force cleanup
                self.qps = None
                self._qp_initialized = False

    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext: torch.Tensor | None, **kwargs: Any) -> torch.Tensor:
        """Solve for the equilibrium point of the network using ProxQP.

        This method uses the ProxQP solver to find the equilibrium point, with
        support for batched inputs and parallel solving.

        Args:
            x: Input tensor of the network.
            i_ext: External current tensor (None for free phase, tensor for nudged phase).
            **kwargs: Additional keyword arguments.

        Returns:
            Equilibrium node potentials.
        """
        batch_size, _ = x.shape
        g = self.rhs(x).cpu().numpy()

        # Initialize similarity transform vector for amp_factor > 1
        if self.amp_factor > 1.0 and not hasattr(self, "_s_vector"):
            s_diag_elements = []
            for layer_idx, dim_size in enumerate(self.dims):
                scale_factor = (1 / self.amp_factor) ** layer_idx
                s_diag_elements.extend([scale_factor] * dim_size)
            self._s_vector = np.array(s_diag_elements)

        # Free phase: initialize QP problems
        if i_ext is None:
            # Clean up previous QP problems to prevent memory leaks
            self._cleanup_qps()
            # Get Hessian matrix
            H = self.laplacian().cpu().numpy()

            # Apply similarity transform if amp_factor > 1
            if self.amp_factor > 1.0:
                s_np = self._s_vector
                s_inv_np = 1.0 / s_np
                # Transform: H_hat = S * H * S^-1
                H = s_np[:, np.newaxis] * H * s_inv_np[np.newaxis, :]
                # Transform RHS: g_hat = S * g
                g = g * s_np[np.newaxis, :]

            n = n_ineq = H.shape[0]

            # No equality or inequality constraints (only box constraints)
            A = b = C = lower = upper = None

            # Set box constraints based on activation function limits
            if hasattr(self.OTS, "get_box_constraints"):
                # DifferentialPairRectifier case - apply constraints only to hidden layers
                # Exclude output layer to match ResistiveNetworkStrategy behavior
                hidden_dims = self.dims[:-1]  # All layers except output
                output_dim = self.dims[-1]

                if hidden_dims:
                    # Get constraints for hidden layers only
                    lb_hidden, ub_hidden = self.OTS.get_box_constraints(hidden_dims)
                    # Create unconstrained bounds for output layer
                    lb_output = torch.full((output_dim,), float("-inf"))
                    ub_output = torch.full((output_dim,), float("inf"))
                    # Concatenate hidden and output bounds
                    lb_tensor = torch.cat([lb_hidden, lb_output])
                    ub_tensor = torch.cat([ub_hidden, ub_output])
                else:
                    # If no hidden layers, all bounds are unconstrained
                    lb_tensor = torch.full((sum(self.dims),), float("-inf"))
                    ub_tensor = torch.full((sum(self.dims),), float("inf"))

                self.lb = lb_tensor.cpu().numpy()
                self.ub = ub_tensor.cpu().numpy()
                # Handle infinite bounds for numerical stability
                self.lb = np.clip(self.lb, -1e20, 1e20)
                self.ub = np.clip(self.ub, -1e20, 1e20)
            else:
                # Traditional rectifier case
                self.lb = self.OTS.Vl * np.ones(n)
                self.ub = self.OTS.Vr * np.ones(n)

            # Transform box constraints if using similarity transform
            if self.amp_factor > 1.0:
                s_np = self._s_vector
                # Transform bounds: x_hat = S * x, so bounds need to be transformed
                self.lb = self.lb * s_np
                self.ub = self.ub * s_np

            # Create vector of QP problems (one per batch item)
            self.qps = proxsuite.proxqp.dense.VectorQP()
            for i in range(batch_size):
                qp = proxsuite.proxqp.dense.QP(
                    n, 0, n_ineq, True
                )  # dim, n_eq, n_ineq, is_box_constrained
                qp.init(H, g[i], A, b, C, lower, upper, self.lb, self.ub)
                self.qps.append(qp)
            self._qp_initialized = True
        # Nudged phase: update existing QP problems
        else:
            # Add external current to RHS
            g[:, -self.dims[-1] :] += i_ext.cpu().numpy()
            # Apply similarity transform to RHS if amp_factor > 1
            if self.amp_factor > 1.0:
                s_np = self._s_vector
                g = g * s_np[np.newaxis, :]

            # Update each QP problem with new RHS and warm start
            for idx, qp in enumerate(self.qps):
                qp.settings.initial_guess = (
                    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
                qp.update(g=g[idx], l_box=self.lb, u_box=self.ub)

        # Solve all QP problems in parallel
        proxsuite.proxqp.dense.solve_in_parallel(self.qps, self.num_threads)

        # Collect results
        nodes_list = []
        for i in range(batch_size):
            vout = self.qps[i].results.x.copy()

            # Apply inverse similarity transform if amp_factor > 1
            if self.amp_factor > 1.0:
                vout /= self._s_vector

            nodes_list.append(torch.from_numpy(vout).type_as(x))

        # Stack results into a batch
        nodes = torch.stack(nodes_list, dim=0)
        return nodes

    def reset(self) -> None:
        """Reset cached computations.

        This method clears all cached matrices, vectors, and QP problems.
        """
        super().reset()
        # Clean up QP problems with proper cleanup
        self._cleanup_qps()


class ResistiveNetworkStrategy(FirstOrderStrategy):
    """Strategy for solving equilibrium points using resistive network dynamics.

    This strategy implements the quadratic coordinate descent algorithm from
    energy-based learning for resistive networks. It uses analytical solutions
    for quadratic energy functions to achieve fast convergence.

    Args:
        activation: The activation function rectifier to use.
        update_mode: Update mode ('forward', 'backward', 'synchronous', 'asynchronous').
        num_iterations: Number of coordinate descent iterations.
        initialization: Initialization method ('linear', 'zero', 'random').
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        activation: activation.AbstractRectifier,
        update_mode: str = "asynchronous",
        num_iterations: int = 6,
        initialization: str = "zero",
        **kwargs: Any,
    ) -> None:
        super().__init__(activation=activation, **kwargs)
        self.update_mode = update_mode
        self.num_iterations = num_iterations
        self.initialization = initialization
        self._layer_states: list[torch.Tensor] = []

    @torch.no_grad()
    def quadratic_update(
        self, x: torch.Tensor, layer_idx: int, i_ext: torch.Tensor | None
    ) -> torch.Tensor:
        """Perform quadratic coordinate descent update for a single layer.

        This method computes the analytical minimum of the quadratic energy function
        for a specific layer based on the resistive network energy:
        E = 0.5 * Σ((z_pre - z_post)² * W)

        Args:
            x: Input tensor of the network.
            layer_idx: Index of the layer to update.
            i_ext: External current tensor (None for free phase).

        Returns:
            Updated layer activations.
        """
        if layer_idx >= len(self.dims):
            raise ValueError(f"Layer index {layer_idx} exceeds network depth {len(self.dims)}")

        batch_size = x.size(0)
        layer_dim = self.dims[layer_idx]
        device = x.device
        dtype = x.dtype

        # Initialize quadratic and linear coefficients
        a = torch.zeros(batch_size, layer_dim, device=device, dtype=dtype)
        b = torch.zeros(batch_size, layer_dim, device=device, dtype=dtype)

        # Add bias contribution (linear term only)
        if self.B and layer_idx < len(self.B):
            b += self.B[layer_idx].unsqueeze(0).expand(batch_size, -1)

        # Get the res_scale for the current layer's incoming weights
        res_scale = 1.0  # Default to 1 if no shortcut is defined
        if self.shortcuts is not None:
            for from_idx, to_idx, scale in self.shortcuts:
                if to_idx == layer_idx:
                    res_scale = scale
                    break

        # Connection to previous layer (this layer is post-synaptic)
        if layer_idx == 0:
            # First layer connects to input, not scaled by res_scale
            weight = self.W[0]
            a += 0.5 * weight.sum(dim=1).unsqueeze(0)
            b -= torch.matmul(x, weight.T)
        else:
            # Hidden layer connects to previous layer, scaled by res_scale
            weight = self.W[layer_idx] * res_scale
            a += 0.5 * weight.sum(dim=1).unsqueeze(0)
            b -= torch.matmul(self._layer_states[layer_idx - 1], weight.T)

        # Connection to next layer (this layer is pre-synaptic)
        if layer_idx < len(self.dims) - 1:
            # The connection *from* this layer *to* the next uses the next layer's res_scale
            next_res_scale = 1.0
            if self.shortcuts is not None:
                for from_idx, to_idx, scale in self.shortcuts:
                    if to_idx == layer_idx + 1:
                        next_res_scale = scale
                        break
            weight = self.W[layer_idx + 1] * next_res_scale
            a += 0.5 * weight.sum(dim=0).unsqueeze(0)
            b -= torch.matmul(self._layer_states[layer_idx + 1], weight)

        # Add identity shortcut connection if it exists
        if self.shortcuts is not None:
            for from_idx, to_idx, scale in self.shortcuts:
                if to_idx == layer_idx and from_idx == layer_idx - 1:
                    identity_scale = 1.0  # Use a fixed scale of 1.0 for the identity part
                    z_prev = self._layer_states[from_idx]
                    # Energy term: 0.5 * identity_scale * (z_prev - z_curr)^2
                    # This adds 0.5 * identity_scale to 'a' and -identity_scale * z_prev to 'b'
                    a += 0.5 * identity_scale
                    b -= identity_scale * z_prev
                    break  # Assume only one adjacent shortcut

        # Add external current for nudged phase (output layer only)
        if i_ext is not None and layer_idx == len(self.dims) - 1:
            if i_ext.shape[1] == layer_dim:
                b += i_ext
            else:
                start_idx = sum(self.dims[:layer_idx])
                end_idx = start_idx + layer_dim
                b += i_ext[:, start_idx:end_idx]

        # Analytical solution: z = -b / (2 * a)
        a = torch.clamp(a, min=1e-12)
        z_new = -b / (2.0 * a)

        # Apply activation constraints
        is_output_layer = layer_idx == len(self.dims) - 1
        if hasattr(self.activation, "get_box_constraints") and not is_output_layer:
            if self.dims[layer_idx] % 2 != 0:
                raise ValueError(f"Layer {layer_idx} must have even nodes for differential pairs")
            half_dim = self.dims[layer_idx] // 2
            z_new[:, :half_dim] = z_new[:, :half_dim].clamp(min=0.0)
            z_new[:, half_dim:] = z_new[:, half_dim:].clamp(max=0.0)
        else:
            if hasattr(self.activation, "Vl") and hasattr(self.activation, "Vr"):
                z_new = torch.clamp(z_new, min=self.activation.Vl, max=self.activation.Vr)

        return z_new

    @torch.no_grad()
    def initialize_states(self, x: torch.Tensor, i_ext: torch.Tensor | None) -> list[torch.Tensor]:
        """Initialize layer states according to the specified initialization method.

        Args:
            x: Input tensor of the network.
            i_ext: External current tensor (None for free phase).

        Returns:
            List of initialized layer state tensors.
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if self.initialization == "linear":
            # Use linear solve for initialization (original behavior)
            try:
                v_linear = self.lin_solve(x, i_ext)
                layer_states = torch.split(v_linear, self.dims, dim=-1)
                return [layer.clone() for layer in layer_states]
            except torch._C._LinAlgError:
                # Fallback to zero initialization if linear solve fails
                print(
                    "Warning: Linear solve failed (matrix not positive definite), falling back to zero initialization"
                )
                layer_states = []
                for dim in self.dims:
                    layer_states.append(torch.zeros(batch_size, dim, dtype=dtype, device=device))
                return layer_states

        elif self.initialization == "zero":
            # Zero initialization (energy-based-learning compatible)
            layer_states = []
            for dim in self.dims:
                layer_states.append(torch.zeros(batch_size, dim, dtype=dtype, device=device))
            return layer_states

        elif self.initialization == "random":
            # Random initialization with small values
            layer_states = []
            for dim in self.dims:
                layer_states.append(torch.randn(batch_size, dim, dtype=dtype, device=device) * 0.01)
            return layer_states

        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")

    @cache_free_solution
    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext: torch.Tensor | None, **kwargs: Any) -> torch.Tensor:
        """Solve for the equilibrium point using resistive network dynamics.

        This method implements coordinate descent with analytical updates for
        each layer, following the resistive network energy minimization approach.

        Args:
            x: Input tensor of the network.
            i_ext: External current tensor (None for free phase).
            **kwargs: Additional keyword arguments.

        Returns:
            Equilibrium node potentials.
        """
        # Check and set attributes
        self.check_and_set_attrs(kwargs)

        # Initialize layer states according to the specified method
        self._layer_states = self.initialize_states(x, i_ext)

        # Coordinate descent iterations
        for iteration in range(self.num_iterations):
            # Determine update order based on mode
            if self.update_mode == "forward":
                layer_indices = list(range(len(self.dims)))
            elif self.update_mode == "backward":
                layer_indices = list(range(len(self.dims) - 1, -1, -1))
            elif self.update_mode == "synchronous":
                # Update all layers simultaneously
                new_states = []
                for layer_idx in range(len(self.dims)):
                    z_new = self.quadratic_update(x, layer_idx, i_ext)
                    new_states.append(z_new)
                self._layer_states = new_states
                continue
            else:  # asynchronous (default)
                # Alternate between forward and backward passes
                if iteration % 2 == 0:
                    layer_indices = list(range(len(self.dims)))
                else:
                    layer_indices = list(range(len(self.dims) - 1, -1, -1))

            # Update layers in the determined order
            for layer_idx in layer_indices:
                z_new = self.quadratic_update(x, layer_idx, i_ext)
                self._layer_states[layer_idx] = z_new

        # Concatenate all layer states
        v_final = torch.cat(self._layer_states, dim=-1)
        return v_final

    def reset(self) -> None:
        """Reset cached computations and layer states."""
        super().reset()
        self._layer_states = []
