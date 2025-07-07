import numpy as np
import torch
import scipy.linalg  # Added import
import scipy.linalg.lapack as lapack  # Added import

from .strategies import SecondOrderStrategy
from src.utils import _SCIPY_AVAILABLE, RankedLogger

if _SCIPY_AVAILABLE:
    from scipy.optimize import fsolve, approx_fprime, minimize, newton_krylov, root

log = RankedLogger(__name__, rank_zero_only=True)


class ScipyStrategy(SecondOrderStrategy):
    """Solve for the equilibrium point of the network with Scipy."""

    def __init__(self, method="hybr", **kwargs) -> None:
        super().__init__(**kwargs)
        self.method = method

    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network using SciPy's root method.

        Currently only supports cpu. & does not support batched
        operation in parallel.
        """
        if x.device != torch.device("cpu"):
            raise ValueError("ScipyStrategy only supports cpu.")

        self.check_and_set_attrs(kwargs)
        v_init = self.lin_solve(x, i_ext).numpy()
        # Get default options for root method
        options = {
            "xtol": self.atol,
            "maxiter": self.max_iter,
        }

        # Update with any additional options from kwargs
        if "root_options" in kwargs:
            options.update(kwargs["root_options"])
        self.reset()
        vout_list = []
        # scipy strategy does not support batched operation in parallel
        for xi, v0i, i_exti in zip(x, v_init, i_ext):
            # minimize the residual w.r.t v, hidden potential
            # unsqueeze the first dimension to match the expected input shape
            res = root(
                lambda v, x, i_ext: self.residual(v, x, i_ext).squeeze(),
                x0=v0i,  # x0 represents the initial guess of first argument, v
                args=(xi, i_exti),
                jac=lambda v, _, __: self.jacobian(v).squeeze(),
                method=self.method,
                options=options,
            )

            if not res.success:
                log.warning(f"{self.__class__.__name__} did not converge: {res.message}")

            vout_list.append(res.x)
            self.reset()

        vout = np.stack(vout_list, axis=0)
        vout = torch.from_numpy(vout).type_as(x)

        if i_ext is None:
            self._free_solution = vout.detach().clone()

        return vout


class TrustRegionStrategy(ScipyStrategy):
    """
    SciPy의 trust-constr를 활용하여 F(x)=Lx+I_+i(x)=0 문제를 풀기 위한 Strategy.
    (box constraint: x in [alpha, beta]는 필요에 따라 추가할 수 있음)
    """

    def __init__(self, gamma=0.1, alpha=-1.0, beta=1.0, **kwargs):
        super().__init__(method="trust-krylov", **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def residual(self, x_np, xi, i_ext):
        """목적 함수: F(x) = Lx + I_ + i(x)"""
        L = self.laplacian().cpu().numpy()
        I_ = self.rhs(xi).squeeze(0).cpu().numpy()

        x_tensor = torch.from_numpy(x_np).float()
        fx = L @ x_np + I_ + self.OTS.i(x_tensor).cpu().numpy()
        return fx

    def jacobian(self, x_np, xi, i_ext):
        """Jacobian 계산: J(x) = L + diag(a(x))"""
        L = self.laplacian().cpu().numpy()

        x_tensor = torch.from_numpy(x_np).float()
        a_x = self.OTS.a(x_tensor).cpu().numpy()
        return L + np.diag(a_x)

    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network using Trust-Region method."""
        # Add trust-krylov specific options
        root_options = kwargs.get("root_options", {})
        root_options.update({"gtol": self.atol})  # trust-krylov uses gtol instead of xtol
        kwargs["root_options"] = root_options

        return super().solve(x, i_ext, **kwargs)


class QuasiNewtonStrategy(ScipyStrategy):
    """
    BFGS를 활용하여 f(x)=1/2||F(x)||^2 (F(x)=Lx+I_+i(x)) 최소화를 통해 해 F(x)=0를 찾는 Strategy.
    """

    def __init__(self, gamma=0.1, alpha=-1.0, beta=1.0, **kwargs):
        # Note: We're still using root() but with a different method
        super().__init__(method="lm", **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def residual(self, x_np, xi, i_ext):
        """F(x) = Lx + I_ + i(x)"""
        L = self.laplacian().cpu().numpy()
        I_ = self.rhs(xi).squeeze(0).cpu().numpy()

        x_tensor = torch.from_numpy(x_np).float()
        fx = L @ x_np + I_ + self.OTS.i(x_tensor).cpu().numpy()
        return fx

    def jacobian(self, x_np, xi, i_ext):
        """Jacobian 계산: J(x) = L + diag(a(x))"""
        L = self.laplacian().cpu().numpy()

        x_tensor = torch.from_numpy(x_np).float()
        a_x = self.OTS.a(x_tensor).cpu().numpy()
        return L + np.diag(a_x)

    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network using Levenberg-Marquardt method."""
        # Add Levenberg-Marquardt specific options
        root_options = kwargs.get("root_options", {})
        root_options.update(
            {
                "ftol": self.atol,
                "factor": 100,  # Levenberg-Marquardt damping factor
            }
        )
        kwargs["root_options"] = root_options

        return super().solve(x, i_ext, **kwargs)


class NewtonKrylovStrategy(ScipyStrategy):
    """
    SciPy의 newton_krylov 함수를 사용하여 F(x)=Lx+I_+i(x)=0 문제를 해결하는 Strategy.
    """

    def __init__(self, gamma=0.1, alpha=-1.0, beta=1.0, **kwargs):
        super().__init__(method="krylov", **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def residual(self, x_np, xi, i_ext):
        """F(x) = Lx + I_ + i(x)"""
        L = self.laplacian().cpu().numpy()
        I_ = self.rhs(xi).squeeze(0).cpu().numpy()

        x_tensor = torch.from_numpy(x_np).float()
        fx = L @ x_np + I_ + self.OTS.i(x_tensor).cpu().numpy()
        return fx

    def jacobian(self, x_np, xi, i_ext):
        """Jacobian 계산: J(x) = L + diag(a(x))"""
        L = self.laplacian().cpu().numpy()

        x_tensor = torch.from_numpy(x_np).float()
        a_x = self.OTS.a(x_tensor).cpu().numpy()
        return L + np.diag(a_x)

    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network using Newton-Krylov method."""
        # Add Krylov specific options
        root_options = kwargs.get("root_options", {})
        root_options.update(
            {
                "f_tol": self.atol,
                "f_solver": "lgmres",  # Linear solver to use
            }
        )
        kwargs["root_options"] = root_options

        try:
            return super().solve(x, i_ext, **kwargs)
        except Exception as e:
            log.warning(f"NewtonKrylovStrategy failed: {e}")
            # 실패 시 초기값 반환
            batch_size = x.size(0)
            n = sum(self.dims)
            x0_np = np.zeros((batch_size, n))
            return torch.from_numpy(x0_np).float().to(x.device)


class ScipyNewtonStrategy(ScipyStrategy):
    """
    Solve J*dv = -f with Newton's method using SciPy for linear algebra.
    This strategy processes samples in a batch iteratively due to SciPy's
    typical non-batched operation for low-level routines.
    It attempts to use pivoted Cholesky factorization (pstrf) and falls back to LU / pseudo-inverse.
    """

    def __init__(
        self, clip_threshold: float, attn_factor: float = 1.0, momentum: float = 0.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)  # method from ScipyStrategy.__init__ is not used here
        self.clip_threshold = clip_threshold
        self.attn_factor = attn_factor
        self._initial_attn_factor = attn_factor  # Store for resetting per sample
        self.momentum = momentum

    def jacobian(
        self,
        v_np: np.ndarray,  # xi_torch and i_exti_torch are no longer needed here
    ) -> np.ndarray:
        """Compute the Jacobian J(v) = L + diag(a(v)) using NumPy, via parent method.
        This includes the eps stabilization term.

        Args:
            v_np: Current state vector (NumPy array).

        Returns:
            Jacobian matrix (NumPy array).
        """
        # Call SecondOrderStrategy.jacobian
        # It expects v: NpOrTensor. If v_np is (total_dim,), parent will process it as batch_size=1.
        # If input to parent is NumPy array, it returns a NumPy array.
        # The result J_np_batched will be (1, total_dim, total_dim)
        J_np_batched = super().jacobian(v_np)

        return J_np_batched.squeeze(0)

    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network using Newton's method with SciPy.
        Processes samples iteratively.
        """
        original_x_device = x.device
        original_x_dtype = x.dtype

        # Ensure tensors are on CPU for SciPy processing if not already
        # Store original device and dtype to return tensor on the original device/dtype
        current_x_device = x.device
        if current_x_device != torch.device("cpu"):
            log.warning(
                "ScipyNewtonStrategy currently forces CPU usage for internal computations. Tensors will be moved to CPU."
            )
            x_cpu = x.cpu()
            i_ext_cpu = i_ext.cpu() if i_ext is not None else None
        else:
            x_cpu = x
            i_ext_cpu = i_ext

        self.check_and_set_attrs(kwargs)

        # Batched initial guess, on CPU
        # self.lin_solve expects x and i_ext to be on the same device as model parameters.
        # Temporarily move them to original device if they were moved to CPU for the strategy overall.
        # Or, ensure lin_solve can handle CPU inputs if model is on GPU (might involve moving model params temporarily or lin_solve itself moving data)
        # For now, assume lin_solve is called with CPU tensors if the strategy forces CPU.
        v_init_torch = self.lin_solve(x_cpu, i_ext_cpu)

        vout_list = []

        for i in range(x_cpu.shape[0]):  # Iterate over batch
            self.reset()  # Reset after lin_solve to clear any internal state
            # Prepare single sample tensors (still on CPU)
            xi_torch_sample = x_cpu[i : i + 1]
            i_exti_torch_sample = i_ext_cpu[i : i + 1] if i_ext_cpu is not None else None

            v_np = v_init_torch[i].cpu().numpy()  # Ensure v_init is on CPU

            current_attn_factor = self._initial_attn_factor
            p_np = np.zeros_like(v_np)

            # Pass CPU tensors to residual and jacobian
            v = torch.from_numpy(v_np)
            residual_v_np = (
                self.residual(v, xi_torch_sample, i_exti_torch_sample).squeeze(0).cpu().numpy()
            )

            iter_idx = 0
            for iter_idx in range(self.max_iter):
                if np.max(np.abs(residual_v_np)) <= self.atol:
                    break

                J_np = self.jacobian(v_np)  # Pass only v_np

                try:
                    # Attempt pivoted Cholesky factorization (pstrf)
                    # J_np should be float32 or float64 based on v_np and L
                    current_dtype = J_np.dtype
                    if current_dtype == np.float32:
                        pstrf_routine = lapack.spstrf
                    elif current_dtype == np.float64:
                        pstrf_routine = lapack.dpstrf
                    else:
                        # Attempt to convert to float64 if not supported, or raise error
                        log.warning(
                            f"Unsupported dtype {J_np.dtype} for pstrf. Attempting conversion to float64."
                        )
                        J_np_converted = J_np.astype(np.float64)
                        if J_np_converted.dtype == np.float64:
                            pstrf_routine = lapack.dpstrf
                            current_dtype = np.float64
                            J_np = J_np_converted  # Use the converted matrix
                        else:  # Should not happen if astype worked
                            raise TypeError(
                                f"Unsupported dtype {J_np.dtype} for pstrf and conversion failed."
                            )

                    # pstrf overwrites the input matrix, so pass a Fortran-ordered copy
                    J_np_fortran_copy = np.asfortranarray(J_np, dtype=current_dtype)

                    # lower=0 means factor U such that P A P^T = U^T U (U is upper triangular)
                    c_factor, piv_arr, rank, info = pstrf_routine(
                        J_np_fortran_copy,
                        lower=0,
                        tol=-1.0,  # tol=-1.0 for default LAPACK tolerance
                    )

                    if info == 0:  # Successful factorization
                        U_factor = np.triu(c_factor)
                        p_idx = piv_arr - 1  # 0-indexed pivot array

                        # Validate p_idx before using it for indexing
                        N_dim = J_np.shape[0]
                        if np.any(p_idx >= N_dim) or np.any(p_idx < 0):
                            log.warning(
                                f"pstrf reported success (info=0) for J_np shape {J_np.shape}, "
                                f"but pivot array p_idx ({p_idx}) contains out-of-bounds indices. Falling back."
                            )
                            raise scipy.linalg.LinAlgError(
                                "pstrf (info=0) resulted in out-of-bounds pivot indices."
                            )
                        # Additional check for permutation validity (optional but good practice)
                        if len(np.unique(p_idx)) != N_dim:
                            log.warning(
                                f"pstrf reported success (info=0) for J_np shape {J_np.shape}, "
                                f"but pivot array p_idx ({p_idx}) is not a valid permutation "
                                f"(unique elements: {len(np.unique(p_idx))}, expected: {N_dim}). Falling back."
                            )
                            raise scipy.linalg.LinAlgError(
                                "pstrf (info=0) resulted in a non-permutation pivot array."
                            )

                        b_target_perm = (-residual_v_np)[p_idx]

                        U_rank = U_factor[:rank, :rank]
                        b_perm_rank = b_target_perm[:rank]

                        z = scipy.linalg.solve_triangular(
                            U_rank, b_perm_rank, trans="T", lower=False, check_finite=False
                        )
                        y_rank = scipy.linalg.solve_triangular(
                            U_rank, z, trans="N", lower=False, check_finite=False
                        )

                        dv_perm = np.zeros_like(-residual_v_np, dtype=current_dtype)
                        dv_perm[:rank] = y_rank

                        dv_np = np.zeros_like(-residual_v_np, dtype=current_dtype)
                        dv_np[p_idx] = dv_perm
                    elif info > 0:
                        log.warning(
                            f"pstrf: matrix not positive definite (info={info}, rank={rank}). Sample {i}, iter {iter_idx}. Falling back."
                        )
                        raise scipy.linalg.LinAlgError("Matrix not positive definite for pstrf.")
                    else:  # info < 0
                        log.error(
                            f"pstrf: illegal argument (info={info}). Sample {i}, iter {iter_idx}. Falling back."
                        )
                        raise ValueError("Illegal argument for pstrf.")

                except (scipy.linalg.LinAlgError, ValueError, TypeError) as e_pstrf:
                    log.debug(
                        f"Pivoted Cholesky (pstrf) failed for sample {i} iter {iter_idx}: {e_pstrf}. Trying LU."
                    )
                    try:
                        lu, piv_lu = scipy.linalg.lu_factor(J_np, check_finite=False)
                        dv_np = scipy.linalg.lu_solve(
                            (lu, piv_lu), -residual_v_np, check_finite=False
                        )
                    except (np.linalg.LinAlgError, ValueError) as e_lu:
                        log.warning(
                            f"LU solve failed for sample {i} iter {iter_idx}: {e_lu}. Trying pseudo-inverse."
                        )
                        try:
                            dv_np = np.linalg.lstsq(J_np, -residual_v_np, rcond=None)[0]
                        except np.linalg.LinAlgError as e_lstsq:
                            log.error(
                                f"Pseudo-inverse failed for sample {i} iter {iter_idx}: {e_lstsq}. Using zero update."
                            )
                            dv_np = np.zeros_like(v_np)

                if np.any(np.isnan(dv_np)) or np.any(np.isinf(dv_np)):
                    log.warning(
                        f"dv_np contains NaN/Inf for sample {i} at iter {iter_idx}. Using zero update."
                    )
                    dv_np = np.zeros_like(v_np)  # Fallback

                dv_np = np.clip(dv_np, -self.clip_threshold, self.clip_threshold)

                p_np = p_np * self.momentum + current_attn_factor * dv_np
                v_new_np = v_np + p_np

                # Pass CPU tensors to residual
                v_new = torch.from_numpy(v_new_np)
                residual_v_new_np = (
                    self.residual(v_new, xi_torch_sample, i_exti_torch_sample)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                if (
                    np.linalg.norm(residual_v_new_np) > np.linalg.norm(residual_v_np)
                    and np.linalg.norm(residual_v_new_np) > self.atol
                ):  # Don't reduce step if already converged but norm slightly higher due to precision
                    log.debug(
                        f"Sample {i}, iter {iter_idx}: residual norm increased from {np.linalg.norm(residual_v_np):.2e} to {np.linalg.norm(residual_v_new_np):.2e}. Reducing step."
                    )
                    current_attn_factor *= 0.25
                    # Do not update v_np, p_np, residual_v_np; retry with smaller step on next iteration implicitly by not updating p_np with full dv_np
                    p_np = p_np * 0  # Reset momentum if step was bad
                else:
                    v_np = v_new_np
                    residual_v_np = residual_v_new_np
                    if current_attn_factor < 1.0:
                        current_attn_factor = min(1.0, current_attn_factor * 1.1)

                if (iter_idx % 10 == 0) and (log.logger.getEffectiveLevel() <= 10):  # DEBUG
                    log.debug(
                        f"Sample {i}, iter {iter_idx}: max_abs_res: {np.max(np.abs(residual_v_np)):.3e}, attn_factor: {current_attn_factor:.2e}"
                    )

            # Final log for the sample
            if iter_idx == self.max_iter - 1 and np.max(np.abs(residual_v_np)) > self.atol:
                log.warning(
                    f"ScipyNewtonStrategy for sample {i} did not converge in {self.max_iter} iterations. "
                    f"Final max_abs_residual={np.max(np.abs(residual_v_np)):.3e}"
                )
            else:
                log.debug(
                    f"ScipyNewtonStrategy for sample {i} converged in {iter_idx + 1} iterations. Final max_abs_residual={np.max(np.abs(residual_v_np)):.3e}"
                )

            vout_list.append(v_np)

        self.reset()

        vout_np = np.stack(vout_list, axis=0)
        # Convert back to original device and dtype
        vout_torch = torch.from_numpy(vout_np).to(dtype=original_x_dtype, device=original_x_device)

        if i_ext is None:  # Check original i_ext
            self._free_solution = vout_torch.detach().clone()

        return vout_torch

    def reset(self):
        """Reset internal state."""
        super().reset()
        # Reset attention factor to its initial value if needed for multiple .solve() calls on the same instance
        self.attn_factor = self._initial_attn_factor
