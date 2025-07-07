from typing import Literal

import torch
import torch.nn.functional as F

from .strategies import FirstOrderStrategy, SecondOrderStrategy, cache_free_solution
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TriangularPreconditioner:
    """
    Efficient triangular preconditioner for systems where the Jacobian
    has a Laplacian structure L = L_lower + α*L_lower^T.

    This preconditioner uses the triangular structure to perform
    efficient forward/backward substitution operations.
    """

    def __init__(self, strategy: FirstOrderStrategy, alpha: float = 1.0):
        """
        Initialize the triangular preconditioner.

        Args:
            strategy: The FirstOrderStrategy instance that contains the model
            alpha: Scaling factor for the upper triangular part (defaults to 1.0)
        """
        self.strategy = strategy
        self.alpha = alpha
        self.L_lower = None
        self.diag = None
        self.initialized = False

    def setup(self, v: torch.Tensor):
        """
        Set up the preconditioner using the current model state.

        Args:
            v: Current solution estimate, used for computing nonlinear terms
        """
        # Get the lower triangular part of the Laplacian
        self.L_lower = self.strategy.build_lower_triangular()

        # Add diagonal terms
        dims = self.strategy.dims
        size = sum(dims)
        D0 = (
            -self.L_lower.sum(-2)
            - self.L_lower.sum(-1)
            + F.pad(self.strategy.W[0].sum(-1), (0, size - dims[0]))
        )

        # Add bias term to diagonal if present
        if self.strategy.B:
            B = self.strategy.bias()
            D0 += B

        # Add nonlinear derivative terms to diagonal
        if len(v.shape) == 2:  # batched input
            a_diag = self.strategy.OTS.a(v)
            batch_size = v.shape[0]
            # We'll store diagonal for each batch item
            self.diag = torch.zeros(batch_size, size, device=v.device, dtype=v.dtype)
            for i in range(batch_size):
                self.diag[i] = D0 + a_diag[i]
        else:
            a_diag = self.strategy.OTS.a(v.unsqueeze(0)).squeeze(0)
            self.diag = D0 + a_diag

        # Add small regularization term for numerical stability
        eps = self.strategy.eps if hasattr(self.strategy, "eps") else 1e-8
        self.diag += eps

        self.initialized = True

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Apply the preconditioner by solving M⁻¹b using triangular factors.
        Handles both single and batched right-hand side vectors.

        Args:
            b: Right-hand side vector or batch of vectors.
               Shape [n] or [batch_size, n].

        Returns:
            Solution to M⁻¹b, with the same shape as b.
        """
        if not self.initialized:
            raise RuntimeError("Preconditioner not initialized. Call setup() first.")

        original_b_ndim = b.ndim
        if original_b_ndim == 1:
            b = b.unsqueeze(0)  # Treat as a batch of 1

        batch_size, n_dim = b.shape

        # Prepare diagonal for batch operation (current_diag will be [batch_size, n_dim])
        current_diag: torch.Tensor
        if self.diag.ndim == 1:
            # self.diag is [n], expand to [batch_size, n]
            current_diag = self.diag.unsqueeze(0).expand(batch_size, n_dim)
        elif self.diag.ndim == 2:
            # self.diag is [setup_batch_size, n]
            if (
                self.diag.shape[0] == 1
            ):  # Covers setup_batch_size == 1, regardless of solve batch_size
                # setup was for batch 1, expand to current solve batch_size
                current_diag = self.diag.expand(batch_size, n_dim)
            elif self.diag.shape[0] == batch_size:
                current_diag = self.diag
            else:
                raise ValueError(
                    f"Batch size of b ({batch_size}) does not match "
                    f"preconditioner's diagonal batch size ({self.diag.shape[0]}) "
                    "and cannot be broadcasted from a single precomputed diagonal."
                )
        else:
            raise ValueError(f"self.diag has unexpected number of dimensions: {self.diag.ndim}")

        # Determine if a single matrix solve (broadcasting M_fwd/M_bwd) is possible
        # This is true if self.diag represents a single diagonal vector (or batch-1)
        use_single_matrix_path = self.diag.ndim == 1 or (
            self.diag.ndim == 2 and self.diag.shape[0] == 1
        )

        if use_single_matrix_path:
            # Path 1: M_fwd_single, M_bwd_single are [n_dim, n_dim] and broadcast by solve_triangular
            # This saves memory by not forming [batch_size, n_dim, n_dim] matrices for M_fwd/M_bwd.
            if self.diag.ndim == 1:
                single_diag_vector = self.diag
            else:  # self.diag.ndim == 2 and self.diag.shape[0] == 1
                single_diag_vector = self.diag[0]

            D_common = torch.diag(single_diag_vector)  # Shape: [n_dim, n_dim]

            # 1. First solve (D_common + L_lower)y = b (forward substitution)
            # M_fwd_op is [n_dim, n_dim]. b.unsqueeze(-1) is [batch_size, n_dim, 1].
            # solve_triangular broadcasts M_fwd_op.
            M_fwd_op = self.L_lower + D_common
            y = torch.linalg.solve_triangular(M_fwd_op, b.unsqueeze(-1), upper=False).squeeze(-1)

            # 2. Apply diagonal scaling: z = D^-1 y
            # current_diag is [batch_size, n_dim], y is [batch_size, n_dim]
            z = y / current_diag

            # 3. Then solve (D_common + alpha*L_lower^T)x = z (backward substitution)
            # M_bwd_op is [n_dim, n_dim]. z.unsqueeze(-1) is [batch_size, n_dim, 1].
            # solve_triangular broadcasts M_bwd_op.
            M_bwd_op = self.alpha * self.L_lower.T + D_common
            x = torch.linalg.solve_triangular(M_bwd_op, z.unsqueeze(-1), upper=True).squeeze(-1)
        else:
            # Path 2: M_fwd, M_bwd are [batch_size, n_dim, n_dim] (original logic)
            # This path is taken if self.diag.ndim == 2 and self.diag.shape[0] == batch_size (and batch_size > 1).
            # current_diag is [batch_size, n_dim]
            D_batch = torch.diag_embed(current_diag)  # Shape: [batch_size, n_dim, n_dim]

            # 1. First solve (D_batch + L_lower)y = b (forward substitution)
            # self.L_lower ([n_dim, n_dim]) is broadcast to [batch_size, n_dim, n_dim] for the sum.
            # M_fwd is [batch_size, n_dim, n_dim].
            M_fwd = self.L_lower + D_batch
            y = torch.linalg.solve_triangular(M_fwd, b.unsqueeze(-1), upper=False).squeeze(-1)

            # 2. Apply diagonal scaling: z = D^-1 y
            z = y / current_diag

            # 3. Then solve (D_batch + alpha*L_lower^T)x = z (backward substitution)
            # self.L_lower.T ([n_dim, n_dim]) is broadcast for the sum.
            # M_bwd is [batch_size, n_dim, n_dim].
            M_bwd = self.alpha * self.L_lower.T + D_batch
            x = torch.linalg.solve_triangular(M_bwd, z.unsqueeze(-1), upper=True).squeeze(-1)

        if original_b_ndim == 1:
            x = x.squeeze(0)  # Restore original shape if input was 1D

        return x


class NewtonStrategy(SecondOrderStrategy):
    r"""Solve J\Delta{X}=-f with Newton's method.

    Args:
        clip_threshold (float): threshold for voltage change.
        attn_factor (float): attenuation factor for voltage change.
        solver_type (str): Type of inner solver to use. Options:
                          'cholesky' - Dense Cholesky decomposition
                          'bicg' - BiCGSTAB iterative solver
                          'log_transformed' - Log-transformed solver for better conditioning
    """

    def __init__(
        self,
        clip_threshold: float = 1.0,
        attn_factor: float = 1,
        momentum: float = 0,
        solver_type: Literal["cholesky", "bicg", "log_transformed"] = "log_transformed",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.init_attn_factor = attn_factor
        self.attn_factor = attn_factor
        self.momentum = momentum
        self._s_vector = None
        self.solver_type = solver_type.lower()

    def reset(self):
        """Reset the strategy state."""
        super().reset()
        self._s_vector = None
        self.attn_factor = self.init_attn_factor

    @cache_free_solution
    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network.

        Args:
            x (torch.Tensor): input of the network.
            i_ext (torch.Tensor): external current.
        KwArgs:
            params (tuple[list[torch.Tensor], list[torch.Tensor]]): weights and biases of the model.
            OTS (eqprop_utils.P3OTS): nonlinearity.
            dims (list): dimensions of the model.
            max_iter (int): maximum number of iterations.
            atol (float): absolute tolerance.
            amp_factor (float): inter-layer potential amplifying factor.
        """
        self.check_and_set_attrs(kwargs)

        # Select solver based on solver_type parameter
        if self.solver_type == "cholesky":
            vout = self._densecholsol(x, i_ext)
        elif self.solver_type == "bicg":
            vout = self._densebicgsol(x, i_ext)
        elif self.solver_type == "log_transformed":
            vout = self._log_transformed_solver(x, i_ext)
        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unknown solver type: {self.solver_type}")

        return vout

    @torch.no_grad()
    def _densecholsol(
        self,
        x: torch.Tensor,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with dense cholesky/LU decomposition. uses adaptive step size.
        For every iteration, the residual is calculated.
        If the residual is not reduced, the step size is quartered. Else, it is increased by 10%.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices. Each element of the list size (dim, dim).
            B (list, optional): List of bias vectors. Defaults to None. Each element of the list size (batchsize, dim).
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_utils.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.

        """
        v = self.lin_solve(x, i_ext)
        residual_v = self.residual(v, x, i_ext).unsqueeze(-1)
        if self.amp_factor > 1.0 and self._s_vector is None:
            s_diag_elements = []
            for layer_idx, dim_size in enumerate(self.dims):
                scale_factor = self.amp_factor ** (layer_idx / 2.0)
                s_diag_elements.extend([scale_factor] * dim_size)
            self._s_vector = torch.tensor(s_diag_elements, device=v.device, dtype=v.dtype)
        if self.amp_factor > 1.0:
            # map eqn to SPD space
            # broadcast (size) to (batchsize, size, 1)
            residual_v *= self._s_vector.unsqueeze(0).unsqueeze(-1)

        J = None
        idx = 0
        p = torch.zeros_like(v)
        while (residual_v.abs().max() > self.atol) and (idx < self.max_iter):
            J = self.jacobian(v)  # (batchsize, size, size)
            # or SPOSV
            if self.amp_factor == 1.0:
                lo, info = torch.linalg.cholesky_ex(J)
                dv = torch.cholesky_solve(-residual_v, lo).squeeze(-1)
            else:
                s_inv = 1.0 / self._s_vector
                J_hat = self._s_vector.unsqueeze(1) * J * s_inv
                lo, info = torch.linalg.cholesky_ex(J_hat)
                dv_hat = torch.cholesky_solve(-residual_v, lo).squeeze(-1)
                dv = s_inv * dv_hat
            # limit the voltage change
            if torch.isnan(dv).any():
                raise ValueError("dv contains NaN")
            dv.clamp_(min=-self.clip_threshold, max=self.clip_threshold)
            p = p * self.momentum + self.attn_factor * dv
            v_new = v + p
            residual_v_new = self.residual(v_new, x, i_ext).unsqueeze(-1)  # (batchsize, size, 1)
            if self.amp_factor > 1.0:
                residual_v_new *= self._s_vector.unsqueeze(0).unsqueeze(-1)

            if torch.any(residual_v_new.norm() > residual_v.norm()):
                log.debug(f"idx:{idx}, residual increased")
                self.attn_factor *= 0.25
            else:
                v = v_new
                residual_v = residual_v_new
                if self.attn_factor < 1:
                    self.attn_factor *= 1.1
            if (idx % 10 == 0) & (log.logger.getEffectiveLevel() == 10):
                eigvals = torch.linalg.eigvals(J)
                pd = (torch.min(eigvals.real) > 0) & torch.all(eigvals.imag == 0)
                log.debug(f"idx:{idx} residual: {dv.abs().max():.3e}, PDness: {pd}")
            idx += 1
        (
            log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
            if J is not None
            else None
        )
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, residual={residual_v.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v

    @torch.no_grad()
    def _densebicgsol(
        self,
        x: torch.Tensor,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with BiCG (Biconjugate Gradient Stabilized) method.
        Uses adaptive step size and triangular preconditioning for better convergence.
        Implementation uses batched operations for improved performance.

        Args:
            x (torch.Tensor): Input.
            i_ext ([type], optional): External current. Defaults to None.

        Returns:
            torch.Tensor: Solution vector.
        """
        v = self.lin_solve(x, i_ext)
        residual_v = self.residual(v, x, i_ext).unsqueeze(-1)
        J = None
        idx = 0
        p = torch.zeros_like(v)

        # Maximum BiCG iterations and tolerance
        max_bicg_iter = 200
        bicg_tol = 1e-6

        # Create triangular preconditioner
        preconditioner = TriangularPreconditioner(self, alpha=self.amp_factor)

        while (residual_v.abs().max() > self.atol) and (idx < self.max_iter):
            J = self.jacobian(v)

            # Set up preconditioner with current solution
            preconditioner.setup(v)

            # Implementation of Batched Preconditioned BiCGSTAB
            batch_size = v.shape[0]
            dim = v.shape[1]  # Get the dimension size

            # Initialize negative residual
            # Ensure r has shape [batch_size, dim]
            r = -residual_v.squeeze(-1)

            # Apply preconditioner to initial residual (batched)
            # Preconditioner expects [batch_size, n]
            z = preconditioner.solve(r)
            # Use preconditioned residual as shadow residual, ensure correct shape
            r_tilde = z.clone()

            # Initialize BiCGSTAB variables
            rho = torch.sum(r_tilde * z, dim=1)  # [batch_size]
            rho_prev = torch.ones_like(rho)
            alpha = torch.ones_like(rho)
            omega = torch.ones_like(rho)
            # Ensure correct shapes for vectors
            p_bicg = torch.zeros(batch_size, dim, device=v.device, dtype=v.dtype)
            v_bicg = torch.zeros(batch_size, dim, device=v.device, dtype=v.dtype)
            dv = torch.zeros(batch_size, dim, device=v.device, dtype=v.dtype)  # Solution

            # Batch matrix-vector product function
            def matvec_batched(vec):
                # vec shape: [batch_size, dim]
                # J shape: [batch_size, dim, dim]
                # result shape: [batch_size, dim]
                return torch.bmm(J, vec.unsqueeze(-1)).squeeze(-1)

            # BiCGSTAB iteration
            converged = False  # Track convergence
            bicg_iterations = 0
            for i in range(max_bicg_iter):
                bicg_iterations = i + 1
                # Update p_bicg
                if i == 0:
                    p_bicg = z.clone()
                else:
                    # Ensure beta calculation is safe
                    beta_num = rho
                    beta_den = rho_prev * omega
                    # Avoid division by zero or near-zero
                    beta = (beta_num / (beta_den + 1e-15)) * alpha
                    p_bicg = z + beta.unsqueeze(1) * (p_bicg - omega.unsqueeze(1) * v_bicg)

                # Apply J matrix (batched)
                v_bicg = matvec_batched(p_bicg)

                # Apply preconditioner (batched)
                z_tilde = preconditioner.solve(v_bicg)

                # Compute alpha (batched)
                r_tilde_z_tilde = torch.sum(r_tilde * z_tilde, dim=1)
                # Avoid division by zero
                alpha = rho / (r_tilde_z_tilde + 1e-15)

                # Update solution estimate (batched)
                h = dv + alpha.unsqueeze(1) * p_bicg

                # Compute residual (batched)
                s = r - alpha.unsqueeze(1) * v_bicg

                # Check if early convergence based on s norm
                s_norm = torch.norm(s, dim=1)
                if s_norm.max() < bicg_tol:
                    dv = h
                    converged = True
                    break

                # Apply preconditioner to s (batched)
                s_hat = preconditioner.solve(s)

                # Calculate Az for momentum term (batched)
                t = matvec_batched(s_hat)

                # Apply preconditioner to t (batched)
                t_hat = preconditioner.solve(t)

                # Compute omega (batched)
                t_hat_s_hat = torch.sum(t_hat * s_hat, dim=1)
                t_hat_t_hat = torch.sum(t_hat * t_hat, dim=1)
                # Avoid division by zero
                omega = t_hat_s_hat / (t_hat_t_hat + 1e-15)

                # Update solution (batched)
                dv = h + omega.unsqueeze(1) * s_hat

                # Update residual (batched)
                r = s - omega.unsqueeze(1) * t

                # Calculate new preconditioned residual (batched)
                z = preconditioner.solve(r)

                # Check convergence based on r norm
                residual_norm = torch.norm(r, dim=1)
                if residual_norm.max() < bicg_tol:
                    converged = True
                    break

                # Update rho for next iteration
                rho_prev = rho.clone()  # Ensure rho_prev is updated correctly
                rho = torch.sum(r_tilde * z, dim=1)

                # Check for numerical breakdown
                if torch.any(torch.abs(rho) < 1e-15) or torch.any(torch.abs(omega) < 1e-15):
                    log.debug(
                        f"BiCG breakdown at inner iteration {i}, rho_min={rho.min():.2e}, omega_min={omega.min():.2e}"
                    )
                    # Use the current best solution if breakdown occurs
                    break

            if not converged:
                log.debug(
                    f"BiCG inner loop did not converge in {max_bicg_iter} iterations, final residual norm: {residual_norm.max():.3e}"
                )

            # Limit the voltage change
            if torch.isnan(dv).any():
                log.error("NaN detected in BiCG solution (dv)")
                break  # Exit outer loop if NaN occurs
            if torch.isinf(dv).any():
                log.error("Inf detected in BiCG solution (dv)")
                break  # Exit outer loop if Inf occurs

            dv.clamp_(min=-self.clip_threshold, max=self.clip_threshold)
            p = p * self.momentum + self.attn_factor * dv
            v_new = v + p

            # Calculate new residual (ensure correct shape for norm calculation)
            residual_v_new = self.residual(v_new, x, i_ext)  # Shape [batch_size, dim]

            # Check if residual norm is reduced
            # Compare norms of residuals (which are [batch_size, dim])
            current_residual_norm = residual_v.squeeze(-1).norm(dim=1)
            new_residual_norm = residual_v_new.norm(dim=1)

            if torch.any(new_residual_norm > current_residual_norm):
                log.debug(f"idx:{idx}, residual norm increased")
                self.attn_factor *= 0.25
            else:
                v = v_new
                residual_v = residual_v_new.unsqueeze(
                    -1
                )  # Add back the dimension for next iteration check
                if self.attn_factor < 1:
                    self.attn_factor = min(
                        1.0, self.attn_factor * 1.1
                    )  # Ensure attn_factor doesn't exceed 1

            # Check convergence for the outer loop
            if residual_v.abs().max() <= self.atol:
                log.debug(f"Outer loop converged at iteration {idx}")
                break

            if (idx % 10 == 0) & (log.logger.getEffectiveLevel() <= 10):  # Use <= for DEBUG level
                try:
                    eigvals = torch.linalg.eigvals(J[0])  # Check first batch item
                    pd = (torch.min(eigvals.real) > -1e-9) & torch.all(
                        torch.abs(eigvals.imag) < 1e-9
                    )  # Relax PD check slightly
                    log.debug(
                        f"idx:{idx} outer_res: {residual_v.abs().max():.3e}, inner_res: {residual_norm.max():.3e} ({bicg_iterations} iters), PDness: {pd}, attn: {self.attn_factor:.2f}"
                    )
                except Exception as e:
                    log.debug(
                        f"idx:{idx} outer_res: {residual_v.abs().max():.3e}, inner_res: {residual_norm.max():.3e} ({bicg_iterations} iters), attn: {self.attn_factor:.2f}. Eigval check failed: {e}"
                    )

            idx += 1
        # --- End of outer Newton loop ---

        # Store iterations count if needed
        self.iterations = idx

        (
            log.debug(f"Final condition number of J: {torch.linalg.cond(J[0]):.2f}")
            if J is not None and J.numel() > 0 and J.shape[0] > 0
            else None
        )

        if idx == self.max_iter:
            log.warning(
                f"bicgsol did not converge in {self.max_iter} iterations, residual={residual_v.abs().max():.3e}"
            )
        else:
            log.debug(f"bicgsol converged in {idx} iterations")

        return v

    @torch.no_grad()
    def _log_transformed_solver(
        self,
        x: torch.Tensor,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with log-transformed residuals for better conditioning.
        Uses the transformation r(x) = sign(f)*log(1+|f|) to handle large residuals.

        Args:
            x (torch.Tensor): Input.
            i_ext ([type], optional): External current. Defaults to None.

        Returns:
            torch.Tensor: Solution vector.
        """
        v = self.lin_solve(x, i_ext).clamp(min=self.OTS.Vl, max=self.OTS.Vr)
        residual_v = self.residual(v, x, i_ext)

        # Convert to log-transformed residual
        log_residual = torch.sign(residual_v) * torch.log1p(residual_v.abs())

        idx = 0
        p = torch.zeros_like(v)

        while (residual_v.abs().max() > self.atol) and (idx < self.max_iter):
            # Get Jacobian of the original residual function
            J = self.jacobian(v)

            # Create scaling diagonal matrix D = diag(1/(1+|f_i|))
            D_diag = 1.0 / (1 + residual_v.abs())

            # Scale the Jacobian: J_r = D @ J
            scaled_J = J * D_diag.unsqueeze(1)
            # Solve the transformed system
            try:
                if self.amp_factor == 1.0:
                    lo, info = torch.linalg.cholesky_ex(scaled_J)
                    if torch.any(info != 0):
                        raise RuntimeError("Cholesky decomposition failed")
                    dv = torch.cholesky_solve(-log_residual.unsqueeze(-1), lo).squeeze(-1)
                else:
                    lo, piv, info = torch.linalg.lu_factor_ex(scaled_J)
                    if torch.any(info != 0):
                        raise RuntimeError("LU decomposition failed")
                    dv = torch.linalg.lu_solve(lo, piv, -log_residual.unsqueeze(-1)).squeeze(-1)
            except RuntimeError as e:
                log.debug(f"Linear system solve failed: {e}")
                # Fallback to least squares
                dv = torch.linalg.lstsq(scaled_J, -log_residual.unsqueeze(-1)).solution.squeeze(-1)

            # Limit the voltage change
            if torch.isnan(dv).any():
                raise ValueError("dv contains NaN")

            dv.clamp_(min=-self.clip_threshold, max=self.clip_threshold)
            p = p * self.momentum + self.attn_factor * dv

            # Update solution
            v_new = v + p
            residual_v_new = self.residual(v_new, x, i_ext)

            # Check if residual is reduced
            if torch.any(residual_v_new.norm() > residual_v.norm()):
                log.debug(f"idx:{idx}, residual increased")
                self.attn_factor *= 0.25
            else:
                v = v_new
                residual_v = residual_v_new
                log_residual = torch.sign(residual_v) * torch.log1p(residual_v.abs())
                # self.attn_factor *= 1.1
                if self.attn_factor < 1:
                    self.attn_factor *= 1.1

            if (idx % 10 == 0) & (log.logger.getEffectiveLevel() == 10):
                eigvals = torch.linalg.eigvals(scaled_J)
                pd = (torch.min(eigvals.real) > 0) & torch.all(eigvals.imag == 0)
                log.debug(f"idx:{idx} residual: {residual_v.abs().max():.3e}, PDness: {pd}")

            idx += 1

        # Log convergence information
        if idx == self.max_iter:
            log.warning(
                f"log_transformed_solver did not converge in {self.max_iter} iterations, "
                f"residual={residual_v.abs().max():.3e}"
            )
        else:
            log.debug(f"log_transformed_solver converged in {idx} iterations")

        return v


class TorchProxQPStrategy(SecondOrderStrategy):
    """
    Quadratic Programming (QP) solver based on proximal augmented Lagrangian approach.
    Solves problems of the form:
        min 0.5 x^T H x + g^T x
        s.t. Ax = b, Cx <= d, x_lb <= x <= x_ub
    """

    def __init__(
        self,
        rho=1.0,
        mu_e=1e-3,
        mu_i=1e-3,
        mu_f=0.5,
        eps0=1e-6,
        eps0_ext=1e-3,
        alpha_bcl=0.9,
        beta_bcl=0.8,
        eps_abs=1e-5,
        k_max=10,
        max_outer_iter=50,
        max_sub_iter=20,
        sub_tol=1e-4,
        use_box=True,
        use_precond=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Algorithm parameters
        self.rho = rho  # Proximal term weight
        self.mu_e = mu_e  # Penalty parameter for equality constraints
        self.mu_i = mu_i  # Penalty parameter for inequality constraints
        self.mu_f = mu_f  # Reduction factor for penalty parameters
        self.eps0 = eps0  # Initial tolerance
        self.eps0_ext = eps0_ext  # Initial extrapolation tolerance
        self.alpha_bcl = alpha_bcl  # BCL parameter
        self.beta_bcl = beta_bcl  # BCL parameter
        self.eps_abs = eps_abs  # Absolute tolerance for constraints
        self.k_max = k_max  # Maximum number of iterations before dual update
        self.max_outer_iter = max_outer_iter  # Maximum number of outer iterations
        self.max_sub_iter = max_sub_iter  # Maximum number of inner iterations
        self.sub_tol = sub_tol  # Tolerance for inner solves
        self.use_box = use_box  # Use box constraints
        self.use_precond = use_precond  # Use preconditioning

    def ruiz_equilibration(self, H, A, b, C, u, num_iter=5):
        """
        Ruiz equilibration for better conditioning.
        Scales matrices H, A, C by diagonal scaling matrices.
        """
        B, d, _ = H.shape
        s = torch.ones(B, d, device=H.device, dtype=H.dtype)
        for _ in range(num_iter):
            norm = torch.sqrt(torch.sum(H**2, dim=(-2, -1)) + 1e-8)  # (B,)
            s = 1.0 / (norm.unsqueeze(-1).expand(B, d) + 1e-8)
        S = torch.diag_embed(s)  # (B, d, d)
        scaled_H = S @ H @ S
        scaled_A = A @ S if A.numel() > 0 else A
        scaled_C = C @ S if C.numel() > 0 else C
        return scaled_H, scaled_A, b, scaled_C, u, s

    def initialize_qp(self, H, g, A, b):
        """
        Initialize QP solution by solving a relaxed KKT system.
        Only considers equality constraints.
        """
        B, d, _ = H.shape
        n_e = A.shape[1] if A.numel() > 0 else 0
        device = H.device
        dtype = H.dtype

        I_d = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(B, d, d)
        if n_e > 0:
            I_e = torch.eye(n_e, device=device, dtype=dtype).unsqueeze(0).expand(B, n_e, n_e)
            top = torch.cat([H + self.rho * I_d, A.transpose(1, 2)], dim=-1)  # (B, d, d+n_e)
            bottom = torch.cat([A, self.mu_e * I_e], dim=-1)  # (B, n_e, d+n_e)
            K = torch.cat([top, bottom], dim=1)  # (B, d+n_e, d+n_e)
            rhs_top = -g.unsqueeze(-1)  # (B, d, 1)
            rhs_bottom = b.unsqueeze(-1)  # (B, n_e, 1)
            rhs = torch.cat([rhs_top, rhs_bottom], dim=1)  # (B, d+n_e, 1)
            sol = torch.linalg.solve(K, rhs)  # (B, d+n_e, 1)
            x0 = sol[:, :d, 0]
            y0 = sol[:, d:, 0]
        else:
            # Simple unconstrained solution if no equality constraints
            x0 = torch.zeros(B, d, device=device, dtype=dtype)
            y0 = torch.zeros(B, 0, device=device, dtype=dtype)

        z0 = torch.zeros(B, A.shape[1] if A.numel() > 0 else 0, device=device, dtype=dtype)
        return x0, y0, z0

    def project_box(self, x, x_lb, x_ub):
        """Project x onto the box [x_lb, x_ub]"""
        return torch.max(torch.min(x, x_ub), x_lb)

    def qp_objective(self, x, H, g, A, b, C, u, z, x_prev):
        """
        Compute the proximal augmented Lagrangian objective function.
        which is:
        0.5 * x^T H x + g^T x + 0.5 / mu_e * ||Ax - b||^2 + 0.5 / mu_i * ||Cx - u + mu_i * z||^2
        + 0.5 * rho * ||x - x_prev||^2
        """
        quad = 0.5 * torch.sum(x * (H @ x.unsqueeze(-1)).squeeze(-1), dim=-1)
        lin = torch.sum(g * x, dim=-1)

        if A.numel() > 0:
            Ax_minus_b = (A @ x.unsqueeze(-1)).squeeze(-1) - b
            eq_term = 0.5 / self.mu_e * torch.sum(Ax_minus_b**2, dim=-1)
        else:
            eq_term = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if C.numel() > 0:
            Cx = (C @ x.unsqueeze(-1)).squeeze(-1)
            ineq_input = Cx - u + self.mu_i * z
            proj = torch.clamp(ineq_input, min=0.0)
            ineq_term = 0.5 / self.mu_i * torch.sum(proj**2, dim=-1)
        else:
            ineq_term = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        prox_term = 0.5 * self.rho * torch.sum((x - x_prev) ** 2, dim=-1)
        return quad + lin + eq_term + ineq_term + prox_term

    def qp_gradient(self, x, H, g, A, b, C, u, z, x_prev):
        """
        Compute the gradient of the proximal augmented Lagrangian.
        """
        grad_quad = (H @ x.unsqueeze(-1)).squeeze(-1)
        grad_lin = g
        grad_prox = self.rho * (x - x_prev)

        if A.numel() > 0:
            Ax_minus_b = (A @ x.unsqueeze(-1)).squeeze(-1) - b
            grad_eq = (A.transpose(1, 2) @ Ax_minus_b.unsqueeze(-1)).squeeze(-1) / self.mu_e
        else:
            grad_eq = 0.0

        if C.numel() > 0:
            Cx = (C @ x.unsqueeze(-1)).squeeze(-1)
            ineq_term = Cx - u + self.mu_i * z
            mask = (ineq_term > 0).float()
            grad_ineq = (C.transpose(1, 2) @ (mask * ineq_term).unsqueeze(-1)).squeeze(
                -1
            ) / self.mu_i
        else:
            grad_ineq = 0.0

        return grad_quad + grad_lin + grad_eq + grad_prox + grad_ineq

    def qp_hessian(self, x, H, A, C, u, z):
        """
        Compute the Hessian of the proximal augmented Lagrangian.
        """
        B, d = x.shape
        device = x.device
        dtype = x.dtype

        Iden = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(B, d, d)

        if A.numel() > 0:
            hess_eq = A.transpose(1, 2) @ A / self.mu_e
        else:
            hess_eq = torch.zeros(B, d, d, device=device, dtype=dtype)

        if C.numel() > 0:
            Cx = (C @ x.unsqueeze(-1)).squeeze(-1)
            ineq_term = Cx - u + self.mu_i * z
            mask = (ineq_term > 0).float()
            hess_ineq = (1.0 / self.mu_i) * torch.einsum("bnd,bnq->bdq", C, C * mask.unsqueeze(-1))
        else:
            hess_ineq = torch.zeros(B, d, d, device=device, dtype=dtype)

        return H + hess_eq + self.rho * Iden + hess_ineq

    def newton_step(self, x, H, g, A, b, C, u, z, x_prev):
        """
        Perform a semi-smooth Newton step to solve the proximal subproblem.
        """
        x_new = x.clone()
        for it in range(self.max_sub_iter):
            grad = self.qp_gradient(x_new, H, g, A, b, C, u, z, x_prev)
            if torch.max(torch.abs(grad)) < self.sub_tol:
                break

            hess = self.qp_hessian(x_new, H, A, C, u, z)
            dx = torch.linalg.solve(hess, -grad.unsqueeze(-1)).squeeze(-1)

            # Line search with Armijo condition
            alpha = 1.0
            c = 1e-4
            f_val = self.qp_objective(x_new, H, g, A, b, C, u, z, x_prev)
            dir_deriv = torch.sum(grad * dx, dim=-1)

            for ls in range(20):
                x_candidate = x_new + alpha * dx
                f_candidate = self.qp_objective(x_candidate, H, g, A, b, C, u, z, x_prev)
                if torch.all(f_candidate <= f_val + c * alpha * dir_deriv):
                    break
                alpha *= 0.5

            x_new = x_new + alpha * dx

        return x_new

    @cache_free_solution
    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext=None, **kwargs) -> torch.Tensor:
        """
        Solve the QP problem derived from the EqProp equilibrium conditions.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim)
            i_ext (torch.Tensor, optional): External current
            **kwargs: Additional parameters

        Returns:
            torch.Tensor: Equilibrium state solution
        """
        self.check_and_set_attrs(kwargs)

        # Extract batch dimension and device
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Use the Jacobian and residual functions from the parent class
        # to construct QP matrices
        J = self.jacobian(x)  # This will be our H matrix
        H = J

        # Generate coefficients for the QP problem
        g = self.residual(x, x, i_ext)

        # Set up constraint matrices - in this implementation we don't use constraints
        # from the parent class, but you could add them here
        A = torch.empty(batch_size, 0, H.shape[1], device=device, dtype=dtype)
        b = torch.empty(batch_size, 0, device=device, dtype=dtype)
        C = torch.empty(batch_size, 0, H.shape[1], device=device, dtype=dtype)
        u = torch.empty(batch_size, 0, device=device, dtype=dtype)

        # Define box constraints
        if self.use_box:
            x_lb = torch.full((batch_size, H.shape[1]), self.OTS.Vl, device=device, dtype=dtype)
            x_ub = torch.full((batch_size, H.shape[1]), self.OTS.Vr, device=device, dtype=dtype)
        else:
            x_lb = None
            x_ub = None

        # Optional preconditioning
        if self.use_precond and (A.numel() > 0 or C.numel() > 0):
            H, A, b, C, u, scaling = self.ruiz_equilibration(H, A, b, C, u, num_iter=5)
        else:
            scaling = torch.ones(batch_size, H.shape[1], device=device, dtype=dtype)

        # Initialize solution
        x0, y0, z0 = self.initialize_qp(H, g, A, b)

        # Main optimization loop
        x_sol = x0.clone()
        y = y0.clone()
        z = z0.clone()
        eps = self.eps0
        eps_ext = self.eps0_ext

        for k in range(self.max_outer_iter):
            x_new = self.newton_step(x_sol, H, g, A, b, C, u, z, x_sol)

            if self.use_box:
                x_new = self.project_box(x_new, x_lb, x_ub)

            # Check constraint violations
            if A.numel() > 0 or C.numel() > 0:
                eq_res = (
                    torch.abs((A @ x_new.unsqueeze(-1)).squeeze(-1) - b)
                    if A.numel() > 0
                    else torch.zeros(batch_size, device=device)
                )
                ineq_res = (
                    torch.clamp((C @ x_new.unsqueeze(-1)).squeeze(-1) - u, min=0.0)
                    if C.numel() > 0
                    else torch.zeros(batch_size, device=device)
                )

                if eq_res.numel() > 0 or ineq_res.numel() > 0:
                    combined = (
                        torch.cat([eq_res, ineq_res], dim=-1)
                        if (eq_res.numel() and ineq_res.numel())
                        else (eq_res if eq_res.numel() else ineq_res)
                    )
                    p = torch.max(combined, dim=-1)[0]
                else:
                    p = torch.zeros(batch_size, device=device, dtype=dtype)

            else:
                p = torch.zeros(batch_size, device=device, dtype=dtype)

            p_max = torch.max(p)

            if p_max < self.eps_abs:
                break

            # Update multipliers or penalties
            if (p_max < eps_ext) or (k >= self.k_max):
                # Update multipliers
                if A.numel() > 0:
                    Ax_minus_b = (A @ x_new.unsqueeze(-1)).squeeze(-1) - b
                    y = y + Ax_minus_b / self.mu_e

                if C.numel() > 0:
                    Cx_minus_u = (C @ x_new.unsqueeze(-1)).squeeze(-1) - u
                    z = torch.clamp(z + Cx_minus_u / self.mu_i, min=0.0)

                eps = eps * self.mu_i
                eps_ext = eps_ext * (self.mu_i**self.beta_bcl)
            else:
                # Update penalty parameters
                self.mu_i = max(self.mu_i * self.mu_f, 1e-8)
                self.mu_e = max(self.mu_e * self.mu_f, 1e-9)
                eps = self.eps0 * self.mu_i
                eps_ext = self.eps0_ext * (self.mu_i**self.alpha_bcl)

            x_sol = x_new

        # Undo preconditioning if applied
        if self.use_precond:
            x_sol = x_sol / scaling

        return x_sol

    def reset(self):
        """Reset internal state."""
        super().reset()
        # Reset penalty parameters to initial values
        self.mu_e = 1e-3
        self.mu_i = 1e-3


class PrimalDualStrategy(SecondOrderStrategy):
    """
    Perturbed KKT 시스템을 Newton 선형화로 풀어,
      Lx + I_ + a(x) - s + t = 0,
      s_i (x_i - alpha) = gamma,  t_i (beta - x_i) = gamma,
    (s,t > 0, x in (alpha,beta)) 조건을 만족하는 해를 찾는 solver.
    """

    def __init__(self, gamma=0.1, alpha=-1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network using Primal-Dual method."""
        self.check_and_set_attrs(kwargs)

        # 초기값 설정
        batch_size = x.size(0)
        n = sum(self.dims)

        # 초기 추정치: 중앙값으로 초기화
        x0 = torch.zeros(n, device=x.device)
        s0 = torch.ones(n, device=x.device)  # dual 변수 초기값 (양의 값)
        t0 = torch.ones(n, device=x.device)

        # 각 배치에 대해 계산
        vout_list = []
        for i in range(batch_size):
            xi = x[i : i + 1]  # 배치 차원 유지
            L = self.laplacian()
            I_ = self.rhs(xi).squeeze(0)  # 배치 차원 제거

            # 각 배치 항목에 대해 PrimalDual 알고리즘 실행
            x_sol, _, _ = self._primal_dual_solve(L, I_, x0, s0, t0)
            vout_list.append(x_sol)

        vout = torch.stack(vout_list, dim=0)
        if i_ext is None:
            self._free_solution = vout.detach().clone()

        return vout

    def _primal_dual_solve(self, L, I_, x0, s0, t0):
        """Primal-Dual 알고리즘 구현"""
        x = x0.clone()
        s = s0.clone()
        t = t0.clone()
        n = x.numel()

        for k in range(self.max_iter):
            # residual 계산
            a_val = self.OTS.a(x)
            r1 = L @ x + I_ + a_val - s + t
            r2 = s * (x - self.alpha) - self.gamma
            r3 = t * (self.beta - x) - self.gamma

            # 전체 residual norm
            res_norm = torch.norm(torch.cat([r1, r2, r3]))
            if res_norm < self.atol:
                log.debug(
                    f"Primal-Dual converged in {k} iterations, res_norm = {res_norm.item():.3e}"
                )
                break

            # Jacobian 구성 (각 블록)
            # r1 관련: dF/dx = L + diag(a'(x)), dF/ds = -I_, dF/dt = I_.
            J11 = L + torch.diag_embed(a_val)
            J12 = -torch.eye(n, device=x.device)
            J13 = torch.eye(n, device=x.device)

            # r2 관련: d/dx = diag(s), d/ds = diag(x-alpha)
            J21 = torch.diag_embed(s)
            J22 = torch.diag_embed(x - self.alpha)
            J23 = torch.zeros((n, n), device=x.device)

            # r3 관련: d/dx = -diag(t), d/dt = diag(beta-x)
            J31 = -torch.diag_embed(t)
            J32 = torch.zeros((n, n), device=x.device)
            J33 = torch.diag_embed(self.beta - x)

            # 블록 행렬로 Jacobian 구성
            J_top = torch.cat([J11, J12, J13], dim=1)
            J_mid = torch.cat([J21, J22, J23], dim=1)
            J_bot = torch.cat([J31, J32, J33], dim=1)
            J_full = torch.cat([J_top, J_mid, J_bot], dim=0)

            r_full = torch.cat([r1, r2, r3], dim=0)

            try:
                delta = torch.linalg.solve(J_full, -r_full)
            except RuntimeError as e:
                log.warning(f"Linear solve failed: {e}")
                break

            delta_x = delta[:n]
            delta_s = delta[n : 2 * n]
            delta_t = delta[2 * n :]

            # 간단한 damping (여기서는 step size=1 사용)
            x += delta_x
            s += delta_s
            t += delta_t

        if k == self.max_iter - 1:
            log.warning(
                f"Primal-Dual did not converge in {self.max_iter} iterations, res_norm = {res_norm.item():.3e}"
            )

        return x, s, t

    def reset(self):
        """Reset the internal states."""
        super().reset()
