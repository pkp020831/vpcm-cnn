from typing import Literal

import torch

from .strategies import SecondOrderStrategy, cache_free_solution
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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
            self._s_vector = torch.tensor(
                s_diag_elements, device=v.device, dtype=v.dtype
            )
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
            residual_v_new = self.residual(v_new, x, i_ext).unsqueeze(
                -1
            )  # (batchsize, size, 1)
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
                    dv = torch.cholesky_solve(-log_residual.unsqueeze(-1), lo).squeeze(
                        -1
                    )
                else:
                    lo, piv, info = torch.linalg.lu_factor_ex(scaled_J)
                    if torch.any(info != 0):
                        raise RuntimeError("LU decomposition failed")
                    dv = torch.linalg.lu_solve(
                        lo, piv, -log_residual.unsqueeze(-1)
                    ).squeeze(-1)
            except RuntimeError as e:
                log.debug(f"Linear system solve failed: {e}")
                # Fallback to least squares
                dv = torch.linalg.lstsq(
                    scaled_J, -log_residual.unsqueeze(-1)
                ).solution.squeeze(-1)

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
                log.debug(
                    f"idx:{idx} residual: {residual_v.abs().max():.3e}, PDness: {pd}"
                )

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
