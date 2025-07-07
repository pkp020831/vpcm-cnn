import torch
import torch.nn.functional as F

from .strategies import PythonStrategy
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LMStrategy(PythonStrategy):
    r"""Solve J\Delta{X}=-f with Levenberg-Marquardt method."""

    def __init__(self, clip_threshold, lambda_factor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.lambda_factor = lambda_factor

    @torch.no_grad()
    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
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
        (W, B) = kwargs.get("params", (self.W, self.B))
        if i_ext is None:
            self.free_solution = None
        vout = self._LMdensecholsol(x, W, B, i_ext)
        if i_ext is None:
            self.free_solution = vout
        return vout

    def _LMdensecholsol(
        self,
        x: torch.Tensor,
        W: list,
        B: list | None = None,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with dense cholesky decomposition.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices.
            dims (list): List of dimensions.
            B (list, optional): List of bias vectors. Defaults to None.
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_utils.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.

        """
        dims = self.dims
        batchsize = x.size(0)
        size = sum(dims)
        lambda_ = self.lambda_factor
        # construct the laplacian
        paddedG = [torch.zeros(dims[0], size).type_as(x)]
        [
            paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
            for i, g in enumerate(W[1:])
        ]
        Ll = torch.cat(paddedG, dim=-2)
        L = Ll + Ll.mT * self.amp_factor
        # construct the RHS
        B = (
            torch.zeros((x.size(0), size)).type_as(x)
            if not B
            else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
        )
        B[:, : dims[0]] += x @ W[0].T
        if i_ext is not None:
            B[:, -dims[-1] :] += i_ext
        B *= self.amp_factor
        # construct the diagonal
        D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[0]))
        L += D0.diag()
        # initial solution
        # lo, info = torch.linalg.cholesky_ex(L)
        # v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
        L = L.expand(batchsize, *L.shape)
        # initial solution
        if self._free_solution is not None and i_ext is not None:
            v = self.free_solution
        else:
            v = torch.linalg.lstsq(L, B.unsqueeze(-1)).solution.squeeze()
        dv = torch.ones(1).type_as(x)
        # L = L.expand(batchsize, *L.shape)
        idx = 1
        residuals = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
        residuals[:, :, 0] += self.OTS.i(v)
        while idx < self.max_iter:
            # nonlinearity comes here
            A = self.OTS.a(v)
            jacobian = L.clone()
            jacobian += torch.stack([a.diag() for a in A])  # expensive?

            if torch.all(torch.norm(residuals, dim=(1, 2)) < self.atol):
                break

            # Update rule
            JTJ = torch.bmm(jacobian.mT, jacobian)
            while True:
                try:
                    # Try to update parameters
                    dv = -torch.linalg.solve(
                        torch.bmm(jacobian.mT, residuals),
                        JTJ + lambda_ * torch.eye(v.size(-1)),
                    )
                    break
                except RuntimeError:
                    # In case of singular matrix, increase damping and try again
                    log.debug(f"Jacobian is singular, lambda={lambda_}")
                    lambda_ *= 10
                    if lambda_ > 1e2:
                        raise RuntimeError("Jacobian is singular")
            # or SPOSV
            # lo, info = torch.linalg.cholesky_ex(J)
            # if any(info):  # singular
            #     log.debug(residuals"J is singular, info={info}")
            #     lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
            #     dv = torch.linalg.lu_solve(lo, piv, -residuals).squeeze(-1)
            # else:
            #     dv = torch.cholesky_solve(-residuals, lo).squeeze(-1)

            v += dv
            new_residuals = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
            new_residuals[:, : -dims[-1], 0] += self.OTS.i(v[:, : -dims[-1]])
            if torch.all(torch.norm(new_residuals, dim=(1, 2)) < torch.norm(residuals, dim=(1, 2))):
                lambda_ /= 10

            idx += 1
            residuals = new_residuals

        log.debug(f"condition number of J: {torch.linalg.cond(jacobian[0]):.2f}")
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, dv={dv.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v


def _inv_L(W: torch.Tensor, D0: torch.Tensor, dims: list) -> torch.Tensor:
    """Compute the inverse of the Laplacian matrix."""
    hidden_len = dims[0]
    D1 = D0[hidden_len:]
    D2 = D0[:hidden_len]
    schur_L = D1.diag() - W.T @ D2.diag() @ W
    inv_schur_L = schur_L.inverse()
    A = inv_schur_L @ W.T @ D2.diag()
    B = -inv_schur_L @ W.T @ D2.diag() @ W
    AB = torch.cat((A, B), dim=-1)
    CD = torch.cat((B.mT, inv_schur_L), dim=-1)
    inv_L = torch.cat((AB, CD), dim=-2)
    return inv_L


def _sparsecholsol(x, W, dims, B, i_ext):
    """Use torch.block_diag to construct the sparse matrix.

    Args:
        x (_type_): _description_
        W (_type_): _description_
        B (_type_): _description_
        i_ext (_type_): _description_
    """
    raise NotImplementedError


# TODO: custom CUDA kernel
def _block_tri_cholesky(W: list[torch.Tensor]):
    """Blockwise cholesky decomposition for a size varying block tridiagonal
    matrix. see spftrf() in LAPACK.

    Args:
        W (List[torch.Tensor]): List of lower triangular blocks.

    Returns:
        L (List[torch.Tensor]): List of lower triangular blocks.
        C (List[torch.Tensor]): List of diagonal blocks. as column vectors.
    """
    n = len(W)
    C = [torch.zeros_like(W[i]) for i in range(n)]
    L = [None] * (n + 1)
    W.append(0)
    L[0] = torch.sqrt(W[0].sum(dim=-1))
    for i in range(n):
        C[i] = W[i] / L[i]  # C[i] = W[i] @ D_prev^-T, trsm()
        D = W[i].sum(dim=-2) + W[i + 1].sum(dim=-1) - torch.bmm(C[i], C[i].mT)
        L[i + 1] = torch.sqrt(D)

    return L, C


def _block_tri_cholesky_solve(L, C, B):
    """Blockwise cholesky solve for a size varying block tridiagonal matrix.

    Args:
        L (List[torch.Tensor]): List of lower triangular blocks.
        C (List[torch.Tensor]): List of diagonal blocks.
        B (torch.Tensor): RHS.

    Returns:
        X (torch.Tensor): Solution.
    """
    n = len(L)
    X = torch.zeros_like(B)
    for i in range(n):
        X[:, i * C[i].size(-1) : (i + 1) * C[i].size(-1)] = torch.cholesky_solve(
            B[:, i * C[i].size(-1) : (i + 1) * C[i].size(-1)],
            L[i + 1] + torch.bmm(C[i].transpose(-1, -2), C[i]),
        )

    return X
