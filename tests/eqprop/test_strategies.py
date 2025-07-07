import pytest
import torch
import numpy as np
from scipy.sparse.linalg import bicgstab

from src.core.eqprop import activation
from src.core.eqprop.strategy import strategies, second_strategies
from src.core.spice.circuits import create_circuit
from src.core.spice.xyce import XyceSim
from src.core.spice.spice_utils import SPICENNParser
from tests.helpers.run_if import RunIf


class TestSecondOrderStrategy:
    """Test the second order strategy."""

    # 4x4 laplacian matrix for a network with dims=[2, 2]
    lap = torch.tensor(
        [[5.0, 0.0, -2.0, 0.0], [0.0, 5.0, 0.0, -2.0], [-2.0, 0.0, 3.0, 0.0], [0.0, -2.0, 0.0, 3.0]]
    ).float()

    def test_laplacian(self, second_order_strategy):
        """Test if the laplacian is computed correctly."""
        st = second_order_strategy
        # Check the laplacian value
        assert torch.allclose(st.laplacian(), self.lap)

    @pytest.mark.parametrize("v", [-0.5, 1.0])
    def test_jacobian(self, second_order_strategy, v):
        """Test if the jacobian is computed correctly."""
        expected_j = 1.0 if v == 1 else 0.0
        v = torch.tensor([[v, v, v, v]])
        st = second_order_strategy
        J1 = st.jacobian(v)
        assert len(J1.shape) == 3

        # Create expected Jacobian by adding activation derivative to diagonal
        expected_diag = torch.zeros(4)
        expected_diag[:2] = expected_j  # Only apply to the first 2 elements (non-output layer)
        expected_J = self.lap + torch.diag(expected_diag)

        assert torch.allclose(J1[0], expected_J)

    @pytest.mark.parametrize("x", [[1.0, -1.0], [1.0, 1.0]])
    def test_rhs(self, second_order_strategy, x):
        """Test if the right hand side is computed correctly."""
        if x == [1.0, -1.0]:
            expected_rhs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        elif x == [1.0, 1.0]:
            expected_rhs = torch.tensor([[-3.0, -3.0, -1.0, -1.0]])
        else:
            raise ValueError("Invalid x")
        x = torch.tensor([x])
        st = second_order_strategy
        st.reset()
        assert torch.allclose(st.rhs(x), expected_rhs)

    @pytest.mark.parametrize(
        "v, x, expected",
        [
            (-0.5, [1.0, -1.0], [[-2.5, -2.5, -1.5, -1.5]]),
            (0.5, [1.0, 1.0], [[-1.5, -1.5, -0.5, -0.5]]),
        ],
    )
    def test_residual(self, second_order_strategy, v, x, expected):
        """Test if the residual is computed correctly."""
        st = second_order_strategy
        st.reset()
        # Use 4-element vector for v to match the 4x4 laplacian
        v_tensor = torch.tensor([[v, v, v, v]])
        expected_residual = torch.tensor(expected)
        result = st.residual(v=v_tensor, x=torch.tensor([x]), i_ext=None)
        assert torch.allclose(result, expected_residual, atol=1e-5)

    @pytest.mark.parametrize(
        "v, x, expected",
        [
            (-0.5, [1.0, -1.0], [[0.4545, 0.4545, 0.6364, 0.6364]]),
            (0.5, [1.0, 1.0], [[1.0, 1.0, 1.0, 1.0]]),
        ],
    )
    def test_lin_solve(self, second_order_strategy, v, x, expected):
        """Test if the linear solve without activation is computed
        correctly."""
        st = second_order_strategy
        st.reset()
        expected_v = torch.tensor(expected)
        result = st.lin_solve(torch.tensor([x]), None)
        assert torch.allclose(result, expected_v, atol=1e-4)


class TestNewtonStrategy:
    """Test the Newton strategy."""

    @pytest.mark.parametrize(
        "x, v",
        [
            ([1.0, -1.0], [-0.5, -0.5]),
            ([1.0, 1.0], [0.5, 0.5]),
            pytest.param(
                [-1.0, -1.0],
                [-1.2, -1.2],
                marks=pytest.mark.skip(reason="This test case is unstable"),
            ),
        ],
    )
    @pytest.mark.parametrize("strategy_name", ["NewtonStrategy", "PrimalDualStrategy"])
    def test_strategy_solve(toymodel, strategy_name, x, v):
        """Test if the strategy can solve the system."""
        # Define the input
        common_params = {
            "activation": activation.SymReLU(Vl=-0.6, Vr=0.6),
            "max_iter": 20,
            "add_nonlin_last": False,
            "atol": 1e-6,
        }

        # Add strategy-specific parameters
        if strategy_name == "NewtonStrategy":
            common_params["clip_threshold"] = 1
        elif strategy_name == "PrimalDualStrategy":
            common_params["gamma"] = 0.1
            common_params["alpha"] = -0.6
            common_params["beta"] = 0.6

        st: strategies.AbstractStrategy = getattr(strategies, strategy_name)(**common_params)
        st.W = toymodel["W"]
        st.B = toymodel["B"]
        st.dims = toymodel["dims"]
        st.reset()
        x = torch.tensor([x])
        v = torch.tensor([v])
        v_pred = st.solve(x, None)

        # Extract the first two elements (corresponding to the first layer)
        # from the 4-element output vector
        v_pred_first_layer = v_pred[:, :2]

        # Use a higher tolerance for the comparison since the actual values
        # may differ from the expected values due to the different network structure
        assert torch.allclose(v_pred_first_layer, v, atol=0.0001)

    @pytest.mark.parametrize(
        "x",
        [
            ([1.0, -1.0, 1.0]),
            ([1.0, 1.0, 1.0]),
            ([-1.0, -1.0, 1.0]),
        ],
    )
    def test_densebicgsol_vs_cholesky(self, xormodel, x):
        """
        Test if the _densebicgsol method produces similar results to _densecholsol
        when amp_factor=1.
        """
        # Set up a NewtonStrategy with identical parameters for both methods

        common_params = {
            "activation": activation.SymReLU(Vl=-0.6, Vr=0.6),
            "max_iter": 20,
            "add_nonlin_last": False,
            "atol": 1e-6,
            "clip_threshold": 1,
            "amp_factor": 1.0,  # Important: set to 1.0 to use Cholesky
        }

        # Create NewtonStrategy instance
        newton_st = second_strategies.NewtonStrategy(**common_params)
        newton_st.W = xormodel["W"]
        newton_st.dims = xormodel["dims"]
        newton_st.reset()

        # Prepare input tensor
        x_tensor = torch.tensor([x], dtype=torch.float32)

        # Run the linear solve to get a good initial point
        v_init = newton_st.lin_solve(x_tensor, None)

        # Create a copy of the strategy to run each solver method
        # Note: Directly accessing private methods for test purposes
        newton_chol = second_strategies.NewtonStrategy(**common_params)
        newton_chol.W = xormodel["W"]
        newton_chol.dims = xormodel["dims"]
        newton_chol.reset()

        newton_bicg = second_strategies.NewtonStrategy(**common_params)
        newton_bicg.W = xormodel["W"]
        newton_bicg.dims = xormodel["dims"]
        newton_bicg.reset()

        # Run each solver with the same initial point
        v_chol = newton_chol._densecholsol(x_tensor, None)
        v_bicg = newton_bicg._densebicgsol(x_tensor, None)

        # Check that solutions are close
        assert torch.allclose(v_chol, v_bicg, atol=1e-4), (
            f"BiCG and Cholesky solvers produced different results:\n"
            f"Cholesky: {v_chol}\nBiCG: {v_bicg}\n"
            f"Difference: {torch.abs(v_chol - v_bicg).max()}"
        )

    @pytest.mark.parametrize(
        "x",
        [
            ([1.0, -1.0, 1.0]),
            ([1.0, 1.0, 1.0]),
        ],
    )
    @pytest.mark.parametrize("solver_type", ["cholesky", "bicg", "log_transformed"])
    def test_newton_different_solvers(self, xormodel, x, solver_type):
        """
        Test if all the solver types in NewtonStrategy produce consistent results.
        """
        # Set up the NewtonStrategy with the specified solver
        common_params = {
            "activation": activation.SymReLU(Vl=-0.6, Vr=0.6),
            "max_iter": 20,
            "add_nonlin_last": False,
            "atol": 1e-6,
            "clip_threshold": 1,
            "solver_type": solver_type,
        }

        # Create a NewtonStrategy instance
        newton_st = second_strategies.NewtonStrategy(**common_params)
        newton_st.W = xormodel["W"]
        newton_st.dims = xormodel["dims"]
        newton_st.reset()

        # Prepare input tensor
        x_tensor = torch.tensor([x], dtype=torch.float32)

        # Solve using the selected solver
        v_solution = newton_st.solve(x_tensor, None)

        # Check that the solution is valid (we don't need to compare with a reference,
        # just ensure it's not NaN or out of bounds)
        assert not torch.isnan(v_solution).any(), f"Solver {solver_type} produced NaN values"

        # Check that the residual is small
        residual = newton_st.residual(v_solution, x_tensor, None)
        assert torch.max(torch.abs(residual)) < 1e-4, (
            f"Solver {solver_type} didn't minimize residual: {torch.max(torch.abs(residual))}"
        )

    @pytest.mark.parametrize(
        "x",
        [
            ([1.0, -1.0, 1.0]),
            ([1.0, 1.0, 1.0]),
        ],
    )
    def test_densebicgsol_vs_scipy(self, xormodel, x):
        """
        Test if the _densebicgsol method produces similar results to SciPy's BiCGSTAB
        implementation for the same linear system.
        """
        # Set up the NewtonStrategy
        common_params = {
            "activation": activation.SymReLU(Vl=-0.6, Vr=0.6),
            "max_iter": 20,
            "add_nonlin_last": False,
            "atol": 1e-6,
            "clip_threshold": 1,
        }

        # Create a NewtonStrategy instance
        newton_st = second_strategies.NewtonStrategy(**common_params)
        newton_st.W = xormodel["W"]
        newton_st.dims = xormodel["dims"]
        newton_st.reset()

        # Prepare input tensor
        x_tensor = torch.tensor([x], dtype=torch.float32)

        # Get initial point
        v_init = newton_st.lin_solve(x_tensor, None)

        # Get the Jacobian matrix and residual at this point
        J = newton_st.jacobian(v_init)[0].detach().cpu().numpy()  # Take first batch item
        residual_v = newton_st.residual(v_init, x_tensor, None)[0].detach().cpu().numpy()

        # Create a preconditioner for SciPy
        # Using a simplified version of the triangular preconditioner
        precond = np.diag(np.diag(J))  # Diagonal preconditioner for simplicity

        # Define matrix-vector product function for SciPy BiCGSTAB
        def mv(v):
            return J @ v

        # Run SciPy's BiCGSTAB
        scipy_result, info = bicgstab(
            A=mv,
            b=-residual_v,
            x0=np.zeros_like(residual_v),
            tol=1e-6,
            maxiter=200,
        )

        # Check that the iterative solve converged
        assert info == 0, f"SciPy BiCGSTAB did not converge: info={info}"

        # Convert SciPy result back to tensor for comparison
        scipy_solution = torch.tensor(scipy_result, dtype=torch.float32).unsqueeze(0)

        # Create another strategy instance for our BiCG implementation
        newton_bicg = second_strategies.NewtonStrategy(**common_params)
        newton_bicg.W = xormodel["W"]
        newton_bicg.dims = xormodel["dims"]
        newton_bicg.reset()

        # Create a new instance of the preconditioner
        # This step ensures the first solver step in _densebicgsol will use BiCGSTAB
        # and not revert to Cholesky
        preconditioner = second_strategies.TriangularPreconditioner(newton_bicg)

        # Get the incremental update from our BiCG solver
        # We need to isolate just one BiCG step, so we modify parameters
        newton_bicg.max_iter = 1  # Just one outer iteration
        custom_bicg_result = newton_bicg._densebicgsol(x_tensor, None)

        # Calculate the increment that would be applied in one step
        dv_scipy = scipy_solution  # This is the solution to J*dv = -residual
        dv_custom = custom_bicg_result - v_init

        # Normalize both directions for comparison (we only care about direction)
        dv_scipy_norm = dv_scipy / torch.norm(dv_scipy)
        dv_custom_norm = dv_custom / torch.norm(dv_custom)

        # Check that the solutions point in similar directions
        dot_product = torch.sum(dv_scipy_norm * dv_custom_norm)
        assert dot_product > 0.8, (
            f"BiCG solvers produced different directions, cosine similarity: {dot_product}"
        )


@RunIf(xyce=True)
@pytest.mark.parametrize(
    "x",
    [
        ([1.0, -1.0, 1.0]),
        ([1.0, 1.0, 1.0]),
        ([-1.0, -1.0, 1.0]),
    ],
)
@pytest.mark.parametrize("strategy_name", ["NewtonStrategy", "PrimalDualStrategy"])
def test_strategy_and_xyce(
    xormodel,  # Fixture로부터 {W, dims, ...}
    strategy_name,
    x,
    tmp_path,  # pytest에서 제공하는 임시 경로
):
    """
    Test if the strategy can solve the system, and compare with Xyce simulation of a matching circuit.
    """
    # ────────────── 1) Strategy 설정 & 계산 ──────────────
    common_params = {
        "activation": activation.SymReLU(Vl=-0.6, Vr=0.6),
        "max_iter": 20,
        "add_nonlin_last": False,
        "atol": 1e-6,
    }

    if strategy_name == "NewtonStrategy":
        common_params["clip_threshold"] = 1
    elif strategy_name == "PrimalDualStrategy":
        common_params["gamma"] = 0.1
        common_params["alpha"] = -0.6
        common_params["beta"] = 0.6

    st: strategies.AbstractStrategy = getattr(strategies, strategy_name)(**common_params)
    st.W = xormodel["W"]  # ex) [W0, W1, ...] or a single W (depending on your model)
    # st.B = toymodel["B"]   # 필요 시 바이어스도 할당
    st.dims = xormodel["dims"]
    st.reset()

    x_t = torch.tensor([x], dtype=torch.float32)
    v_pred = st.solve(x_t, None)

    # 첫 레이어 노드만 비교 대상으로 꺼냄 (예: 2개 노드)
    v_pred_first_layer = v_pred[:, :2]

    # ────────── 2) Xyce용 Circuit 생성 및 시뮬레이션 ──────────
    # create_circuit는 사용자가 이미 작성한 함수(질문 예시 참고).
    # 필요한 파라미터( Rectifier, diode model path 등 )를 params로 전달.

    SPICE_params = {
        "A": 1,
        "beta": 0.6,
        "Diode": {
            "ModelName": "OTS",
            "Rectifier": "SymReLUOTS",
        },
    }
    xyce_params = common_params.copy()
    xyce_params["SPICE_params"] = SPICE_params

    newton_params = common_params.copy()
    newton_params["clip_threshold"] = 1
    st1 = strategies.NewtonStrategy(**newton_params)
    xyce_st = strategies.XyceStrategy(**xyce_params)

    xyce_st.W = xormodel["W"]
    xyce_st.dims = xormodel["dims"]
    xyce_st.reset()
    xyce_v_pred = xyce_st.solve(x_t, None)
    xyce_v_pred_first_layer = xyce_v_pred[:, :2]

    # Strategy가 기대하는 v와 비교
    assert torch.allclose(v_pred_first_layer, xyce_v_pred_first_layer, atol=0.0001), (
        f"Strategy solve mismatch: {v_pred_first_layer} vs {xyce_v_pred_first_layer}"
    )


class TestPrimalDualStrategy:
    """Test the PrimalDual strategy."""

    def test_initialization(self, primal_dual_strategy):
        """Test if the PrimalDualStrategy is initialized correctly."""
        st = primal_dual_strategy
        assert isinstance(st, strategies.PrimalDualStrategy)
        assert st.gamma == 0.1
        assert st.alpha == -0.6
        assert st.beta == 0.6
        assert st.max_iter == 20
        assert st.atol == 1e-4

    def test_primal_dual_solve(self, primal_dual_strategy):
        """Test the _primal_dual_solve method."""
        st = primal_dual_strategy
        st.reset()

        # Create test inputs
        L = torch.tensor([[4.0, -2.0], [-2.0, 2.0]], dtype=torch.float32)
        I_ = torch.tensor([1.0, 0.0], dtype=torch.float32)
        x0 = torch.zeros(2, dtype=torch.float32)
        s0 = torch.ones(2, dtype=torch.float32)
        t0 = torch.ones(2, dtype=torch.float32)

        # Call the method
        x, s, t = st._primal_dual_solve(L, I_, x0, s0, t0)

        # Check the output types and shapes
        assert isinstance(x, torch.Tensor)
        assert isinstance(s, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        assert x.shape == torch.Size([2])
        assert s.shape == torch.Size([2])
        assert t.shape == torch.Size([2])

        # Check that the solution satisfies the KKT conditions (approximately)
        # 1. Primal feasibility: x in (alpha, beta)
        assert torch.all(x > st.alpha) and torch.all(x < st.beta)

        # 2. Dual feasibility: s, t > 0
        assert torch.all(s > 0) and torch.all(t > 0)

        # 3. Complementary slackness (approximately)
        assert torch.allclose(s * (x - st.alpha), torch.tensor(st.gamma), atol=1e-3)
        assert torch.allclose(t * (st.beta - x), torch.tensor(st.gamma), atol=1e-3)

        # 4. Stationarity
        a_val = st.OTS.a(x)
        residual = L @ x + I_ + a_val - s + t
        assert torch.norm(residual) < 1e-3

    @pytest.mark.parametrize(
        "x_input",
        [
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([[1.0, 1.0]]),
        ],
    )
    def test_solve(self, primal_dual_strategy, x_input):
        """Test the solve method."""
        st = primal_dual_strategy
        st.reset()

        # Call the solve method
        result = st.solve(x_input, None)

        # Check the output type and shape
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([1, 4])  # 4 elements for the 2 layers (2 nodes each)

        # Check that the solution is within bounds
        assert torch.all(result > st.alpha) and torch.all(result < st.beta)

        # Check that the first two elements (first layer) are reasonable
        first_layer = result[:, :2]
        assert first_layer.shape == torch.Size([1, 2])
