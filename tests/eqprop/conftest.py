import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import open_dict

from src._eqprop.direct_backbone import AnalogEP2
from src.core.eqprop import activation
from src.core.eqprop.strategy import strategies, second_strategies


@pytest.fixture(scope="module")
def toymodel():
    """A pytest fixture for creating a toy model for testing strategies.

    Returns:
        dict: A dictionary containing model parameters.
    """
    # Create a simple model with 2 inputs and 2 hidden units
    W = [torch.tensor([[1.0, 1.0], [1.0, 1.0]]), torch.tensor([[2.0, 0.0], [0.0, 2.0]])]
    # Need to provide bias for each layer
    B = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    dims = [2, 2]

    return {"W": W, "B": B, "dims": dims}


@pytest.fixture(scope="module")
def xor():
    """A pytest fixture for creating a toy model for testing strategies.

    Returns:
        dict: A dictionary containing model parameters.
    """
    # Create a simple model with 2 inputs and 2 hidden units
    W = [torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), torch.tensor([[2.0, 1.0], [1.0, 2.0]])]
    # Need to provide bias for each layer
    B = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    dims = [2, 2]

    return {"W": W, "B": B, "dims": dims}


@pytest.fixture(scope="module")
def xormodel():
    """A pytest fixture for creating a toy model for testing strategies.

    Returns:
        dict: A dictionary containing model parameters.
    """
    # Create a simple model with 2 inputs and 2 hidden units
    W = [torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), torch.tensor([[2.0, 1.0], [1.0, 2.0]])]
    # Need to provide bias for each layer
    B = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    dims = [2, 2]

    return {"W": W, "B": B, "dims": dims}


@pytest.fixture(scope="module")
def second_order_strategy(toymodel) -> second_strategies.SecondOrderStrategy:
    """A pytest fixture for instantiating a second order EqProp strategy.

    Args:
        toymodel (dict): Dictionary containing model parameters.

    Returns:
        strategy.SecondOrderStrategy: A second order strategy instance.
    """
    st = second_strategies.NewtonStrategy(
        activation=activation.SymReLU(Vl=-0.6, Vr=0.6),
        amp_factor=1.0,
        add_nonlin_last=False,
        max_iter=1,
        atol=1e-4,
        clip_threshold=1,
    )
    st.W = toymodel["W"]
    st.B = toymodel["B"]
    st.dims = toymodel["dims"]
    return st


@pytest.fixture(scope="module")
def primal_dual_strategy(toymodel) -> second_strategies.PrimalDualStrategy:
    """A pytest fixture for instantiating a PrimalDual EqProp strategy.

    Args:
        toymodel (dict): Dictionary containing model parameters.

    Returns:
        strategy.PrimalDualStrategy: A PrimalDual strategy instance.
    """
    st = second_strategies.PrimalDualStrategy(
        activation=activation.SymReLU(Vl=-0.6, Vr=0.6),
        amp_factor=1.0,
        add_nonlin_last=False,
        max_iter=20,
        atol=1e-4,
        gamma=0.1,
        alpha=-0.6,
        beta=0.6,
    )
    st.W = toymodel["W"]
    st.B = toymodel["B"]
    st.dims = toymodel["dims"]
    return st


@pytest.fixture(scope="module")
def toy_backbone(cfg_train_global) -> AnalogEP2:
    """A pytest fixture for instantiating a toy EqProp backbone.

    Args:
        cfg_train_global (_type_): _description_

    Returns:
        AnalogEP2: _description_
    """
    cfg = cfg_train_global.copy()
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        # cfg.paths.out
        cfg.model.net.batch_size = 1
        cfg.model.net.dims = [2, 1, 1]
        cfg.model.net.beta = 0.1
        cfg.model.net.solver.strategy.add_nonlin_last = False
        cfg.model.net.solver.strategy.clip_threshold = 1
        cfg.model.net.solver.strategy.max_iter = 20
        cfg.model.net.solver.strategy.activation = {
            "_target_": "src.utils.eqprop_utils.SymReLU",
            "Vl": -0.6,
            "Vr": 0.6,
        }
    backbone_: AnalogEP2 = instantiate(cfg.model.net)
    backbone = backbone_(hyper_params={"bias": True})
    backbone.model[0].weight.data = torch.tensor([[1.0, 1.0]])
    backbone.model[0].bias.data = torch.tensor([1.0])
    backbone.model[1].weight.data = torch.tensor([[2.0]])
    backbone.model[1].bias.data = torch.tensor([0.0])
    return backbone
