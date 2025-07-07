from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.data.mnist_datamodule import MNISTDataModule
from tests.helpers.run_if import RunIf


def load_batch():
    """Load a batch of MNIST."""
    data_dir = "data/"
    dm = MNISTDataModule(data_dir=data_dir, batch_size=1)
    dm.prepare_data()
    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    dm_it = iter(dm.train_dataloader())
    yield next(dm_it)


@pytest.fixture(scope="function")
def eqprop_torch(batch, model, beta):
    """Run eqprop for 1 iter in torch."""
    # run 1 free & nudge phase together
    vout_torch = model(batch)
    return vout_torch


@RunIf(xyce=True)
@pytest.fixture(scope="function")
def eqprop_xyce(batch, model, beta):
    """Run eqprop for 1 iter in xyce."""
    # reshape vout to match torch version
    vout_xyce = model(batch)
    return vout_xyce


def test_eqprop_precision(cfg_train, ckpt_path):
    """Compare eqprop_torch with eqprop_xyce.

    1 free & nudge phase each.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.ckpt_path = str(ckpt_path / "checkpoints" / "last.ckpt")

    # compare when initialized with same weights
    beta = cfg_train.model.beta
    model = torch.load(cfg_train.ckpt_path)  # nosec B614
    batch = load_batch()
    vout_torch = eqprop_torch(batch, model, beta)
    vout_xyce = eqprop_xyce(batch, model, beta)
    assert vout_torch.shape == vout_xyce.shape, "Shapes do not match. check eqprop_xyce."
    assert torch.allclose(vout_torch, vout_xyce, atol=cfg_train.model.net.solver.atol), (
        "Values do not match. check eqprop solver."
    )


class TestAnalogEP2XOR:
    """Test EqProp on a simple XOR problem."""

    @pytest.mark.parametrize("x", torch.tensor([[-1.0, -1.0]]))
    def test_free(self, toy_backbone, x):
        """Test whether output is computed correctly in free phase."""
        ypred = toy_backbone(x).detach()
        assert torch.allclose(ypred.squeeze(), torch.tensor(-1.2))

    @pytest.mark.parametrize("x", torch.tensor([[-1.0, -1.0]]))
    def test_nudge(self, toy_backbone, x):
        """Test whether output grad and node potentials are computed correctly
        in nudge phase."""
        toy_backbone.reset_nodes()
        ypred = toy_backbone(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(ypred, torch.tensor([0.3]))
        loss.backward()
        assert torch.allclose(ypred.grad, torch.tensor([-3.0]))
        reversed_nodes = toy_backbone.solver(x, grad=ypred.grad)
        assert torch.allclose(reversed_nodes[0], torch.tensor(-0.95))
        assert torch.allclose(reversed_nodes[1], torch.tensor(-1.1))

    @pytest.mark.parametrize("x", [[-1.0, -1.0]])
    def test_update(self, toy_backbone, x):
        """Test whether weights are updated correctly."""
        x = torch.tensor(x).reshape(1, -1)
        toy_backbone.reset_nodes()
        toy_backbone(x)
        ypred = toy_backbone(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(ypred, torch.tensor([0.3]))
        loss.backward()
        toy_backbone.eqprop(x)
        net = toy_backbone.model
        assert torch.allclose(net.lin1.weight.grad, torch.tensor([[0.5, 0.5]]))
        # assert torch.allclose(net.lin1.bias.grad, torch.tensor([0.0]))
        assert torch.allclose(net.last.weight.grad.item(), torch.tensor([[0.225]]), atol=1e-4)
        # assert torch.allclose(net.last.bias.grad, torch.tensor([0.0]))

    @pytest.mark.slow
    def test_train_XOR(self, toy_backbone, XOR):
        """Test whether the network can learn XOR."""
        optimizer = torch.optim.SGD(toy_backbone.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss(reduction="sum")
        for epoch in range(10):
            for x, y in XOR:
                toy_backbone.reset_nodes()
                x = torch.tensor(x).reshape(1, -1)
                y = torch.tensor(y).reshape(1, -1)
                y_hat = toy_backbone(x)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                toy_backbone.eqprop(x, y)
                optimizer.step()
