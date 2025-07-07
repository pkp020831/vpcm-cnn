import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


class XORDataModule(LightningDataModule):
    """Example of LightningDataModule for XOR dataset."""

    def __init__(
        self, batch_size: int = 1, num_workers: int = 0, pin_memory: bool = False, transform=None
    ):
        """_summary_

        Args:
            batch_size (int, optional): _description_. Defaults to 1.
            num_workers (int, optional): _description_. Defaults to 0.
            pin_memory (bool, optional): _description_. Defaults to False.
            transform (_type_, optional): transform inputs. If set to None, input values become {-1,1} Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms = transform if transform else transforms.Normalize((0.5,), (0.5))

    @property
    def num_classes(self):
        return 2

    def setup(self, stage=None) -> None:
        """Setup dataset."""
        inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).float().unsqueeze(1)
        targets = torch.tensor([0, 1, 1, 0])
        self.dataset = TensorDataset(self.transforms(inputs), targets)

    def train_dataloader(self):
        """train_dataloader.

        Returns:
            _type_: _description_
        """
        # Create the dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def val_dataloader(self):
        """val_dataloader.

        Returns:
            _type_: _description_
        """
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader


class XORwithBiasDataModule(XORDataModule):
    """XOR dataset with bias.

    Args:
        XORDataModule (_type_): _description_
    """

    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        transform=None,
        scale_factor=1,
    ):
        """_summary_

        Args:
            batch_size (int, optional): _description_. Defaults to 1.
            num_workers (int, optional): _description_. Defaults to 0.
            pin_memory (bool, optional): _description_. Defaults to False.
            transform (_type_, optional): transform inputs. If set to None, input values become {-1,1} Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transforms = transform if transform else transforms.Normalize((0.5,), (0.25))
        self.scale_factor = scale_factor

    def __getitem__(self, index: int) -> EVAL_DATALOADERS:
        """__getitem__.

        Args:
            index (int): _description_

        Returns:
            EVAL_DATALOADERS: _description_
        """
        return self.train_dataloader(), self.val_dataloader()

    def setup(self, stage=None) -> None:
        """Setup dataset for XOR with inputs as c=1, w=1, h=3 format."""
        assert self.batch_size <= 4, "batch_size must be less than or equal to 4"

        # (4, 3) 형태의 inputs를 만듦 (batch_size, h=3)
        inputs = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        ).float()

        # 필요한 차원을 추가하여 (batch_size, c=1, w=1, h=3)로 변환
        inputs = inputs.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, 3)

        # targets는 그대로 사용
        targets = torch.tensor([0, 1, 1, 0])

        # scale_factor 적용
        inputs = inputs * self.scale_factor
        targets = targets * self.scale_factor

        # transforms 적용 후 dataset 생성
        self.dataset = TensorDataset(self.transforms(inputs), targets)
