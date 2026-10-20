"""Dataset split enumeration and dataset wrappers."""

from __future__ import annotations

import enum

import torch
from torch.utils.data import Dataset, Subset as TorchSubset

from src.data.structures import Inputs, Targets


class Partitions(enum.Enum):
    """Named dataset splits."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class DataSplit(Dataset[tuple[Inputs, Targets]]):
    """Dataset split yielding inputs and targets."""


class Subset(DataSplit, TorchSubset[tuple[Inputs, Targets]]):
    """Subset of a dataset split that preserves the DataSplit interface."""


class MnistSplit(DataSplit):
    """MNIST subset returning the structures the model consumes."""

    def __init__(self, dataset: TorchSubset[tuple[torch.Tensor, int]]) -> None:
        super().__init__()
        self.dataset = dataset
        return

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        img_tensor, target = self.dataset[index]
        label = torch.tensor(target, dtype=torch.long)
        return Inputs(image=img_tensor, condition=label), Targets(label=label, ref=img_tensor)
