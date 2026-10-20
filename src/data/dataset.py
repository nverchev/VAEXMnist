"""Dataset classes and factory for the training/eval split."""

import torch.distributed as dist

from src.config.experiment import Experiment
from src.config.options import Datasets
from src.config.specs import PhaseDataConfig
from src.data.databuilder import DataBuilder
from src.data.mnist import MnistDataBuilder
from src.data.split import DataSplit, Partitions


def get_databuilder() -> DataBuilder:
    """Return the dataset builder instance."""
    builders: dict[Datasets, type[DataBuilder]] = {Datasets.MNIST: MnistDataBuilder}
    return builders[Experiment.get_config().data.dataset.name]()


def get_split(partition: Partitions, *, data: PhaseDataConfig) -> DataSplit:
    """Return dataset instance for a given partition.

    Args:
        partition: the partition to load.
        data: the data of the phase that uses the split.
    """
    databuilder = get_databuilder()
    return databuilder.get_split(partition, data=data)


def _get_splits(data: PhaseDataConfig) -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    return get_split(train_partition, data=data), get_split(eval_partition, data=data)


def get_splits(*, data: PhaseDataConfig) -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets in a multiprocess-safe way.

    Args:
        data: the data of the phase that uses the splits.
    """
    cfg = Experiment.get_config()
    if not cfg.user.n_subprocesses:
        return _get_splits(data)

    rank = dist.get_rank()
    splits: tuple[DataSplit, DataSplit]
    for i in range(cfg.user.n_subprocesses):
        if rank == i:
            splits = _get_splits(data)

        dist.barrier()

    return splits
