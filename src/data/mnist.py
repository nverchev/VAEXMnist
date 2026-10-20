"""MNIST data builder, with the stroke-thickness bias it can apply."""

from typing import Any, ClassVar, override, cast

from collections.abc import Callable

import torch

from torch.nn import functional
from torch.utils.data import Subset
from torchvision import datasets as vision_datasets, transforms

from src.config.specs import PhaseDataConfig
from src.data.databuilder import DataBuilder
from src.data.split import MnistSplit, Partitions


__all__ = [
    'MnistDataBuilder',
    'get_random_class_mask',
]

RAW_SIZE = 28
TRAIN_SIZE = 50000
VAL_SIZE = 10000


class _MnistDataset(vision_datasets.MNIST):
    """Torchvision MNIST dataset with typed tensor attributes."""

    data: torch.Tensor
    targets: torch.Tensor


class MnistDataBuilder(DataBuilder):
    """Data builder managing MNIST loading, partition splits and the bias.

    The bias removes, in each class, the samples whose stroke width disagrees
    with the label: the thick ones from the classes expected to be thin and the
    other way round. The thresholds are per-class quantiles of the stroke width
    measured on the training partition, so every partition is cut at the same
    absolute widths, and a quantile of one keeps the standard dataset. Training
    and validation are cut, so the model learns and is selected on the biased
    distribution, while the test partition stays whole to measure the shortcut.
    Each phase chooses its data, so the cut depends on the phase that asks.
    """

    folder_name: ClassVar[str] = 'mnist'

    full_train_dataset: _MnistDataset
    test_dataset: _MnistDataset
    thresholds: torch.Tensor
    _partition_maps: dict[bool, dict[Partitions, tuple[_MnistDataset, torch.Tensor]]]
    _stroke_widths: dict[int, torch.Tensor]

    def get_alignment(self, partition: Partitions, *, data: PhaseDataConfig) -> torch.Tensor:
        """Get which samples of a partition have the stroke width their class group expects.

        Args:
            partition: the partition, in the order of its split.
            data: the data of the phase that uses the split.

        Returns:
            A boolean mask, true for the bias-aligned samples and false for the conflicting ones.
        """
        dataset, indices = self._get_partition_map(data)[partition]
        return self._get_alignment_mask(self._stroke_widths[id(dataset)][indices], dataset.targets[indices])

    def get_labels(self, partition: Partitions, *, data: PhaseDataConfig) -> torch.Tensor:
        """Get the labels of a partition.

        Args:
            partition: the partition, in the order of its split.
            data: the data of the phase that uses the split.

        Returns:
            The class of each sample.
        """
        dataset, indices = self._get_partition_map(data)[partition]
        return dataset.targets[indices]

    @override
    def get_split(self, partition: Partitions, *, data: PhaseDataConfig) -> MnistSplit:
        dataset, indices = self._get_partition_map(data)[partition]
        return MnistSplit(Subset(dataset, indices.tolist()))

    def _build_partition_map(self, random: bool) -> dict[Partitions, tuple[_MnistDataset, torch.Tensor]]:
        partition_map: dict[Partitions, tuple[_MnistDataset, torch.Tensor]] = {}
        for partition in Partitions:
            dataset, indices = self._get_partition_indices(partition)
            if partition is not Partitions.test:
                widths = self._stroke_widths[id(dataset)][indices]
                labels = dataset.targets[indices]
                if random:
                    mask = get_random_class_mask(labels, fraction=self.bias_cfg.quantile)
                else:
                    mask = self._get_alignment_mask(widths, labels)

                indices = indices[mask]

            partition_map[partition] = (dataset, indices)

        return partition_map

    def _compute_class_thresholds(self, thickness: dict[int, torch.Tensor]) -> torch.Tensor:
        train_partition = Partitions.train_val if self.cfg.final else Partitions.train
        dataset, indices = self._get_partition_indices(train_partition)
        widths = thickness[id(dataset)][indices]
        labels = dataset.targets[indices]
        n_classes = len(self.cfg.data.dataset.classes)
        thresholds = torch.empty(n_classes, dtype=widths.dtype)
        for label in range(n_classes):
            quantile = self.bias_cfg.quantile if label in self.bias_cfg.thin_classes else 1.0 - self.bias_cfg.quantile
            class_widths = widths[labels == label]
            thresholds[label] = class_widths.quantile(quantile) if class_widths.numel() else torch.nan

        return thresholds

    def _compute_stroke_widths(self) -> dict[int, torch.Tensor]:
        return {
            id(dataset): self.estimate_stroke_thickness(dataset.data.float() / 255.0, self.bias_cfg.ink_threshold)
            for dataset in (self.full_train_dataset, self.test_dataset)
        }

    @override
    def _download(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        padding = (self.cfg.data.x_dim - RAW_SIZE) // 2
        steps: list[Callable[..., Any]] = [transforms.ToTensor()]
        if padding:
            steps.append(transforms.Pad(padding))

        self.transform = transforms.Compose(steps)
        self.full_train_dataset = _MnistDataset(root=self.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = _MnistDataset(root=self.data_dir, train=False, download=True, transform=self.transform)
        return

    def _get_alignment_mask(self, thickness: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        thin_expected = torch.isin(labels, self.thin_classes)
        limit = self.thresholds[labels]
        return torch.where(thin_expected, thickness <= limit, thickness >= limit)

    def _get_partition_map(self, data: PhaseDataConfig) -> dict[Partitions, tuple[_MnistDataset, torch.Tensor]]:
        if data.random not in self._partition_maps:
            self._partition_maps[data.random] = self._build_partition_map(data.random)

        return self._partition_maps[data.random]

    def _get_partition_indices(self, partition: Partitions) -> tuple[_MnistDataset, torch.Tensor]:
        if partition is Partitions.test:
            return self.test_dataset, torch.arange(len(self.test_dataset))

        if partition is Partitions.val:
            return self.full_train_dataset, torch.arange(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)

        if partition is Partitions.train_val:
            return self.full_train_dataset, torch.arange(len(self.full_train_dataset))

        return self.full_train_dataset, torch.arange(TRAIN_SIZE)

    @override
    def _preprocess(self) -> None:
        self._stroke_widths = self._compute_stroke_widths()
        self.thresholds = self._compute_class_thresholds(self._stroke_widths)
        self._partition_maps = {}
        return

    @staticmethod
    def estimate_stroke_thickness(images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Estimate the average stroke width of each image.

        The foreground is eroded with a 3x3 structuring element; the pixels lost by
        the erosion form the stroke border. For a stroke of width ``w`` and length
        ``l`` the foreground holds ``w * l`` pixels and the border ``2 * l``, so
        twice the area-to-border ratio recovers ``w`` independently of how long or
        how curved the stroke is.

        Args:
            images: Batch of grayscale images shaped ``(n, h, w)`` or ``(n, 1, h, w)``
                with values in ``[0, 1]``.
            threshold: Intensity above which a pixel counts as ink.

        Returns:
            Tensor of shape ``(n,)`` with the estimated stroke width in pixels.
        """
        foreground = (images > threshold).float()
        if foreground.dim() == 3:
            foreground = foreground.unsqueeze(1)

        eroded = -functional.max_pool2d(-functional.pad(foreground, (1, 1, 1, 1)), kernel_size=3, stride=1)
        area = foreground.sum(dim=(1, 2, 3))
        border = (foreground - eroded).sum(dim=(1, 2, 3))
        return 2.0 * area / border.clamp(min=1.0)


def get_random_class_mask(labels: torch.Tensor, /, *, fraction: float, seed: int = 0) -> torch.Tensor:
    """Select the same fraction of each class at random.

    Args:
        labels: the class of each sample.
        fraction: the share of each class to select.
        seed: the seed of the random selection, fixed so that the subset does not change.

    Returns:
        A boolean mask over the samples.
    """
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label in labels.unique():
        class_indices = (labels == cast(torch.Tensor, label)).nonzero().squeeze(1)
        n_selected = round(fraction * len(class_indices))
        order = torch.randperm(len(class_indices), generator=generator)
        mask[class_indices[order[:n_selected]]] = True

    return mask
