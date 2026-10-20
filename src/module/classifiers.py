"""Classifier of the digits."""

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.data.structures import Inputs

__all__ = [
    'BaseClassifier',
    'get_classifier',
]


class BaseClassifier(nn.Module):
    """Small CNN mapping the images to the logits of the classes.

    Attributes:
        n_classes: Number of classes.
        layers: Two convolutions, a pooling and two linear layers.
    """

    n_classes: Final[int]
    layers: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        self.n_classes = len(cfg.data.dataset.classes)
        pooled_dim = cfg.data.x_dim // 2
        self.layers = nn.Sequential(
            nn.Conv2d(cfg.data.n_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * pooled_dim * pooled_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes),
        )
        return

    @override
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the logits of the classes.

        Args:
            inputs: the images.

        Returns:
            The logits.
        """
        return self.layers(inputs.image)


def get_classifier() -> BaseClassifier:
    """Get the classifier of the digits.

    Returns:
        The classifier.
    """
    return BaseClassifier()
