"""Measurement of the stroke width of images."""

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.data.mnist import MnistDataBuilder
from src.data.structures import Inputs

__all__ = [
    'StrokeWidth',
]


class StrokeWidth(nn.Module):
    """Measure the stroke width of the images, with the estimator that biases the dataset.

    Attributes:
        ink_threshold: Intensity above which a pixel counts as ink.
    """

    ink_threshold: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.ink_threshold = Experiment.get_config().data.dataset.bias.ink_threshold
        return

    @override
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the stroke width of each image.

        Args:
            inputs: the images.

        Returns:
            The stroke widths in pixels.
        """
        return MnistDataBuilder.estimate_stroke_thickness(inputs.image, self.ink_threshold)
