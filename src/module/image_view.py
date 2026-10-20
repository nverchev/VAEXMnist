"""Module returning the images of its inputs."""

from typing import override

import torch
import torch.nn as nn

from src.data.structures import Inputs

__all__ = [
    'ImageView',
]


class ImageView(nn.Module):
    """Return the images of the inputs, so that metrics can compare them with a reference."""

    @override
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the images.

        Args:
            inputs: the images.

        Returns:
            The images.
        """
        return inputs.image
