"""Tests for the layer building blocks."""

import torch

from src.module.layers import Mean, ResampleSkip


class TestResampleSkip:
    """Test the skip connection of the residual blocks."""

    def test_halving_the_channels_averages_them_in_pairs(self) -> None:
        """Each output channel is the mean of two consecutive input channels."""
        x = torch.randn(2, 4, 3, 3)
        like = torch.empty(2, 2, 3, 3)

        out = ResampleSkip(4, 2, stride=1)(x, like)

        torch.testing.assert_close(out, torch.stack([x[:, 0:2].mean(1), x[:, 2:4].mean(1)], dim=1))

    def test_doubling_the_channels_repeats_them(self) -> None:
        """The channels are repeated as a block."""
        x = torch.randn(2, 2, 3, 3)
        like = torch.empty(2, 4, 3, 3)

        out = ResampleSkip(2, 4, stride=1)(x, like)

        torch.testing.assert_close(out, x.repeat(1, 2, 1, 1))

    def test_a_stride_resizes_the_input_to_the_output(self) -> None:
        """With a stride, the input takes the resolution of the output."""
        x = torch.randn(2, 3, 8, 8)
        like = torch.empty(2, 3, 4, 4)

        out = ResampleSkip(3, 3, stride=2)(x, like)

        assert out.shape == like.shape


def test_the_representation_shows_the_settings() -> None:
    """The modules show their settings in their representation."""
    text = repr(torch.nn.Sequential(Mean(2, 3), ResampleSkip(4, 2, stride=2)))

    for setting in ('dims=(2, 3)', 'h_in=4, h_out=2, stride=2'):
        assert setting in text, setting
