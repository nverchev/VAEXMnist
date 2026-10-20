"""Tests for the MNIST data builder."""

import torch

from src.data.mnist import get_random_class_mask


def test_the_random_mask_selects_the_same_share_of_each_class() -> None:
    """Each class keeps its share of the samples, rounded."""
    labels = torch.tensor([0] * 10 + [1] * 6 + [2] * 3)

    mask = get_random_class_mask(labels, fraction=0.5)

    assert [int(mask[labels == label].sum()) for label in range(3)] == [5, 3, 2]


def test_the_random_mask_does_not_change_with_the_same_seed() -> None:
    """The same seed selects the same samples."""
    labels = torch.arange(100) % 10

    first = get_random_class_mask(labels, fraction=0.5)
    second = get_random_class_mask(labels, fraction=0.5)

    assert torch.equal(first, second)
