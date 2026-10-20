"""Options and choices for configuration settings."""

import enum


class Datasets(enum.StrEnum):
    """Dataset choices."""

    MNIST = enum.auto()
    """Standard MNIST dataset."""
    """Biased MNIST dataset with spurious patches."""


class Models(enum.StrEnum):
    """Autoencoder VAE model choices."""

    CVAE = enum.auto()
    """Conditional Variational Autoencoder (Conditions on the attribute)."""
    VAEX = enum.auto()
    """Explainable Variational Autoencoder (Conditions on the classifier's attribute evalution)."""


class GradOp(enum.StrEnum):
    """Gradient operation names."""

    PARAM_NORMALIZER = enum.auto()
    """Normalize gradients by parameter norm."""
    Z_SCORE_NORMALIZER = enum.auto()
    """Standardize gradients using running Z-score statistics."""
    NORM_CLIPPER = enum.auto()
    """Clip global gradient norm."""
    VALUE_CLIPPER = enum.auto()
    """Clip elementwise gradient values."""
    HIST_CLIPPER = enum.auto()
    """Histogram-based global gradient clipper."""
    PARAM_HIST_CLIPPER = enum.auto()
    """Histogram-based per-parameter gradient clipper."""


class ClipCriterion(enum.StrEnum):
    """Clipping criterion names for histogram-based gradient clippers."""

    Z_STAT = enum.auto()
    """Z-score statistical clipping threshold."""
    EMA = enum.auto()
    """Exponential moving average threshold."""


class Schedulers(enum.StrEnum):
    """Learning-rate scheduler names."""

    CONSTANT = enum.auto()
    """Fixed constant learning rate."""
    COSINE = enum.auto()
    """Cosine annealing learning rate schedule."""
    EXPONENTIAL = enum.auto()
    """Exponential decay learning rate schedule."""
    POLYNOMIAL = enum.auto()
    """Polynomial decay learning rate schedule."""
    STEP = enum.auto()
    """Step-wise learning rate decay schedule."""
