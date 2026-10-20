"""Torch config utilities."""

import math
import random

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'ActClass',
    'GroupedNorm',
    'NormClass',
    'get_activation_cls',
    'get_device',
    'get_mixed_precision',
    'get_norm_cls',
    'get_optim_cls',
    'set_seed',
]

type ActClass = type[nn.Module]
type NormClass = type[nn.Module]


class GroupedNorm(nn.GroupNorm):
    """Group normalization with automatic number of groups for channels."""

    def __init__(self, num_channels: int) -> None:
        """Initialize.

        Args:
            num_channels: number of channels to normalize.
        """
        n_groups = int(math.sqrt(num_channels))
        while num_channels % n_groups != 0:
            n_groups -= 1

        super().__init__(num_groups=n_groups, num_channels=num_channels)
        return


def get_activation_cls(act_name: str | ActClass, /) -> ActClass:
    """Get an activation class from the ones available in PyTorch.

    Args:
        act_name: name of the class, or the class itself, which is looked up by its name.

    Returns:
        The activation class.

    Raises:
        ValueError: if no activation class has that name.
    """
    if isinstance(act_name, type):
        act_name = act_name.__name__

    try:
        return getattr(nn.modules.activation, act_name)
    except AttributeError as ae:
        raise ValueError(f'Input act_name "{act_name}" is not the name of a pytorch activation.') from ae


def get_device(device: str | torch.device, /) -> torch.device | Literal['auto']:
    """Get the device where the models run.

    Args:
        device: ``auto`` for the accelerator when available, or a device, as a name such as ``cpu`` or ``cuda:1``.

    Returns:
        The device, or ``'auto'``.

    Raises:
        ValueError: if the name is not ``auto`` or the name of a device.
    """
    if isinstance(device, torch.device) or device == 'auto':
        return device

    try:
        return torch.device(device)
    except RuntimeError as re:
        raise ValueError(f'Input device "{device}" is not auto or the name of a device.') from re


def get_mixed_precision(precision: str | torch.dtype | None, /) -> torch.dtype | None:
    """Get the autocast precision of mixed precision training.

    Args:
        precision: ``float16`` or ``bfloat16``, as a name or a dtype; ``None`` for no mixed precision.

    Returns:
        The dtype, or ``None`` without mixed precision.

    Raises:
        ValueError: if the precision is not ``float16`` or ``bfloat16``.
    """
    if precision is None:
        return None

    if isinstance(precision, torch.dtype):
        precision = str(precision).removeprefix('torch.')

    if precision == 'float16':
        return torch.float16
    elif precision == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f'Input precision "{precision}" is not float16 or bfloat16.')


def get_norm_cls(norm_name: str | NormClass, /) -> NormClass:
    """Get a normalization class from the custom ones and the ones available in PyTorch.

    Args:
        norm_name: name of the class, or the class itself, which is looked up by its name.

    Returns:
        The normalization class.

    Raises:
        ValueError: if no normalization class has that name.
    """
    if isinstance(norm_name, type):
        norm_name = norm_name.__name__

    if norm_name == 'GroupedNorm':
        return GroupedNorm

    try:
        return getattr(nn, norm_name)
    except AttributeError as ae:
        raise ValueError(f'Input norm_name "{norm_name}" is not the name of a pytorch nn.Module.') from ae


def get_optim_cls(optimizer_name: str | type[torch.optim.Optimizer], /) -> type[torch.optim.Optimizer]:
    """Get an optimizer class from the ones available in PyTorch.

    Args:
        optimizer_name: name of the class, or the class itself, which is looked up by its name.

    Returns:
        The optimizer class.

    Raises:
        ValueError: if no optimizer class has that name.
    """
    if isinstance(optimizer_name, type):
        optimizer_name = optimizer_name.__name__

    try:
        return getattr(torch.optim, optimizer_name)
    except AttributeError as ae:
        raise ValueError(f'Input opt_name "{optimizer_name}" is not the name of a pytorch optimizer.') from ae


def set_seed(seed: int, /) -> None:
    """Set seed for reproducibility.

    Args:
        seed: seed for the random generators of Python, NumPy and PyTorch.

    Side Effects:
        Seeds the random generators and makes cuDNN choose deterministic algorithms.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return
