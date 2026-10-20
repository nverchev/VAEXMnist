"""Neural network layer building blocks."""

import abc
import functools
import warnings

from collections.abc import Callable
from typing import Final, Protocol, cast, override, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.environment import DEBUG_MODE
from src.config.torch import ActClass, NormClass


type _grad_t = tuple[torch.Tensor, ...] | torch.Tensor


@runtime_checkable
class HasInplace(Protocol):
    """Protocol for classes that have an inplace option."""

    inplace: bool


@runtime_checkable
class HasResetParamPublic(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""


@runtime_checkable
class HasResetParamPrivate(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def _reset_parameters(self) -> None:
        """Reset the parameters of the module."""


@runtime_checkable
class Weighted(Protocol):
    """Protocol for classes that have a weight attribute."""

    weight: torch.Tensor
    bias: torch.Tensor | None


class Mean(nn.Module):
    """Average over some dimensions, which the result loses.

    Attributes:
        dims: dimensions to average.
    """

    dims: Final[tuple[int, ...]]

    def __init__(self, *dims: int) -> None:
        """Initialize.

        Args:
            dims: dimensions to average.
        """
        super().__init__()
        self.dims = dims
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(self.dims)

    @override
    def extra_repr(self) -> str:
        return f'dims={self.dims}'


class BaseLayer(nn.Module, metaclass=abc.ABCMeta):
    """A class that includes a layer followed by an optional activation and grouped normalization."""

    in_dim: int
    out_dim: int
    n_groups_layer: int
    use_bias: bool
    use_trunc_init: bool
    init_weight_fn: Callable[[torch.Tensor], torch.Tensor]
    init_bias_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        n_groups_layer: int = 1,
        use_trunc_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups_layer = n_groups_layer
        self.use_bias = True if norm_cls is None else False
        self.module: nn.Module = self.get_module()
        self.norm: nn.Module | None = self.get_norm_layer(norm_cls, out_dim)
        self.act: nn.Module | None = self.get_activation(act_cls)
        self.use_trunc_init = use_trunc_init
        self.init_weight_fn = self.get_init_weight(self.act, use_trunc_init)
        self.init_bias_fn = self.get_init_bias(use_trunc_init)
        self.reset_parameters()
        if DEBUG_MODE:
            warnings.filterwarnings('ignore', message='.*Full backward hook is firing.*')
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    def reset_parameters(self) -> None:
        self.init_weight_fn(cast(torch.Tensor, self.module.weight))
        if self.norm is None:
            self.init_bias_fn(cast(torch.Tensor, self.module.bias))
            return

        _reset_parameters_or_warn(self.norm)
        self.init_bias_fn(cast(torch.Tensor, self.norm.bias))
        return

    @abc.abstractmethod
    def get_module(self) -> nn.Module:
        """Get wrapped module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.module(x)
        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

        return y

    @staticmethod
    def get_activation(act_cls: ActClass | None) -> nn.Module | None:
        if act_cls is None:
            return None

        act = act_cls()
        if isinstance(act, HasInplace):
            act.inplace = True

        return act

    @staticmethod
    def get_init_weight(act: nn.Module | None, use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        if use_trunc_init:
            return functools.partial(nn.init.trunc_normal_, mean=0.0, std=0.02, a=-0.04, b=0.04)

        if act is None:
            return functools.partial(nn.init.xavier_normal_, gain=1)

        if isinstance(act, nn.Hardtanh):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('tanh'))

        if isinstance(act, nn.Sigmoid):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('sigmoid'))

        if isinstance(act, nn.ReLU):
            return functools.partial(nn.init.kaiming_normal_, nonlinearity='relu')

        if isinstance(act, nn.LeakyReLU):
            return functools.partial(nn.init.kaiming_normal_, a=act.negative_slope, nonlinearity='leaky_relu')

        if isinstance(act, nn.GELU):
            return functools.partial(nn.init.xavier_normal_, gain=1)

        return functools.partial(nn.init.xavier_normal_, gain=1)

    @staticmethod
    def get_init_bias(use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.nn.init.zeros_ if use_trunc_init else _keep_unchanged

    @staticmethod
    def get_norm_layer(norm_cls: NormClass | None, out_dim: int) -> nn.Module | None:
        if norm_cls is None:
            return None

        return norm_cls(out_dim)


class LinearLayer(BaseLayer):
    """Linear layer block."""

    @override
    def get_module(self) -> nn.Module:
        if self.n_groups_layer > 1:
            raise NotImplementedError('nn.Linear has no option for groups')

        return nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)


class Conv2dLayer(BaseLayer):
    """2D convolutional layer built on BaseLayer for proper weight initialisation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        use_trunc_init: bool = False,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        super().__init__(in_dim, out_dim, act_cls=act_cls, norm_cls=norm_cls, use_trunc_init=use_trunc_init)
        return

    @override
    def get_module(self) -> nn.Module:
        return nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.use_bias,
            groups=self.groups,
        )


class ConvTranspose2dLayer(BaseLayer):
    """2D transposed convolutional layer built on BaseLayer for proper weight initialisation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        use_trunc_init: bool = False,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        super().__init__(in_dim, out_dim, act_cls=act_cls, norm_cls=norm_cls, use_trunc_init=use_trunc_init)
        return

    @property
    def output_padding(self) -> int:
        """Padding that makes the output exactly *stride* times the input.

        A transposed convolution maps a size ``n`` to
        ``(n - 1) * stride - 2 * padding + kernel_size``, which undershoots
        ``n * stride`` unless the difference is added back. Without it a
        stride-2 block yields ``2n - 1`` and a 1x1 input never grows.
        """
        exact = self.stride + 2 * self.padding - self.kernel_size
        return max(0, min(self.stride - 1, exact))

    @override
    def get_module(self) -> nn.Module:
        return nn.ConvTranspose2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=self.use_bias,
            groups=self.groups,
        )


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block for 4D feature maps."""

    gate: nn.Sequential

    def __init__(self, h_dim: int, reduction_ratio: float = 0.1) -> None:
        """Initialize.

        Args:
            h_dim: channels of the feature maps.
            reduction_ratio: width of the hidden layer, relative to the channels.
        """
        super().__init__()
        h_reduced = max(1, int(h_dim * reduction_ratio))
        self.gate = nn.Sequential(
            Mean(2, 3),
            LinearLayer(h_dim, h_reduced, act_cls=nn.LeakyReLU),
            LinearLayer(h_reduced, h_dim, act_cls=nn.Sigmoid),
            nn.Unflatten(1, (h_dim, 1, 1)),
        )
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mul(self.gate(inputs))


class ConvBlock(nn.Module):
    """Post-activation convolutional block: Conv2dLayer → norm → act.

    Args:
        use_trunc_init: if True, skips norm/act to keep a near-zero output.
    """

    block: BaseLayer

    def __init__(
        self,
        h_in: int,
        h_out: int,
        deconv: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        *,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        use_trunc_init: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        LayerCls = ConvTranspose2dLayer if deconv else Conv2dLayer
        self.block = LayerCls(
            h_in,
            h_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_cls=None if use_trunc_init else act_cls,
            norm_cls=None if use_trunc_init else norm_cls,
            use_trunc_init=use_trunc_init,
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResampleSkip(nn.Module):
    """Bring the input of a residual block to the shape of its output.

    The input is resized when the block changes the resolution, and its
    channels are averaged in pairs or repeated when the block halves or
    doubles them.

    Attributes:
        h_in: channels of the input.
        h_out: channels of the output.
        stride: stride of the block.
    """

    h_in: Final[int]
    h_out: Final[int]
    stride: Final[int]

    def __init__(self, h_in: int, h_out: int, stride: int) -> None:
        """Initialize.

        Args:
            h_in: channels of the input.
            h_out: channels of the output.
            stride: stride of the block.
        """
        super().__init__()
        self.h_in = h_in
        self.h_out = h_out
        self.stride = stride
        return

    def forward(self, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Reshape the input ``x`` as the output ``like`` of the block."""
        output_size = like.size()
        if self.stride != 1:
            x = F.interpolate(x, size=output_size[2:], mode='bilinear', align_corners=False)

        if self.h_out < self.h_in:
            x = x.view(output_size[0], output_size[1], 2, output_size[2], output_size[3]).mean(2)
        elif self.h_out > self.h_in:
            x = x.repeat(1, 2, 1, 1)

        return x

    @override
    def extra_repr(self) -> str:
        return f'h_in={self.h_in}, h_out={self.h_out}, stride={self.stride}'


class ResBlock(nn.Module):
    """Sequence of blocks whose output is added to the input, resampled to its shape.

    The last block starts from a truncated Gaussian, without normalization and activation,
    so the residual block starts close to its skip connection.

    Attributes:
        layers: the blocks, in order.
        se: the Squeeze-and-Excitation applied to the output of the blocks.
        skip: the connection carrying the input to the output of the blocks.
    """

    layers: nn.Sequential
    se: SqueezeExcite
    skip: ResampleSkip

    def __init__(
        self,
        Block: type[nn.Module],
        h_in: int,
        h_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        deconv: bool = False,
        depth: int = 0,
        *,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
    ) -> None:
        """Initialize.

        Args:
            Block: the class of the blocks.
            h_in: channels of the input.
            h_out: channels of the output.
            kernel_size: kernel size of the last block.
            stride: stride of the last block.
            padding: padding of the last block.
            deconv: whether the last block is a transposed convolution.
            depth: number of blocks before the last, keeping the channels of the input.
            act_cls: the class of the activation.
            norm_cls: the class of the normalization.
        """
        super().__init__()
        modules: list[nn.Module] = [Block(h_in, h_in, act_cls=act_cls, norm_cls=norm_cls) for _ in range(depth)]
        modules.append(
            Block(
                h_in,
                h_out,
                deconv=deconv,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                act_cls=act_cls,
                norm_cls=norm_cls,
                use_trunc_init=True,
            )
        )
        self.layers = nn.Sequential(*modules)
        self.se = SqueezeExcite(h_out)
        self.skip = ResampleSkip(h_in, h_out, stride)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the resampled input to the output of the blocks."""
        y = self.layers(x)
        return self.skip(x, y) + self.se(y)


def debug_check(_not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t) -> None:
    """Debug hook to check for NaN or Inf values in the output tensor."""

    import pdb

    if isinstance(tensor_out, tuple):
        tensor = tensor_out[0]
    else:
        tensor = tensor_out

    if torch.any(torch.isnan(tensor)):
        breakpoint()
        pdb.set_trace()
    elif torch.any(torch.isinf(tensor)):
        breakpoint()
        pdb.set_trace()

    return None


def _reset_parameters_or_warn(module: nn.Module) -> None:
    if isinstance(module, HasResetParamPublic):
        module.reset_parameters()
        return

    if isinstance(module, HasResetParamPrivate):
        module._reset_parameters()
        return

    if isinstance(module, nn.Module):
        warnings.warn(f'Module {module.__class__.__name__} does not implement reset_parameters().', stacklevel=3)

    return


def _keep_unchanged(tensor: torch.Tensor, /) -> torch.Tensor:
    return tensor
