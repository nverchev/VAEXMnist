"""Conditional VAE architectures: the CVAE and VAEX."""

import abc
import itertools

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import Models
from src.config.torch import ActClass, NormClass
from src.data.structures import CounterfactualOutputs, Inputs, Outputs
from src.module.layers import ConvBlock, ResBlock

__all__ = [
    'CVAE',
    'VAEX',
    'BaseVAE',
    'ConditionalVAE',
    'get_autoencoder',
]


class BaseVAE(nn.Module, abc.ABC):
    """Abstract base for VAEs conditioned on a probability vector over the classes.

    Attributes:
        n_channels: Number of channels of the images.
        n_classes: Number of classes, the size of the condition.
        x_dim: Spatial dimension of the images.
        z_dim: Dimension of the latent space.
        sampling_variance: Multiplier on the standard deviation of the latent samples at inference.
    """

    n_channels: Final[int]
    n_classes: Final[int]
    x_dim: Final[int]
    z_dim: Final[int]
    sampling_variance: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        self.n_channels = cfg.data.n_channels
        self.n_classes = len(cfg.data.dataset.classes)
        self.x_dim = cfg.data.x_dim
        self.z_dim = cfg.autoencoding.model.z_dim
        self.sampling_variance = cfg.autoencoding.objective.sampling_variance
        return

    @override
    def forward(self, inputs: Inputs, /) -> Outputs:
        """Reconstruct the images with their own condition.

        Args:
            inputs: the images, with their labels or the log-probabilities of the classifier.

        Returns:
            The reconstructions and the posterior of the latent.
        """
        condition = self.get_condition(inputs)
        mu, log_var = self.encode(inputs.image, condition)
        z = self.sample_latent(mu, log_var)
        return Outputs(recon=self.decode(z, condition), mu=mu, log_var=log_var)

    @abc.abstractmethod
    def decode(self, z: torch.Tensor, condition: torch.Tensor, /) -> torch.Tensor:
        """Decode latent samples into images.

        Args:
            z: the latent samples.
            condition: the probability vectors over the classes.

        Returns:
            The images.
        """

    @abc.abstractmethod
    def encode(self, image: torch.Tensor, condition: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images into the posterior of the latent.

        Args:
            image: the images.
            condition: the probability vectors over the classes.

        Returns:
            The mean and the log-variance of the posterior.
        """

    @abc.abstractmethod
    def generate_counterfactual(self, inputs: Inputs, target_condition: torch.Tensor, /) -> CounterfactualOutputs:
        """Generate counterfactuals: encode the images with their condition, decode them with the target condition.

        Args:
            inputs: the images, with their labels or the log-probabilities of the classifier.
            target_condition: the probability vector over the classes each image is decoded with.

        Returns:
            The counterfactuals, the reconstructions decoded from the same latent samples, and the posterior.
        """

    @abc.abstractmethod
    def generate_sample(self, batch_size: int, /, *, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Generate random images from the prior of the latent.

        Args:
            batch_size: the number of images.
            condition: the probability vectors over the classes. Default draws random ones.

        Returns:
            The images.
        """

    @abc.abstractmethod
    def get_condition(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the condition of the inputs.

        Args:
            inputs: the images, with their labels or the log-probabilities of the classifier.

        Returns:
            The probability vectors over the classes.
        """

    @abc.abstractmethod
    def get_random_condition(self, batch_size: int, /) -> torch.Tensor:
        """Get random conditions.

        Args:
            batch_size: the number of conditions.

        Returns:
            The probability vectors over the classes.
        """

    def get_target_condition(self, targets: torch.Tensor, /, *, value: float) -> torch.Tensor:
        """Get conditions that give each target class the probability ``value`` and share the rest equally.

        Args:
            targets: the target class of each condition.
            value: the probability of the target class: ``1`` for the class alone, ``1 / n_classes`` for none.

        Returns:
            The probability vectors over the classes.
        """
        condition = torch.full((len(targets), self.n_classes), (1 - value) / (self.n_classes - 1))
        condition = condition.to(targets.device)
        condition[torch.arange(len(targets)), targets.long()] = value
        return condition

    def sample_latent(self, mu: torch.Tensor, log_var: torch.Tensor, /) -> torch.Tensor:
        """Sample the posterior of the latent, scaled by ``sampling_variance`` at inference.

        Args:
            mu: the mean of the posterior.
            log_var: the log-variance of the posterior.

        Returns:
            The latent samples.
        """
        std = torch.exp(0.5 * log_var)
        scale = 1.0 if self.training else self.sampling_variance
        return mu + torch.randn_like(std) * std * scale


class ConditionalVAE(BaseVAE, abc.ABC):
    """Convolutional VAE that concatenates the condition to the encoding and to the latent.

    Attributes:
        h_dims: Number of channels at each resolution.
        k_dims: Kernel sizes of the resolution changes.
        strides: Strides of the resolution changes.
        paddings: Paddings of the resolution changes.
        depths: Residual layers in each resolution change.
        act_cls: Activation of the blocks.
        norm_cls: Normalization of the blocks.
        final_res: Spatial dimension at the lowest resolution.
        flat_dim: Size of the flattened encoding at the lowest resolution.
        encoder: Convolutional encoder, from the images to the flattened encoding.
        fc_mu: Linear layer from the encoding and the condition to the mean of the posterior.
        fc_log_var: Linear layer from the encoding and the condition to the log-variance of the posterior.
        decoder: Decoder, from the latent and the condition to the images.
    """

    h_dims: Final[tuple[int, ...]]
    k_dims: Final[tuple[int, ...]]
    strides: Final[tuple[int, ...]]
    paddings: Final[tuple[int, ...]]
    depths: Final[tuple[int, ...]]
    act_cls: Final[ActClass]
    norm_cls: Final[NormClass]
    final_res: Final[int]
    flat_dim: Final[int]
    encoder: Final[nn.Sequential]
    fc_mu: Final[nn.Linear]
    fc_log_var: Final[nn.Linear]
    decoder: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        arch = Experiment.get_config().autoencoding.model.architecture
        self.h_dims = arch.h_dims
        self.k_dims = arch.k_dims
        self.strides = arch.strides
        self.paddings = arch.paddings
        self.depths = arch.depths
        self.act_cls = arch.act_cls
        self.norm_cls = arch.norm_cls
        self.final_res = self._get_final_resolution()
        self.flat_dim = self.h_dims[-1] * self.final_res**2
        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self.flat_dim + self.n_classes, self.z_dim)
        self.fc_log_var = nn.Linear(self.flat_dim + self.n_classes, self.z_dim)
        self.decoder = self._build_decoder()
        return

    @override
    def decode(self, z: torch.Tensor, condition: torch.Tensor, /) -> torch.Tensor:
        return self.decoder(torch.cat([z, condition], dim=1))

    @override
    def encode(self, image: torch.Tensor, condition: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = torch.cat([self.encoder(image), condition], dim=1)
        return self.fc_mu(encoding), self.fc_log_var(encoding)

    @override
    @torch.inference_mode()
    def generate_counterfactual(self, inputs: Inputs, target_condition: torch.Tensor, /) -> CounterfactualOutputs:
        condition = self.get_condition(inputs)
        mu, log_var = self.encode(inputs.image, condition)
        z = self.sample_latent(mu, log_var)
        return CounterfactualOutputs(
            counterfactual=self.decode(z, target_condition), recon=self.decode(z, condition), mu=mu, log_var=log_var
        )

    @override
    @torch.inference_mode()
    def generate_sample(self, batch_size: int, /, *, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            condition = self.get_random_condition(batch_size)

        z = torch.randn(batch_size, self.z_dim, device=condition.device)
        return self.decode(z, condition)

    def _build_decoder(self) -> nn.Sequential:
        modules: list[nn.Module] = [
            nn.Linear(self.z_dim + self.n_classes, self.flat_dim),
            nn.Unflatten(1, (self.h_dims[-1], self.final_res, self.final_res)),
        ]
        for (h_in, h_out), k_dim, stride, padding, depth in zip(
            itertools.pairwise(reversed(self.h_dims)),
            reversed(self.k_dims),
            reversed(self.strides),
            reversed(self.paddings),
            reversed(self.depths),
            strict=True,
        ):
            modules.append(
                ResBlock(
                    ConvBlock,
                    h_in,
                    h_out,
                    k_dim,
                    stride,
                    padding,
                    deconv=True,
                    depth=depth,
                    act_cls=self.act_cls,
                    norm_cls=self.norm_cls,
                )
            )

        modules.extend([ConvBlock(self.h_dims[0], self.n_channels, use_trunc_init=True), nn.Sigmoid()])
        return nn.Sequential(*modules)

    def _build_encoder(self) -> nn.Sequential:
        modules: list[nn.Module] = [nn.Conv2d(self.n_channels, self.h_dims[0], 1, bias=False)]
        for (h_in, h_out), k_dim, stride, padding, depth in zip(
            itertools.pairwise(self.h_dims), self.k_dims, self.strides, self.paddings, self.depths, strict=True
        ):
            modules.append(
                ResBlock(
                    ConvBlock,
                    h_in,
                    h_out,
                    k_dim,
                    stride,
                    padding,
                    deconv=False,
                    depth=depth,
                    act_cls=self.act_cls,
                    norm_cls=self.norm_cls,
                )
            )

        modules.append(nn.Flatten())
        return nn.Sequential(*modules)

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_final_resolution(self) -> int:
        resolution = self.x_dim
        for k_dim, stride, padding in zip(self.k_dims, self.strides, self.paddings, strict=True):
            resolution = (resolution + 2 * padding - k_dim) // stride + 1

        return resolution


class CVAE(ConditionalVAE):
    """Conditional VAE, conditioned on the one-hot label."""

    @override
    def get_condition(self, inputs: Inputs, /) -> torch.Tensor:
        condition = inputs.condition
        if condition.dim() == 1:
            return nn.functional.one_hot(condition.long(), num_classes=self.n_classes).float()

        return condition

    @override
    def get_random_condition(self, batch_size: int, /) -> torch.Tensor:
        random_classes = torch.randint(0, self.n_classes, (batch_size,), device=self._get_device())
        return nn.functional.one_hot(random_classes, num_classes=self.n_classes).float()


class VAEX(ConditionalVAE):
    """VAE conditioned on the probabilities of an external classifier.

    Attributes:
        temperature: Divisor of the log-probabilities of the classifier before the softmax.
    """

    temperature: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.temperature = Experiment.get_config().autoencoding.model.temperature
        return

    @override
    def get_condition(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the softmax of the log-probabilities of the classifier, divided by the temperature.

        Args:
            inputs: the images, with the log-probabilities of the classifier.

        Returns:
            The probability vectors over the classes.

        Raises:
            ValueError: if the inputs have no log-probabilities.
        """
        if inputs.log_probs.numel() == 0:
            raise ValueError('VAEX needs the log-probabilities of the classifier in the inputs.')

        return torch.softmax(inputs.log_probs / self.temperature, dim=1)

    @override
    def get_random_condition(self, batch_size: int, /) -> torch.Tensor:
        """Get conditions drawn uniformly from the probability simplex.

        Args:
            batch_size: the number of conditions.

        Returns:
            The probability vectors over the classes.
        """
        concentration = torch.ones(self.n_classes, device=self._get_device())
        return torch.distributions.Dirichlet(concentration=concentration).sample((batch_size,))


def get_autoencoder() -> BaseVAE:
    """Get the VAE that the configuration selects.

    Returns:
        The CVAE or VAEX.
    """
    if Experiment.get_config().autoencoding.model.name == Models.CVAE:
        return CVAE()

    return VAEX()
