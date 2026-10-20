"""Typed I/O structures for the model, loss and metrics."""

import dataclasses
from typing import NamedTuple

import torch


class Inputs(NamedTuple):
    """Input batch containing image tensor, condition tensor, and classifier log-probabilities."""

    image: torch.Tensor
    condition: torch.Tensor = torch.empty(0)
    log_probs: torch.Tensor = torch.empty(0)

    @property
    def features(self) -> torch.Tensor:
        """Alias for image compatibility with generic linear layers."""
        return self.image


class Targets(NamedTuple):
    """Ground-truth target labels and reference image for loss functions and metrics."""

    label: torch.Tensor
    ref: torch.Tensor = torch.empty(0)


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs of the VAE.

    Attributes:
        recon: Reconstructed (or counterfactual) images.
        mu: Mean of the posterior of the latent.
        log_var: Log-variance of the posterior of the latent.
    """

    recon: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor


@dataclasses.dataclass(slots=True)
class CounterfactualOutputs:
    """Counterfactuals of the VAE, with the reconstructions decoded from the same latent samples.

    Attributes:
        counterfactual: Images decoded with the target condition.
        recon: Images decoded with their own condition, from the same latent samples.
        mu: Mean of the posterior of the latent.
        log_var: Log-variance of the posterior of the latent.
    """

    counterfactual: torch.Tensor
    recon: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
