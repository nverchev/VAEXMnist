"""Evaluation metrics and loss specifications."""

from __future__ import annotations

import math

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from drytorch.core import protocols as p
from drytorch.lib.objectives import DictLoss, Loss, LossWithMetrics, Metric, join
from src.config.experiment import Experiment
from src.data.mnist import MnistDataBuilder
from src.data.structures import Outputs, Targets


class NLLLoss:
    """Callable class for Gaussian negative log-likelihood reconstruction loss component."""

    def __init__(self, variance: float = 0.1) -> None:
        self.adj_nll: float = 0.5 * math.log(2 * math.pi)
        self.var: float = variance
        self.logvar: float = math.log(self.var)
        return

    def __call__(self, outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        ref_image = targets.ref
        SE = F.mse_loss(outputs.recon, ref_image, reduction='none')
        NLL = 0.5 * (SE / self.var + self.logvar).mean(0).sum() + self.adj_nll * SE[0].numel()
        MSE = SE.mean()
        return {'NLL': NLL, 'MSE': MSE}


class KLDLoss:
    """Callable class for KL divergence loss."""

    def __init__(self) -> None:
        return

    def __call__(self, outputs: Outputs, _: Targets) -> torch.Tensor:
        mu = outputs.mu
        log_var = outputs.log_var
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_mean = kld.mean()
        return kld_mean


def get_accuracy() -> Metric[torch.Tensor, Targets]:
    """Return top-1 per-sample accuracy metric."""
    return Metric(_predicts_label, name='Accuracy')


def get_cross_entropy() -> Loss[torch.Tensor, Targets]:
    """Return cross-entropy loss objective."""
    ce = nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return ce(outputs, targets.label)

    return Loss(_cross_entropy, name='CrossEntropy')


def get_nll_loss() -> DictLoss[Outputs, Targets]:
    """Get Gaussian negative log-likelihood reconstruction loss component with MSE metric."""
    cfg = Experiment.get_config()
    objective_cfg = cfg.autoencoding.objective
    return DictLoss(NLLLoss(variance=objective_cfg.sigma_2), name='NLL')


def get_kld_loss() -> Loss[Outputs, Targets]:
    """Get KL divergence loss component."""
    return Loss(KLDLoss(), name='KLD')


def get_annealing(model: p.ModelProtocol[Any, Any], /) -> Loss[Outputs, Targets]:
    """Get the cyclical annealing weight on the KL term.

    Args:
        model: the model whose epoch sets the weight.
    """
    cfg = Experiment.get_config()
    annealing_period = cfg.autoencoding.train.annealing_period
    cycles = cfg.autoencoding.train.annealing_cycles

    min_weight = 0.01

    def _annealing(outputs: Outputs, _: Targets) -> torch.Tensor:
        device = outputs.recon.device
        progress = torch.tensor(model.epoch / annealing_period, device=device)
        if progress >= 1.0:
            return torch.ones((), device=device)

        phase = torch.remainder(progress * cycles, 1.0)
        weight = 0.5 * (1.0 - torch.cos(phase * math.pi))
        return torch.clamp(weight, min=min_weight)

    return Loss(_annealing, name='Annealing')


def get_classifier_loss() -> p.LossProtocol[torch.Tensor, Targets]:
    """Get main loss objective based on active experiment configuration."""
    return join(get_cross_entropy(), get_accuracy())


def get_change() -> Metric[torch.Tensor, Targets]:
    """Get the per-sample mean squared difference between the images and their reference, the size of a change."""

    def _change(images: torch.Tensor, targets: Targets) -> torch.Tensor:
        return (images - targets.ref).pow(2).flatten(1).mean(dim=1)

    return Metric(_change, name='Change')


def get_counterfactual_success() -> Metric[torch.Tensor, Targets]:
    """Get the per-sample success of the counterfactuals: the classifier predicts their target class."""
    return Metric(_predicts_label, name='Success')


def get_group_accuracy() -> Metric[torch.Tensor, Targets]:
    """Get the per-sample accuracy of the predicted group of digits, thin or thick."""
    thin_classes = torch.tensor(Experiment.get_config().data.dataset.bias.thin_classes)

    def _group_accuracy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        thin = thin_classes.to(outputs.device)
        return (torch.isin(outputs.argmax(dim=-1), thin) == torch.isin(targets.label, thin)).float()

    return Metric(_group_accuracy, name='GroupAccuracy')


def get_shortcut_loss() -> p.LossProtocol[torch.Tensor, Targets]:
    """Get the classifier loss with the accuracy and the accuracy of the group of digits."""
    return join(get_cross_entropy(), get_accuracy(), get_group_accuracy())


def get_width_change() -> Metric[torch.Tensor, Targets]:
    """Get the per-sample change of stroke width from the reference images to the measured ones."""
    ink_threshold = Experiment.get_config().data.dataset.bias.ink_threshold

    def _width_change(widths: torch.Tensor, targets: Targets) -> torch.Tensor:
        return widths - MnistDataBuilder.estimate_stroke_thickness(targets.ref, ink_threshold)

    return Metric(_width_change, name='WidthChange')


def get_vae_loss(model: p.ModelProtocol[Any, Any], /) -> LossWithMetrics[Outputs, Targets]:
    """Get VAE loss composed with drytorch natural syntax.

    Args:
        model: the model the loss trains, whose epoch sets the annealing.
    """
    cfg = Experiment.get_config()
    nll = get_nll_loss()
    kld = get_kld_loss()
    if cfg.autoencoding.train.annealing_period > 0:
        loss = nll + kld * get_annealing(model)
    else:
        loss = nll + kld

    return loss


def _predicts_label(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
    return (outputs.argmax(dim=-1) == targets.label).float()
