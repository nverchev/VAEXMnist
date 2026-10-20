"""Tests for the losses and metrics."""

import pathlib

import pytest
import torch

from src.config import Experiment, get_experiment_config
from src.data.structures import Outputs, Targets
from src.lib.objectives import get_annealing, get_classifier_loss, get_counterfactual_success


class _EpochModel:
    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        return


def test_the_classifier_loss_reports_the_accuracy() -> None:
    """The cross entropy is joined with the accuracy, which is reported but not optimized."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = Targets(label=torch.tensor([0, 1, 1]))

    result = get_classifier_loss().calculate(logits, targets)

    assert set(result) == {'CrossEntropy', 'Accuracy'}
    torch.testing.assert_close(result['Accuracy'], torch.tensor([1.0, 1.0, 0.0]))


def test_a_counterfactual_succeeds_when_the_classifier_predicts_its_target() -> None:
    """A counterfactual succeeds only when the top class of the classifier is its target class."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = Targets(label=torch.tensor([1, 1]))

    result = get_counterfactual_success().calculate(logits, targets)

    torch.testing.assert_close(result['Success'], torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize(('epoch', 'expected'), [(0, 0.01), (1_000, 1.0)])
def test_the_annealing_weight_starts_low_and_ends_at_one(tmp_path: pathlib.Path, epoch: int, expected: float) -> None:
    """The weight is floored at the start of the period and reaches one after it."""
    outputs = Outputs(recon=torch.zeros(1), mu=torch.zeros(1), log_var=torch.zeros(1))
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    with exp.create_run(record=False):
        annealing = get_annealing(_EpochModel(epoch))

    weight = annealing.calculate(outputs, Targets(label=torch.zeros(1)))['Annealing']

    assert float(weight) == pytest.approx(expected)
