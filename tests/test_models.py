"""Tests for the model wrappers."""

import pathlib

import torch

from torch import nn

from src.config import Experiment, get_experiment_config
from src.data.structures import Inputs, Outputs
from src.lib.models import VAEModel


class _Reconstructing(nn.Module):
    def forward(self, inputs: Inputs) -> Outputs:
        empty = torch.empty(0)
        return Outputs(recon=inputs.image.half(), mu=empty, log_var=empty)


def test_the_reconstructions_are_in_full_precision_from_the_module_in_evaluation_mode(tmp_path: pathlib.Path) -> None:
    """The module runs in evaluation mode and the reconstructions come back as float32."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    with exp.create_run(record=False):
        model = VAEModel(_Reconstructing(), name='vae', device=torch.device('cpu'))
        outputs = model.reconstruct(Inputs(image=torch.zeros(1, 3)))

    assert not model.module.training
    assert outputs.recon.dtype == torch.float32
