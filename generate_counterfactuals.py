"""Generate counterfactuals of selected samples towards a target class and render them.

Usage::

    uv run generate_counterfactuals.py
    uv run generate_counterfactuals.py user.generate.target_class=7 user.generate.target_value=0.5
"""

import logging

import torch

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.lib.events import ImageEvent
from src.config.options import Models
from src.data import Inputs, Partitions, get_split
from src.lib.models import get_autoencoder_model, get_classifier_model, load_trained_state


@torch.inference_mode()
def generate_counterfactuals() -> None:
    """Turn selected samples towards the target class, and publish each with its original and its reconstruction."""
    cfg = Experiment.get_config()
    cfg_generate = cfg.user.generate
    partition = Partitions.test if cfg.final else Partitions.val
    split = get_split(partition, data=cfg.autoencoding.data)
    vae_model = get_autoencoder_model()
    load_trained_state(vae_model, epoch=cfg.user.load or -1)
    device = vae_model.device
    if cfg.user.plot.sample_indices:
        sample_indices, (samples, targets) = loading.take_from_dataset(
            split, indices=cfg.user.plot.sample_indices, device=device
        )
    else:
        sample_indices, (samples, targets) = loading.take_from_dataset(
            split, n_samples=cfg_generate.batch_size, preserve_order=False, device=device
        )

    inputs = Inputs(image=samples.image, condition=targets.label.float())
    classifier = get_classifier_model()
    load_trained_state(classifier)
    if cfg.autoencoding.model.name == Models.VAEX:
        inputs = inputs._replace(log_probs=classifier.log_probabilities(inputs))

    logging.info('Generating counterfactuals of %d samples...', len(sample_indices))
    target_classes = torch.full_like(targets.label, cfg_generate.target_class)
    target_condition = vae_model.module.get_target_condition(target_classes, value=cfg_generate.target_value)
    outputs = vae_model.generate_counterfactual(inputs, target_condition)
    title = f'target_{cfg_generate.target_class}'
    if cfg_generate.target_value != 1:
        title += f'_value_{cfg_generate.target_value:g}'

    original_captions = classifier.describe_predictions(inputs.image)
    recon_captions = classifier.describe_predictions(outputs.recon)
    counterfactual_captions = classifier.describe_predictions(outputs.counterfactual)
    for i, index in enumerate(sample_indices):
        folder = f'{partition.name}/sample_{index}'
        label = cfg.data.dataset.classes[int(targets.label[i])]
        ImageEvent(image=inputs.image[i], title='original', caption=f'{label}, {original_captions[i]}', folder=folder)
        ImageEvent(image=outputs.recon[i], title='reconstruction', caption=recon_captions[i], folder=folder)
        ImageEvent(image=outputs.counterfactual[i], title=title, caption=counterfactual_captions[i], folder=folder)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the counterfactual generation."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        generate_counterfactuals()

    return


if __name__ == '__main__':
    main()
