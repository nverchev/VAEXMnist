"""Generate counterfactuals of misclassified samples towards their labels and render them.

Usage::

    uv run generate_misclassified_counterfactuals.py
    uv run generate_misclassified_counterfactuals.py user.generate.target_value=0.5
"""

import logging

import torch

from drytorch import DataLoader
from drytorch.lib import loading
from drytorch.utils.apply_ops import apply_to

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.lib.events import ImageEvent
from src.data import Inputs, Partitions, get_split
from src.data.split import DataSplit, Subset
from src.lib.models import ClassifierModel, get_autoencoder_model, get_classifier_model, load_trained_state


@torch.inference_mode()
def generate_misclassified_counterfactuals() -> None:
    """Turn misclassified samples towards their labels, and publish each with its original and its reconstruction."""
    cfg = Experiment.get_config()
    cfg_generate = cfg.user.generate
    partition = Partitions.test if cfg.final else Partitions.val
    split = get_split(partition, data=cfg.autoencoding.data)
    vae_model = get_autoencoder_model()
    load_trained_state(vae_model, epoch=cfg.user.load or -1)
    classifier = get_classifier_model()
    load_trained_state(classifier)
    misclassified = Subset(split, _get_misclassified(classifier, split))
    subset_indices, (samples, targets) = loading.take_from_dataset(
        misclassified, n_samples=cfg_generate.batch_size, preserve_order=False, device=vae_model.device
    )
    sample_indices = [misclassified.indices[index] for index in subset_indices]
    inputs = Inputs(image=samples.image, condition=targets.label.float())
    inputs = inputs._replace(log_probs=classifier.log_probabilities(inputs))
    logging.info('Generating counterfactuals of %d misclassified samples...', len(sample_indices))
    target_condition = vae_model.module.get_target_condition(targets.label, value=cfg_generate.target_value)
    outputs = vae_model.generate_counterfactual(inputs, target_condition)
    original_captions = classifier.describe_predictions(inputs.image)
    recon_captions = classifier.describe_predictions(outputs.recon)
    counterfactual_captions = classifier.describe_predictions(outputs.counterfactual)
    for i, index in enumerate(sample_indices):
        folder = f'{partition.name}/sample_{index}'
        label = int(targets.label[i])
        title = f'target_{label}'
        if cfg_generate.target_value != 1:
            title += f'_value_{cfg_generate.target_value:g}'

        original_caption = f'{cfg.data.dataset.classes[label]}, {original_captions[i]}'
        ImageEvent(image=inputs.image[i], title='original', caption=original_caption, folder=folder)
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
        generate_misclassified_counterfactuals()

    return


def _get_misclassified(classifier: ClassifierModel, split: DataSplit) -> list[int]:
    batch_size = Experiment.get_config().classification.train.batch_size
    misclassified: list[torch.Tensor] = []
    for inputs, targets in DataLoader(dataset=split, batch_size=batch_size, pin_memory=False):
        inputs, targets = apply_to((inputs, targets), classifier.device)
        misclassified.append(classifier.log_probabilities(inputs).argmax(dim=1) != targets.label)

    return torch.cat(misclassified).nonzero().squeeze(1).tolist()


if __name__ == '__main__':
    main()
