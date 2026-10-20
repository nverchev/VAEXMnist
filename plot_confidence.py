"""Plot the distribution of the probability of the class the classifier predicts.

The probabilities are those VAEX receives as its condition: the softmax of the log-probabilities of the
classifier divided by ``autoencoding.model.temperature``.

Usage::

    uv run plot_confidence.py final=true
    uv run plot_confidence.py --config-name biased final=true user.plot.use_train=true
"""

import logging

import torch

from drytorch import DataLoader
from drytorch.utils.apply_ops import apply_to

from src.config import Experiment, ExperimentConfig, hydra_main
from src.data import Partitions, get_split
from src.trackers import get_trackers
from src.lib.events import ConfidenceEvent
from src.lib.models import get_classifier_model, load_trained_state


@torch.inference_mode()
def plot_confidence() -> None:
    """Publish the probability of the predicted class of each image, and whether the prediction is correct."""
    cfg = Experiment.get_config()
    if cfg.user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    split = get_split(partition, data=cfg.autoencoding.data)
    classifier = get_classifier_model()
    load_trained_state(classifier)
    temperature = cfg.autoencoding.model.temperature
    top_probabilities: list[torch.Tensor] = []
    correct: list[torch.Tensor] = []
    loader = DataLoader(dataset=split, batch_size=cfg.classification.train.batch_size, pin_memory=False)
    for inputs, targets in loader:
        inputs, targets = apply_to((inputs, targets), classifier.device)
        probabilities, predictions = torch.softmax(classifier.log_probabilities(inputs) / temperature, dim=1).max(dim=1)
        top_probabilities.append(probabilities.cpu())
        correct.append((predictions == targets.label).cpu())

    top = torch.cat(top_probabilities)
    is_correct = torch.cat(correct)
    for threshold in (0.9, 0.99, 0.999):
        share = 100 * (top > threshold).float().mean().item()
        logging.info('Top probability above %g: %.1f %% of the predictions', threshold, share)

    title = f'confidence_temperature_{temperature:g}'
    ConfidenceEvent(probabilities=top, correct=is_correct, temperature=temperature, title=title, folder=partition.name)
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and plot the distribution of the probabilities."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        plot_confidence()

    return


if __name__ == '__main__':
    main()
