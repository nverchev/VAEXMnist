"""Evaluate the classifier on the shortcut, the VAE, and the counterfactuals of the VAE.

The evaluation follows section 5 of ``docs/biased_mnist.md``: the evaluation partition is
split into the images whose stroke width agrees with their class group (aligned) and the
others (conflicting). Run it on the test partition, which the bias leaves whole; without
``final``, it uses the validation partition of the data the VAE trains on.

Usage::

    uv run evaluate_counterfactuals.py final=true
    uv run evaluate_counterfactuals.py --config-name biased final=true
    uv run evaluate_counterfactuals.py --config-name biased_classifier final=true
    uv run evaluate_counterfactuals.py --config-name biased_vae final=true
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from drytorch import DataLoader, Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.options import Models
from src.data import Partitions, get_databuilder
from src.data.derived_datasets import (
    CounterfactualDataset,
    EvaluatedDataset,
    ReconstructedDataset,
    ReconstructedEvaluatedDataset,
)
from src.data.mnist import MnistDataBuilder
from src.data.split import DataSplit, Subset
from src.data.structures import Inputs, Targets
from src.trackers import get_trackers
from src.lib.models import (
    ClassifierModel,
    VAEModel,
    get_autoencoder_model,
    get_classifier_model,
    get_image_model,
    get_stroke_width_model,
    load_trained_state,
)
from src.lib.objectives import (
    get_change,
    get_counterfactual_success,
    get_shortcut_loss,
    get_vae_loss,
    get_width_change,
)


def evaluate_shortcut(classifier_model: ClassifierModel, split: DataSplit, aligned: Tensor) -> Tensor:
    """Test the classifier on all the images, on the aligned ones and on the conflicting ones.

    Args:
        classifier_model: the trained classifier.
        split: the images of the evaluation partition.
        aligned: whether each image has the stroke width its class group expects.

    Returns:
        The classes the classifier predicts for the images.
    """
    batch_size = Experiment.get_config().classification.train.batch_size
    loader = DataLoader(dataset=split, batch_size=batch_size, pin_memory=False)
    test = Test(classifier_model, name='Classification', loader=loader, metric=get_shortcut_loss())
    test(store_outputs=True)
    for name, mask in (('aligned', aligned), ('conflicting', ~aligned)):
        subset = Subset(split, _get_indices(mask))
        subset_loader = DataLoader(dataset=subset, batch_size=batch_size, pin_memory=False)
        Test(classifier_model, name=f'Classification_{name}', loader=subset_loader, metric=get_shortcut_loss())()

    return torch.cat(list(test.outputs_list)).argmax(dim=1).cpu()


def evaluate_counterfactuals(
    classifier_model: ClassifierModel,
    vae_model: VAEModel,
    split: DataSplit,
    predictions: Tensor,
    labels: Tensor,
) -> None:
    """Test the success of the counterfactuals, how much they change the images, and how much the stroke width.

    Each image is turned towards every class the classifier does not predict for it, and each
    misclassified image also towards its label. Both changes are measured against the reconstruction
    decoded from the same latent sample: the mean squared difference, and the change of stroke width of
    the counterfactuals that move an image to the other group.

    Args:
        classifier_model: the trained classifier.
        vae_model: the trained VAE.
        split: the images of the evaluation partition.
        predictions: the classes the classifier predicts for the images.
        labels: the classes of the images.
    """
    cfg = Experiment.get_config()
    thin_classes = cfg.data.dataset.bias.thin_classes
    class_iter = range(len(cfg.data.dataset.classes))
    pairs = [(image, target) for target in class_iter for image in _get_indices(predictions != target)]
    image_model = get_image_model()
    misclassified_pairs = [(image, int(labels[image])) for image in _get_indices(predictions != labels)]
    for name, test_pairs in (('Counterfactuals', pairs), ('Counterfactuals_misclassified', misclassified_pairs)):
        loader = _get_counterfactual_loader(classifier_model, vae_model, split, test_pairs)
        Test(classifier_model, name=name, loader=loader, metric=get_counterfactual_success())()
        Test(image_model, name=f'{name}_change', loader=loader, metric=get_change())()

    width_model = get_stroke_width_model()
    thin_predicted = torch.isin(predictions, torch.tensor(thin_classes))
    for group, to_thin in (('thin', True), ('thick', False)):
        sources = _get_indices(thin_predicted != to_thin)
        pairs = [(image, target) for target in class_iter if (target in thin_classes) == to_thin for image in sources]
        loader = _get_counterfactual_loader(classifier_model, vae_model, split, pairs)
        Test(width_model, name=f'Width_change_to_{group}_group', loader=loader, metric=get_width_change())()

    return


@torch.inference_mode()
def evaluate() -> None:
    """Evaluate the classifier on the shortcut, the VAE, its reconstructions and its counterfactuals."""
    cfg = Experiment.get_config()
    partition = Partitions.test if cfg.final else Partitions.val
    builder = get_databuilder()
    if not isinstance(builder, MnistDataBuilder):
        raise TypeError('The evaluation needs the MNIST data builder.')

    classifier_model = get_classifier_model()
    load_trained_state(classifier_model)
    vae_model = get_autoencoder_model()
    load_trained_state(vae_model, epoch=cfg.user.load or -1)
    data = cfg.autoencoding.data
    split = builder.get_split(partition, data=data)
    predictions = evaluate_shortcut(classifier_model, split, builder.get_alignment(partition, data=data))
    labels = builder.get_labels(partition, data=data)
    is_vaex = cfg.autoencoding.model.name == Models.VAEX
    vae_split = EvaluatedDataset(split, classifier_model) if is_vaex else split
    vae_loader = DataLoader(dataset=vae_split, batch_size=cfg.autoencoding.train.batch_size, pin_memory=False)
    Test(vae_model, name='Reconstruction', loader=vae_loader, metric=get_vae_loss(vae_model))()
    if is_vaex:
        recon_split = ReconstructedEvaluatedDataset(split, autoencoder=vae_model, classifier=classifier_model)
    else:
        recon_split = ReconstructedDataset(split, autoencoder=vae_model)

    batch_size = cfg.classification.train.batch_size
    recon_loader = DataLoader(dataset=recon_split, batch_size=batch_size, pin_memory=False)
    Test(classifier_model, name='Classification_reconstructions', loader=recon_loader, metric=get_shortcut_loss())()
    evaluate_counterfactuals(classifier_model, vae_model, split, predictions, labels)
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and launch the evaluation."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        evaluate()

    return


def _get_counterfactual_loader(
    classifier_model: ClassifierModel,
    vae_model: VAEModel,
    split: DataSplit,
    pairs: Sequence[tuple[int, int]],
) -> DataLoader[tuple[Inputs, Targets]]:
    images, targets = zip(*pairs, strict=True)
    counterfactuals = CounterfactualDataset(
        Subset(split, list(images)),
        autoencoder=vae_model,
        classifier=classifier_model,
        targets=list(targets),
    )
    batch_size = Experiment.get_config().autoencoding.train.batch_size
    return DataLoader(dataset=counterfactuals, batch_size=batch_size, pin_memory=False)


def _get_indices(mask: Tensor) -> list[int]:
    return mask.nonzero().squeeze(1).tolist()


if __name__ == '__main__':
    main()
