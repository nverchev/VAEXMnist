"""Models of the experiment."""

import torch

from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib.models import Model

from src.config.experiment import Experiment
from src.data.structures import CounterfactualOutputs, Inputs, Outputs
from src.module.autoencoders import BaseVAE, get_autoencoder
from src.module.classifiers import BaseClassifier, get_classifier
from src.module.image_view import ImageView
from src.module.stroke_width import StrokeWidth

__all__ = [
    'ClassifierModel',
    'VAEModel',
    'get_autoencoder_model',
    'get_classifier_model',
    'get_image_model',
    'get_stroke_width_model',
    'load_trained_state',
]


class ClassifierModel(Model[Inputs, torch.Tensor]):
    """Model wrapping a classifier that maps inputs to logits.

    Attributes:
        module: the classifier.
    """

    module: BaseClassifier

    @torch.inference_mode()
    def log_probabilities(self, inputs: Inputs, /) -> torch.Tensor:
        """Get the log-probabilities of the classes, in the precision of the model.

        Args:
            inputs: the images to classify.

        Returns:
            The log-probabilities, in full precision.
        """
        self.module.eval()
        return torch.log_softmax(self(inputs).float(), dim=-1)

    @torch.inference_mode()
    def describe_predictions(self, images: torch.Tensor, /) -> list[str]:
        """Describe the top class of each image and its probability, as in ``predicted 7 (0.98)``.

        Args:
            images: the images to classify.

        Returns:
            One description per image.
        """
        class_names = Experiment.get_config().data.dataset.classes
        probabilities, predictions = self.log_probabilities(Inputs(image=images)).exp().max(dim=1)
        return [
            f'predicted {class_names[prediction]} ({probability:.2f})'
            for prediction, probability in zip(predictions.tolist(), probabilities.tolist(), strict=True)
        ]


class VAEModel(Model[Inputs, Outputs]):
    """Model wrapping the VAE.

    Attributes:
        module: the VAE.
    """

    module: BaseVAE

    @torch.inference_mode()
    def generate_counterfactual(self, inputs: Inputs, target_condition: torch.Tensor, /) -> CounterfactualOutputs:
        """Generate counterfactuals of the images for their target conditions, in the precision of the model.

        Args:
            inputs: the images, with their conditions.
            target_condition: the probability vector over the classes each image is decoded with.

        Returns:
            The counterfactuals and the reconstructions from the same latent samples, in full precision.
        """
        self.module.eval()
        with self._autocast():
            outputs = self.module.generate_counterfactual(inputs, target_condition)

        outputs.counterfactual = outputs.counterfactual.float()
        outputs.recon = outputs.recon.float()
        return outputs

    @torch.inference_mode()
    def generate_sample(self, batch_size: int, /) -> torch.Tensor:
        """Generate random images from the prior, with random conditions, in the precision of the model.

        Args:
            batch_size: the number of images.

        Returns:
            The images, in full precision.
        """
        self.module.eval()
        with self._autocast():
            images = self.module.generate_sample(batch_size)

        return images.float()

    @torch.inference_mode()
    def reconstruct(self, inputs: Inputs, /) -> Outputs:
        """Reconstruct the images, in the precision of the model.

        Args:
            inputs: the images, with their conditions.

        Returns:
            The outputs, with the reconstructions in full precision.
        """
        self.module.eval()
        with self._autocast():
            outputs = self.module(inputs)

        outputs.recon = outputs.recon.float()
        return outputs

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=self.mixed_precision, enabled=self.mixed_precision is not None)


def get_autoencoder_model() -> VAEModel:
    """Get the VAE of the experiment.

    Returns:
        The VAE, wrapped for training, evaluation and checkpointing.
    """
    cfg = Experiment.get_config()
    vae_cfg = cfg.autoencoding
    return VAEModel(
        get_autoencoder(),
        device=cfg.user.device,
        mixed_precision=vae_cfg.train.mixed_precision,
        name=vae_cfg.model.name,
    )


def get_classifier_model() -> ClassifierModel:
    """Get the classifier of the experiment.

    Returns:
        The classifier, wrapped for training, evaluation and checkpointing.
    """
    cfg = Experiment.get_config()
    return ClassifierModel(get_classifier(), name=cfg.classification.model.name, device=cfg.user.device)


def get_image_model() -> Model[Inputs, torch.Tensor]:
    """Get a model returning the images of its inputs, so that metrics can compare them.

    Returns:
        The model returning the images.
    """
    return Model(ImageView(), name='images', device=Experiment.get_config().user.device)


def get_stroke_width_model() -> Model[Inputs, torch.Tensor]:
    """Get the measurement of the stroke width, wrapped for evaluation.

    Returns:
        The model returning the stroke width of each image.
    """
    return Model(StrokeWidth(), name='stroke_width', device=Experiment.get_config().user.device)


def load_trained_state(model: Model, /, *, epoch: int = -1) -> None:
    """Load the weights of a model trained by an earlier phase of the experiment.

    Args:
        model: the model to load.
        epoch: the epoch to load. Default loads the last saved epoch.

    Raises:
        RuntimeError: if the experiment holds no trained model.
    """
    try:
        model.load_state(epoch)
    except ModelNotFoundError as error:
        cfg = Experiment.get_config()
        msg = (
            f'No trained {model.name} for the experiment {cfg.name} in {cfg.par_dir}: '
            'train it first with the same configuration and overrides.'
        )
        raise RuntimeError(msg) from error

    return
