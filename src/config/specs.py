"""Specification for the configuration files.

Root config hierarchy::

    ExperimentConfig
    ├── classification: ClassificationConfig
    │   ├── name
    │   └── train: TrainingConfig
    ├── autoencoding: AutoencodingConfig
    │   ├── name
    │   ├── train: AutoencoderTrainingConfig
    │   ├── model: ModelConfig
    │   │   └── architecture: VAEArchitectureConfig
    │   └── objective: ObjectiveConfig
    ├── user: UserSettings
    │   ├── trackers: TrackerList
    │   ├── path: PathSpecs          ← populated from .env
    │   └── hydra: HydraSettings     ← populated at runtime
    └── data: DataConfig
        └── dataset: DatasetConfig

Each phase, classification and autoencoding, has its own PhaseConfig subclass.
"""

import dataclasses
import pathlib
from functools import cached_property

from typing import Annotated, Any, Literal, Self

import torch

from pydantic import BeforeValidator, Field, PlainValidator, model_validator
from pydantic.dataclasses import dataclass

from src.config.environment import EnvSettings, PROJECT_NAME, VERSION, get_current_branch
from src.config.torch import (
    ActClass,
    NormClass,
    get_activation_cls,
    get_device,
    get_mixed_precision,
    get_norm_cls,
    get_optim_cls,
    set_seed,
)
from src.config.options import (
    GradOp,
    ClipCriterion,
    Schedulers,
    Models,
    Datasets,
)


ActivationClass = Annotated[ActClass, BeforeValidator(get_activation_cls)]
Device = Annotated[torch.device | Literal['auto'], PlainValidator(get_device)]
MixedPrecision = Annotated[torch.dtype | None, PlainValidator(get_mixed_precision)]
NormalizationClass = Annotated[NormClass, BeforeValidator(get_norm_cls)]
OptimizerClass = Annotated[type[torch.optim.Optimizer], BeforeValidator(get_optim_cls)]
PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


@dataclass(frozen=True, kw_only=True)
class BaseArchitectureConfig:
    """Base specification for an architecture with activation and normalisation.

    Attributes:
        norm_cls: Class of the normalization, written by its name.
        act_cls: Class of the activation, written by its name.
    """

    norm_cls: NormalizationClass
    act_cls: ActivationClass


@dataclass(frozen=True, kw_only=True)
class VAEArchitectureConfig(BaseArchitectureConfig):
    """Convolutional VAE architecture parameters.

    Attributes:
        norm_cls: Class of the normalization, written by its name.
        act_cls: Class of the activation, written by its name.
        h_dims: Number of channels at each resolution.
        k_dims: Kernel sizes of the resolution changes.
        strides: Strides of the resolution changes.
        paddings: Paddings of the resolution changes.
        depths: Residual layers in each resolution change.
    """

    h_dims: tuple[StrictlyPositiveInt, ...]
    k_dims: tuple[StrictlyPositiveInt, ...]
    strides: tuple[StrictlyPositiveInt, ...]
    paddings: tuple[PositiveInt, ...]
    depths: tuple[PositiveInt, ...]


@dataclass(frozen=True, kw_only=True)
class ObjectiveConfig:
    """Specification for Autoencoder / VAE loss objective and metrics parameters.

    Attributes:
        sigma_2: Variance of the Gaussian NLL reconstruction loss.
        sampling_variance: Multiplier on the sampling standard deviation at inference.
    """

    sigma_2: float
    sampling_variance: float


@dataclass(frozen=True)
class ModelConfig:
    """Model class selector and non-architectural settings.

    Attributes:
        name: Model architecture variant to instantiate.
        architecture: Convolutional architecture of the VAE.
        z_dim: Dimension of the latent space.
        temperature: Divisor of the log-probabilities of the classifier before the softmax (VAEX).
    """

    name: Models
    architecture: VAEArchitectureConfig
    z_dim: StrictlyPositiveInt
    temperature: float


@dataclass(frozen=True, kw_only=True)
class BiasConfig:
    """Specification of the stroke-thickness bias injected into MNIST.

    Samples are ranked by estimated stroke width within their own class, so the
    thresholds do not inherit the natural ink differences between digits. Thin
    strokes are the expected appearance of ``thin_classes`` and thick strokes
    that of every other class; samples contradicting that rule are dropped. A
    phase can drop the same share of each class at random instead (see
    :class:`PhaseDataConfig`), which gives an unbiased dataset of the same size.

    Attributes:
        thin_classes: Classes whose thin samples are kept and thick ones removed.
        quantile: Fraction of each class the bias keeps, as a quantile of the
            stroke width. ``0.5`` removes the disagreeing half and ``1.0`` keeps
            the standard dataset.
        ink_threshold: Intensity above which a pixel counts as ink.
    """

    thin_classes: tuple[PositiveInt, ...]
    quantile: float
    ink_threshold: float

    @model_validator(mode='after')
    def _check_quantile(self) -> Self:
        if not 0.0 <= self.quantile <= 1.0:
            msg = 'The quantile must satisfy 0 <= quantile <= 1.'
            raise ValueError(msg)

        return self


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset identity and paths.

    Attributes:
        name: Name of the dataset to use.
        classes: The names of the classes/unique values expected for the target attribute.
        n_train_samples: Expected number of training samples.
        n_val_samples: Expected number of validation samples.
        n_test_samples: Expected number of test samples.
        bias: Stroke-thickness bias applied by the biased MNIST builder.
    """

    name: Datasets
    classes: list[str]
    n_train_samples: StrictlyPositiveInt
    n_val_samples: StrictlyPositiveInt
    n_test_samples: StrictlyPositiveInt
    bias: BiasConfig


@dataclass(frozen=True)
class DataConfig:
    """Full data configuration.

    Attributes:
        dataset: The dataset-specific configuration and splits.
        x_dim: Spatial dimension of the input images. Must be divisible by
            ``2 ** len(strides)`` so the encoder and decoder resolutions match.
        crop_size: Center crop size before resizing to x_dim.
    """

    dataset: DatasetConfig
    x_dim: StrictlyPositiveInt
    crop_size: StrictlyPositiveInt
    n_channels: StrictlyPositiveInt


@dataclass(frozen=True)
class SchedulerConfig:
    """Specification for the learning-rate scheduler.

    Attributes:
        function: Which scheduler to use (see :class:`~src.config.options.Schedulers`).
        restart_interval: Epochs between cosine restarts (0 = no restart).
        restart_fraction: Fraction of base LR applied at each restart.
        warmup_steps: Linear warm-up epochs at the start of training.
        settings: Additional keyword arguments forwarded to the scheduler
            constructor (e.g. ``decay_steps``, ``min_decay``).
    """

    function: Schedulers
    restart_interval: PositiveInt
    restart_fraction: PositiveFloat
    warmup_steps: PositiveInt
    settings: dict[str, Any]


@dataclass(frozen=True)
class LearningConfig:
    """Specification for the optimiser and learning schedule.

    Attributes:
        optimizer: PyTorch optimizer, written by its name.
        learning_rate: Global learning rate (or per-parameter group base LR).
        scheduler: Learning-rate scheduler configuration.
        grad_op: Optional gradient pre-processing operation applied before
            each optimiser step (see :class:`~src.config.options.GradOp`).
        clip_criterion: Criterion for histogram-based clippers.
        opt_settings: Extra keyword arguments forwarded to the optimiser
            constructor (e.g. ``weight_decay``).
    """

    optimizer: OptimizerClass
    learning_rate: PositiveFloat
    scheduler: SchedulerConfig
    grad_op: GradOp | None
    clip_criterion: ClipCriterion | None
    opt_settings: dict[str, Any]

    @model_validator(mode='after')
    def _check_clip_criterion(self) -> Self:
        if self.clip_criterion is None and self.grad_op in (GradOp.HIST_CLIPPER, GradOp.PARAM_HIST_CLIPPER):
            msg = 'Must specify clip_criterion when using HistClipper or ParamHistClipper.'
            raise ValueError(msg)

        return self


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Specification for early stopping.

    Attributes:
        active: Whether to enable early stopping.
        window: Number of recent validation metrics to average.
        patience: Epochs to wait for improvement before stopping.
    """

    active: bool
    window: int
    patience: int


@dataclass(frozen=True)
class TrainingConfig:
    """Specification for the training loop.

    Attributes:
        batch_size: Total batch size across all devices.
        learn: Optimiser and scheduler configuration.
        n_epochs: Maximum number of training epochs.
        early_stopping: Early stopping configuration.
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig
    _n_subprocesses: PositiveInt

    @model_validator(mode='after')
    def _check_batch_divisibility(self) -> Self:
        if self._n_subprocesses and self.batch_size % self._n_subprocesses != 0:
            msg = 'Global batch size {} not divisible by number of devices {}.'
            raise ValueError(msg.format(self.batch_size, self._n_subprocesses))

        return self

    @property
    def batch_size_per_device(self) -> int:
        """Batch size assigned to each individual device."""
        if self._n_subprocesses == 0:
            return self.batch_size

        return self.batch_size // self._n_subprocesses


@dataclass(frozen=True, kw_only=True)
class AutoencoderTrainingConfig(TrainingConfig):
    """Specification for the training loop of the VAE.

    Attributes:
        batch_size: Total batch size across all devices.
        learn: Optimiser and scheduler configuration.
        n_epochs: Maximum number of training epochs.
        early_stopping: Early stopping configuration.
        annealing_period: Epochs over which the KL weight is annealed. Past it the weight
            stays at one.
        annealing_cycles: Number of times the KL weight ramps back to one within that
            period, so that reconstruction pressure returns periodically instead of once.
        mixed_precision: Autocast precision, ``float16`` (with gradient scaling) or ``bfloat16``; ``None`` for none.
    """

    annealing_period: PositiveInt
    annealing_cycles: StrictlyPositiveInt
    mixed_precision: MixedPrecision


@dataclass(frozen=True)
class PhaseDataConfig:
    """Data a phase trains and is validated on.

    Attributes:
        random: Whether to drop the same share of each class at random (``half_mnist``) instead of the samples
            whose stroke width disagrees with their class group (``biased_mnist``).
    """

    random: bool


@dataclass(frozen=True)
class PhaseConfig:
    """Specification for a single phase.

    Attributes:
        train: Training loop configuration.
        data: Data the phase trains and is validated on.
    """

    train: TrainingConfig
    data: PhaseDataConfig


@dataclass(frozen=True)
class ClassifierModelConfig:
    """Classifier model.

    Attributes:
        name: Classifier architecture.
    """

    name: str


@dataclass(frozen=True)
class ClassificationConfig(PhaseConfig):
    """Specification for the classifier.

    Attributes:
        model: Classifier model.
    """

    model: ClassifierModelConfig


@dataclass(frozen=True)
class AutoencodingConfig(PhaseConfig):
    """Specification for the VAE phase.

    Attributes:
        train: Training loop configuration, with the annealing and the mixed precision of the VAE.
        data: Data the phase trains and is validated on.
        model: VAE model configuration.
        objective: VAE loss objective and metrics configuration.
    """

    train: AutoencoderTrainingConfig
    model: ModelConfig
    objective: ObjectiveConfig


@dataclass(frozen=True)
class TrackerList:
    """Flags controlling which experiment trackers are active.

    Attributes:
        hydra: Saves a copy of the resolved config to the Hydra output dir.
        csv: Appends metrics to a CSV file in the experiment directory.
        tensorboard: Writes the metrics and the images the scripts publish to TensorBoard logs.
        wandb: Uploads the metrics and the images the scripts publish to Weights & Biases (requires ``track_all``
            extra).
        sqlalchemy: Persists metrics in a SQLite database (requires ``track_all`` extra).
        matplotlib: Saves the image grids the scripts publish, and a PNG figure of each metric after training.
        plotly: Saves an HTML figure of each metric after training (requires plotly).
    """

    hydra: bool
    csv: bool
    tensorboard: bool
    wandb: bool
    sqlalchemy: bool
    matplotlib: bool
    plotly: bool


@dataclass(frozen=True)
class PlottingSettings:
    """Options for plotting and sample visualization.

    Attributes:
        sample_indices: Indices of samples in the dataset to plot.
        use_train: Whether to draw samples from the training set instead of the test set.
        show: Whether to also display the figures: the image grids and the learning curves.
    """

    sample_indices: list[int]
    use_train: bool
    show: bool


@dataclass(frozen=True)
class GenerateOptions:
    """Options for sample generation.

    Attributes:
        batch_size: Number of samples generated in a batch, and of random samples the visualizations show
            when no sample indices are given.
        target_class: Class the counterfactual grid turns the samples towards.
        target_value: Probability of the target class in the condition of the counterfactuals; the other classes
            share the rest equally, so ``1`` is the target class alone and ``1 / n_classes`` the uniform condition.
    """

    batch_size: PositiveInt
    target_class: PositiveInt
    target_value: Probability


@dataclasses.dataclass(frozen=True)
class HydraSettings:
    """Subset of Hydra runtime settings, populated automatically at startup.

    Attributes:
        output_dir: Directory where Hydra stores logs and config dumps.
        job_logging: Hydra job-logging DictConfig used to configure Python logging.
        config_name: Name of the file in ``configs/experiment`` the configuration starts from.
    """

    output_dir: pathlib.Path = pathlib.Path('outputs')
    job_logging: Any = None
    config_name: str = ''


@dataclass(frozen=True)
class PathSpecs:
    """Filesystem path specifications populated from environment variables.

    Attributes:
        data_dir: Root directory holding the datasets, shared by all versions.
        root_exp_dir: Root directory under which all experiment artefacts are stored.

    Note:
        Paths are loaded from the ``.env`` file via :class:`~src.config.environment.EnvSettings`.
    """

    _env = EnvSettings()
    data_dir: pathlib.Path = _env.data_dir
    root_exp_dir: pathlib.Path = _env.root_exp_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """Version-scoped experiment directory: ``<root_exp_dir>/v<VERSION>``."""
        return self.root_exp_dir / f'v{VERSION}'


@dataclass(frozen=True)
class UserSettings:
    """User-specific runtime preferences.

    Attributes:
        device: Device where the models run: ``auto`` for the accelerator when available, or a name such as
            ``cpu`` or ``cuda:1``.
        n_workers: DataLoader worker processes. ``0`` runs loading in the main process.
        n_subprocesses: Number of DDP processes. ``0`` disables distributed training.
        trackers: Active experiment-tracking backends.
        seed: Global random seed for reproducibility. ``None`` leaves it unset.
        checkpoint_every: Save a model checkpoint every *n* epochs; ``0`` saves only at the end of training.
        load: Checkpoint epoch to resume from:``0`` skip; ``-1`` loads last checkpoint.
        run_id: Run to resume; ``None`` resumes the last run of the experiment.
        plot: Options of the figures.
        generate: Options of the random samples.
        hydra: Hydra runtime settings.
        path: Paths of the data and the experiments.
    """

    device: Device
    n_workers: PositiveInt
    n_subprocesses: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    load: int
    run_id: str | None
    plot: PlottingSettings
    generate: GenerateOptions
    hydra: HydraSettings = dataclasses.field(default_factory=HydraSettings)
    path: PathSpecs = dataclasses.field(default_factory=PathSpecs)

    def __post_init__(self) -> None:
        """Apply the random seed immediately after the config is constructed."""
        if self.seed is not None:
            set_seed(self.seed)

        return


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object for all experiment settings.

    Attributes:
        classification: Classification phase configuration.
        autoencoding: Autoencoding phase configuration.
        user: User-specific options and runtime preferences.
        data: Data pre-processing and dataset configuration.
        final: If True, uses validation data for training and test data for evaluation.
        variation: Variation name for the experiment (defaults to git branch if None).
        tags: List of tags describing active configuration overrides.
    """

    classification: ClassificationConfig
    autoencoding: AutoencodingConfig
    user: UserSettings
    data: DataConfig
    final: bool
    variation: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = dataclasses.field(default=f'v{VERSION}', init=False)

    @cached_property
    def git_branch(self) -> str:
        """Current git branch (used when variation is unset)."""
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Set 'variation' in the configuration file."
            raise ValueError(msg) from ve

    @property
    def resolved_variation(self) -> str:
        """variation if set, otherwise the current git branch."""
        return self.variation or self.git_branch

    @property
    def name(self) -> str:
        """Full experiment name embedded into tracker runs."""
        suffix = '_final' if self.final else ''
        return self.resolved_variation[:249] + suffix

    @property
    def par_dir(self) -> pathlib.Path:
        """Directory of the experiments of this configuration: ``<version_dir>/<config_name>``."""
        return self.user.path.version_dir / self.user.hydra.config_name

    @property
    def project(self) -> str:
        """Tracker project name, one per version."""
        return f'{PROJECT_NAME}-{self.version}'
