"""Hyperparameter tuning entry point for the VAE using Optuna.

Usage::

    uv run tune_vae.py
    uv run tune_vae.py tune=architecture
"""

import pathlib
import warnings

from collections.abc import Callable

import hydra
import optuna
import torch

from omegaconf import DictConfig

from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError, ModelNotFoundError
from drytorch.core.registering import register_model, unregister_model

from src.config import ConfigPath, Experiment, ExperimentConfig, get_experiment_config
from src.config.options import Models
from src.trackers import set_tuning_logging
from src.lib.models import ClassifierModel
from src.utils.tuning import get_study_name, impute_failed_trial, log_study_settings

from train_vae import load_classifier, train_vae


def set_objective(tune_cfg: DictConfig) -> Callable[[optuna.Trial], float]:
    """Set the objective function following the study configuration.

    For VAEX, the trials read the classifier trained with the fixed overrides, which is
    loaded once and registered to each trial in turn.
    """
    cfg = get_experiment_config(tune_cfg.config_name, overrides=list(tune_cfg.overrides))
    classifier = _load_classifier(cfg) if cfg.autoencoding.model.name == Models.VAEX else None

    def objective(trial: optuna.Trial) -> float:
        """Set up the experiment and launch the training cycle."""
        overrides = suggest_overrides(tune_cfg, trial)
        seed = 'null' if tune_cfg.seed is None else tune_cfg.seed
        overrides += [f'autoencoding.train.n_epochs={tune_cfg.n_epochs}', f'user.seed={seed}']
        if tune_cfg.mixed_precision is not None:
            overrides.append(f'autoencoding.train.mixed_precision={tune_cfg.mixed_precision}')

        trial_cfg = get_experiment_config(tune_cfg.config_name, overrides=overrides)
        exp = Experiment(trial_cfg, name=trial_cfg.name, par_dir=trial_cfg.par_dir, tags=trial_cfg.tags)
        with exp.create_run(record=False):
            if classifier is not None:
                register_model(classifier)

            try:
                train_vae(classifier, trial=trial)
            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                if classifier is not None:
                    unregister_model(classifier)

                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        return get_final_value(trial)

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNING.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig) -> None:
    """Set up the study and launch optimization."""
    set_tuning_logging()
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    if tune_cfg.tune.use_pruner:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=tune_cfg.tune.min_resource,
            max_resource=tune_cfg.tune.max_resource,
            reduction_factor=tune_cfg.tune.reduction_factor,
        )

    else:
        pruner = optuna.pruners.NopPruner()

    with warnings.catch_warnings():
        # multivariate TPE and the constant liar are stable in practice; Optuna still labels them experimental
        warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)

    study_name = get_study_name(tune_cfg.config_name, tune_cfg.tune.study_name, tune_cfg.overrides)
    study = optuna.create_study(
        study_name=study_name, storage=tune_cfg.storage, sampler=sampler, pruner=pruner, load_if_exists=True
    )
    log_study_settings(study, tune_cfg)
    # the trials of the study, from every worker and every run, stop at n_trials
    max_trials = optuna.study.MaxTrialsCallback(tune_cfg.tune.n_trials, states=None)
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials, callbacks=[max_trials])
    return


def _load_classifier(cfg: ExperimentConfig) -> ClassifierModel:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=False):
        try:
            classifier = load_classifier()
        except ModelNotFoundError as error:
            msg = f'No trained classifier for the experiment {cfg.name}: train it first with the same overrides.'
            raise RuntimeError(msg) from error

        unregister_model(classifier)

    return classifier


if __name__ == '__main__':
    main()
