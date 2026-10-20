"""Visualize hyperparameter optimization results.

Usage::

    uv run plot_tuning_results.py
    uv run plot_tuning_results.py tune=learn
"""

import hydra
import optuna

from omegaconf import DictConfig

from src.config import ConfigPath
from src.utils.tuning import get_study_name, visualize_study


@hydra.main(version_base=None, config_path=ConfigPath.TUNING.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig) -> None:
    """Load Optuna study and show visualization plots."""
    study_name = get_study_name(tune_cfg.config_name, tune_cfg.tune.study_name, tune_cfg.overrides)
    study = optuna.load_study(study_name=study_name, storage=tune_cfg.storage)
    renderer = tune_cfg.get('renderer', 'browser')
    visualize_study(study, renderer=renderer)
    return


if __name__ == '__main__':
    main()
