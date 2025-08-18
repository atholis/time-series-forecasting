from typing import Optional

import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.simulation import walk_forward_simulation

# TODO:
# - have simulation interface and inject it to the constructor
# - do I need validation data as well?

TEST_SIZE_IN_MONTHS = 1
TRAIN_SIZE_IN_MONTHS = 1

REGISTRY = {
    "ridge": lambda trial: Ridge(
        alpha=trial.suggest_float("alpha", 1e-4, 1e2, log=True),
        fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
    ),
    "random_forest": lambda trial: RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        n_jobs=-1,
        random_state=42,
    ),
}


def make_model(trial: optuna.Trial, name: str) -> BaseEstimator:
    try:
        model = REGISTRY[name](trial)
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}"
        ) from exc
    return model


class OptunaModelTuner:

    def __init__(
        self,
        direction: str,
        pruner: optuna.pruners.BasePruner,
        study_name: str,
        storage: Optional[str] = None,
    ):
        self._study = optuna.create_study(
            direction=direction,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=bool(storage),
        )

    def tune(
        self,
        df_features: pd.DataFrame,
        df_target: pd.DataFrame,
        n_trials: int,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> optuna.Study:
        self._study.optimize(
            lambda trial: self._objective(trial, df_features, df_target),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )
        return self._study

    @staticmethod
    def _objective(
        trial: optuna.Trial,
        df_features: pd.DataFrame,
        df_target: pd.DataFrame,
        model_name: str,
    ) -> float:
        model = make_model(trial=trial, name=model_name)

        simulation_result = walk_forward_simulation(
            df_features=df_features,
            df_target=df_target,
            model=model,
            test_size_months=TEST_SIZE_IN_MONTHS,
            train_size_months=TRAIN_SIZE_IN_MONTHS,
        )

        df_metrics_aggregated = pd.concat(simulation_result.df_metrics)

        return float(df_metrics_aggregated.mean())
