import os
from typing import Any, Dict

from dataclasses import dataclass
import mlflow
import pandas as pd

from src.experiment_tracking.flavor_manager import FlavorManager

PREDICTION_DATA_NAME = "predictions.parquet"
TEST_METRICS_TABLE_FILE_NAME = "test_metrics.json"
TRAINING_METRICS_TABLE_FILE_NAME = "training_metrics.json"


@dataclass
class ExperimentData:
    parameters_dict: Dict[str, Any]
    tags_dict: Dict[str, Any]
    df_predictions: pd.DataFrame
    df_validation_metrics: pd.DataFrame
    df_training_metrics: pd.DataFrame


class MLFlowExperimentTracker:

    def __init__(self, tracking_uri: str, flavor_manager: FlavorManager):
        mlflow.set_tracking_uri(tracking_uri)
        self._flavor_manager = flavor_manager

    def log_experiment(
        self,
        experiment_name: str,
        run_name: str,
        experiment_data: ExperimentData,
        model: Any = None,
    ):
        experiment = mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment.experiment_id,
        ):
            mlflow.set_tags(experiment_data.tags_dict)
            mlflow.log_params(experiment_data.parameters_dict)
            avg_metrics_dict = self._calculate_avg_metrics_dict(
                df_validation_metrics=experiment_data.df_validation_metrics,
                df_training_metrics=experiment_data.df_training_metrics,
            )
            mlflow.log_metrics(avg_metrics_dict)
            self._log_prediction_data(experiment_data.df_predictions)
            self._log_tables(
                df_training_metrics=experiment_data.df_training_metrics,
                df_validation_metrics=experiment_data.df_validation_metrics,
            )

    @staticmethod
    def _log_prediction_data(df_predictions: pd.DataFrame):
        # TODO: use temp path
        df_predictions.to_parquet(
            PREDICTION_DATA_NAME, engine="pyarrow", compression="brotli"
        )
        mlflow.log_artifact(PREDICTION_DATA_NAME)
        os.remove(PREDICTION_DATA_NAME)

    @staticmethod
    def _log_tables(
        df_validation_metrics: pd.DataFrame,
        df_training_metrics: pd.DataFrame,
    ):
        for df_table, path in zip(
            [df_validation_metrics, df_training_metrics],
            [TEST_METRICS_TABLE_FILE_NAME, TRAINING_METRICS_TABLE_FILE_NAME],
        ):
            mlflow.log_table(
                data=df_table.reset_index(),
                artifact_file=path,
            )

    # TODO: think whether this is needed
    @staticmethod
    def _calculate_avg_metrics_dict(
        df_validation_metrics: pd.DataFrame,
        df_training_metrics: pd.DataFrame,
    ) -> Dict[str, str]:
        metrics_dict = {}
        metrics_dict.update(df_validation_metrics.mean(axis=0).to_dict())
        metrics_dict.update(df_training_metrics.mean(axis=0).to_dict())
        return metrics_dict
