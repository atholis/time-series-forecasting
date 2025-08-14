import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import list_artifacts
import numpy as np
import pandas as pd

from src.experiment_tracking.experiment_tracker import (
    ExperimentData,
    MLFlowExperimentTracker,
)
from src.experiment_tracking.helpers import get_run_ids_from_name

from typing import Dict

# TODO:
# - Test if the run is logged under the correct experiment with the correct name
# - Add parametrize testing for various models


def test_logging(
    scenario_setup: Dict[str, str],
    mlflow_tracker: MLFlowExperimentTracker,
    experiment_data: ExperimentData,
    mlflow_client: MlflowClient,
):
    mlflow_tracker.log_experiment(
        experiment_name=scenario_setup.get("experiment_name"),
        run_name=scenario_setup.get("run_name"),
        experiment_data=experiment_data,
    )

    df_logged_runs = mlflow.search_runs(
        experiment_names=[scenario_setup.get("experiment_name")]
    )

    target_experimentation_run_id = get_run_ids_from_name(
        df_logged_runs=df_logged_runs,
        run_name=scenario_setup.get("run_name"),
    ).values[0]

    logged_data_dict = mlflow_client.get_run(
        target_experimentation_run_id
    ).data.to_dictionary()

    assert logged_data_dict.get("tags").items() >= experiment_data.tags_dict.items()
    assert logged_data_dict.get("params") == {"parameter": "0.0"}

    logged_mean_metrics_array = pd.DataFrame(
        logged_data_dict.get("metrics"), index=[0]
    ).values
    mean_metrics_array = experiment_data.df_training_metrics.mean().values.reshape(
        1, len(experiment_data.df_training_metrics)
    )
    assert np.array_equal(logged_mean_metrics_array, mean_metrics_array)

    logged_artifact_paths = [
        file.path for file in list_artifacts(f"runs:/{target_experimentation_run_id}/")
    ]
    assert set(
        [
            "predictions.parquet",
            "test_metrics.json",
            "training_metrics.json",
        ]
    ) == set(logged_artifact_paths)
