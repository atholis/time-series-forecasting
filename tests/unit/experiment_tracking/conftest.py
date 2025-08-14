import os

from mlflow import MlflowClient
import pandas as pd
import pytest

from src.experiment_tracking.experiment_tracker import (
    ExperimentData,
    MLFlowExperimentTracker,
)
from src.experiment_tracking.flavor_manager import FlavorManager
from src.experiment_tracking.helpers import delete_model_run

MLFLOW_URI = "file:///" + os.getcwd().replace("\\", "/") + "/mlruns"


@pytest.fixture(scope="module")
def scenario_setup():
    yield {"run_name": "run_name", "experiment_name": "test"}
    delete_model_run(
        run_name="run_name",
        experiment_name="test",
        remote_server_uri=MLFLOW_URI,
    )


@pytest.fixture(scope="module")
def mlflow_tracker() -> MLFlowExperimentTracker:
    return MLFlowExperimentTracker(
        tracking_uri=MLFLOW_URI, flavor_manager=FlavorManager()
    )


@pytest.fixture(scope="module")
def experiment_data() -> ExperimentData:
    return ExperimentData(
        df_predictions=pd.DataFrame(
            data=[i for i in range(10)],
            columns=["column_a"],
            index=pd.date_range(
                start="2022-01-01 00:00:00+01:00",
                end="2022-01-01 00:09:00+01:00",
                freq="1Min",
            ),
        ),
        df_training_metrics=pd.DataFrame(
            data=[[1.0, 1.0], [1.0, 1.0]], columns=["r2", "mae"]
        ),
        df_validation_metrics=pd.DataFrame(
            data=[[1.0, 1.0], [1.0, 1.0]], columns=["r2", "mae"]
        ),
        parameters_dict={"parameter": 0.0},
        tags_dict={"tag": "values"},
    )


@pytest.fixture(scope="module")
def mlflow_client() -> MlflowClient:
    return MlflowClient(MLFLOW_URI)
