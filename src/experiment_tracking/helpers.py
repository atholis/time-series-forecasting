from enum import Enum

import mlflow
import pandas as pd


class _MLFlowLoggedRunsColumnName(Enum):
    RUN_NAME = "tags.mlflow.runName"
    RUN_ID = "run_id"
    PARENT_RUN_ID = "tags.mlflow.parentRunId"


def get_run_ids_from_name(run_name: str, df_logged_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves all logged run ids that have a match to the passed name

    Parameters
    ----------
    run_name:
        name of the run to be searched
    df_logged_runs:
        DataFrame of runs as returned from the search_runs call,
        where each metric, parameter, and tag is expanded into
        its own column named metrics.*, params.*, or tags.* respectively.
        For runs that don't have a particular metric, parameter, or tag, the
        value for the corresponding column is (NumPy) Nan, None, or None respectively.
    Returns
    -------
    pandas.DataFrame:
        DataFrame that contains the logged run data

    Raises
    ------
    ValueError:
        if df_logged_runs is empty
    """
    if df_logged_runs.empty:
        raise ValueError(
            "df_logged_runs is empty, probably because the experiment contains no runs!"
        )

    return df_logged_runs[
        df_logged_runs[_MLFlowLoggedRunsColumnName.RUN_NAME.value] == run_name
    ].loc[:, _MLFlowLoggedRunsColumnName.RUN_ID.value]


def delete_model_run(run_name: str, experiment_name: str, remote_server_uri: str):
    """
    Removes a logged run from a mlflow experiment

    Parameters
    ----------
    run_name:
        name of the model run
    experiment_name:
        name of the experiment
    remote_server_uri:
        uri of the remote tracking server
    """
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow_client = mlflow.MlflowClient(remote_server_uri)

    df_logged_runs = mlflow.search_runs(experiment_names=[experiment_name])
    ds = get_run_ids_from_name(run_name=run_name, df_logged_runs=df_logged_runs)

    run_id = ds.values[0]
    mlflow_client.delete_run(run_id)
