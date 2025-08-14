from dataclasses import dataclass
from enum import Enum
from typing import Any

import mlflow
from mlflow.models.model import ModelInfo


class FlavorName(Enum):
    """ "Supported Flavor Names"""

    SP_FORECASTING_MODELS = "sp_forecasting_models"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    PROPHET = "prophet"
    KERAS = "keras"
    TENSORFLOW = "tensorflow"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ONNX = "onnx"
    STATSMODELS = "statsmodels"
    FASTAI = "fastai"


FLAVOR_NAMES_TO_MODULES_MAP = {
    FlavorName.PYTORCH: mlflow.pytorch,
    FlavorName.SKLEARN: mlflow.sklearn,
    FlavorName.XGBOOST: mlflow.xgboost,
    FlavorName.PROPHET: mlflow.prophet,
    FlavorName.KERAS: mlflow.keras,
    FlavorName.TENSORFLOW: mlflow.tensorflow,
    FlavorName.LIGHTGBM: mlflow.lightgbm,
    FlavorName.CATBOOST: mlflow.catboost,
    FlavorName.LIGHTGBM: mlflow.lightgbm,
    FlavorName.ONNX: mlflow.onnx,
    FlavorName.STATSMODELS: mlflow.statsmodels,
}


@dataclass
class MLFlowTrainedModel:
    model: Any
    flavor_name: FlavorName
    artifact_path: str

    def __post_init__(self):
        if not isinstance(self.flavor_name, FlavorName):
            raise ValueError(
                f"flavor_name should be of {type(FlavorName)} not {type(self.flavor_name)}"
            )
        if not isinstance(self.artifact_path, str):
            raise ValueError(
                f"artifact_path should be of {type(str)} not {type(self.artifact_path)}"
            )


@dataclass
class MLFlowTrainedModelReference:
    flavor_name: FlavorName
    model_uri: str
    dst_path: str = None

    def __post_init__(self):
        if not isinstance(self.flavor_name, FlavorName):
            raise ValueError(
                f"flavor_name should be of {type(FlavorName)} not {type(self.flavor_name)}"
            )
        if not isinstance(self.model_uri, str):
            raise ValueError(
                f"model_uri should be of {type(str)} not {type(self.model_uri)}"
            )

        if not isinstance(self.dst_path, str) and self.dst_path is not None:
            raise ValueError(
                f"dst_path should be of {type(str)} not {type(self.dst_path)}"
            )


class FlavorManager:
    """Flavor agnostic manager for model logging/loading"""

    def __init__(self) -> None:
        """Instantiates the class"""
        self._flavor_name_to_flavor_module_map = FLAVOR_NAMES_TO_MODULES_MAP

    def log_model(self, trained_model: MLFlowTrainedModel) -> ModelInfo:
        """Logs a trained model to the ml flow server

        Parameters
        ----------
        trained model
            trained model to be logged

        Returns
        -------
        ModelInfo:
            instance that contains the
            metadata of the logged model
        """
        flavor_wrapper = self._flavor_name_to_flavor_module_map.get(
            trained_model.flavor_name
        )
        return flavor_wrapper.log_model(
            model=trained_model.model,
            artifact_path=trained_model.artifact_path,
        )

    def load_model(self, model_reference: MLFlowTrainedModelReference) -> Any:
        """Loads a trained model from the ml flow server

        Parameters
        ----------
        model_reference
            reference to a trained model

        Returns
        -------
        Any
            trained model
        """
        flavor_wrapper = self._flavor_name_to_flavor_module_map.get(
            model_reference.flavor_name
        )
        return flavor_wrapper.load_model(
            model_reference.model_uri, model_reference.dst_path
        )
