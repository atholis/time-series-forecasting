from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# TODO:
# - is the dataset structure enough? - especially for NN models
# - should I push the responsibility of creating the torch inputs to the model? - probably because the interface should be clean of technologies


@dataclass
class InferenceDataset:
    df_features: pd.DataFrame


@dataclass
class TrainingDataset(InferenceDataset):
    df_targets: pd.DataFrame
    df_weights: Optional[pd.DataFrame] = None


class BaseModel(ABC):

    @abstractmethod
    def fit(self, training_set: TrainingDataset):
        pass

    @abstractmethod
    def predict(self, inference_set: InferenceDataset) -> pd.DataFrame:
        pass
