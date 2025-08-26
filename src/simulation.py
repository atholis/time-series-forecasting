from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.metrics import MLMetricName
from src.models.base import TrainingDataset

# TODO:
# - Make retraining frequency an option
# - add model interface - when available
# - Make simulator a class with an interface, getting a metrics factory
# - Make metrics configurable?


@dataclass
class SimulationResult:
    df_predictions: List[pd.DataFrame]
    df_actuals: List[pd.DataFrame]
    df_metrics: List[pd.DataFrame]


class BaseSimulator(ABC):

    @abstractmethod
    def simulate(self, dataset: TrainingDataset, model: Any) -> SimulationResult:
        pass


class WalkForwardSimulator(BaseSimulator):

    def __init__(self, train_size_months: int, test_size_months: int):
        super().__init__()

        self._train_size_months = train_size_months
        self._test_size_months = test_size_months

    def simulate(self, dataset: TrainingDataset, model: Any) -> SimulationResult:
        """
        Perform walk-forward validation on time series data with lagged and future features.
        Train size and test size are specified in months, and the sliding windows move by 1 month.

        Parameters:
            df_features: pd.DataFrame
                Feature matrix (time series with lagged/future features).
            df_target: pd.DataFrame
                Target variable.
            train_size_months: int
                Number of months for the training window.
            test_size_months: int
                Number of months for the test window.
            model: Any
                Machine learning model with fit() and predict() methods.

        Returns:
            SimulationResults dataclass
        """

        if not isinstance(dataset.df_features.index, pd.DatetimeIndex):
            raise ValueError("df_features must have a DatetimeIndex")
        if not isinstance(dataset.df_targets.index, pd.DatetimeIndex):
            raise ValueError("df_target must have a DatetimeIndex")

        df_predictions = []
        df_actuals = []
        df_metrics = []

        start_date = dataset.df_features.index[0]
        end_date = dataset.df_features.index[-1]
        current_train_start = start_date

        while True:
            current_train_end = current_train_start + pd.DateOffset(
                months=self._train_size_months
            )
            current_test_start = current_train_end
            current_test_end = current_test_start + pd.DateOffset(
                months=self._test_size_months
            )

            if current_test_end > end_date:
                break

            # Select train/test sets
            df_features_train = dataset.df_features.loc[
                current_train_start:current_train_end
            ]
            df_target_train = dataset.df_targets.loc[
                current_train_start:current_train_end
            ]
            df_features_test = dataset.df_features.loc[
                current_test_start:current_test_end
            ]
            df_target_test = dataset.df_targets.loc[current_test_start:current_test_end]

            # Fit model and predict
            model.fit(df_features_train, df_target_train)
            y_pred = pd.DataFrame(
                model.predict(df_features_test),
                index=df_features_test.index,
                columns=df_target_test.columns,
            )

            # Compute metrics
            mae = mean_absolute_error(df_target_test, y_pred)
            metrics = pd.DataFrame(
                {MLMetricName.MAE.value: [mae]}, index=[current_test_start]
            )

            # Append results
            df_predictions.append(y_pred)
            df_actuals.append(df_target_test)
            df_metrics.append(metrics)

            # Print fold details
            print(
                f"Train: {current_train_start.date()} to {current_train_end.date()} | "
                f"Test: {current_test_start.date()} to {current_test_end.date()} - {MLMetricName.MAE.value}: {mae:.4f}"
            )

            # Move the train window 1 month forward
            current_train_start += pd.DateOffset(months=1)

        return SimulationResult(
            df_predictions=df_predictions, df_actuals=df_actuals, df_metrics=df_metrics
        )
