from enum import Enum
import pandas as pd

# TODO: optimize inputs

import pandas as pd

pd.options.plotting.backend = "plotly"


class MLMetricName(Enum):
    BIAS = "bias"
    CFE = "cfe"
    MAE = "mae"
    MSE = "mse"
    rMSE = "rmse"
    TRACKING_SIGNAL = "tracking_signal"


def tracking_signal(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    per_horizon: bool,
    eps: float = 1e-9,
):
    """
    Classic tracking signal on wide data (columns = horizons 0..H-1).

    Parameters
    ----------
    df_actual : DataFrame with actual values, columns = horizons
    df_forecast : DataFrame with forecast values, columns = horizons
    per_horizon : if True, return TS per horizon; if False, return one aggregated TS
    eps : small value to avoid division by zero

    Returns
    -------
    DataFrame (per_horizon=True) or Series (per_horizon=False)
    """

    # signed errors per horizon
    errors = df_actual - df_forecast.values

    if per_horizon:
        cfe = errors.cumsum()
        mae = errors.abs().expanding().mean()
        # cfe = errors.rolling(window="14d").sum()
        # mae = errors.abs().rolling(window="14d").mean()
        ts = cfe / (mae + eps)
        ts.columns = [f"h_{c}" for c in ts.columns]  # valid Python identifiers
        return ts
    else:
        # aggregate across horizons first (simple mean), then compute one TS
        err_agg = errors.mean(axis=1)
        cfe = err_agg.cumsum()
        mae = err_agg.abs().expanding().mean()
        ts = cfe / (mae + eps)
        ts.name = "ts_aggregate"
        return ts


def calculate_tracking_signal(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    actual_col: str,
    forecast_col: str,
    agg_freq="1h",
) -> pd.DataFrame:
    """
    Returns:
      tracking_signal  : cumulative tracking signal since reset (APICS style)
    """

    df = pd.merge(
        df_actual[[actual_col]],
        df_forecast[[forecast_col]],
        left_index=True,
        right_index=True,
        how="inner",
    ).sort_index()
    err_min = df[actual_col] - df[forecast_col]

    err = err_min.resample(agg_freq).mean()

    cfe_cum = err.cumsum()
    mae_cum = err.abs().expanding().mean()
    tracking_signal = cfe_cum / (mae_cum + 1e-9)

    return tracking_signal


def calculate_bias(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    actual_col: str,
    forecast_col: str,
) -> float:
    return (df_actual.loc[actual_col] - df_forecast.loc[forecast_col]).mean()


def calculate_cfe(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    actual_col: str,
    forecast_col: str,
) -> float:
    return (df_actual.loc[actual_col] - df_forecast.loc[forecast_col]).sum()
