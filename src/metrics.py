import pandas as pd

# TODO: optimize inputs


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

    cfe = err.cumsum()
    mad_cum = err.abs().expanding().mean()
    tracking_signal = cfe / (mad_cum + 1e-9)

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
