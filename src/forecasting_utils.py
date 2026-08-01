from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


FORECAST_WINDOW_YEARS = 10


@dataclass(frozen=True)
class ForecastSummary:
    model_name: str
    latest_actual_year: int
    target_year: int
    latest_actual_value: float
    final_forecast_value: float


def one_step_forecast_from_values(train_values: np.ndarray, model_name: str) -> float:
    """Fit a one-step forecast using the model labels already used in the project."""
    if len(train_values) == 0:
        raise ValueError("Training window is empty.")

    if model_name == "Naive":
        return float(train_values[-1])

    if not HAS_STATSMODELS:
        raise ValueError("statsmodels is required for this forecast model.")

    if model_name == "AutoReg":
        if len(train_values) < 5:
            return float(train_values[-1])
        lags = min(3, len(train_values) - 1)
        fit = AutoReg(train_values, lags=lags, old_names=False, trend="c").fit()
        return float(fit.forecast(steps=1)[0])

    if model_name == "ARIMA":
        if len(train_values) < 6:
            return float(train_values[-1])
        fit = ARIMA(
            train_values,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        return float(fit.forecast(steps=1)[0])

    if model_name == "Holt":
        if len(train_values) < 3:
            return float(train_values[-1])
        fit = ExponentialSmoothing(
            train_values,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        return float(fit.forecast(1)[0])

    if model_name == "LogHolt":
        if len(train_values) < 3:
            return float(train_values[-1])
        safe_values = np.clip(train_values, a_min=1e-9, a_max=None)
        fit = ExponentialSmoothing(
            np.log(safe_values),
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        return float(np.exp(fit.forecast(1)[0]))

    raise ValueError(f"Unsupported forecast model: {model_name}")


def forecast_series_to_target_year(
    series_df: pd.DataFrame,
    value_col: str,
    model_name: str,
    target_year: int,
    window_size: int = FORECAST_WINDOW_YEARS,
) -> pd.DataFrame:
    """Recursively forecast a single annual series to a future target year."""
    clean_df = (
        series_df[["year", value_col]]
        .dropna()
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .copy()
    )

    if clean_df.empty:
        raise ValueError("No historical values are available for this series.")

    last_year = int(clean_df["year"].max())
    if target_year <= last_year:
        raise ValueError(f"Target year must be greater than {last_year}.")
    if len(clean_df) < window_size:
        raise ValueError(
            f"This country has only {len(clean_df)} usable years, which is fewer "
            f"than the required {window_size}-year forecast window."
        )

    working_df = clean_df.copy()
    forecast_rows: list[dict[str, float | int]] = []

    for forecast_year in range(last_year + 1, int(target_year) + 1):
        train_window = working_df.tail(window_size).copy()
        train_values = train_window[value_col].astype(float).to_numpy()
        next_value = one_step_forecast_from_values(train_values, model_name)

        forecast_rows.append(
            {
                "forecast_year": int(forecast_year),
                "predicted_value": float(next_value),
                "window_start_year": int(train_window["year"].min()),
                "window_end_year": int(train_window["year"].max()),
            }
        )

        working_df = pd.concat(
            [working_df, pd.DataFrame({"year": [forecast_year], value_col: [next_value]})],
            ignore_index=True,
        )

    return pd.DataFrame(forecast_rows)


def build_history_and_forecast_frame(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Create a single frame for plotting actual history and future forecasts."""
    actual = (
        history_df[["year", value_col]]
        .dropna()
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .rename(columns={"year": "plot_year", value_col: "value"})
        .assign(series="Actual")
    )

    forecast = (
        forecast_df.rename(columns={"forecast_year": "plot_year", "predicted_value": "value"})
        [["plot_year", "value"]]
        .assign(series="Forecast")
    )

    return pd.concat([actual, forecast], ignore_index=True)
