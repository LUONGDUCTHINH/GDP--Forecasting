from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import build_model_test_table, get_best_main_model
from src.charts import build_forecast_chart, build_main_model_yearly_chart, build_multi_forecast_chart
from src.components import (
    render_insight_box,
    render_metric_card,
    render_note_box,
    render_page_header,
    render_pipeline,
)
from src.dashboard_data import (
    EVENT_TIMELINE_PATH,
    GDP_TS_FIG_PATH,
    LIFE_TS_FIG_PATH,
    POP_TS_FIG_PATH,
    load_benchmark_metrics,
    load_main_model_metrics,
    load_main_model_sample,
    load_main_model_specifications,
    load_main_model_yearly,
    load_panel,
    load_ts_best,
    load_ts_summary,
)
from src.forecasting_utils import FORECAST_WINDOW_YEARS, HAS_STATSMODELS, forecast_series_to_target_year
from src.formatting import format_currency, format_integer, format_percent, format_years

try:
    import statsmodels.formula.api as smf

    HAS_STATSMODELS_FORMULA = True
except ImportError:
    HAS_STATSMODELS_FORMULA = False



GDP_BENCHMARK_FIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "figure18_gdp_benchmark.png"
)

DATASET_META = {
    "gdp": {
        "label": "GDP",
        "value_col": "gdp_per_capita_usd",
        "display": "GDP per Capita (current US$)",
        "chart_title": "GDP per capita",
    },
    "population": {
        "label": "Population",
        "value_col": "population_total",
        "display": "Population Total",
        "chart_title": "Population",
    },
    "life_expectancy": {
        "label": "Life Expectancy",
        "value_col": "life_expectancy_years",
        "display": "Life Expectancy (years)",
        "chart_title": "Life expectancy",
    },
}
DATASET_ORDER = ["gdp", "population", "life_expectancy"]
MAIN_DYNAMIC_SPECS = [
    {
        "model": "Model 1 - Baseline Dynamic",
        "formula": "target_log_gdp_next_year ~ log_gdp_per_capita + log_population_total + life_expectancy_years",
    },
    {
        "model": "Model 2 - Extended Dynamic",
        "formula": (
            "target_log_gdp_next_year ~ log_gdp_per_capita + log_population_total + life_expectancy_years "
            "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean"
        ),
    },
    {
        "model": "Model 3 - Full Dynamic",
        "formula": (
            "target_log_gdp_next_year ~ log_gdp_per_capita + log_population_total + life_expectancy_years "
            "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean "
            "+ asian_financial_crisis_9798 + global_financial_crisis_0809 + C(wb_region) + year_trend"
        ),
    },
]
MAIN_DYNAMIC_FEATURE_MODELS = {
    "population_total": "LogHolt",
    "life_expectancy_years": "Holt",
    "inflation_pct_clean": "ARIMA",
    "unemployment_pct_clean": "AutoReg",
    "internet_users_pct_clean": "Holt",
}
FUTURE_PROJECTION_MODELS = {
    "gdp": "ARIMA",
    "population": "LogHolt",
    "life_expectancy": "Holt",
}

FINAL_SELECTED_MAIN_MODEL = "Model 3 - Full Dynamic"



def _format_indicator_value(dataset_key: str, value: float) -> str:
    if dataset_key == "gdp":
        return format_currency(value)
    if dataset_key == "population":
        return format_integer(value)
    return format_years(value)


def _format_indicator_change(dataset_key: str, value: float) -> str:
    if dataset_key == "gdp":
        return format_currency(value)
    if dataset_key == "population":
        if pd.isna(value):
            return "N/A"
        return f"{int(round(float(value))):+,}"
    if pd.isna(value):
        return "N/A"
    return f"{float(value):+,.2f} years"


def _forecast_shape_explanation(dataset_key: str, model_name: str) -> tuple[str, str]:
    dataset_label = DATASET_META[dataset_key]["label"]

    if model_name == "Naive":
        return (
            "Why does this forecast path look flat?",
            f"The selected {dataset_label.lower()} projection model is Naive. In a recursive forecast, Naive copies the most recent "
            "observed value into the next year, then repeats that same value for later years. That makes the future path look flat "
            "instead of continuing a trend.",
        )

    if model_name == "LogHolt":
        return (
            "Why does this forecast path stay smooth?",
            f"The selected {dataset_label.lower()} projection model is LogHolt. It extends the long-run trend on a log scale, so the "
            "future path usually stays smooth and gradually compounding rather than flat.",
        )

    if model_name == "Holt":
        return (
            "Why does this forecast path stay linear?",
            f"The selected {dataset_label.lower()} projection model is Holt trend smoothing. It projects the recent level and slope "
            "forward, so the future path often continues in a steady upward or downward direction.",
        )

    return (
        "How should this forecast path be read?",
        f"The selected {dataset_label.lower()} projection model uses the most recent {FORECAST_WINDOW_YEARS}-year window and "
        "forecasts recursively, so each new predicted year becomes part of the next forecasting window.",
    )


def _best_ts_model_name(dataset_key: str) -> str:
    best_df = load_ts_best(dataset_key)
    if not best_df.empty and "Model" in best_df.columns:
        return str(best_df["Model"].iloc[0])

    summary_df = load_ts_summary(dataset_key)
    if summary_df.empty:
        fallback = {"gdp": "Naive", "life_expectancy": "Naive", "population": "LogHolt"}
        return fallback[dataset_key]

    return str(summary_df.sort_values(["RMSE", "MAPE_pct", "MAE"]).iloc[0]["Model"])


def _future_projection_model_name(dataset_key: str) -> str:
    return FUTURE_PROJECTION_MODELS[dataset_key]


def _best_ts_metrics_row(dataset_key: str) -> pd.Series | None:
    model_name = _best_ts_model_name(dataset_key)
    return _ts_metrics_row(dataset_key, model_name)


def _ts_metrics_row(dataset_key: str, model_name: str) -> pd.Series | None:
    summary_df = load_ts_summary(dataset_key)
    if summary_df.empty:
        return None
    temp = summary_df[summary_df["Model"].astype(str) == model_name].copy()
    if temp.empty:
        return None
    return temp.iloc[0]


def _available_ts_models(dataset_key: str) -> list[str]:
    summary_df = load_ts_summary(dataset_key)
    if summary_df.empty or "Model" not in summary_df.columns:
        return [_best_ts_model_name(dataset_key)]
    return summary_df["Model"].dropna().astype(str).drop_duplicates().tolist()


def _build_country_series_frame(panel_df: pd.DataFrame, country_code: str, value_col: str) -> pd.DataFrame:
    return (
        panel_df[panel_df["country_code"] == country_code][["year", value_col]]
        .dropna()
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .copy()
    )


@st.cache_resource(show_spinner=False)
def _fit_main_dynamic_models() -> dict[str, object]:
    if not (HAS_STATSMODELS and HAS_STATSMODELS_FORMULA):
        return {}

    model_df = load_main_model_sample().copy()
    model_df["wb_region"] = model_df["wb_region"].astype(str)

    fitted_models: dict[str, object] = {}
    for spec in MAIN_DYNAMIC_SPECS:
        fitted_models[spec["model"]] = smf.ols(formula=spec["formula"], data=model_df).fit(cov_type="HC3")

    return fitted_models


def _build_main_dynamic_future_forecasts(
    panel_df: pd.DataFrame,
    country_code: str,
    target_year: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, object]]:
    if not (HAS_STATSMODELS and HAS_STATSMODELS_FORMULA):
        raise ValueError("statsmodels is required to generate future forecasts for the three main GDP models.")

    fitted_models = _fit_main_dynamic_models()
    if not fitted_models:
        raise ValueError("Main GDP models could not be fitted for future forecasting.")

    country_rows = panel_df[panel_df["country_code"] == country_code].copy()
    if country_rows.empty:
        raise ValueError(f"Country code {country_code} was not found in the panel dataset.")

    country_name = str(country_rows["country_name"].dropna().iloc[0])
    wb_region = str(country_rows["wb_region"].dropna().iloc[0])

    gdp_history_df = _build_country_series_frame(panel_df, country_code, "gdp_per_capita_usd")
    if gdp_history_df.empty:
        raise ValueError("No GDP history is available for the selected country.")

    latest_actual_gdp_year = int(gdp_history_df["year"].max())
    latest_actual_gdp_value = float(gdp_history_df["gdp_per_capita_usd"].iloc[-1])
    if target_year <= latest_actual_gdp_year:
        raise ValueError(
            f"Target year must be greater than the latest actual GDP year ({latest_actual_gdp_year})."
        )

    feature_series_frames = {
        col: _build_country_series_frame(panel_df, country_code, col)
        for col in MAIN_DYNAMIC_FEATURE_MODELS.keys()
    }

    required_last_years: dict[str, int] = {}
    for feature_col, feature_df in feature_series_frames.items():
        if feature_df.empty:
            raise ValueError(f"{feature_col} history is missing for the selected country.")
        required_last_years[feature_col] = int(feature_df["year"].max())

    target_feature_year = target_year - 1

    feature_maps: dict[str, dict[int, float]] = {}
    for feature_col, model_name in MAIN_DYNAMIC_FEATURE_MODELS.items():
        hist_df = feature_series_frames[feature_col]
        current_map = {int(row["year"]): float(row[feature_col]) for _, row in hist_df.iterrows()}
        if target_feature_year > int(hist_df["year"].max()):
            fc_df = forecast_series_to_target_year(
                hist_df,
                value_col=feature_col,
                model_name=model_name,
                target_year=target_feature_year,
                window_size=FORECAST_WINDOW_YEARS,
            )
            for _, row in fc_df.iterrows():
                current_map[int(row["forecast_year"])] = float(row["predicted_value"])
        feature_maps[feature_col] = current_map

    actual_gdp_map = {int(row["year"]): float(row["gdp_per_capita_usd"]) for _, row in gdp_history_df.iterrows()}
    year_base = int(panel_df["year"].min())

    forecast_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    latest_observed_driver_year = min(required_last_years.values())

    for spec in MAIN_DYNAMIC_SPECS:
        model_name = spec["model"]
        fitted_model = fitted_models[model_name]
        model_gdp_map = actual_gdp_map.copy()
        model_rows: list[dict[str, object]] = []

        for forecast_year in range(latest_actual_gdp_year + 1, int(target_year) + 1):
            feature_year = forecast_year - 1
            gdp_input = float(model_gdp_map[feature_year])
            population_used = float(feature_maps["population_total"][feature_year])
            life_used = float(feature_maps["life_expectancy_years"][feature_year])
            inflation_used = float(feature_maps["inflation_pct_clean"][feature_year])
            unemployment_used = float(feature_maps["unemployment_pct_clean"][feature_year])
            internet_used = float(feature_maps["internet_users_pct_clean"][feature_year])

            pred_input = pd.DataFrame(
                [
                    {
                        "country_name": country_name,
                        "country_code": country_code,
                        "wb_region": wb_region,
                        "year": int(feature_year),
                        "gdp_per_capita_usd": gdp_input,
                        "log_gdp_per_capita": np.log(max(gdp_input, 1e-9)),
                        "population_total": population_used,
                        "log_population_total": np.log(max(population_used, 1e-9)),
                        "life_expectancy_years": life_used,
                        "inflation_pct_clean": inflation_used,
                        "unemployment_pct_clean": unemployment_used,
                        "internet_users_pct_clean": internet_used,
                        "asian_financial_crisis_9798": int(feature_year in [1997, 1998]),
                        "global_financial_crisis_0809": int(feature_year in [2008, 2009]),
                        "year_trend": int(feature_year - year_base + 1),
                    }
                ]
            )

            predicted_log = float(fitted_model.predict(pred_input).iloc[0])
            predicted_gdp = float(np.exp(predicted_log))
            model_gdp_map[int(forecast_year)] = predicted_gdp

            model_rows.append(
                {
                    "forecast_year": int(forecast_year),
                    "predicted_value": predicted_gdp,
                    "feature_year": int(feature_year),
                    "gdp_input_used": gdp_input,
                    "population_used": population_used,
                    "life_expectancy_used": life_used,
                    "inflation_used": inflation_used,
                    "unemployment_used": unemployment_used,
                    "internet_used": internet_used,
                }
            )

        model_forecast_df = pd.DataFrame(model_rows)
        forecast_frames[model_name] = model_forecast_df

        final_forecast_value = float(model_forecast_df["predicted_value"].iloc[-1])
        absolute_change = final_forecast_value - latest_actual_gdp_value
        total_growth_pct = (absolute_change / latest_actual_gdp_value) * 100 if latest_actual_gdp_value > 0 else np.nan
        cagr_pct = (
            ((final_forecast_value / latest_actual_gdp_value) ** (1 / (target_year - latest_actual_gdp_year)) - 1) * 100
            if latest_actual_gdp_value > 0
            else np.nan
        )

        summary_rows.append(
            {
                "Model": model_name,
                "Forecast in target year": format_currency(final_forecast_value),
                "Absolute change": format_currency(absolute_change),
                "Total growth (%)": format_percent(total_growth_pct),
                "CAGR (%)": format_percent(cagr_pct),
                "_target_value_raw": final_forecast_value,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("_target_value_raw", ascending=False).reset_index(drop=True)

    metadata = {
        "country_name": country_name,
        "country_code": country_code,
        "wb_region": wb_region,
        "latest_observed_driver_year": latest_observed_driver_year,
        "latest_actual_gdp_year": latest_actual_gdp_year,
        "latest_actual_gdp_value": latest_actual_gdp_value,
        "first_future_gdp_year": latest_actual_gdp_year + 1,
        "feature_models": {
            "population_total": MAIN_DYNAMIC_FEATURE_MODELS["population_total"],
            "life_expectancy_years": MAIN_DYNAMIC_FEATURE_MODELS["life_expectancy_years"],
            "inflation_pct_clean": MAIN_DYNAMIC_FEATURE_MODELS["inflation_pct_clean"],
            "unemployment_pct_clean": MAIN_DYNAMIC_FEATURE_MODELS["unemployment_pct_clean"],
            "internet_users_pct_clean": MAIN_DYNAMIC_FEATURE_MODELS["internet_users_pct_clean"],
        },
    }

    return forecast_frames, summary_df, metadata


def _build_compare_summary(
    dataset_key: str,
    model_names: list[str],
    forecast_frames: dict[str, pd.DataFrame],
    latest_actual_value: float,
    latest_actual_year: int,
    target_year: int,
    best_model_name: str,
    projection_model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_name in model_names:
        forecast_df = forecast_frames.get(model_name, pd.DataFrame())
        if forecast_df.empty:
            continue

        final_forecast_value = float(forecast_df["predicted_value"].iloc[-1])
        absolute_change = final_forecast_value - latest_actual_value
        pct_change = (absolute_change / latest_actual_value) * 100 if latest_actual_value > 0 else float("nan")
        cagr_pct = ((final_forecast_value / latest_actual_value) ** (1 / (target_year - latest_actual_year)) - 1) * 100
        metrics_row = _ts_metrics_row(dataset_key, model_name)
        role_parts: list[str] = []
        if model_name == projection_model_name:
            role_parts.append("Future projection model")
        if model_name == best_model_name:
            role_parts.append("Best backtest model")

        rows.append(
            {
                "Model": model_name,
                "Role": " + ".join(role_parts) if role_parts else "Alternative model",
                "Target year forecast": _format_indicator_value(dataset_key, final_forecast_value),
                "Absolute change": _format_indicator_change(dataset_key, absolute_change),
                "Forecast growth (%)": pct_change,
                "CAGR (%)": cagr_pct,
                "Test RMSE": metrics_row["RMSE"] if metrics_row is not None else pd.NA,
                "Test MAPE (%)": metrics_row["MAPE_pct"] if metrics_row is not None else pd.NA,
                "_target_value_raw": final_forecast_value,
            }
        )

    return pd.DataFrame(rows)


def render() -> None:
    panel_df = load_panel()
    main_metrics_df = load_main_model_metrics()
    main_yearly_df = load_main_model_yearly()
    main_specs_df = load_main_model_specifications()
    benchmark_df = load_benchmark_metrics()

    render_page_header(
        active_key="forecasting",
        title="GDP Forecasting Models",
        question=(
            "Which models perform best for indicator-level forecasting, and how well do "
            "the main GDP specifications predict next-year GDP per capita?"
        ),
        description=(
            "This page follows the two-stage modelling structure used in the report. "
            "Stage 1 evaluates separate time-series models for GDP per capita, population, "
            "and life expectancy using rolling 10-year backtesting. Stage 2 evaluates "
            "progressively richer next-year GDP specifications on a shared holdout period "
            "and then uses forecasted drivers to demonstrate future GDP scenarios."
        ),
        chips=[
            "Stage 1: rolling 10-year indicator backtesting",
            "Stage 2: next-year GDP per capita prediction",
            "Shared GDP holdout: feature years 2018-2022",
            "Evaluation: MAE, RMSE, MAPE, and R²",
        ],
    )

    render_note_box(
        "Two distinct forecasting tasks",
        (
            "Indicator-level time-series forecasting and multivariate GDP prediction are "
            "related but methodologically different. The first asks which temporal model "
            "best predicts each individual indicator. The second asks whether demographic, "
            "macroeconomic, regional, and structural information improves next-year GDP "
            "per capita prediction. Their metrics should therefore be interpreted separately."
        ),
    )

    gdp_best_model = _best_ts_model_name("gdp")
    life_best_model = _best_ts_model_name("life_expectancy")
    pop_best_model = _best_ts_model_name("population")

    foundation_rows = {
        "gdp": _best_ts_metrics_row("gdp"),
        "life_expectancy": _best_ts_metrics_row("life_expectancy"),
        "population": _best_ts_metrics_row("population"),
    }

    st.markdown("### Stage 1 — Indicator-level time-series forecasting")
    st.caption(
        "GDP per capita, population, and life expectancy are evaluated separately because "
        "their temporal behaviour differs. The reported winner for each indicator is selected "
        "from rolling one-step-ahead backtesting with a 10-year historical window."
    )
    metric_cols = st.columns(3)
    for col, dataset_key in zip(metric_cols, DATASET_ORDER):
        row = foundation_rows[dataset_key]
        with col:
            if row is None:
                render_metric_card(
                    DATASET_META[dataset_key]["label"],
                    _best_ts_model_name(dataset_key),
                    "No summary metrics file was found",
                )
            else:
                render_metric_card(
                    DATASET_META[dataset_key]["label"],
                    str(row["Model"]),
                    (
                        f"RMSE: {row['RMSE']:.3f} | MAPE: {row['MAPE_pct']:.3f}% | "
                        f"Countries: {int(row['n_countries_modeled'])}"
                    ),
                )

    st.markdown("### Interactive indicator forecast explorer")
    render_note_box(
        "Backtest winner versus future projection model",
        (
            "The model with the lowest one-step backtest error is retained as the formal "
            "historical benchmark. For long-horizon demonstration, a different projection "
            "model may be used when the benchmark winner would produce an implausibly flat "
            "recursive path. This distinction is shown explicitly rather than hidden."
        ),
    )
    indicator_labels = [DATASET_META[key]["label"] for key in DATASET_ORDER]
    if hasattr(st, "segmented_control"):
        selected_indicator_label = st.segmented_control(
            "Choose an indicator to forecast",
            indicator_labels,
            default=indicator_labels[0],
            selection_mode="single",
            key="forecast_indicator_selector",
        )
        if selected_indicator_label is None:
            selected_indicator_label = indicator_labels[0]
    else:
        selected_indicator_label = st.radio(
            "Choose an indicator to forecast",
            indicator_labels,
            horizontal=True,
            key="forecast_indicator_selector",
        )

    selector_cols = st.columns([0.9, 1.1], gap="large")

    selected_dataset_key = next(
        key for key in DATASET_ORDER if DATASET_META[key]["label"] == selected_indicator_label
    )
    selected_meta = DATASET_META[selected_dataset_key]
    selected_best_model = _best_ts_model_name(selected_dataset_key)
    selected_projection_model = _future_projection_model_name(selected_dataset_key)
    selected_value_col = selected_meta["value_col"]
    available_models = _available_ts_models(selected_dataset_key)
    if selected_projection_model not in available_models:
        available_models.append(selected_projection_model)

    with selector_cols[0]:
        if hasattr(st, "segmented_control"):
            display_mode = st.segmented_control(
                "Forecast display mode",
                ["Best model only", "Compare models"],
                default="Best model only",
                selection_mode="single",
                key=f"forecast_display_mode_{selected_dataset_key}",
            )
            if display_mode is None:
                display_mode = "Best model only"
        else:
            display_mode = st.radio(
                "Forecast display mode",
                ["Best model only", "Compare models"],
                horizontal=True,
                key=f"forecast_display_mode_{selected_dataset_key}",
            )

    with selector_cols[1]:
        st.caption(
            f"Backtest winner: {selected_best_model}. Future projection model in Best mode: {selected_projection_model}. "
            "Compare mode overlays all saved time-series models so the benchmark winner and the deployed projection model can be viewed together."
        )

    history_counts = (
        panel_df.dropna(subset=[selected_value_col])
        .groupby(["country_name", "country_code"], as_index=False)
        .size()
        .rename(columns={"size": "n_obs"})
    )
    eligible_countries = history_counts[history_counts["n_obs"] >= FORECAST_WINDOW_YEARS].copy()
    if eligible_countries.empty:
        st.warning(
            f"No country has enough {selected_meta['label'].lower()} history for the {FORECAST_WINDOW_YEARS}-year forecast window."
        )
        return

    country_labels = (
        eligible_countries.assign(label=lambda df: df["country_name"] + " (" + df["country_code"] + ")")
        .sort_values("country_name")
        .reset_index(drop=True)
    )
    default_index = 0
    if "VNM" in country_labels["country_code"].values:
        default_index = int(country_labels.index[country_labels["country_code"] == "VNM"][0])

    selected_label = st.selectbox(
        f"Select a country for the {selected_meta['label']} future path",
        country_labels["label"].tolist(),
        index=default_index,
        key=f"forecast_country_label_{selected_dataset_key}",
    )
    selected_row = country_labels[country_labels["label"] == selected_label].iloc[0]
    country_code = str(selected_row["country_code"])
    country_name = str(selected_row["country_name"])

    history_df = (
        panel_df[panel_df["country_code"] == country_code]
        .dropna(subset=[selected_value_col])
        .sort_values("year")
        .copy()
    )
    latest_actual_year = int(history_df["year"].max())

    target_year = st.slider(
        f"Select a future target year for {selected_meta['label']}",
        min_value=latest_actual_year + 1,
        max_value=latest_actual_year + 15,
        value=min(latest_actual_year + 10, latest_actual_year + 15),
        step=1,
        key=f"forecast_target_year_{selected_dataset_key}",
    )

    forecast_model_names = [selected_projection_model] if display_mode == "Best model only" else available_models
    forecast_frames: dict[str, pd.DataFrame] = {}
    forecast_errors: dict[str, str] = {}

    for model_name in forecast_model_names:
        try:
            forecast_frames[model_name] = forecast_series_to_target_year(
                history_df,
                value_col=selected_value_col,
                model_name=model_name,
                target_year=target_year,
                window_size=FORECAST_WINDOW_YEARS,
            )
        except ValueError as exc:
            forecast_errors[model_name] = str(exc)

    if not forecast_frames:
        first_error = next(iter(forecast_errors.values()), "No forecast could be generated for the selected setup.")
        st.warning(first_error)
    else:
        latest_actual_value = float(history_df[selected_value_col].iloc[-1])
        compare_summary_df = _build_compare_summary(
            selected_dataset_key,
            forecast_model_names,
            forecast_frames,
            latest_actual_value,
            latest_actual_year,
            target_year,
            selected_best_model,
            selected_projection_model,
        )
        if compare_summary_df.empty:
            st.warning("No forecast summary could be created for the selected setup.")
            return

        projection_available_model = (
            selected_projection_model
            if selected_projection_model in compare_summary_df["Model"].tolist()
            else str(compare_summary_df.iloc[0]["Model"])
        )
        benchmark_available_model = (
            selected_best_model
            if selected_best_model in compare_summary_df["Model"].tolist()
            else projection_available_model
        )
        projection_row = compare_summary_df[compare_summary_df["Model"] == projection_available_model].iloc[0]
        benchmark_row = compare_summary_df[compare_summary_df["Model"] == benchmark_available_model].iloc[0]

        if display_mode == "Best model only":
            backtest_row = _ts_metrics_row(selected_dataset_key, projection_available_model)
            final_forecast_value = float(projection_row["_target_value_raw"])
            absolute_change = final_forecast_value - latest_actual_value
            pct_change = float(projection_row["Forecast growth (%)"])
            cagr_pct = float(projection_row["CAGR (%)"])

            forecast_cols = st.columns(5)
            with forecast_cols[0]:
                render_metric_card(
                    f"{selected_meta['label']} projection model",
                    projection_available_model,
                    "Used for the future path shown below",
                )
            with forecast_cols[1]:
                render_metric_card(
                    "Backtest winner",
                    selected_best_model,
                    "Repository benchmark result from rolling backtesting",
                )
            with forecast_cols[2]:
                render_metric_card(
                    "Latest actual year",
                    str(latest_actual_year),
                    f"{country_name} has {len(history_df)} valid {selected_meta['label'].lower()} observations",
                )
            with forecast_cols[3]:
                render_metric_card(
                    "Forecast in target year",
                    _format_indicator_value(selected_dataset_key, final_forecast_value),
                    f"{country_name} in {target_year}",
                )
            with forecast_cols[4]:
                render_metric_card(
                    "Absolute change",
                    _format_indicator_change(selected_dataset_key, absolute_change),
                    f"Relative to the latest actual {selected_meta['label'].lower()} year",
                )

            growth_cols = st.columns(2)
            with growth_cols[0]:
                render_metric_card(
                    "Forecast growth",
                    format_percent(pct_change),
                    f"CAGR: {format_percent(cagr_pct)}",
                )
            with growth_cols[1]:
                if projection_available_model != selected_best_model:
                    render_metric_card(
                        "Why not use the winner here?",
                        "Long-horizon path updated",
                        "Best mode prefers a multi-step projection model when the backtest winner creates an implausibly flat future path.",
                    )
                else:
                    render_metric_card(
                        "Benchmark and projection",
                        "Same model",
                        "The backtest winner is also the deployed future-projection model for this indicator.",
                    )

            if backtest_row is not None:
                st.caption(
                    f"Backtest metrics for the selected {selected_meta['label']} projection model: "
                    f"MAE {backtest_row['MAE']:.2f}, RMSE {backtest_row['RMSE']:.2f}, "
                    f"MAPE {backtest_row['MAPE_pct']:.2f}%, R² {backtest_row['R_squared']:.4f}."
                )

            if projection_available_model != selected_best_model:
                render_note_box(
                    "Why do the benchmark winner and projection model differ?",
                    f"The rolling backtest winner for {selected_meta['label'].lower()} is {selected_best_model}, but the future path shown here uses "
                    f"{projection_available_model}. This dashboard keeps the benchmark result for evaluation, while using a more plausible multi-step "
                    "projection model for long-horizon forecasting.",
                )

            note_title, note_copy = _forecast_shape_explanation(selected_dataset_key, projection_available_model)
            render_note_box(note_title, note_copy)

            st.plotly_chart(
                build_forecast_chart(
                    history_df,
                    forecast_frames[projection_available_model],
                    value_col=selected_value_col,
                    title=f"{selected_meta['chart_title']} forecast path for {country_name} ({country_code})",
                    y_label=selected_meta["display"],
                    forecast_label=f"Forecast ({projection_available_model})",
                ),
                use_container_width=True,
            )
        else:
            spread_raw = float(compare_summary_df["_target_value_raw"].max() - compare_summary_df["_target_value_raw"].min())
            forecast_cols = st.columns(5)
            with forecast_cols[0]:
                render_metric_card(
                    "Forecast display mode",
                    "Compare models",
                    f"{selected_meta['label']} paths are overlaid for the same country and target year",
                )
            with forecast_cols[1]:
                render_metric_card(
                    "Compared models",
                    str(len(compare_summary_df)),
                    f"Saved {selected_meta['label'].lower()} time-series models in the repository",
                )
            with forecast_cols[2]:
                render_metric_card(
                    "Best backtest model",
                    benchmark_available_model,
                    f"Repository winner for {selected_meta['label'].lower()}",
                )
            with forecast_cols[3]:
                render_metric_card(
                    "Future projection model",
                    projection_available_model,
                    f"Primary long-horizon model used in Best model only mode",
                )
            with forecast_cols[4]:
                render_metric_card(
                    "Forecast spread",
                    _format_indicator_change(selected_dataset_key, spread_raw),
                    "Gap between the highest and lowest target-year model outputs",
                )

            st.caption(
                "Compare mode overlays all saved time-series models for the selected indicator. "
                "Use this to compare the benchmark winner against the model used for multi-step future projection."
            )

            if projection_available_model != benchmark_available_model:
                render_note_box(
                    "How should Compare mode be interpreted?",
                    f"{benchmark_available_model} is the best backtest model, while {projection_available_model} is the deployed future projection model. "
                    "Overlaying both helps show why a benchmark winner can differ from the model chosen for long-horizon forecasting.",
                )
            else:
                compare_note_title, compare_note_copy = _forecast_shape_explanation(
                    selected_dataset_key,
                    projection_available_model,
                )
                render_note_box(compare_note_title, compare_note_copy)

            st.plotly_chart(
                build_multi_forecast_chart(
                    history_df,
                    forecast_frames,
                    value_col=selected_value_col,
                    title=f"{selected_meta['chart_title']} forecast comparison for {country_name} ({country_code})",
                    y_label=selected_meta["display"],
                    best_model_name=projection_available_model,
                ),
                use_container_width=True,
            )

            display_summary_df = compare_summary_df.drop(columns=["_target_value_raw"]).copy()
            for col in ["Forecast growth (%)", "CAGR (%)", "Test MAPE (%)"]:
                if col in display_summary_df.columns:
                    display_summary_df[col] = display_summary_df[col].apply(format_percent)
            if "Test RMSE" in display_summary_df.columns:
                if selected_dataset_key == "gdp":
                    display_summary_df["Test RMSE"] = display_summary_df["Test RMSE"].apply(format_currency)
                elif selected_dataset_key == "population":
                    display_summary_df["Test RMSE"] = display_summary_df["Test RMSE"].apply(format_integer)
                else:
                    display_summary_df["Test RMSE"] = display_summary_df["Test RMSE"].apply(format_years)
            st.dataframe(
                display_summary_df,
                use_container_width=True,
                hide_index=True,
            )

        if forecast_errors:
            error_lines = [f"{model}: {message}" for model, message in forecast_errors.items()]
            st.info("Some model paths could not be generated.\n\n" + "\n".join(error_lines))

    render_note_box(
        "How this time-series future path works",
        f"The interactive forecast above uses a rolling {FORECAST_WINDOW_YEARS}-year history and the best-performing "
        f"{selected_meta['label'].lower()} time-series models saved in the repository. "
        f"The repository benchmark winner is {selected_best_model}, while Best model only mode may use a different deployment model "
        f"({selected_projection_model}) for more plausible long-horizon forecasting. Compare models mode overlays all saved model paths on the same future horizon.",
    )

    st.markdown("### Stage 2 — Main GDP prediction models")

    render_note_box(
        "How Stage 1 connects to Stage 2",
        (
            "Stage 1 forecasts the individual drivers required for future scenarios. "
            "Stage 2 then combines lagged GDP with demographic, macroeconomic, regional, "
            "and structural information to predict next-year GDP per capita."
        ),
    )

    render_pipeline(
        [
            "Forecast future indicator values",
            "Construct next-year GDP inputs",
            "Apply Model 1, Model 2, and Model 3",
            "Compare shared-holdout performance",
            "Generate future GDP scenarios",
        ]
    )

    st.markdown("#### Three progressive GDP prediction specifications")
    model_cols = st.columns(3, gap="large")

    with model_cols[0]:
        render_metric_card(
            "Model 1 — Baseline Dynamic",
            "OLS",
            (
                "Lagged GDP + Population + Life Expectancy. "
                "Purpose: establish the simplest interpretable benchmark."
            ),
        )

    with model_cols[1]:
        render_metric_card(
            "Model 2 — Extended Dynamic",
            "Elastic Net",
            (
                "Model 1 inputs + Inflation + Unemployment + Internet Usage. "
                "Purpose: test whether macroeconomic and digital indicators improve prediction."
            ),
        )

    with model_cols[2]:
        render_metric_card(
            "Model 3 — Full Dynamic",
            "OLS",
            (
                "Model 2 inputs + Region Effects + Event Dummies + Year Trend. "
                "Purpose: capture structural differences and major global shocks."
            ),
        )

    render_note_box(
        "Why are variables added progressively?",
        (
            "The models are intentionally nested from simple to complex. This allows the project "
            "to test whether each additional information layer—demographic, macroeconomic, digital, "
            "regional, and structural—adds useful predictive or explanatory value without losing "
            "the interpretability of the baseline comparison."
        ),
    )

    indicator_role_df = pd.DataFrame(
        [
            {
                "Indicator": "Lagged GDP per capita",
                "Reason for inclusion": (
                    "Captures persistence in a country's recent economic level."
                ),
            },
            {
                "Indicator": "Population",
                "Reason for inclusion": (
                    "Represents demographic scale, labour supply, and market size."
                ),
            },
            {
                "Indicator": "Life expectancy",
                "Reason for inclusion": (
                    "Acts as a proxy for long-run human development and population health."
                ),
            },
            {
                "Indicator": "Inflation",
                "Reason for inclusion": (
                    "Captures macroeconomic price stability and purchasing-power pressure."
                ),
            },
            {
                "Indicator": "Unemployment",
                "Reason for inclusion": (
                    "Reflects labour-market utilisation and short-run economic conditions."
                ),
            },
            {
                "Indicator": "Internet usage",
                "Reason for inclusion": (
                    "Provides a proxy for digital adoption, connectivity, and productive capacity."
                ),
            },
            {
                "Indicator": "Region effects",
                "Reason for inclusion": (
                    "Control for persistent structural differences between World Bank regions."
                ),
            },
            {
                "Indicator": "Event dummies",
                "Reason for inclusion": (
                    "Represent major global shocks that may shift GDP beyond normal trends."
                ),
            },
            {
                "Indicator": "Year trend",
                "Reason for inclusion": (
                    "Captures broad long-run movement not explained by the observed predictors."
                ),
            },
        ]
    )

    with st.expander(
        "Why are these indicators included in the GDP models?",
        expanded=False,
    ):
        st.dataframe(
            indicator_role_df,
            use_container_width=True,
            hide_index=True,
        )

    render_note_box(
        "Why OLS for Models 1 and 3, but Elastic Net for Model 2?",
        (
            "OLS is used for Models 1 and 3 because direct coefficient interpretation is central "
            "to the baseline and full structural specifications. Elastic Net is used for Model 2 "
            "because its regularisation can stabilise prediction when correlated macroeconomic "
            "and digital predictors are introduced together."
        ),
    )

    render_note_box(
        "Transition to model evaluation",
        (
            "The next benchmark compares fifteen model-and-algorithm configurations on the same "
            "chronological holdout. This ensures that the final decision is based on common "
            "out-of-sample evidence rather than model complexity alone."
        ),
    )

    st.markdown("#### Shared-holdout model benchmark")
    st.caption(
        "Fifteen GDP model configurations were evaluated on the same chronological "
        "holdout sample. Panel A ranks configurations by RMSE, while Panel B compares "
        "MAE, MAPE, and R² for the same ordering."
    )

    if GDP_BENCHMARK_FIG_PATH.exists():
        st.image(
            str(GDP_BENCHMARK_FIG_PATH),
            caption="Common-holdout benchmark across fifteen GDP model configurations",
            use_container_width=True,
        )

        render_note_box(
            "Benchmark interpretation",
            (
                "Model 2 + Elastic Net achieves the lowest holdout RMSE and the highest "
                "R², making it the strongest predictive configuration. Model 3 + OLS "
                "achieves the lowest MAE and MAPE while retaining direct coefficient, "
                "regional-effect, and event-dummy interpretation. The project therefore "
                "reports separate predictive and explanatory winners."
            ),
        )
    main_test_table = build_model_test_table(main_metrics_df)
    best_main = get_best_main_model(main_metrics_df)
    if best_main is not None:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card("Best benchmark model", str(best_main["model"]), "Selected by test RMSE, then MAPE, then MAE")
        with metric_cols[1]:
            render_metric_card("Test MAE", format_currency(best_main["MAE"]), "GDP per capita test error")
        with metric_cols[2]:
            render_metric_card("Test RMSE", format_currency(best_main["RMSE"]), "GDP per capita test error")
        with metric_cols[3]:
            render_metric_card("Test MAPE", format_percent(best_main["MAPE_pct"]), f"R²: {best_main['R_squared']:.4f}")

    if not main_test_table.empty:
        formatted_table = main_test_table[["model", "MAE", "RMSE", "MAPE_pct", "R_squared", "n_obs"]].rename(
            columns={
                "model": "Model",
                "MAE": "MAE",
                "RMSE": "RMSE",
                "MAPE_pct": "MAPE (%)",
                "R_squared": "R²",
                "n_obs": "Test observations",
            }
        )
        st.dataframe(formatted_table, use_container_width=True, hide_index=True)

    st.plotly_chart(build_main_model_yearly_chart(main_yearly_df), use_container_width=True)

    st.markdown("### Future GDP scenarios from the three main specifications")
    render_note_box(
        "Evaluation evidence versus future scenario",
        (
            "The shared-holdout results above are the formal evidence of predictive performance. "
            "The future paths below are forward-looking scenario demonstrations that depend on "
            "recursively forecasted GDP and driver values; they should not be interpreted as "
            "certainty or as replacements for the test-set results."
        ),
    )
    st.caption(
        "This scenario extends GDP per capita beyond the observed panel. Each future GDP year uses the previous year's GDP "
        "plus forecasted drivers from the feature year immediately before it."
    )

    required_future_cols = ["gdp_per_capita_usd", *MAIN_DYNAMIC_FEATURE_MODELS.keys()]
    future_country_summary = (
        panel_df.groupby(["country_name", "country_code", "wb_region"], as_index=False)
        .agg(
            **{
                f"{col}_obs": (col, lambda s: int(s.notna().sum()))
                for col in required_future_cols
            }
        )
        .copy()
    )
    future_obs_cols = [f"{col}_obs" for col in required_future_cols]
    future_country_summary["min_required_obs"] = future_country_summary[future_obs_cols].min(axis=1)
    eligible_future_countries = future_country_summary[
        future_country_summary["min_required_obs"] >= FORECAST_WINDOW_YEARS
    ].copy()

    if eligible_future_countries.empty:
        st.warning(
            "No country has enough GDP and driver history to generate the future GDP scenario for the three main models."
        )
    else:
        eligible_future_countries["label"] = (
            eligible_future_countries["country_name"]
            + " ("
            + eligible_future_countries["country_code"]
            + ")"
        )
        eligible_future_countries = eligible_future_countries.sort_values("country_name").reset_index(drop=True)

        future_default_index = 0
        if "VNM" in eligible_future_countries["country_code"].values:
            future_default_index = int(
                eligible_future_countries.index[eligible_future_countries["country_code"] == "VNM"][0]
            )

        future_country_col, future_target_col = st.columns([1.3, 1.0], gap="large")
        with future_country_col:
            future_country_label = st.selectbox(
                "Select a country for future main-model GDP forecasting",
                eligible_future_countries["label"].tolist(),
                index=future_default_index,
                key="future_main_model_country_label",
            )

        future_country_row = eligible_future_countries[
            eligible_future_countries["label"] == future_country_label
        ].iloc[0]
        future_country_code = str(future_country_row["country_code"])
        future_country_name = str(future_country_row["country_name"])

        future_country_panel = panel_df[panel_df["country_code"] == future_country_code].copy()
        latest_actual_gdp_year = int(
            future_country_panel.loc[future_country_panel["gdp_per_capita_usd"].notna(), "year"].max()
        )
        max_future_target_year = latest_actual_gdp_year + 30
        default_future_target_year = min(
            latest_actual_gdp_year + 10,
            max_future_target_year,
        )

        with future_target_col:
            future_target_year = int(
                st.number_input(
                    "Enter a future GDP target year",
                    min_value=latest_actual_gdp_year + 1,
                    max_value=max_future_target_year,
                    value=default_future_target_year,
                    step=1,
                    key="future_main_model_target_year",
                    help=(
                        "Enter a year after the latest observed GDP year. "
                        "The dashboard recursively forecasts all required inputs "
                        "and produces GDP-per-capita scenarios from Models 1, 2, and 3."
                    ),
                )
            )
            st.caption(
                f"Available range: {latest_actual_gdp_year + 1}-"
                f"{max_future_target_year}. Longer horizons carry greater uncertainty."
            )

        try:
            future_forecast_map, future_summary_df, future_meta = _build_main_dynamic_future_forecasts(
                panel_df,
                country_code=future_country_code,
                target_year=future_target_year,
            )

            # Model 3 is the final selected model of the project.
            # Model 2 + Elastic Net remains the strongest RMSE-based benchmark,
            # but the future scenario layer highlights Model 3 because it combines
            # competitive accuracy with full structural interpretability.
            best_main_model_name = FINAL_SELECTED_MAIN_MODEL

            future_history_df = (
                future_country_panel[["year", "gdp_per_capita_usd"]]
                .dropna()
                .drop_duplicates(subset=["year"])
                .sort_values("year")
                .copy()
            )

            future_spread = (
                float(future_summary_df["_target_value_raw"].max())
                - float(future_summary_df["_target_value_raw"].min())
            )

            summary_by_model = future_summary_df.set_index("Model")
            selected_model_row = summary_by_model.loc[FINAL_SELECTED_MAIN_MODEL]

            context_cols = st.columns(4)
            with context_cols[0]:
                render_metric_card(
                    "Selected country",
                    future_country_code,
                    f"{future_country_name} | {future_meta['wb_region']}",
                )
            with context_cols[1]:
                render_metric_card(
                    "Latest actual GDP year",
                    str(future_meta["latest_actual_gdp_year"]),
                    format_currency(future_meta["latest_actual_gdp_value"]),
                )
            with context_cols[2]:
                render_metric_card(
                    "Target forecast year",
                    str(future_target_year),
                    f"First forecast year: {future_meta['first_future_gdp_year']}",
                )
            with context_cols[3]:
                render_metric_card(
                    "Model spread",
                    format_currency(future_spread),
                    "Highest minus lowest target-year scenario",
                )

            st.markdown(f"#### GDP per capita forecasts for {future_target_year}")
            forecast_cols = st.columns(3)

            ordered_models = [
                "Model 1 - Baseline Dynamic",
                "Model 2 - Extended Dynamic",
                "Model 3 - Full Dynamic",
            ]

            for col, model_name in zip(forecast_cols, ordered_models):
                model_row = summary_by_model.loc[model_name]
                with col:
                    is_selected = model_name == FINAL_SELECTED_MAIN_MODEL
                    render_metric_card(
                        model_name,
                        str(model_row["Forecast in target year"]),
                        (
                            "Final selected model: lowest MAE and MAPE with "
                            "full structural interpretability"
                            if is_selected
                            else (
                                "Strongest RMSE-based predictive benchmark"
                                if model_name == "Model 2 - Extended Dynamic"
                                else "Baseline scenario for comparison"
                            )
                        ),
                        delta="FINAL SELECTED MODEL" if is_selected else None,
                        tone="positive" if is_selected else "neutral",
                    )

            render_note_box(
                "Final model selection",
                (
                    f"Model 3 - Full Dynamic is the final selected model. Its "
                    f"{future_target_year} GDP-per-capita forecast for "
                    f"{future_country_name} is "
                    f"{selected_model_row['Forecast in target year']}. "
                    "Model 3 is highlighted because it records the lowest MAE and "
                    "MAPE while retaining direct interpretation of regional effects, "
                    "event dummies, macroeconomic drivers, and the time trend. "
                    "Model 2 + Elastic Net remains the strongest RMSE-based benchmark."
                ),
            )

            st.plotly_chart(
                build_multi_forecast_chart(
                    future_history_df,
                    future_forecast_map,
                    value_col="gdp_per_capita_usd",
                    title=(
                        f"Future GDP per capita comparison from the three main models for "
                        f"{future_country_name} ({future_country_code})"
                    ),
                    y_label="GDP per Capita (current US$)",
                    best_model_name=best_main_model_name,
                ),
                use_container_width=True,
            )

            display_future_summary = (
                future_summary_df.set_index("Model")
                .reindex(
                    [
                        "Model 1 - Baseline Dynamic",
                        "Model 2 - Extended Dynamic",
                        "Model 3 - Full Dynamic",
                    ]
                )
                .reset_index()
            )
            display_future_summary["Role"] = [
                "Baseline comparison",
                "Best RMSE-based benchmark",
                "Final selected model",
            ]

            st.dataframe(
                display_future_summary.drop(columns=["_target_value_raw"]),
                use_container_width=True,
                hide_index=True,
            )

            feature_assumptions = future_meta["feature_models"]
            render_note_box(
                "How the future main-model GDP scenario is generated",
                (
                    f"For each future GDP year t, the dashboard feeds the main GDP model with GDP(t-1) plus forecasted "
                    f"drivers from year t-1. Driver models used here are: population via {feature_assumptions['population_total']}, "
                    f"life expectancy via {feature_assumptions['life_expectancy_years']}, inflation via "
                    f"{feature_assumptions['inflation_pct_clean']}, unemployment via "
                    f"{feature_assumptions['unemployment_pct_clean']}, and internet usage via "
                    f"{feature_assumptions['internet_users_pct_clean']}. The test-set metrics above still remain the formal "
                    "test-set evidence, while this block is the forward-looking scenario layer for demo and decision support."
                ),
            )

            with st.expander("Future main-model forecast details", expanded=False):
                detailed_future_rows = []
                for model_name, forecast_df in future_forecast_map.items():
                    if forecast_df.empty:
                        continue
                    temp = forecast_df.copy()
                    temp.insert(0, "Model", model_name)
                    temp["predicted_value"] = temp["predicted_value"].apply(format_currency)
                    temp["gdp_input_used"] = temp["gdp_input_used"].apply(format_currency)
                    temp["population_used"] = temp["population_used"].apply(format_integer)
                    temp["life_expectancy_used"] = temp["life_expectancy_used"].apply(format_years)
                    temp["inflation_used"] = temp["inflation_used"].apply(format_percent)
                    temp["unemployment_used"] = temp["unemployment_used"].apply(format_percent)
                    temp["internet_used"] = temp["internet_used"].apply(format_percent)
                    detailed_future_rows.append(temp)

                if detailed_future_rows:
                    detailed_future_df = pd.concat(detailed_future_rows, ignore_index=True).rename(
                        columns={
                            "forecast_year": "Forecast year",
                            "predicted_value": "Predicted GDP",
                            "feature_year": "Feature year used",
                            "gdp_input_used": "GDP input used",
                            "population_used": "Population used",
                            "life_expectancy_used": "Life expectancy used",
                            "inflation_used": "Inflation used",
                            "unemployment_used": "Unemployment used",
                            "internet_used": "Internet usage used",
                        }
                    )
                    st.dataframe(detailed_future_df, use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.warning(str(exc))

    with st.expander("How were the three main GDP models specified?", expanded=False):
        if main_specs_df.empty:
            st.info("No saved specification table was found in the repository.")
        else:
            st.dataframe(main_specs_df, use_container_width=True, hide_index=True)
        if EVENT_TIMELINE_PATH.exists():
            st.image(str(EVENT_TIMELINE_PATH), caption="Global event dummies used in the GDP panel models")

    with st.expander("Time-series model comparison figures", expanded=False):
        figure_cols = st.columns(3, gap="large")
        for col, fig_path, caption in zip(
            figure_cols,
            [GDP_TS_FIG_PATH, LIFE_TS_FIG_PATH, POP_TS_FIG_PATH],
            ["GDP time-series comparison", "Life expectancy time-series comparison", "Population time-series comparison"],
        ):
            with col:
                if fig_path.exists():
                    st.image(str(fig_path), caption=caption)
                else:
                    st.info(f"{caption} figure is not available.")

    if not benchmark_df.empty:
        with st.expander("Algorithm benchmark across saved GDP configurations", expanded=False):
            benchmark_table = benchmark_df[
                (benchmark_df["split"].astype(str).str.lower() == "test")
                & (benchmark_df["scale"].astype(str).str.lower() == "level_gdp_usd")
            ][["model", "MAE", "RMSE", "MAPE_pct", "R_squared", "n_obs"]].rename(
                columns={
                    "model": "Model",
                    "MAE": "MAE",
                    "RMSE": "RMSE",
                    "MAPE_pct": "MAPE (%)",
                    "R_squared": "R²",
                    "n_obs": "Test observations",
                }
            )
            st.dataframe(benchmark_table, use_container_width=True, hide_index=True)
            st.caption(
                "These saved test results compare the available algorithm configurations on the "
                "GDP prediction task. Interpret them together with specification logic, data coverage, "
                "and model transparency rather than selecting a model on complexity alone."
            )

    render_insight_box(
        "Forecasting interpretation",
        [
            (
                f"Rolling backtesting identifies {gdp_best_model} for GDP per capita, "
                f"{life_best_model} for life expectancy, and {pop_best_model} for population "
                "within the saved indicator-level results."
            ),
            (
                "The best one-step backtest model is not automatically the most suitable "
                "model for a recursive multi-year projection."
            ),
            (
                "The main GDP benchmark evaluates predictive accuracy on a shared holdout, "
                "while richer specifications are also assessed for interpretability and "
                "structural relevance."
            ),
            (
                "Future GDP outputs are scenario paths conditional on forecasted inputs and "
                "historical model structure, not guaranteed economic outcomes."
            ),
        ],
    )

    render_note_box(
        "Next stage",
        (
            "The final page consolidates the empirical findings from EDA, country comparison, "
            "relationship analysis, time-series forecasting, and the main GDP models. It also "
            "states the project's limitations, critical evaluation, and future improvements."
        ),
    )