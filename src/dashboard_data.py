from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "Data" / "Raw"
CLEANED_DIR = BASE_DIR / "Data" / "Cleaned"
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"

PROJECT_TITLE = "Global GDP per Capita Forecasting and Economic Drivers"
DASHBOARD_TITLE = "Global GDP Trends and Forecasting Dashboard"
DASHBOARD_SUBTITLE = (
    "Exploring economic growth, demographic relationships and future GDP patterns "
    "across countries."
)

PANEL_PATH = CLEANED_DIR / "panel_with_event_dummies_and_extra_drivers.csv"
CLEANED_CORE_PATH = CLEANED_DIR / "cleaned_data.csv"
MAIN_METRICS_PATH = CLEANED_DIR / "gdp_main_models_rebuilt_with_lag_metrics.csv"
MAIN_YEARLY_PATH = CLEANED_DIR / "gdp_main_models_rebuilt_with_lag_yearly_summary.csv"
MAIN_BIAS_PATH = CLEANED_DIR / "gdp_main_models_rebuilt_with_lag_bias_summary.csv"
MAIN_TEST_PRED_PATH = CLEANED_DIR / "gdp_main_models_rebuilt_with_lag_test_predictions.csv"
MAIN_SPEC_PATH = CLEANED_DIR / "gdp_main_models_rebuilt_with_lag_specifications.csv"

GDP_TS_SUMMARY_PATH = CLEANED_DIR / "gdp_time_series_model_selection_summary_10y.csv"
LIFE_TS_SUMMARY_PATH = CLEANED_DIR / "life_time_series_model_selection_summary_10y.csv"
POP_TS_SUMMARY_PATH = CLEANED_DIR / "population_time_series_model_selection_summary_10y.csv"

GDP_TS_BEST_PATH = CLEANED_DIR / "gdp_time_series_best_model_10y.csv"
LIFE_TS_BEST_PATH = CLEANED_DIR / "life_time_series_best_model_10y.csv"
POP_TS_BEST_PATH = CLEANED_DIR / "population_time_series_best_model_10y.csv"

RF_MODEL_3_METRICS_PATH = CLEANED_DIR / "gdp_random_forest_model_3_metrics.csv"
XGB_MODEL_3_METRICS_PATH = CLEANED_DIR / "gdp_xgboost_model_3_metrics.csv"

EVENT_TIMELINE_PATH = FIG_DIR / "figure_4_3_global_event_dummies_timeline.png"
GDP_TS_FIG_PATH = FIG_DIR / "figure_5_1_gdp_ts_model_comparison.png"
LIFE_TS_FIG_PATH = FIG_DIR / "figure_5_2_life_ts_model_comparison.png"
POP_TS_FIG_PATH = FIG_DIR / "figure_5_3_population_ts_model_comparison.png"

GDP_MAP_1960_PATH = OUTPUT_DIR / "gdp per capita by year (1960).png"
GDP_MAP_2023_PATH = OUTPUT_DIR / "gdp per capita by year (2023).png"
LIFE_MAP_1960_PATH = OUTPUT_DIR / "life_expectancy heatmap 1960.png"
LIFE_MAP_2023_PATH = OUTPUT_DIR / "life_expectancy heatmap 2023.png"
POP_MAP_1960_PATH = OUTPUT_DIR / "population heatmap 1960.png"
POP_MAP_2023_PATH = OUTPUT_DIR / "population heatmap 2023.png"


def resolve_existing_path(*candidates: Path) -> Path:
    """Return the first path that exists."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


RAW_SOURCE_CONFIG = {
    "gdp": {
        "path": RAW_DIR / "gdp.csv",
        "display_name": "GDP per Capita",
        "role": "Primary dataset",
        "value_col": "gdp_per_capita_usd",
        "unit_label": "current US$",
    },
    "population": {
        "path": RAW_DIR / "population.csv",
        "display_name": "Population",
        "role": "Supporting dataset",
        "value_col": "population_total",
        "unit_label": "people",
    },
    "life_expectancy": {
        "path": RAW_DIR / "life_expectancy.csv",
        "display_name": "Life Expectancy",
        "role": "Supporting dataset",
        "value_col": "life_expectancy_years",
        "unit_label": "years",
    },
}


STATIC_IMAGES = {
    "gdp_maps": [GDP_MAP_1960_PATH, GDP_MAP_2023_PATH],
    "life_maps": [LIFE_MAP_1960_PATH, LIFE_MAP_2023_PATH],
    "population_maps": [POP_MAP_1960_PATH, POP_MAP_2023_PATH],
    "event_timeline": EVENT_TIMELINE_PATH,
    "ts_model_figures": [GDP_TS_FIG_PATH, LIFE_TS_FIG_PATH, POP_TS_FIG_PATH],
}


MAIN_MODEL_REQUIRED_COLUMNS = [
    "gdp_per_capita_usd",
    "life_expectancy_years",
    "population_total",
    "log_gdp_per_capita",
    "log_population_total",
    "target_log_gdp_next_year",
    "inflation_pct_clean",
    "unemployment_pct_clean",
    "internet_users_pct_clean",
]


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[col for col in df.columns if str(col).startswith("Unnamed")],
        errors="ignore",
    )


def _year_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    """Load the final analytical panel used by the dashboard and main models."""
    df = pd.read_csv(PANEL_PATH)
    numeric_cols = [
        "year",
        "gdp_per_capita_usd",
        "life_expectancy_years",
        "population_total",
        "population_growth_pct",
        "log_gdp_per_capita",
        "log_population_total",
        "target_log_gdp_next_year",
        "inflation_pct",
        "unemployment_pct",
        "internet_users_pct",
        "inflation_pct_clean",
        "unemployment_pct_clean",
        "internet_users_pct_clean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["country_name", "country_code", "wb_region"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    valid_total_mask = (
        df["gdp_per_capita_usd"].notna()
        & df["population_total"].notna()
        & (df["population_total"] > 0)
    )
    df["estimated_total_gdp_usd"] = pd.NA
    df.loc[valid_total_mask, "estimated_total_gdp_usd"] = (
        df.loc[valid_total_mask, "gdp_per_capita_usd"]
        * df.loc[valid_total_mask, "population_total"]
    )
    df["estimated_total_gdp_usd"] = pd.to_numeric(df["estimated_total_gdp_usd"], errors="coerce")
    df["year_trend"] = df["year"] - int(df["year"].min()) + 1

    return df.sort_values(["country_name", "year"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_cleaned_core() -> pd.DataFrame:
    """Load the early merged core dataset retained in the repository."""
    if not CLEANED_CORE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLEANED_CORE_PATH)


@st.cache_data(show_spinner=False)
def load_optional_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV if it exists and return an empty frame otherwise."""
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_raw_source_wide(dataset_key: str) -> pd.DataFrame:
    """Load a World Bank raw source in its original wide layout."""
    config = RAW_SOURCE_CONFIG[dataset_key]
    df = pd.read_csv(config["path"], skiprows=4)
    return _drop_unnamed_columns(df)


@st.cache_data(show_spinner=False)
def load_raw_source_long(dataset_key: str) -> pd.DataFrame:
    """Convert a World Bank raw source from wide to long format."""
    config = RAW_SOURCE_CONFIG[dataset_key]
    wide_df = load_raw_source_wide(dataset_key)
    year_cols = _year_columns(wide_df)

    long_df = wide_df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        value_vars=year_cols,
        var_name="year",
        value_name=config["value_col"],
    )
    long_df = long_df.rename(
        columns={
            "Country Name": "country_name",
            "Country Code": "country_code",
            "Indicator Name": "indicator_name",
            "Indicator Code": "indicator_code",
        }
    )
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df[config["value_col"]] = pd.to_numeric(long_df[config["value_col"]], errors="coerce")
    long_df = long_df.dropna(subset=["year"]).copy()
    long_df["year"] = long_df["year"].astype(int)
    return long_df


@st.cache_data(show_spinner=False)
def summarize_raw_source(dataset_key: str) -> dict[str, object]:
    """Return a compact summary of a raw source dataset."""
    config = RAW_SOURCE_CONFIG[dataset_key]
    wide_df = load_raw_source_wide(dataset_key)
    long_df = load_raw_source_long(dataset_key)
    value_col = config["value_col"]

    indicator_name = None
    indicator_code = None
    if not wide_df.empty:
        indicator_name = wide_df["Indicator Name"].dropna().astype(str).iloc[0]
        indicator_code = wide_df["Indicator Code"].dropna().astype(str).iloc[0]

    valid_long = long_df.dropna(subset=[value_col]).copy()

    return {
        "dataset_key": dataset_key,
        "display_name": config["display_name"],
        "role": config["role"],
        "value_col": value_col,
        "indicator_name": indicator_name,
        "indicator_code": indicator_code,
        "wide_rows": int(len(wide_df)),
        "countries": int(wide_df["Country Code"].nunique()),
        "year_min": int(valid_long["year"].min()) if not valid_long.empty else None,
        "year_max": int(valid_long["year"].max()) if not valid_long.empty else None,
        "valid_observations": int(valid_long[value_col].notna().sum()),
        "missing_values": int(long_df[value_col].isna().sum()),
        "unit_label": config["unit_label"],
    }


@st.cache_data(show_spinner=False)
def load_main_model_metrics() -> pd.DataFrame:
    return load_optional_csv(MAIN_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_main_model_yearly() -> pd.DataFrame:
    return load_optional_csv(MAIN_YEARLY_PATH)


@st.cache_data(show_spinner=False)
def load_main_model_bias() -> pd.DataFrame:
    return load_optional_csv(MAIN_BIAS_PATH)


@st.cache_data(show_spinner=False)
def load_main_model_predictions() -> pd.DataFrame:
    return load_optional_csv(MAIN_TEST_PRED_PATH)


@st.cache_data(show_spinner=False)
def load_main_model_specifications() -> pd.DataFrame:
    return load_optional_csv(MAIN_SPEC_PATH)


@st.cache_data(show_spinner=False)
def load_ts_summary(dataset_key: str) -> pd.DataFrame:
    path_map = {
        "gdp": GDP_TS_SUMMARY_PATH,
        "life_expectancy": LIFE_TS_SUMMARY_PATH,
        "population": POP_TS_SUMMARY_PATH,
    }
    return load_optional_csv(path_map[dataset_key])


@st.cache_data(show_spinner=False)
def load_ts_best(dataset_key: str) -> pd.DataFrame:
    path_map = {
        "gdp": GDP_TS_BEST_PATH,
        "life_expectancy": LIFE_TS_BEST_PATH,
        "population": POP_TS_BEST_PATH,
    }
    return load_optional_csv(path_map[dataset_key])


@st.cache_data(show_spinner=False)
def load_benchmark_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rf_df = load_optional_csv(RF_MODEL_3_METRICS_PATH)
    xgb_df = load_optional_csv(XGB_MODEL_3_METRICS_PATH)
    if not rf_df.empty:
        frames.append(rf_df)
    if not xgb_df.empty:
        frames.append(xgb_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_main_model_sample() -> pd.DataFrame:
    """Return the listwise-complete sample used by the rebuilt main GDP models."""
    panel_df = load_panel()
    cols = [col for col in MAIN_MODEL_REQUIRED_COLUMNS if col in panel_df.columns]
    return panel_df.dropna(subset=cols).copy()

