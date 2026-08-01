from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(
    page_title="Population Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path("/Users/tonytony/Final Project")
DATA_DIR = BASE_DIR / "Data" / "Cleaned"

POP_PATH = DATA_DIR / "panel_with_event_dummies_and_extra_drivers.csv"
GDP_MAIN_MODEL_METRICS_PATH = DATA_DIR / "gdp_main_models_train_test_metrics.csv"
GDP_TS_BEST_PATH = DATA_DIR / "gdp_time_series_best_model_10y.csv"
LIFE_TS_BEST_PATH = DATA_DIR / "life_time_series_best_model_10y.csv"
POP_TS_BEST_PATH = DATA_DIR / "population_time_series_best_model_10y.csv"
GDP_TS_SUMMARY_PATH = DATA_DIR / "gdp_time_series_model_selection_summary_10y.csv"
LIFE_TS_SUMMARY_PATH = DATA_DIR / "life_time_series_model_selection_summary_10y.csv"
POP_TS_SUMMARY_PATH = DATA_DIR / "population_time_series_model_selection_summary_10y.csv"
GDP_TS_PRED_PATH = DATA_DIR / "gdp_time_series_model_selection_predictions_10y.csv"
LIFE_TS_PRED_PATH = DATA_DIR / "life_time_series_model_selection_predictions_10y.csv"
POP_TS_PRED_PATH = DATA_DIR / "population_time_series_model_selection_predictions_10y.csv"

FORECAST_CANDIDATES = [
    DATA_DIR / "future_forecasts_by_dataset.csv",
    DATA_DIR / "population_future_forecasts.csv",
]

EVENT_TIMELINE_FIGURE_PATH = BASE_DIR / "output" / "figures" / "figure_4_3_global_event_dummies_timeline.png"

INDICATOR_CONFIG = {
    "Population": {
        "column": "population_total",
        "label": "Population Total",
        "format": "{:,.0f}",
        "hover_format": ":,.0f",
        "color": "#1f7a5c",
        "map_scale": "Tealgrn",
    },
    "GDP per Capita": {
        "column": "gdp_per_capita_usd",
        "label": "GDP per Capita (US$)",
        "format": "{:,.2f}",
        "hover_format": ":,.2f",
        "color": "#d98e04",
        "map_scale": "YlOrBr",
    },
    "Life Expectancy": {
        "column": "life_expectancy_years",
        "label": "Life Expectancy (years)",
        "format": "{:,.2f}",
        "hover_format": ":,.2f",
        "color": "#3b82f6",
        "map_scale": "Viridis",
    },
}

FORECAST_WINDOW_YEARS = 10

LIVE_FORECAST_MODEL_BY_DATASET = {
    "GDP": "ARIMA",
    "Life Expectancy": "Holt",
    "Population": "LogHolt",
}

TIME_SERIES_BACKTEST_CONFIG = {
    "GDP": {
        "column": "gdp_per_capita_usd",
        "label": "GDP per Capita (US$)",
        "models": ["Naive", "ARIMA", "AutoReg"],
    },
    "Life Expectancy": {
        "column": "life_expectancy_years",
        "label": "Life Expectancy (years)",
        "models": ["Naive", "Holt", "AutoReg"],
    },
    "Population": {
        "column": "population_total",
        "label": "Population Total",
        "models": ["LogHolt", "Naive", "AutoReg"],
    },
}

# -------------------------------------------------------------------
# Styles
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #f6f3ec;
            --surface: rgba(255,255,255,0.78);
            --surface-strong: rgba(255,255,255,0.92);
            --text: #1f2a24;
            --muted: #5d6b63;
            --accent: #1f7a5c;
            --accent-2: #d98e04;
            --line: rgba(31, 42, 36, 0.08);
            --shadow: 0 16px 48px rgba(35, 48, 40, 0.10);
            --radius: 22px;
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(217, 142, 4, 0.10), transparent 28%),
                radial-gradient(circle at 100% 0%, rgba(31, 122, 92, 0.14), transparent 22%),
                linear-gradient(180deg, #f9f6ef 0%, #f3efe6 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,255,255,0.68)),
                linear-gradient(135deg, #dcefe8, #f7edd4);
            border: 1px solid rgba(31, 42, 36, 0.06);
            border-radius: 28px;
            padding: 28px 30px 24px 30px;
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            color: #1f7a5c;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 800;
            color: #142019;
            margin-bottom: 0.55rem;
        }

        .hero-copy {
            font-size: 1.02rem;
            color: #516058;
            max-width: 900px;
            line-height: 1.65;
        }

        .section-card {
            background: var(--surface);
            border: 1px solid rgba(31, 42, 36, 0.06);
            border-radius: var(--radius);
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 36px rgba(35, 48, 40, 0.06);
        }

        .metric-card {
            background: var(--surface-strong);
            border: 1px solid rgba(31, 42, 36, 0.06);
            border-radius: 20px;
            padding: 16px 18px;
            box-shadow: 0 10px 30px rgba(35, 48, 40, 0.06);
        }

        .metric-label {
            color: #66756d;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            color: #17241d;
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .metric-sub {
            color: #6d7a73;
            font-size: 0.86rem;
            margin-top: 0.2rem;
        }

        .subtle-note {
            color: #5d6b63;
            font-size: 0.93rem;
            line-height: 1.6;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.72);
            border-radius: 999px;
            padding: 10px 16px;
            border: 1px solid rgba(31, 42, 36, 0.06);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1f7a5c, #2c8e6d);
            color: white;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.94), rgba(246,243,236,0.96));
            border-right: 1px solid rgba(31, 42, 36, 0.06);
        }

        .small-pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(31, 122, 92, 0.10);
            color: #1f7a5c;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
@st.cache_data
def load_population_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "year",
        "population_total",
        "population_growth_pct",
        "gdp_per_capita_usd",
        "life_expectancy_years",
        "inflation_pct",
        "unemployment_pct",
        "internet_users_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_forecast_data(candidates: list[Path]) -> pd.DataFrame:
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


pop_df = load_population_data(POP_PATH)
forecast_df = load_forecast_data(FORECAST_CANDIDATES)
gdp_main_model_metrics_df = load_optional_csv(GDP_MAIN_MODEL_METRICS_PATH)
gdp_ts_best_df = load_optional_csv(GDP_TS_BEST_PATH)
life_ts_best_df = load_optional_csv(LIFE_TS_BEST_PATH)
pop_ts_best_df = load_optional_csv(POP_TS_BEST_PATH)
gdp_ts_summary_df = load_optional_csv(GDP_TS_SUMMARY_PATH)
life_ts_summary_df = load_optional_csv(LIFE_TS_SUMMARY_PATH)
pop_ts_summary_df = load_optional_csv(POP_TS_SUMMARY_PATH)
gdp_ts_pred_df = load_optional_csv(GDP_TS_PRED_PATH)
life_ts_pred_df = load_optional_csv(LIFE_TS_PRED_PATH)
pop_ts_pred_df = load_optional_csv(POP_TS_PRED_PATH)

# -------------------------------------------------------------------
# Safety checks
# -------------------------------------------------------------------
required_cols = {"country_name", "country_code", "wb_region", "year", "population_total"}
missing_cols = required_cols - set(pop_df.columns)
if missing_cols:
    st.error(f"Population file is missing required columns: {sorted(missing_cols)}")
    st.stop()

available_indicators = {
    label: config
    for label, config in INDICATOR_CONFIG.items()
    if config["column"] in pop_df.columns
}
if not available_indicators:
    st.error("No supported indicator columns were found in the panel dataset.")
    st.stop()

TS_DATASET_BY_INDICATOR = {
    "Population": "Population",
    "GDP per Capita": "GDP",
    "Life Expectancy": "Life Expectancy",
}

# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.markdown("## Dashboard Controls")

region_options = sorted(pop_df["wb_region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Region",
    options=region_options,
    default=region_options[:3] if len(region_options) >= 3 else region_options,
)

region_filtered_df = (
    pop_df[pop_df["wb_region"].isin(selected_regions)].copy()
    if selected_regions else pop_df.copy()
)

country_options = sorted(region_filtered_df["country_name"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country",
    options=country_options,
    default=country_options[:4] if len(country_options) >= 4 else country_options,
)

year_min = int(pop_df["year"].min())
year_max = int(pop_df["year"].max())

selected_year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(max(year_min, year_max - 20), year_max),
)

selected_indicator = st.sidebar.selectbox(
    "Indicator",
    options=list(available_indicators.keys()),
    index=list(available_indicators.keys()).index("GDP per Capita") if "GDP per Capita" in available_indicators else 0,
)

chart_template = st.sidebar.selectbox(
    "Chart theme",
    options=["plotly_white", "ggplot2", "simple_white"],
    index=0,
)

compare_year = st.sidebar.selectbox(
    "Comparison year",
    options=sorted(pop_df["year"].dropna().astype(int).unique().tolist()),
    index=len(sorted(pop_df["year"].dropna().astype(int).unique().tolist())) - 1,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="subtle-note">
    This dashboard now combines project overview, time-series model results,
    indicator exploration for population, GDP, and life expectancy, plus
    comparison and forecast-related outputs.
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Derived data
# -------------------------------------------------------------------
filtered_df = region_filtered_df[
    region_filtered_df["year"].between(selected_year_range[0], selected_year_range[1])
].copy()

if selected_countries:
    filtered_df = filtered_df[filtered_df["country_name"].isin(selected_countries)].copy()

selected_indicator_cfg = available_indicators[selected_indicator]
selected_value_col = selected_indicator_cfg["column"]
selected_value_label = selected_indicator_cfg["label"]
selected_value_color = selected_indicator_cfg["color"]
selected_ts_dataset = TS_DATASET_BY_INDICATOR.get(selected_indicator, "Population")

latest_year_df = pop_df[pop_df["year"] == pop_df["year"].max()].copy()
n_countries = pop_df["country_code"].nunique()
n_regions = pop_df["wb_region"].nunique()

ts_summary_df = pd.concat(
    [
        gdp_ts_summary_df.assign(dataset_label="GDP") if not gdp_ts_summary_df.empty else pd.DataFrame(),
        life_ts_summary_df.assign(dataset_label="Life Expectancy") if not life_ts_summary_df.empty else pd.DataFrame(),
        pop_ts_summary_df.assign(dataset_label="Population") if not pop_ts_summary_df.empty else pd.DataFrame(),
    ],
    ignore_index=True,
)

ts_best_df = pd.concat(
    [
        gdp_ts_best_df.assign(dataset_label="GDP") if not gdp_ts_best_df.empty else pd.DataFrame(),
        life_ts_best_df.assign(dataset_label="Life Expectancy") if not life_ts_best_df.empty else pd.DataFrame(),
        pop_ts_best_df.assign(dataset_label="Population") if not pop_ts_best_df.empty else pd.DataFrame(),
    ],
    ignore_index=True,
)

ts_prediction_map = {
    "GDP": gdp_ts_pred_df,
    "Life Expectancy": life_ts_pred_df,
    "Population": pop_ts_pred_df,
}

overview_latest_year = int(latest_year_df["year"].max())
overview_gdp_region_latest = (
    latest_year_df.dropna(subset=["gdp_per_capita_usd"])
    .groupby("wb_region", as_index=False)["gdp_per_capita_usd"]
    .mean()
    .sort_values("gdp_per_capita_usd", ascending=False)
)

overview_dataset_roles = {
    "GDP per Capita": "Primary forecast target for the main panel models.",
    "Population": "Core demographic driver representing market size and labour scale.",
    "Life Expectancy": "Core development driver used as a proxy for human capital and productivity conditions.",
}
overview_core_dataset_cards = []
for dataset_label in ["GDP per Capita", "Population", "Life Expectancy"]:
    dataset_col = available_indicators[dataset_label]["column"]
    dataset_temp = pop_df.dropna(subset=[dataset_col]).copy()
    overview_core_dataset_cards.append(
        {
            "label": dataset_label,
            "column": dataset_col,
            "coverage": int(len(dataset_temp)),
            "countries": int(dataset_temp["country_code"].nunique()),
            "year_min": int(dataset_temp["year"].min()),
            "year_max": int(dataset_temp["year"].max()),
            "role": overview_dataset_roles[dataset_label],
        }
    )

overview_extended_required_cols = [
    col for col in [
        "gdp_per_capita_usd",
        "population_total",
        "life_expectancy_years",
        "inflation_pct",
        "unemployment_pct",
        "internet_users_pct",
    ] if col in pop_df.columns
]
overview_extended_model_df = (
    pop_df.dropna(subset=overview_extended_required_cols).copy()
    if overview_extended_required_cols
    else pd.DataFrame()
)
overview_extended_obs = int(len(overview_extended_model_df))
overview_extended_countries = int(overview_extended_model_df["country_code"].nunique()) if not overview_extended_model_df.empty else 0
overview_extended_year_min = int(overview_extended_model_df["year"].min()) if not overview_extended_model_df.empty else None
overview_extended_year_max = int(overview_extended_model_df["year"].max()) if not overview_extended_model_df.empty else None
overview_sample_retention_pct = (overview_extended_obs / len(pop_df) * 100) if len(pop_df) else np.nan
overview_country_retention_pct = (overview_extended_countries / n_countries * 100) if n_countries else np.nan

overview_best_panel_model_row = None
if not gdp_main_model_metrics_df.empty:
    overview_panel_candidates = gdp_main_model_metrics_df[
        (gdp_main_model_metrics_df["split"].astype(str).str.lower() == "test")
        & (gdp_main_model_metrics_df["scale"].astype(str).str.lower() == "log_gdp")
    ].copy()
    if not overview_panel_candidates.empty:
        overview_best_panel_model_row = overview_panel_candidates.sort_values(
            ["RMSE", "MAE", "MAPE_pct"],
            ascending=[True, True, True],
        ).iloc[0]


def format_metric(value, digits=4):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def format_indicator_value(value, indicator_label: str) -> str:
    if pd.isna(value):
        return "N/A"
    return available_indicators[indicator_label]["format"].format(value)


def format_signed_indicator_delta(value, indicator_label: str) -> str:
    if pd.isna(value):
        return "N/A"
    abs_text = available_indicators[indicator_label]["format"].format(abs(value))
    if value > 0:
        return f"+{abs_text}"
    if value < 0:
        return f"-{abs_text}"
    return abs_text


def get_indicator_agg(indicator_label: str) -> str:
    return "sum" if indicator_label == "Population" else "mean"


def get_indicator_agg_note(indicator_label: str) -> str:
    return "sum across available countries" if indicator_label == "Population" else "mean across available countries"


def get_trend_benchmark_agg(indicator_label: str) -> str:
    return "median" if indicator_label == "Population" else "mean"


def get_trend_benchmark_note(indicator_label: str) -> str:
    return "median country benchmark" if indicator_label == "Population" else "cross-country average benchmark"


def build_indicator_choropleth(
    df: pd.DataFrame,
    indicator_label: str,
    map_year: int,
    template: str,
    title: str,
    height: int = 430,
) -> go.Figure:
    indicator_cfg = available_indicators[indicator_label]
    value_col = indicator_cfg["column"]
    value_label = indicator_cfg["label"]
    hover_format = indicator_cfg.get("hover_format", ":,.2f")
    plot_df = df[df["year"] == map_year].dropna(subset=["country_code", value_col]).copy()

    if plot_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template=template,
            height=height,
            title=title,
            margin=dict(l=10, r=10, t=55, b=10),
            annotations=[
                dict(
                    text=f"No {indicator_label.lower()} data is available for {map_year}.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=15, color="#5d6b63"),
                )
            ],
        )
        return empty_fig

    q_low = float(plot_df[value_col].quantile(0.02))
    q_high = float(plot_df[value_col].quantile(0.98))
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
        q_low = float(plot_df[value_col].min())
        q_high = float(plot_df[value_col].max())

    fig = px.choropleth(
        plot_df,
        locations="country_code",
        color=value_col,
        hover_name="country_name",
        hover_data={
            "country_code": False,
            "wb_region": True,
            "year": True,
            value_col: True,
        },
        color_continuous_scale=indicator_cfg.get("map_scale", "Viridis"),
        range_color=(q_low, q_high) if q_low != q_high else None,
        locationmode="ISO-3",
        projection="natural earth",
        template=template,
        labels={
            value_col: value_label,
            "wb_region": "WB Region",
            "year": "Year",
        },
        title=title,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        coloraxis_colorbar=dict(title=value_label, len=0.78),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(31,42,36,0.28)",
            showland=True,
            landcolor="rgba(255,255,255,0.7)",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_traces(
        marker_line_color="rgba(255,255,255,0.65)",
        marker_line_width=0.35,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Region: %{customdata[0]}<br>"
            "Year: %{customdata[1]}<br>"
            f"{value_label}: %{{z{hover_format}}}<extra></extra>"
        ),
    )
    return fig


def get_best_model_name(dataset_label: str) -> str:
    temp = ts_best_df[ts_best_df["dataset_label"] == dataset_label]
    if temp.empty:
        return "N/A"
    return str(temp["Model"].iloc[0])


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.strip().lower() in normalized_lookup:
            return normalized_lookup[candidate.strip().lower()]
    return None


def one_step_forecast_from_values(train_values: np.ndarray, model_name: str) -> float:
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

    raise ValueError(f"Unsupported model: {model_name}")


@st.cache_data(show_spinner=False)
def build_country_backtest_predictions(
    series_df: pd.DataFrame,
    value_col: str,
    models: tuple[str, ...],
    window_size: int = FORECAST_WINDOW_YEARS,
) -> pd.DataFrame:
    clean_df = (
        series_df[["year", value_col]]
        .dropna()
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .copy()
    )

    if len(clean_df) <= window_size:
        raise ValueError(
            f"This country has only {len(clean_df)} usable years. At least "
            f"{window_size + 1} years are needed for a rolling {window_size}-year backtest."
        )

    rows = []
    for idx in range(window_size, len(clean_df)):
        train_window = clean_df.iloc[idx - window_size:idx].copy()
        target_row = clean_df.iloc[idx]
        train_values = train_window[value_col].astype(float).to_numpy()

        for model_name in models:
            predicted_value = one_step_forecast_from_values(train_values, model_name)
            rows.append(
                {
                    "year": int(target_row["year"]),
                    "actual": float(target_row[value_col]),
                    "predicted": float(predicted_value),
                    "Model": model_name,
                    "window_start_year": int(train_window["year"].min()),
                    "window_end_year": int(train_window["year"].max()),
                }
            )

    return pd.DataFrame(rows)


def summarize_country_backtest_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame()

    metric_rows = []
    for model_name, model_df in backtest_df.groupby("Model"):
        errors = model_df["actual"] - model_df["predicted"]
        abs_errors = errors.abs()
        actual_nonzero = model_df["actual"].replace(0, np.nan)
        mape = (abs_errors / actual_nonzero).mean() * 100
        rmse = float(np.sqrt((errors ** 2).mean()))
        mae = float(abs_errors.mean())
        metric_rows.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE_pct": float(mape) if pd.notna(mape) else np.nan,
                "n_predictions": int(len(model_df)),
            }
        )

    return pd.DataFrame(metric_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)


def forecast_series_to_target_year(
    series_df: pd.DataFrame,
    value_col: str,
    model_name: str,
    target_year: int,
    window_size: int = FORECAST_WINDOW_YEARS,
) -> pd.DataFrame:
    clean_df = (
        series_df[["year", value_col]]
        .dropna()
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .copy()
    )

    if clean_df.empty:
        raise ValueError("No historical values are available for this series.")

    values = clean_df[value_col].astype(float).to_numpy()
    years = clean_df["year"].astype(int).to_numpy()
    last_year = int(years[-1])

    if target_year <= last_year:
        raise ValueError(f"Target year must be greater than the latest available year ({last_year}).")

    if len(clean_df) < window_size:
        raise ValueError(
            f"This country has only {len(clean_df)} usable years, fewer than the required "
            f"{window_size}-year forecast window."
        )

    working_df = clean_df.copy()
    forecast_rows = []

    for forecast_year in range(last_year + 1, int(target_year) + 1):
        train_window = working_df.tail(window_size).copy()
        train_values = train_window[value_col].astype(float).to_numpy()
        train_start_year = int(train_window["year"].min())
        train_end_year = int(train_window["year"].max())
        next_value = one_step_forecast_from_values(train_values, model_name)

        forecast_rows.append(
            {
                "forecast_year": int(forecast_year),
                "predicted_value": float(next_value),
                "window_start_year": train_start_year,
                "window_end_year": train_end_year,
            }
        )

        working_df = pd.concat(
            [
                working_df,
                pd.DataFrame({"year": [forecast_year], value_col: [next_value]}),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(forecast_rows)


latest_indicator_series = latest_year_df[selected_value_col].dropna()
latest_indicator_value = (
    latest_indicator_series.sum()
    if selected_indicator == "Population"
    else latest_indicator_series.mean()
)

region_indicator_latest = (
    latest_year_df.dropna(subset=[selected_value_col])
    .groupby("wb_region", as_index=False)[selected_value_col]
    .agg(get_indicator_agg(selected_indicator))
    .sort_values(selected_value_col, ascending=False)
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Scenario 4 · Interactive Exploration</div>
        <div class="hero-title">Population Intelligence and Project Demo Dashboard</div>
        <div class="hero-copy">
            Explore long-run population, GDP per capita, and life expectancy patterns,
            compare countries and regions, inspect time-series model results, and
            present the main analytical progress of the final project through a
            dashboard designed for clarity, interpretability, and decision support.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Countries</div>
        <div class="metric-value">{n_countries}</div>
        <div class="metric-sub">country-level observations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col2.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Regions</div>
        <div class="metric-value">{n_regions}</div>
        <div class="metric-sub">World Bank regional groups</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col3.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Year Coverage</div>
        <div class="metric-value">{year_min}–{year_max}</div>
        <div class="metric-sub">historical data window</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col4.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Latest {selected_value_label}</div>
        <div class="metric-value">{format_indicator_value(latest_indicator_value, selected_indicator)}</div>
        <div class="metric-sub">{get_indicator_agg_note(selected_indicator)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

tabs = st.tabs([
    "Project Overview",
    "Time-Series Models",
    "Indicator Trends",
    "Comparison",
    "Forecast Explorer",
])

# -------------------------------------------------------------------
# Tab 1: Project Overview
# -------------------------------------------------------------------
with tabs[0]:
    st.caption(
        "This overview is intentionally fixed at the full-project level so the research story stays consistent during the final demo."
    )

    focus_left, focus_right = st.columns([1.15, 0.85])

    with focus_left:
        st.markdown("### Research Focus")
        st.markdown(
            """
            <div class="section-card">
                <span class="small-pill">Primary Target</span>
                <span class="small-pill">GDP_(t+1)</span>
                <span class="small-pill">Country-Year Panel</span>
                <span class="small-pill">Forecast Logic</span>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    The central objective of this project is to forecast <strong>next-year GDP per capita</strong>
                    using a structured country-year panel. Population and life expectancy are treated as the
                    two core explanatory drivers, while inflation, unemployment, internet usage, and
                    global event dummies are added to test whether richer macroeconomic context improves
                    GDP prediction performance.
                </p>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    In other words, the dashboard does not present three unrelated datasets. It presents
                    <strong>one GDP-focused forecasting problem</strong> supported by demographic and development
                    indicators that help explain why GDP differs across countries and over time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with focus_right:
        st.markdown("### GDP Forecast Snapshot")
        if overview_best_panel_model_row is not None:
            snapshot_cols = st.columns(2)
            snapshot_cols[0].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Best GDP Panel Model</div>
                    <div class="metric-value">{overview_best_panel_model_row['model'].replace('Model ', 'M')}</div>
                    <div class="metric-sub">based on test performance at the log-GDP scale</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            snapshot_cols[1].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Test R-squared</div>
                    <div class="metric-value">{overview_best_panel_model_row['R_squared']:.3f}</div>
                    <div class="metric-sub">with RMSE {overview_best_panel_model_row['RMSE']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                f"""
                <div class="section-card">
                    <p class="subtle-note" style="margin:0 0 0.55rem 0;">
                        <strong>Why this matters:</strong> the overview already connects data engineering to
                        modeling evidence. The current strongest GDP model is <strong>{overview_best_panel_model_row['model']}</strong>,
                        with a test MAPE of <strong>{overview_best_panel_model_row['MAPE_pct']:.3f}%</strong> and
                        <strong>{int(overview_best_panel_model_row['n_obs'])}</strong> test observations.
                    </p>
                    <p class="subtle-note" style="margin:0;">
                        This gives a clear transition from data understanding to model evaluation before the
                        user even reaches the detailed forecasting tabs.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("GDP main-model evaluation metrics are not available yet.")

    st.markdown("### Three Core Datasets")
    dataset_cols = st.columns(3)
    for col, dataset_card in zip(dataset_cols, overview_core_dataset_cards):
        col.markdown(
            f"""
            <div class="section-card">
                <div class="metric-label">{dataset_card['label']}</div>
                <div class="metric-value" style="font-size:1.65rem;">{dataset_card['coverage']:,}</div>
                <div class="metric-sub">country-year observations</div>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    {dataset_card['role']}
                </p>
                <p class="subtle-note" style="margin-top:0.55rem;">
                    Coverage: <strong>{dataset_card['countries']}</strong> countries ·
                    <strong>{dataset_card['year_min']}-{dataset_card['year_max']}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Why GDP Is the Main Forecast Target")
    gdp_story_left, gdp_story_right = st.columns([1.15, 0.85])

    with gdp_story_left:
        gdp_region_fig = px.bar(
            overview_gdp_region_latest,
            x="wb_region",
            y="gdp_per_capita_usd",
            color="gdp_per_capita_usd",
            color_continuous_scale=["#dcefe8", "#2c8e6d"],
            template=chart_template,
            labels={
                "wb_region": "World Bank Region",
                "gdp_per_capita_usd": "Mean GDP per Capita (US$)",
            },
            title=f"Mean GDP per Capita by Region in {overview_latest_year}",
            text_auto=".0f",
        )
        gdp_region_fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            coloraxis_showscale=False,
            xaxis_tickangle=-15,
        )
        st.plotly_chart(gdp_region_fig, use_container_width=True)

    with gdp_story_right:
        top_region = overview_gdp_region_latest.iloc[0] if not overview_gdp_region_latest.empty else None
        bottom_region = overview_gdp_region_latest.iloc[-1] if not overview_gdp_region_latest.empty else None
        top_region_name = top_region["wb_region"] if top_region is not None else "N/A"
        top_region_value = top_region["gdp_per_capita_usd"] if top_region is not None else np.nan
        bottom_region_name = bottom_region["wb_region"] if bottom_region is not None else "N/A"
        bottom_region_value = bottom_region["gdp_per_capita_usd"] if bottom_region is not None else np.nan

        st.markdown(
            f"""
            <div class="section-card">
                <span class="small-pill">GDP Focus</span>
                <span class="small-pill">Regional Inequality</span>
                <span class="small-pill">Model Justification</span>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    GDP per capita is the main forecasting target because it exhibits large and persistent
                    structural differences across regions. In the latest available year, the highest regional
                    average is <strong>{top_region_name}</strong> at <strong>{top_region_value:,.0f} US$</strong>,
                    while the lowest is <strong>{bottom_region_name}</strong> at only
                    <strong>{bottom_region_value:,.0f} US$</strong>.
                </p>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    This wide gap motivates three methodological choices used later in the project:
                    forecasting <strong>GDP per capita rather than raw total GDP</strong>, working with the
                    <strong>log-GDP target</strong> for stability, and adding <strong>region effects</strong>
                    in the richest model specification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Data Readiness and Modeling Trade-Off")
    tradeoff_left, tradeoff_right = st.columns([0.92, 1.08])

    with tradeoff_left:
        tradeoff_metrics_row1 = st.columns(2)
        tradeoff_metrics_row1[0].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Full Panel</div>
                <div class="metric-value">{len(pop_df):,}</div>
                <div class="metric-sub">{n_countries} countries · {year_min}-{year_max}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tradeoff_metrics_row1[1].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Extended Model Sample</div>
                <div class="metric-value">{overview_extended_obs:,}</div>
                <div class="metric-sub">{overview_extended_countries} countries · {overview_extended_year_min}-{overview_extended_year_max}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        tradeoff_metrics_row2 = st.columns(2)
        tradeoff_metrics_row2[0].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sample Retention</div>
                <div class="metric-value">{overview_sample_retention_pct:.1f}%</div>
                <div class="metric-sub">rows kept after adding the richer driver set</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tradeoff_metrics_row2[1].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Country Retention</div>
                <div class="metric-value">{overview_country_retention_pct:.1f}%</div>
                <div class="metric-sub">countries still usable in the extended specification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tradeoff_right:
        coverage_df = pd.DataFrame(
            [
                {
                    "Sample": "Core panel (GDP + Population + Life Expectancy)",
                    "Observations": f"{len(pop_df):,}",
                    "Countries": n_countries,
                    "Year range": f"{year_min}-{year_max}",
                },
                {
                    "Sample": "Extended GDP sample (+ Inflation + Unemployment + Internet)",
                    "Observations": f"{overview_extended_obs:,}",
                    "Countries": overview_extended_countries,
                    "Year range": f"{overview_extended_year_min}-{overview_extended_year_max}",
                },
            ]
        )
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class="section-card" style="margin-top:0.9rem;">
                <p class="subtle-note" style="margin:0;">
                    This trade-off is methodologically important. The three core series are complete across
                    the long historical panel, but the richer GDP models rely on later and less complete
                    macroeconomic indicators. As a result, explanatory richness increases while historical
                    coverage decreases.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Global Events Layer")
    event_left, event_right = st.columns([1.15, 0.85])

    with event_left:
        if EVENT_TIMELINE_FIGURE_PATH.exists():
            st.image(
                str(EVENT_TIMELINE_FIGURE_PATH),
                caption="Global event dummy timeline reused from the project figure assets.",
                use_container_width=True,
            )
        else:
            st.info("Global event timeline figure was not found in the output/figures folder.")

    with event_right:
        st.markdown(
            """
            <div class="section-card">
                <span class="small-pill">Event Dummies</span>
                <span class="small-pill">Shock Engineering</span>
                <span class="small-pill">Model 3</span>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    The panel dataset was extended with manually engineered global-shock indicators so the
                    GDP models can capture macro disruptions that are not explained by country-level
                    demographics alone.
                </p>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    Event windows currently included in the project are:
                    <strong>Asian Financial Crisis (1997-1998)</strong>,
                    <strong>Global Financial Crisis (2008-2009)</strong>,
                    <strong>COVID shock (2020)</strong>,
                    <strong>COVID rebound (2021)</strong>, and
                    <strong>Ukraine / energy shock (2022-2024)</strong>.
                </p>
                <p class="subtle-note" style="margin-top:0.8rem;">
                    These features become especially important in <strong>Model 3</strong>, where they are
                    combined with region effects and year effects to reflect common global disturbances.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Modeling Roadmap")
    roadmap_cols = st.columns(3)
    roadmap_cols[0].markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">Stage 1 · Separate Time-Series</div>
            <p class="subtle-note" style="margin-top:0.8rem;">
                Each core dataset is first forecast independently using a rolling 10-year backtesting design.
            </p>
            <p class="subtle-note" style="margin-top:0.8rem;">
                GDP best model: <strong>{get_best_model_name('GDP')}</strong><br>
                Life Expectancy best model: <strong>{get_best_model_name('Life Expectancy')}</strong><br>
                Population best model: <strong>{get_best_model_name('Population')}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    roadmap_cols[1].markdown(
        """
        <div class="section-card">
            <div class="metric-label">Stage 2 · Main GDP Models</div>
            <p class="subtle-note" style="margin-top:0.8rem;">
                <strong>Model 1</strong>: GDP ~ Population + Life Expectancy
            </p>
            <p class="subtle-note" style="margin-top:0.55rem;">
                <strong>Model 2</strong>: GDP ~ Population + Life Expectancy + Inflation + Unemployment + Internet
            </p>
            <p class="subtle-note" style="margin-top:0.55rem;">
                <strong>Model 3</strong>: GDP ~ Population + Life Expectancy + Inflation + Unemployment + Internet + Event Dummies + Region Effects + Year Effects
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    roadmap_cols[2].markdown(
        """
        <div class="section-card">
            <div class="metric-label">Stage 3 · Dashboard Delivery</div>
            <p class="subtle-note" style="margin-top:0.8rem;">
                The dashboard translates the analytical workflow into an interpretable demo surface:
                overview, trend exploration, country and region comparison, model backtesting, and
                target-year forecasting.
            </p>
            <p class="subtle-note" style="margin-top:0.8rem;">
                This makes the project easier to explain to non-technical viewers while still keeping the
                research logic visible.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Modeling Panel Preview")
    preview_source_df = overview_extended_model_df if not overview_extended_model_df.empty else pop_df
    preview_cols = [
        c for c in [
            "country_name",
            "country_code",
            "wb_region",
            "year",
            "gdp_per_capita_usd",
            "population_total",
            "life_expectancy_years",
            "inflation_pct",
            "unemployment_pct",
            "internet_users_pct",
            "covid_shock_2020",
            "ukraine_energy_shock_2022_2024",
        ] if c in preview_source_df.columns
    ]
    st.dataframe(
        preview_source_df[preview_cols].head(15),
        use_container_width=True,
        hide_index=True,
    )

# -------------------------------------------------------------------
# Tab 2: Time-Series Models
# -------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Separate Time-Series Model Selection")

    if ts_best_df.empty:
        st.warning("No time-series summary files were found in the cleaned-data folder.")
    else:
        metric_cols = st.columns(3)
        for idx, dataset_label in enumerate(["GDP", "Life Expectancy", "Population"]):
            temp = ts_best_df[ts_best_df["dataset_label"] == dataset_label]
            if temp.empty:
                metric_cols[idx].info(f"No best-model file found for {dataset_label}.")
            else:
                row = temp.iloc[0]
                metric_cols[idx].markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{dataset_label} Best Model</div>
                        <div class="metric-value">{row['Model']}</div>
                        <div class="metric-sub">
                            RMSE: {format_metric(row['RMSE'])} · MAPE: {format_metric(row['MAPE_pct'])}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Candidate Model Comparison")
        summary_view = ts_summary_df[
            ["dataset_label", "Model", "MAE", "RMSE", "MAPE_pct", "R_squared", "n_predictions", "n_countries_modeled"]
        ].copy()
        summary_view = summary_view.rename(columns={"dataset_label": "Dataset"})
        st.dataframe(
            summary_view.style.format({
                "MAE": "{:,.4f}",
                "RMSE": "{:,.4f}",
                "MAPE_pct": "{:,.4f}",
                "R_squared": "{:,.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Backtest Output Explorer")
        ts_dataset = st.selectbox(
            "Select dataset",
            options=["GDP", "Life Expectancy", "Population"],
            key="ts_dataset_select",
        )
        ts_config = TIME_SERIES_BACKTEST_CONFIG[ts_dataset]
        ts_value_col = ts_config["column"]
        ts_value_label = ts_config["label"]
        ts_model_options = ts_config["models"]

        country_options = sorted(
            pop_df.dropna(subset=[ts_value_col])["country_name"].dropna().unique().tolist()
        )
        default_ts_country = (
            selected_countries[0]
            if selected_countries and selected_countries[0] in country_options
            else country_options[0]
        )

        ts_control_col1, ts_control_col2 = st.columns([1.0, 1.2])
        selected_ts_country = ts_control_col1.selectbox(
            "Select country",
            options=country_options,
            index=country_options.index(default_ts_country),
            key="ts_country_select",
        )
        selected_ts_models = ts_control_col2.multiselect(
            "Select up to 3 models to compare",
            options=ts_model_options,
            default=ts_model_options,
            max_selections=3,
            key="ts_model_multiselect",
        )

        if not selected_ts_models:
            st.info("Choose at least one model to display the backtest comparison.")
        else:
            ts_country_history_df = (
                pop_df[pop_df["country_name"] == selected_ts_country]
                .dropna(subset=[ts_value_col])
                .sort_values("year")
                .copy()
            )

            try:
                with st.spinner("Running rolling 10-year backtest for the selected country..."):
                    ts_backtest_df = build_country_backtest_predictions(
                        series_df=ts_country_history_df,
                        value_col=ts_value_col,
                        models=tuple(selected_ts_models),
                        window_size=FORECAST_WINDOW_YEARS,
                    )
                ts_metric_df = summarize_country_backtest_metrics(ts_backtest_df)
            except Exception as exc:
                st.warning(f"Backtest comparison could not be generated: {exc}")
                ts_backtest_df = pd.DataFrame()
                ts_metric_df = pd.DataFrame()

            if not ts_backtest_df.empty:
                overall_best_model = get_best_model_name(ts_dataset)
                country_best_model = (
                    ts_metric_df.iloc[0]["Model"]
                    if not ts_metric_df.empty
                    else overall_best_model
                )

                best_col1, best_col2 = st.columns(2)
                best_col1.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Best Overall Model</div>
                        <div class="metric-value">{overall_best_model}</div>
                        <div class="metric-sub">based on full multi-country summary metrics</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                best_col2.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Best For {selected_ts_country}</div>
                        <div class="metric-value">{country_best_model}</div>
                        <div class="metric-sub">based on RMSE from the rolling {FORECAST_WINDOW_YEARS}-year backtest</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"Each prediction year is generated using the previous {FORECAST_WINDOW_YEARS} years only. "
                    f"Models shown: {', '.join(selected_ts_models)}."
                )

                ts_fig = go.Figure()
                actual_df = ts_backtest_df[["year", "actual"]].drop_duplicates().sort_values("year")
                ts_fig.add_trace(
                    go.Scatter(
                        x=actual_df["year"],
                        y=actual_df["actual"],
                        mode="lines+markers",
                        name="Actual",
                        line=dict(color="#1f7a5c", width=3.4),
                    )
                )

                model_color_map = {
                    "Naive": "#d98e04",
                    "ARIMA": "#c96a3d",
                    "AutoReg": "#7f6bff",
                    "Holt": "#3b82f6",
                    "LogHolt": "#9b8c5a",
                }
                for model_name in selected_ts_models:
                    model_df = ts_backtest_df[ts_backtest_df["Model"] == model_name].sort_values("year")
                    line_width = 3.0 if model_name == country_best_model else 2.2
                    dash_style = "solid" if model_name == country_best_model else "dot"
                    ts_fig.add_trace(
                        go.Scatter(
                            x=model_df["year"],
                            y=model_df["predicted"],
                            mode="lines+markers",
                            name=f"Predicted ({model_name}){' · best' if model_name == country_best_model else ''}",
                            line=dict(
                                color=model_color_map.get(model_name, "#d98e04"),
                                width=line_width,
                                dash=dash_style,
                            ),
                        )
                    )

                ts_fig.update_layout(
                    template=chart_template,
                    height=560,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Year",
                    yaxis_title=ts_value_label,
                    legend_title_text="Series",
                )
                st.plotly_chart(ts_fig, use_container_width=True)

                metric_view = ts_metric_df.copy()
                metric_view["Overall Best"] = metric_view["Model"].eq(overall_best_model).map({True: "Yes", False: ""})
                metric_view["Country Best"] = metric_view["Model"].eq(country_best_model).map({True: "Yes", False: ""})

                left_table, right_table = st.columns([0.95, 1.05])
                with left_table:
                    st.markdown("### Model Metrics for Selected Country")
                    st.dataframe(
                        metric_view.style.format({
                            "MAE": "{:,.4f}",
                            "RMSE": "{:,.4f}",
                            "MAPE_pct": "{:,.4f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                with right_table:
                    st.markdown("### Backtest Detail")
                    st.dataframe(
                        ts_backtest_df.sort_values(["Model", "year"]),
                        use_container_width=True,
                        hide_index=True,
                    )

# -------------------------------------------------------------------
# Tab 3: Trends
# -------------------------------------------------------------------
with tabs[2]:
    st.markdown(f"### {selected_indicator} Trend Story")

    trend_source_df = region_filtered_df[
        region_filtered_df["year"].between(selected_year_range[0], selected_year_range[1])
    ].copy()
    trend_indicator_df = (
        trend_source_df.dropna(subset=[selected_value_col])
        .sort_values(["country_name", "year"])
        .copy()
    )

    if trend_indicator_df.empty:
        st.warning("No indicator-trend data is available for the current region and year filters.")
    else:
        trend_country_options = sorted(trend_indicator_df["country_name"].dropna().unique().tolist())
        default_trend_country = None
        for candidate in (
            "Viet Nam",
            "United States",
            "China",
            "India",
            selected_countries[0] if selected_countries else None,
            trend_country_options[0] if trend_country_options else None,
        ):
            if candidate in trend_country_options:
                default_trend_country = candidate
                break

        trend_benchmark_intro = (
            "For <strong>Population</strong>, benchmarks use the <strong>median country benchmark</strong> so the chart stays interpretable."
            if selected_indicator == "Population"
            else f"For <strong>{selected_indicator}</strong>, benchmarks use the <strong>{get_trend_benchmark_note(selected_indicator)}</strong>."
        )

        trend_control_col1, trend_control_col2 = st.columns([1.05, 1.15])
        selected_trend_country = trend_control_col1.selectbox(
            "Focus country",
            options=trend_country_options,
            index=trend_country_options.index(default_trend_country),
            key="trend_focus_country_select",
        )
        trend_control_col2.markdown(
            f"""
            <div class="section-card">
                <p class="subtle-note" style="margin:0;">
                    This tab is now optimized for demo storytelling. Instead of plotting many countries at once,
                    it compares <strong>one focus country</strong> with its <strong>regional benchmark</strong> and the
                    <strong>global benchmark</strong>. {trend_benchmark_intro}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        focus_country_df = (
            trend_indicator_df[trend_indicator_df["country_name"] == selected_trend_country]
            .drop_duplicates(subset=["year"])
            .sort_values("year")
            .copy()
        )
        focus_region_name = (
            str(focus_country_df.sort_values("year")["wb_region"].dropna().iloc[-1])
            if not focus_country_df["wb_region"].dropna().empty
            else "Unknown Region"
        )
        benchmark_agg = get_trend_benchmark_agg(selected_indicator)
        benchmark_note = get_trend_benchmark_note(selected_indicator)

        region_benchmark_df = (
            trend_indicator_df[trend_indicator_df["wb_region"] == focus_region_name]
            .groupby("year", as_index=False)[selected_value_col]
            .agg(benchmark_agg)
            .sort_values("year")
        )
        global_benchmark_df = (
            pop_df[
                pop_df["year"].between(selected_year_range[0], selected_year_range[1])
            ]
            .dropna(subset=[selected_value_col])
            .groupby("year", as_index=False)[selected_value_col]
            .agg(benchmark_agg)
            .sort_values("year")
        )

        latest_focus_year = int(focus_country_df["year"].max())
        latest_focus_value = float(focus_country_df.iloc[-1][selected_value_col])
        latest_region_series = region_benchmark_df.loc[
            region_benchmark_df["year"] == latest_focus_year, selected_value_col
        ]
        latest_global_series = global_benchmark_df.loc[
            global_benchmark_df["year"] == latest_focus_year, selected_value_col
        ]
        latest_region_value = float(latest_region_series.iloc[0]) if not latest_region_series.empty else np.nan
        latest_global_value = float(latest_global_series.iloc[0]) if not latest_global_series.empty else np.nan

        summary_metric_cols = st.columns(3)
        summary_metric_cols[0].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{selected_trend_country}</div>
                <div class="metric-value">{format_indicator_value(latest_focus_value, selected_indicator)}</div>
                <div class="metric-sub">latest available value in {latest_focus_year}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary_metric_cols[1].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{focus_region_name}</div>
                <div class="metric-value">{format_indicator_value(latest_region_value, selected_indicator)}</div>
                <div class="metric-sub">{benchmark_note} in {latest_focus_year}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary_metric_cols[2].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">World Benchmark</div>
                <div class="metric-value">{format_indicator_value(latest_global_value, selected_indicator)}</div>
                <div class="metric-sub">{benchmark_note} in {latest_focus_year}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"### {selected_trend_country} vs Region and World")
        benchmark_caption = (
            "Population benchmarks use the median country benchmark rather than regional totals."
            if selected_indicator == "Population"
            else f"{selected_indicator} benchmarks use the {benchmark_note}."
        )
        st.caption(
            f"The main comparison uses {selected_trend_country} as the focus series, {focus_region_name} as the regional benchmark, "
            f"and the full panel as the global benchmark. {benchmark_caption}"
        )

        focus_trend_fig = go.Figure()
        focus_trend_fig.add_trace(
            go.Scatter(
                x=focus_country_df["year"],
                y=focus_country_df[selected_value_col],
                mode="lines+markers",
                name=selected_trend_country,
                line=dict(color=selected_value_color, width=3.6),
            )
        )
        focus_trend_fig.add_trace(
            go.Scatter(
                x=region_benchmark_df["year"],
                y=region_benchmark_df[selected_value_col],
                mode="lines",
                name=f"{focus_region_name} benchmark",
                line=dict(color="#d98e04", width=3.0, dash="solid"),
            )
        )
        focus_trend_fig.add_trace(
            go.Scatter(
                x=global_benchmark_df["year"],
                y=global_benchmark_df[selected_value_col],
                mode="lines",
                name="World benchmark",
                line=dict(color="#5d6b63", width=2.8, dash="dot"),
            )
        )
        focus_trend_fig.update_layout(
            template=chart_template,
            height=560,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Year",
            yaxis_title=selected_value_label,
            legend_title_text="Series",
        )
        st.plotly_chart(focus_trend_fig, use_container_width=True)

        supplementary_cols = st.columns(2)

        with supplementary_cols[0]:
            st.markdown("### Regional Benchmark Context")
            regional_context_df = (
                trend_indicator_df.groupby(["wb_region", "year"], as_index=False)[selected_value_col]
                .agg(benchmark_agg)
                .sort_values(["wb_region", "year"])
            )
            regional_context_fig = go.Figure()
            for region_name, region_df in regional_context_df.groupby("wb_region"):
                is_focus_region = region_name == focus_region_name
                regional_context_fig.add_trace(
                    go.Scatter(
                        x=region_df["year"],
                        y=region_df[selected_value_col],
                        mode="lines",
                        name=region_name,
                        line=dict(
                            width=3.2 if is_focus_region else 1.9,
                            dash="solid" if is_focus_region else "dot",
                        ),
                        opacity=1.0 if is_focus_region else 0.55,
                    )
                )
            regional_context_fig.update_layout(
                template=chart_template,
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Year",
                yaxis_title=selected_value_label,
                legend_title_text="WB Region",
            )
            st.plotly_chart(regional_context_fig, use_container_width=True)

        with supplementary_cols[1]:
            if selected_indicator == "GDP per Capita":
                change_title = f"Annual GDP per Capita Growth for {selected_trend_country}"
                change_label = "GDP per Capita Growth (%)"
                change_df = focus_country_df[["year", selected_value_col]].copy()
                change_df["change_metric"] = change_df[selected_value_col].pct_change() * 100
            elif selected_indicator == "Population":
                change_title = f"Annual Population Growth for {selected_trend_country}"
                change_label = "Population Growth (%)"
                if "population_growth_pct" in focus_country_df.columns and focus_country_df["population_growth_pct"].notna().any():
                    change_df = focus_country_df[["year", "population_growth_pct"]].copy()
                    change_df["change_metric"] = change_df["population_growth_pct"]
                else:
                    change_df = focus_country_df[["year", selected_value_col]].copy()
                    change_df["change_metric"] = change_df[selected_value_col].pct_change() * 100
            else:
                change_title = f"Annual Life Expectancy Gain for {selected_trend_country}"
                change_label = "Life Expectancy Change (years)"
                change_df = focus_country_df[["year", selected_value_col]].copy()
                change_df["change_metric"] = change_df[selected_value_col].diff()

            st.markdown(f"### {change_title}")
            change_df = change_df.dropna(subset=["change_metric"]).copy()
            if change_df.empty:
                st.info("Not enough historical points are available to compute annual change for the focus country.")
            else:
                change_colors = np.where(change_df["change_metric"] >= 0, "#1f7a5c", "#c96a3d")
                change_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=change_df["year"],
                            y=change_df["change_metric"],
                            marker_color=change_colors,
                            name=change_label,
                        )
                    ]
                )
                change_fig.update_layout(
                    template=chart_template,
                    height=420,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Year",
                    yaxis_title=change_label,
                    showlegend=False,
                )
                st.plotly_chart(change_fig, use_container_width=True)

        st.markdown("### Global Heatmaps")
        map_source_df = region_filtered_df[
            region_filtered_df["year"].between(selected_year_range[0], selected_year_range[1])
        ].copy()
        map_year_options = sorted(map_source_df["year"].dropna().astype(int).unique().tolist())

        if map_year_options:
            default_map_year = compare_year if compare_year in map_year_options else map_year_options[-1]
            map_control_col1, map_control_col2 = st.columns([1.0, 1.8])
            map_year = map_control_col1.select_slider(
                "Heatmap year",
                options=map_year_options,
                value=default_map_year,
            )
            map_control_col2.markdown(
                """
                <div class="section-card">
                    <p class="subtle-note" style="margin:0;">
                        Heatmaps are grouped into separate tabs so the dashboard stays clean in the final demo.
                        They follow the current <strong>region</strong> and <strong>year-range</strong> filters,
                        ignore the country multiselect, and clip color scales to the 2nd-98th percentile for readability.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            heatmap_tabs = st.tabs(["GDP per Capita", "Population", "Life Expectancy"])
            with heatmap_tabs[0]:
                gdp_map_fig = build_indicator_choropleth(
                    df=map_source_df,
                    indicator_label="GDP per Capita",
                    map_year=map_year,
                    template=chart_template,
                    title=f"GDP per Capita Heatmap ({map_year})",
                    height=460,
                )
                st.plotly_chart(gdp_map_fig, use_container_width=True)
            with heatmap_tabs[1]:
                pop_map_fig = build_indicator_choropleth(
                    df=map_source_df,
                    indicator_label="Population",
                    map_year=map_year,
                    template=chart_template,
                    title=f"Population Heatmap ({map_year})",
                    height=460,
                )
                st.plotly_chart(pop_map_fig, use_container_width=True)
            with heatmap_tabs[2]:
                life_map_fig = build_indicator_choropleth(
                    df=map_source_df,
                    indicator_label="Life Expectancy",
                    map_year=map_year,
                    template=chart_template,
                    title=f"Life Expectancy Heatmap ({map_year})",
                    height=460,
                )
                st.plotly_chart(life_map_fig, use_container_width=True)
        else:
            st.info("No yearly map data is available for the current region and time filters.")

        st.markdown("### Quick Read")
        if selected_indicator == "GDP per Capita":
            quick_text = (
                f"In {latest_focus_year}, <strong>{selected_trend_country}</strong> recorded "
                f"<strong>{format_indicator_value(latest_focus_value, selected_indicator)}</strong>, compared with "
                f"the <strong>{focus_region_name}</strong> benchmark of "
                f"<strong>{format_indicator_value(latest_region_value, selected_indicator)}</strong> and the "
                f"<strong>world benchmark</strong> of <strong>{format_indicator_value(latest_global_value, selected_indicator)}</strong>. "
                f"This is the core target variable of the project, so the trend view is designed to show whether the country is "
                f"converging toward or diverging from broader GDP patterns before the forecasting models are applied."
            )
        elif selected_indicator == "Population":
            quick_text = (
                f"For <strong>Population</strong>, the focus is on both scale and growth. "
                f"<strong>{selected_trend_country}</strong> is compared with the regional and global "
                f"<strong>{benchmark_note}</strong> rather than total regional population, which keeps the comparison meaningful at the country level. "
                f"The bar chart on the right then shows whether population growth is accelerating, stabilizing, or slowing over time."
            )
        else:
            quick_text = (
                f"<strong>Life Expectancy</strong> usually changes more smoothly than GDP, so this view emphasizes long-run structural movement. "
                f"In {latest_focus_year}, <strong>{selected_trend_country}</strong> reached "
                f"<strong>{format_indicator_value(latest_focus_value, selected_indicator)}</strong>, against "
                f"<strong>{format_indicator_value(latest_region_value, selected_indicator)}</strong> for "
                f"{focus_region_name} and <strong>{format_indicator_value(latest_global_value, selected_indicator)}</strong> globally. "
                f"This helps explain whether development conditions are strengthening in parallel with the GDP story."
            )

        st.markdown(
            f"""
            <div class="section-card">
                <p class="subtle-note">
                    {quick_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------------
# Tab 4: Comparison
# -------------------------------------------------------------------
with tabs[3]:
    st.markdown(f"### {selected_indicator} Comparison in {compare_year}")
    comparison_base_df = pop_df[pop_df["year"] == compare_year].dropna(subset=[selected_value_col]).copy()
    if selected_regions:
        comparison_base_df = comparison_base_df[comparison_base_df["wb_region"].isin(selected_regions)].copy()

    if comparison_base_df.empty:
        st.warning("No comparison data is available for the current year and region filters.")
    else:
        comparison_scope_note = (
            f"Comparison scope: {len(selected_regions)} selected regions in {compare_year}."
            if selected_regions
            else f"Comparison scope: all World Bank regions in {compare_year}."
        )
        st.caption(comparison_scope_note)

        compare_country_options = sorted(comparison_base_df["country_name"].dropna().unique().tolist())
        default_compare_country = None
        for candidate in (
            "Viet Nam",
            "United States",
            "China",
            "India",
            selected_countries[0] if selected_countries else None,
            compare_country_options[0] if compare_country_options else None,
        ):
            if candidate in compare_country_options:
                default_compare_country = candidate
                break

        compare_control_col1, compare_control_col2 = st.columns([1.0, 1.15])
        compare_mode = compare_control_col1.radio(
            "Comparison mode",
            options=["Country ranking", "Regional comparison"],
            horizontal=True,
        )
        spotlight_country = compare_control_col2.selectbox(
            "Spotlight country",
            options=compare_country_options,
            index=compare_country_options.index(default_compare_country),
            key="comparison_spotlight_country_select",
        )

        ranking_df = comparison_base_df.sort_values(selected_value_col, ascending=False).reset_index(drop=True).copy()
        ranking_df["rank"] = ranking_df.index + 1

        spotlight_row = ranking_df[ranking_df["country_name"] == spotlight_country].iloc[0]
        spotlight_region = str(spotlight_row["wb_region"])
        spotlight_value = float(spotlight_row[selected_value_col])
        spotlight_rank = int(spotlight_row["rank"])
        country_benchmark_agg = get_trend_benchmark_agg(selected_indicator)
        country_benchmark_note = get_trend_benchmark_note(selected_indicator)
        selected_region_scope_active = bool(selected_regions) and len(selected_regions) < n_regions
        comparison_scope_label = "Current Scope" if selected_region_scope_active else "World"
        regional_benchmark_value = float(
            comparison_base_df[comparison_base_df["wb_region"] == spotlight_region][selected_value_col].agg(country_benchmark_agg)
        )
        global_benchmark_value = float(comparison_base_df[selected_value_col].agg(country_benchmark_agg))
        gap_vs_region = spotlight_value - regional_benchmark_value
        gap_vs_world = spotlight_value - global_benchmark_value

        metric_cols = st.columns(4)
        metric_cols[0].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{spotlight_country}</div>
                <div class="metric-value">{format_indicator_value(spotlight_value, selected_indicator)}</div>
                <div class="metric-sub">{selected_indicator} in {compare_year}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols[1].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Rank in Scope</div>
                <div class="metric-value">{spotlight_rank}</div>
                <div class="metric-sub">out of {len(ranking_df)} countries in the current comparison set</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols[2].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Gap vs {spotlight_region}</div>
                <div class="metric-value">{format_signed_indicator_delta(gap_vs_region, selected_indicator)}</div>
                <div class="metric-sub">{country_benchmark_note} for the focus region</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols[3].markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Gap vs {comparison_scope_label}</div>
                <div class="metric-value">{format_signed_indicator_delta(gap_vs_world, selected_indicator)}</div>
                <div class="metric-sub">{country_benchmark_note} across the current comparison scope</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="section-card" style="margin-top:0.9rem;">
                <p class="subtle-note" style="margin:0;">
                    <strong>{spotlight_country}</strong> is the current anchor country for the comparison view.
                    This means the chart and table below are not just showing rankings; they are helping explain
                    where the focus country stands relative to its regional benchmark and the broader world pattern.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if compare_mode == "Country ranking":
            top_n = st.slider("Top N countries", min_value=5, max_value=25, value=12, step=1)
            ranking_plot_df = ranking_df.head(top_n).copy()
            if spotlight_country not in ranking_plot_df["country_name"].values:
                ranking_plot_df = pd.concat(
                    [ranking_plot_df, ranking_df[ranking_df["country_name"] == spotlight_country]],
                    ignore_index=True,
                )
            ranking_plot_df = ranking_plot_df.sort_values(selected_value_col, ascending=False).copy()
            ranking_plot_df["display_country"] = ranking_plot_df["country_name"].where(
                ranking_plot_df["country_name"] != spotlight_country,
                ranking_plot_df["country_name"] + " ★",
            )
            ranking_plot_df["bar_color"] = np.where(
                ranking_plot_df["country_name"] == spotlight_country,
                selected_value_color,
                "#bfc9c3",
            )

            ranking_fig = go.Figure(
                data=[
                    go.Bar(
                        x=ranking_plot_df["display_country"],
                        y=ranking_plot_df[selected_value_col],
                        marker_color=ranking_plot_df["bar_color"],
                        customdata=np.stack(
                            [
                                ranking_plot_df["wb_region"],
                                ranking_plot_df["rank"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Region: %{customdata[0]}<br>"
                            "Rank: %{customdata[1]}<br>"
                            f"{selected_value_label}: %{{y{selected_indicator_cfg.get('hover_format', ':,.2f')}}}<extra></extra>"
                        ),
                    )
                ]
            )
            ranking_fig.add_hline(
                y=regional_benchmark_value,
                line_color="#d98e04",
                line_dash="dot",
                annotation_text=f"{spotlight_region} benchmark",
                annotation_position="top left",
            )
            ranking_fig.add_hline(
                y=global_benchmark_value,
                line_color="#5d6b63",
                line_dash="dash",
                annotation_text=f"{comparison_scope_label} benchmark",
                annotation_position="bottom left",
            )
            ranking_fig.update_layout(
                template=chart_template,
                height=560,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Country",
                yaxis_title=selected_value_label,
                title=f"Country Ranking for {selected_indicator} in {compare_year}",
            )
            ranking_fig.update_xaxes(tickangle=-35)
            st.plotly_chart(ranking_fig, use_container_width=True)

            st.markdown("### Comparison Table")
            comparison_table_df = ranking_df[["rank", "country_name", "country_code", "wb_region", selected_value_col]].copy()
            comparison_table_df = comparison_table_df.rename(
                columns={
                    "rank": "Rank",
                    "country_name": "Country",
                    "country_code": "Code",
                    "wb_region": "WB Region",
                    selected_value_col: selected_value_label,
                }
            )
            if spotlight_country not in comparison_table_df.head(top_n)["Country"].values:
                spotlight_table_row = comparison_table_df[comparison_table_df["Country"] == spotlight_country]
                comparison_table_df = pd.concat(
                    [comparison_table_df.head(top_n), spotlight_table_row],
                    ignore_index=True,
                ).drop_duplicates(subset=["Country"])
            else:
                comparison_table_df = comparison_table_df.head(top_n).copy()

            st.dataframe(comparison_table_df, use_container_width=True, hide_index=True)

        else:
            region_year_df = (
                comparison_base_df.groupby("wb_region", as_index=False)[selected_value_col]
                .agg(get_indicator_agg(selected_indicator))
                .sort_values(selected_value_col, ascending=False)
                .reset_index(drop=True)
            )
            region_year_df["rank"] = region_year_df.index + 1
            region_year_df["display_region"] = region_year_df["wb_region"].where(
                region_year_df["wb_region"] != spotlight_region,
                region_year_df["wb_region"] + " ★",
            )
            region_year_df["bar_color"] = np.where(
                region_year_df["wb_region"] == spotlight_region,
                "#d98e04",
                "#bfc9c3",
            )

            region_fig = go.Figure(
                data=[
                    go.Bar(
                        x=region_year_df["display_region"],
                        y=region_year_df[selected_value_col],
                        marker_color=region_year_df["bar_color"],
                        customdata=np.stack([region_year_df["rank"]], axis=-1),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Region rank: %{customdata[0]}<br>"
                            f"{selected_value_label}: %{{y{selected_indicator_cfg.get('hover_format', ':,.2f')}}}<extra></extra>"
                        ),
                    )
                ]
            )
            if selected_indicator != "Population":
                region_fig.add_hline(
                    y=global_benchmark_value,
                    line_color="#5d6b63",
                    line_dash="dash",
                    annotation_text=f"{comparison_scope_label} benchmark",
                    annotation_position="top left",
                )
            region_fig.update_layout(
                template=chart_template,
                height=560,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="World Bank Region",
                yaxis_title=selected_value_label,
                title=f"Regional Comparison for {selected_indicator} in {compare_year}",
                showlegend=False,
            )
            region_fig.update_xaxes(tickangle=-20)
            st.plotly_chart(region_fig, use_container_width=True)

            st.markdown("### Comparison Table")
            region_table_df = region_year_df[["rank", "wb_region", selected_value_col]].copy().rename(
                columns={
                    "rank": "Rank",
                    "wb_region": "WB Region",
                    selected_value_col: selected_value_label,
                }
            )
            st.dataframe(region_table_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# Tab 5: Forecast Explorer
# -------------------------------------------------------------------
with tabs[4]:
    st.markdown("### Forecast Explorer")

    forecast_indicator_options = list(available_indicators.keys())
    forecast_indicator = st.selectbox(
        "Forecast indicator",
        options=forecast_indicator_options,
        index=forecast_indicator_options.index(selected_indicator),
    )
    forecast_indicator_cfg = available_indicators[forecast_indicator]
    forecast_value_col = forecast_indicator_cfg["column"]
    forecast_value_label = forecast_indicator_cfg["label"]
    forecast_ts_dataset = TS_DATASET_BY_INDICATOR.get(forecast_indicator, "Population")
    forecast_best_model = get_best_model_name(forecast_ts_dataset)
    resolved_best_model = forecast_best_model if forecast_best_model != "N/A" else "Naive"
    resolved_live_model = LIVE_FORECAST_MODEL_BY_DATASET.get(forecast_ts_dataset, resolved_best_model)

    st.markdown("### Live Target-Year Forecast")

    live_forecast_country_options = sorted(
        pop_df.dropna(subset=[forecast_value_col])["country_name"].dropna().unique().tolist()
    )
    default_live_country = (
        selected_countries[0]
        if selected_countries and selected_countries[0] in live_forecast_country_options
        else live_forecast_country_options[0]
    )

    live_control_col1, live_control_col2 = st.columns([1.3, 1.0])

    selected_live_country = live_control_col1.selectbox(
        "Live forecast country",
        options=live_forecast_country_options,
        index=live_forecast_country_options.index(default_live_country),
        key="live_forecast_country_select",
    )

    live_country_history_df = (
        pop_df[pop_df["country_name"] == selected_live_country]
        .dropna(subset=[forecast_value_col])
        .sort_values("year")
        .copy()
    )
    live_last_year = int(live_country_history_df["year"].max())

    selected_target_year = live_control_col2.number_input(
        "Target future year",
        min_value=live_last_year + 1,
        max_value=live_last_year + 40,
        value=min(live_last_year + 5, live_last_year + 40),
        step=1,
        key="live_forecast_target_year_input",
    )

    st.caption(
        f"Latest available year for {selected_live_country}: {live_last_year}. "
        f"Best backtest model for {forecast_indicator}: {resolved_best_model}. "
        f"Future forecast model used here: {resolved_live_model}. "
        f"Live forecast uses a rolling {FORECAST_WINDOW_YEARS}-year window before each predicted year."
    )

    try:
        live_forecast_df = forecast_series_to_target_year(
            series_df=live_country_history_df,
            value_col=forecast_value_col,
            model_name=resolved_live_model,
            target_year=int(selected_target_year),
            window_size=FORECAST_WINDOW_YEARS,
        )

        live_forecast_fig = go.Figure()
        live_forecast_fig.add_trace(
            go.Scatter(
                x=live_country_history_df["year"],
                y=live_country_history_df[forecast_value_col],
                mode="lines+markers",
                name=f"Historical {forecast_indicator}",
                line=dict(color=forecast_indicator_cfg["color"], width=3),
            )
        )
        live_forecast_fig.add_trace(
            go.Scatter(
                x=live_forecast_df["forecast_year"],
                y=live_forecast_df["predicted_value"],
                mode="lines+markers",
                name=f"Forecast ({resolved_live_model})",
                line=dict(color="#d98e04", width=3, dash="dot"),
                marker=dict(size=8),
            )
        )
        live_forecast_fig.update_layout(
            template=chart_template,
            height=540,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Year",
            yaxis_title=forecast_value_label,
            legend_title_text="Series",
        )
        st.plotly_chart(live_forecast_fig, use_container_width=True)

        latest_live_value = float(live_forecast_df.iloc[-1]["predicted_value"])
        summary_col1, summary_col2 = st.columns(2)
        summary_col1.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Forecasted {forecast_indicator}</div>
                <div class="metric-value">{format_indicator_value(latest_live_value, forecast_indicator)}</div>
                <div class="metric-sub">target year {int(selected_target_year)} using forecast model {resolved_live_model}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary_col2.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Forecast Horizon</div>
                <div class="metric-value">{int(selected_target_year) - live_last_year} year(s)</div>
                <div class="metric-sub">from {live_last_year} to {int(selected_target_year)} with rolling {FORECAST_WINDOW_YEARS}-year training window</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(live_forecast_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Live forecast could not be generated: {exc}")

    st.markdown("### Saved Forecast / Backtest Output")

    forecast_view_df = forecast_df.copy()
    forecast_mode_label = "Future forecast file"
    dataset_col = pick_column(forecast_view_df, ["Dataset"])

    if not forecast_view_df.empty and dataset_col:
        dataset_aliases = {
            forecast_indicator.lower(),
            forecast_ts_dataset.lower(),
            forecast_value_col.lower(),
        }
        filtered_forecast_df = forecast_view_df[
            forecast_view_df[dataset_col].astype(str).str.strip().str.lower().isin(dataset_aliases)
        ].copy()
        if not filtered_forecast_df.empty:
            forecast_view_df = filtered_forecast_df

    fallback_ts_df = ts_prediction_map.get(forecast_ts_dataset, pd.DataFrame()).copy()

    if forecast_view_df.empty and not fallback_ts_df.empty:
        forecast_mode_label = f"{forecast_indicator} best-model backtest output"
        forecast_view_df = fallback_ts_df.rename(
            columns={"year": "forecast_year", "predicted": "predicted_value"}
        ).copy()
        if "Model" not in forecast_view_df.columns:
            forecast_view_df["Model"] = get_best_model_name(forecast_ts_dataset)
        st.info(
            "No future forecast file was found for this indicator, so this section is "
            "currently showing the available time-series model output instead."
        )

    if forecast_view_df.empty:
        st.warning("No forecast or model-output file was found in the cleaned-data folder.")
    else:
        country_col = pick_column(forecast_view_df, ["country_name"])
        model_col = pick_column(forecast_view_df, ["Model", "model"])
        year_col = pick_column(forecast_view_df, ["forecast_year", "year"])
        pred_col = pick_column(forecast_view_df, ["predicted_value", "predicted", "forecast_value"])

        if not country_col:
            st.warning("The available forecast file does not match the dashboard structure.")
        else:
            st.markdown(f"**Current source:** {forecast_mode_label}")
            forecast_country_options = sorted(forecast_view_df[country_col].dropna().unique().tolist())
            selected_forecast_country = st.selectbox(
                "Forecast country",
                options=forecast_country_options,
            )

            if model_col and year_col and pred_col:
                available_saved_models = sorted(forecast_view_df[model_col].dropna().astype(str).unique().tolist())
                chosen_saved_model = resolved_best_model
                if resolved_best_model in available_saved_models:
                    saved_model_df = forecast_view_df[
                        forecast_view_df[model_col].astype(str) == resolved_best_model
                    ].copy()
                else:
                    chosen_saved_model = available_saved_models[0] if available_saved_models else resolved_best_model
                    saved_model_df = forecast_view_df[
                        forecast_view_df[model_col].astype(str) == chosen_saved_model
                    ].copy()

                hist_country_df = (
                    pop_df[pop_df["country_name"] == selected_forecast_country]
                    .dropna(subset=[forecast_value_col])
                    .sort_values("year")
                    .copy()
                )
                fc_country_df = (
                    saved_model_df[saved_model_df[country_col] == selected_forecast_country]
                    .sort_values(year_col)
                    .copy()
                )

                if fc_country_df.empty:
                    st.info("No forecast rows available for this country.")
                else:
                    st.caption(f"Saved output is locked to best model: {chosen_saved_model}")
                    forecast_figure = go.Figure()

                    forecast_figure.add_trace(
                        go.Scatter(
                            x=hist_country_df["year"],
                            y=hist_country_df[forecast_value_col],
                            mode="lines",
                            name=f"Historical {forecast_indicator}",
                            line=dict(color=forecast_indicator_cfg["color"], width=3),
                        )
                    )

                    forecast_figure.add_trace(
                        go.Scatter(
                            x=fc_country_df[year_col],
                            y=fc_country_df[pred_col],
                            mode="lines+markers",
                            name=f"Output ({chosen_saved_model})",
                            line=dict(color="#d98e04", width=3, dash="dot"),
                            marker=dict(size=8),
                        )
                    )

                    forecast_figure.update_layout(
                        template=chart_template,
                        height=560,
                        margin=dict(l=10, r=10, t=20, b=10),
                        xaxis_title="Year",
                        yaxis_title=forecast_value_label,
                        legend_title_text="Series",
                    )

                    st.plotly_chart(forecast_figure, use_container_width=True)

                    fc_left, fc_right = st.columns([1, 1])

                    with fc_left:
                        latest_forecast_year = int(fc_country_df[year_col].max())
                        latest_forecast_value = float(fc_country_df.loc[fc_country_df[year_col].idxmax(), pred_col])

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Latest Output Year</div>
                                <div class="metric-value">{latest_forecast_year}</div>
                                <div class="metric-sub">{forecast_mode_label}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("")
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Predicted {forecast_indicator}</div>
                                <div class="metric-value">{format_indicator_value(latest_forecast_value, forecast_indicator)}</div>
                                <div class="metric-sub">last available model output</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with fc_right:
                        st.markdown("### Forecast / Output Table")
                        st.dataframe(fc_country_df, use_container_width=True, hide_index=True)
            else:
                st.warning(
                    "The forecast source needs these columns: "
                    "`country_name`, `Model`, `forecast_year`, and `predicted_value` "
                    "(or equivalent forecast output column names)."
                )

# -------------------------------------------------------------------
# Footer actions
# -------------------------------------------------------------------
st.markdown("---")
footer_col1, footer_col2 = st.columns([1, 1])

with footer_col1:
    st.download_button(
        label="Download filtered historical data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=f"filtered_{selected_value_col}_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

with footer_col2:
    if not forecast_df.empty:
        st.download_button(
            label="Download forecast data",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast_outputs.csv",
            mime="text/csv",
            use_container_width=True,
        )
