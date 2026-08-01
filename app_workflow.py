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
    page_title="GDP Intelligence Workflow Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path("/Users/tonytony/Final Project")
DATA_DIR = BASE_DIR / "Data" / "Cleaned"
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"

PANEL_PATH = DATA_DIR / "panel_with_event_dummies_and_extra_drivers.csv"
MAIN_MODEL_METRICS_PATH = DATA_DIR / "gdp_main_models_rebuilt_with_lag_metrics.csv"
MAIN_MODEL_YEARLY_PATH = DATA_DIR / "gdp_main_models_rebuilt_with_lag_yearly_summary.csv"
MAIN_MODEL_BIAS_PATH = DATA_DIR / "gdp_main_models_rebuilt_with_lag_bias_summary.csv"
MAIN_MODEL_TEST_PRED_PATH = DATA_DIR / "gdp_main_models_rebuilt_with_lag_test_predictions.csv"

GDP_TS_BEST_PATH = DATA_DIR / "gdp_time_series_best_model_10y.csv"
LIFE_TS_BEST_PATH = DATA_DIR / "life_time_series_best_model_10y.csv"
POP_TS_BEST_PATH = DATA_DIR / "population_time_series_best_model_10y.csv"

GDP_TS_SUMMARY_PATH = DATA_DIR / "gdp_time_series_model_selection_summary_10y.csv"
LIFE_TS_SUMMARY_PATH = DATA_DIR / "life_time_series_model_selection_summary_10y.csv"
POP_TS_SUMMARY_PATH = DATA_DIR / "population_time_series_model_selection_summary_10y.csv"

EVENT_TIMELINE_PATH = FIG_DIR / "figure_4_3_global_event_dummies_timeline.png"
GDP_TS_FIG_PATH = FIG_DIR / "figure_5_1_gdp_ts_model_comparison.png"
LIFE_TS_FIG_PATH = FIG_DIR / "figure_5_2_life_ts_model_comparison.png"
POP_TS_FIG_PATH = FIG_DIR / "figure_5_3_population_ts_model_comparison.png"

GDP_HEATMAP_1960_PATH = OUTPUT_DIR / "gdp per capita by year (1960).png"
GDP_HEATMAP_2023_PATH = OUTPUT_DIR / "gdp per capita by year (2023).png"
LIFE_HEATMAP_1960_PATH = OUTPUT_DIR / "life_expectancy heatmap 1960.png"
LIFE_HEATMAP_2023_PATH = OUTPUT_DIR / "life_expectancy heatmap 2023.png"
POP_HEATMAP_1960_PATH = OUTPUT_DIR / "population heatmap 1960.png"
POP_HEATMAP_2023_PATH = OUTPUT_DIR / "population heatmap 2023.png"


MAIN_MODEL_FORMULAS = {
    "Model 1 - Baseline Dynamic": (
        "GDP_(t+1) ~ GDP_t + Population_t + Life Expectancy_t"
    ),
    "Model 2 - Extended Dynamic": (
        "GDP_(t+1) ~ GDP_t + Population_t + Life Expectancy_t "
        "+ Inflation_t + Unemployment_t + Internet_t"
    ),
    "Model 3 - Full Dynamic": (
        "GDP_(t+1) ~ GDP_t + Population_t + Life Expectancy_t "
        "+ Inflation_t + Unemployment_t + Internet_t "
        "+ Event Dummies + Region Effects + Year Trend"
    ),
}

INDICATOR_CONFIG = {
    "GDP per Capita": {
        "column": "gdp_per_capita_usd",
        "label": "GDP per Capita (US$)",
        "format": "{:,.2f}",
        "hover_format": ":,.2f",
        "color": "#cb7f18",
        "scale": "YlOrBr",
        "agg": "mean",
    },
    "Population": {
        "column": "population_total",
        "label": "Population Total",
        "format": "{:,.0f}",
        "hover_format": ":,.0f",
        "color": "#1f7a5c",
        "scale": "Tealgrn",
        "agg": "sum",
    },
    "Life Expectancy": {
        "column": "life_expectancy_years",
        "label": "Life Expectancy (years)",
        "format": "{:,.2f}",
        "hover_format": ":,.2f",
        "color": "#2563eb",
        "scale": "Viridis",
        "agg": "mean",
    },
}

TS_CONFIG = {
    "GDP": {"column": "gdp_per_capita_usd", "label": "GDP per Capita (US$)"},
    "Life Expectancy": {"column": "life_expectancy_years", "label": "Life Expectancy (years)"},
    "Population": {"column": "population_total", "label": "Population Total"},
}

GLOBAL_EVENT_LABELS = [
    "Asian Financial Crisis (1997-1998)",
    "Global Financial Crisis (2008-2009)",
    "COVID Shock (2020)",
    "COVID Rebound (2021)",
    "Ukraine / Energy Shock (2022-2024)",
    "High Global Rates (2023-2024)",
]

FORECAST_WINDOW_YEARS = 10
STEP_OPTIONS = [
    "01 Overview",
    "02 Data",
    "03 EDA",
    "04 Modelling",
    "05 Forecasting",
]


# -------------------------------------------------------------------
# Styles
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #f7f3ea;
            --surface: rgba(255,255,255,0.88);
            --surface-strong: rgba(255,255,255,0.96);
            --text: #18211d;
            --muted: #5e6b65;
            --teal: #1f7a5c;
            --amber: #cb7f18;
            --blue: #2563eb;
            --line: rgba(24, 33, 29, 0.08);
            --shadow: 0 18px 50px rgba(32, 42, 38, 0.10);
            --radius: 24px;
        }

        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(203, 127, 24, 0.10), transparent 24%),
                radial-gradient(circle at 100% 0%, rgba(31, 122, 92, 0.12), transparent 20%),
                linear-gradient(180deg, #fbf8f2 0%, #f4efe5 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1360px;
            padding-top: 1.9rem;
            padding-bottom: 2.2rem;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255,255,255,0.88)),
                linear-gradient(135deg, #edf6f0, #fbf1de);
            border: 1px solid rgba(24, 33, 29, 0.05);
            border-radius: 34px;
            padding: 34px 36px 32px 36px;
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.75fr);
            gap: 18px;
            align-items: stretch;
        }

        .hero-kicker {
            color: var(--teal);
            font-size: 0.84rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .hero-title {
            font-family: "Avenir Next", "Helvetica Neue", sans-serif;
            font-size: 2.6rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.55rem;
            color: #101814;
        }

        .hero-copy {
            max-width: 860px;
            color: #52605a;
            font-size: 1.03rem;
            line-height: 1.7;
        }

        .hero-side-card {
            background: linear-gradient(155deg, #173f34, #215f4e);
            color: white;
            border-radius: 28px;
            padding: 22px 22px 20px 22px;
            min-height: 100%;
            box-shadow: 0 18px 40px rgba(23, 63, 52, 0.22);
        }

        .hero-side-kicker {
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            opacity: 0.78;
        }

        .hero-side-value {
            font-size: 1.95rem;
            font-weight: 850;
            line-height: 1.05;
            margin-top: 0.32rem;
        }

        .hero-side-copy {
            margin-top: 0.55rem;
            font-size: 0.94rem;
            line-height: 1.65;
            color: rgba(255,255,255,0.84);
        }

        .hero-stat-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 1rem;
        }

        .hero-mini {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 12px 14px;
        }

        .hero-mini-label {
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.76;
        }

        .hero-mini-value {
            font-size: 1.18rem;
            font-weight: 820;
            margin-top: 0.2rem;
        }

        .hero-mini-sub {
            font-size: 0.85rem;
            margin-top: 0.16rem;
            color: rgba(255,255,255,0.78);
        }

        .section-shell {
            background: var(--surface);
            border: 1px solid rgba(24, 33, 29, 0.06);
            border-radius: 28px;
            box-shadow: 0 12px 40px rgba(30, 39, 35, 0.08);
            padding: 1.1rem 1.2rem 1.3rem 1.2rem;
            margin-bottom: 1rem;
        }

        .section-kicker {
            color: var(--teal);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.25rem;
        }

        .section-title {
            font-family: "Avenir Next", "Helvetica Neue", sans-serif;
            color: #14201a;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: #57655f;
            font-size: 0.98rem;
            line-height: 1.65;
            margin-bottom: 0.2rem;
        }

        .metric-card {
            background: var(--surface-strong);
            border: 1px solid rgba(24, 33, 29, 0.06);
            border-radius: 22px;
            padding: 16px 18px;
            box-shadow: 0 12px 30px rgba(30, 39, 35, 0.06);
            height: 100%;
        }

        .metric-label {
            color: #68756f;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .metric-value {
            color: #132019;
            font-family: "Avenir Next", "Helvetica Neue", sans-serif;
            font-size: 2.15rem;
            font-weight: 850;
            line-height: 1.08;
            margin-top: 0.22rem;
        }

        .metric-sub {
            color: #67756e;
            font-size: 0.88rem;
            margin-top: 0.25rem;
            line-height: 1.45;
        }

        .insight-box {
            background: linear-gradient(135deg, rgba(31,122,92,0.12), rgba(203,127,24,0.08));
            border: 1px solid rgba(24, 33, 29, 0.06);
            border-radius: 22px;
            padding: 16px 18px;
        }

        .insight-title {
            color: #153126;
            font-size: 0.9rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .insight-copy {
            color: #304139;
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .data-chip {
            display: inline-block;
            padding: 6px 11px;
            border-radius: 999px;
            background: rgba(31,122,92,0.10);
            color: #1f7a5c;
            font-size: 0.82rem;
            font-weight: 750;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .formula-card {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(24, 33, 29, 0.06);
            border-radius: 22px;
            padding: 16px 18px;
            box-shadow: 0 10px 26px rgba(30, 39, 35, 0.05);
            height: 100%;
        }

        .formula-card.best {
            border: 2px solid rgba(31,122,92,0.36);
            box-shadow: 0 12px 34px rgba(31,122,92,0.14);
        }

        .formula-name {
            font-size: 1.1rem;
            font-weight: 800;
            color: #14211a;
        }

        .formula-badge {
            display: inline-block;
            margin-top: 0.45rem;
            margin-bottom: 0.55rem;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(203,127,24,0.10);
            color: #9a5d00;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .formula-text {
            font-size: 0.95rem;
            color: #425049;
            line-height: 1.65;
        }

        .formula-metric {
            margin-top: 0.8rem;
            font-size: 0.9rem;
            color: #2e3e37;
            line-height: 1.55;
        }

        .small-note {
            color: #5d6b63;
            font-size: 0.92rem;
            line-height: 1.62;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: none !important;
            padding-bottom: 0.3rem;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        .stTabs button[role="tab"] {
            background: rgba(255,255,255,0.76) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(24, 33, 29, 0.08) !important;
            padding: 12px 18px !important;
            color: #21302a !important;
            font-weight: 760 !important;
            box-shadow: 0 10px 22px rgba(34, 46, 40, 0.05);
        }

        .stTabs button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #173f34, #256f59) !important;
            color: white !important;
            border-color: rgba(23, 63, 52, 0.22) !important;
            box-shadow: 0 16px 28px rgba(23, 63, 52, 0.16);
        }

        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input,
        [data-baseweb="base-input"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
            background: rgba(255,255,255,0.92) !important;
            color: #16211b !important;
            border: 1px solid rgba(24, 33, 29, 0.10) !important;
            border-radius: 16px !important;
        }

        .filter-shell {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(24, 33, 29, 0.06);
            border-radius: 24px;
            padding: 14px 16px 10px 16px;
            box-shadow: 0 12px 28px rgba(30, 39, 35, 0.06);
            margin-bottom: 0.9rem;
        }

        .filter-title {
            color: #173f34;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .soft-panel {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(24, 33, 29, 0.06);
            border-radius: 24px;
            padding: 16px 18px;
            box-shadow: 0 12px 28px rgba(30, 39, 35, 0.05);
        }

        .soft-panel-title {
            color: #183229;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .soft-panel-copy {
            color: #56645e;
            line-height: 1.65;
            font-size: 0.95rem;
        }

        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
            display: none !important;
        }

        a, a:hover, a:focus, a:visited {
            text-decoration: none !important;
            color: inherit !important;
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        footer {
            display: none !important;
        }

        section[data-testid="stSidebar"] {
            display: none !important;
        }

        @media (max-width: 1100px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 2.1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
@st.cache_data
def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    df["country_name"] = df["country_name"].astype(str)
    df["country_code"] = df["country_code"].astype(str)
    df["wb_region"] = df["wb_region"].astype(str)
    return df


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


panel_df = load_panel(PANEL_PATH)
main_metrics_df = load_optional_csv(MAIN_MODEL_METRICS_PATH)
main_yearly_df = load_optional_csv(MAIN_MODEL_YEARLY_PATH)
main_bias_df = load_optional_csv(MAIN_MODEL_BIAS_PATH)
main_test_pred_df = load_optional_csv(MAIN_MODEL_TEST_PRED_PATH)

gdp_ts_best_df = load_optional_csv(GDP_TS_BEST_PATH)
life_ts_best_df = load_optional_csv(LIFE_TS_BEST_PATH)
pop_ts_best_df = load_optional_csv(POP_TS_BEST_PATH)

gdp_ts_summary_df = load_optional_csv(GDP_TS_SUMMARY_PATH)
life_ts_summary_df = load_optional_csv(LIFE_TS_SUMMARY_PATH)
pop_ts_summary_df = load_optional_csv(POP_TS_SUMMARY_PATH)


# -------------------------------------------------------------------
# Safety checks
# -------------------------------------------------------------------
required_cols = {"country_name", "country_code", "wb_region", "year"}
missing_cols = required_cols - set(panel_df.columns)
if missing_cols:
    st.error(f"Panel dataset is missing required columns: {sorted(missing_cols)}")
    st.stop()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def format_number(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def format_indicator_value(value: float, indicator_label: str) -> str:
    cfg = INDICATOR_CONFIG[indicator_label]
    if pd.isna(value):
        return "N/A"
    return cfg["format"].format(value)


def render_metric_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_strip(active_step: str) -> None:
    labels = {
        "01 Overview": "Overview",
        "02 Data": "Data",
        "03 EDA": "EDA",
        "04 Modelling": "Modelling",
        "05 Forecasting": "Forecasting",
    }
    cols = st.columns(5)
    for col, step in zip(cols, STEP_OPTIONS):
        klass = "workflow-step active" if step == active_step else "workflow-step"
        with col:
            st.markdown(
                f"""
                <div class="{klass}">
                    <div class="workflow-num">{step.split()[0]}</div>
                    <div class="workflow-title">{labels[step]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_insight_box(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">{title}</div>
            <div class="insight-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_cloud(chips: list[str]) -> None:
    if not chips:
        return
    chip_html = "".join([f'<span class="data-chip">{chip}</span>' for chip in chips])
    st.markdown(
        f"""
        <div class="section-shell" style="padding-top:0.95rem; padding-bottom:0.9rem;">
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def coverage_summary(df: pd.DataFrame, column: str) -> dict:
    temp = df.dropna(subset=[column]).copy()
    return {
        "column": column,
        "observations": int(len(temp)),
        "countries": int(temp["country_code"].nunique()) if not temp.empty else 0,
        "year_min": int(temp["year"].min()) if not temp.empty else None,
        "year_max": int(temp["year"].max()) if not temp.empty else None,
    }


def get_best_main_model_row() -> pd.Series | None:
    if main_metrics_df.empty:
        return None
    temp = main_metrics_df[
        (main_metrics_df["split"].astype(str).str.lower() == "test")
        & (main_metrics_df["scale"].astype(str).str.lower() == "level_gdp_usd")
    ].copy()
    if temp.empty:
        return None
    return temp.sort_values(["RMSE", "MAPE_pct", "MAE"]).iloc[0]


def build_indicator_trend(df: pd.DataFrame, indicator_label: str, template: str) -> go.Figure:
    cfg = INDICATOR_CONFIG[indicator_label]
    value_col = cfg["column"]
    agg = cfg["agg"]

    temp = df.dropna(subset=[value_col]).copy()
    if temp.empty:
        return go.Figure()

    if agg == "sum":
        trend_df = temp.groupby("year", as_index=False)[value_col].sum()
    else:
        trend_df = temp.groupby("year", as_index=False)[value_col].mean()

    fig = px.line(
        trend_df,
        x="year",
        y=value_col,
        template=template,
        markers=True,
        title=f"Global {indicator_label} Trend",
        color_discrete_sequence=[cfg["color"]],
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    return fig


def build_country_spotlight(df: pd.DataFrame, country_name: str, indicator_label: str, template: str) -> go.Figure:
    cfg = INDICATOR_CONFIG[indicator_label]
    value_col = cfg["column"]
    temp = (
        df[df["country_name"] == country_name]
        .dropna(subset=[value_col])
        .sort_values("year")
        .copy()
    )
    if temp.empty:
        return go.Figure()

    fig = px.line(
        temp,
        x="year",
        y=value_col,
        template=template,
        markers=True,
        title=f"{country_name}: {indicator_label} over Time",
        color_discrete_sequence=[cfg["color"]],
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    return fig


def build_region_bar(df: pd.DataFrame, year_value: int, indicator_label: str, template: str) -> go.Figure:
    cfg = INDICATOR_CONFIG[indicator_label]
    value_col = cfg["column"]
    agg = cfg["agg"]
    temp = df[df["year"] == year_value].dropna(subset=[value_col]).copy()
    if temp.empty:
        return go.Figure()

    if agg == "sum":
        bar_df = temp.groupby("wb_region", as_index=False)[value_col].sum()
    else:
        bar_df = temp.groupby("wb_region", as_index=False)[value_col].mean()

    bar_df = bar_df.sort_values(value_col, ascending=False)
    fig = px.bar(
        bar_df,
        x="wb_region",
        y=value_col,
        color=value_col,
        color_continuous_scale=cfg["scale"],
        template=template,
        title=f"Regional {indicator_label} Comparison in {year_value}",
        labels={"wb_region": "World Bank Region", value_col: cfg["label"]},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False)
    return fig


def build_indicator_choropleth(
    df: pd.DataFrame,
    indicator_label: str,
    year_value: int,
    template: str,
    title: str,
) -> go.Figure:
    cfg = INDICATOR_CONFIG[indicator_label]
    value_col = cfg["column"]

    plot_df = df[df["year"] == year_value].dropna(subset=["country_code", value_col]).copy()
    if plot_df.empty:
        empty = go.Figure()
        empty.update_layout(
            template=template,
            height=430,
            title=title,
            margin=dict(l=10, r=10, t=60, b=10),
            annotations=[
                dict(
                    text=f"No {indicator_label.lower()} data for {year_value}.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=15, color="#5d6b63"),
                )
            ],
        )
        return empty

    fig = px.choropleth(
        plot_df,
        locations="country_code",
        color=value_col,
        hover_name="country_name",
        hover_data={"wb_region": True, "year": True, value_col: True},
        color_continuous_scale=cfg["scale"],
        locationmode="ISO-3",
        projection="natural earth",
        template=template,
        labels={value_col: cfg["label"], "wb_region": "WB Region", "year": "Year"},
        title=title,
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=60, b=10),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(24,33,29,0.28)",
            showland=True,
            landcolor="rgba(255,255,255,0.7)",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_traces(marker_line_color="rgba(255,255,255,0.65)", marker_line_width=0.35)
    return fig


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

    last_year = int(clean_df["year"].max())
    if target_year <= last_year:
        raise ValueError(f"Target year must be greater than {last_year}.")
    if len(clean_df) < window_size:
        raise ValueError(
            f"This country has only {len(clean_df)} usable years, fewer than the "
            f"required {window_size}-year forecast window."
        )

    working_df = clean_df.copy()
    rows = []
    for forecast_year in range(last_year + 1, int(target_year) + 1):
        train_window = working_df.tail(window_size).copy()
        train_values = train_window[value_col].astype(float).to_numpy()
        next_value = one_step_forecast_from_values(train_values, model_name)
        rows.append(
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
    return pd.DataFrame(rows)


def get_best_ts_model(dataset_label: str) -> str:
    lookup = {
        "GDP": gdp_ts_best_df,
        "Life Expectancy": life_ts_best_df,
        "Population": pop_ts_best_df,
    }
    df = lookup.get(dataset_label, pd.DataFrame())
    if df.empty:
        fallback = {"GDP": "Naive", "Life Expectancy": "Naive", "Population": "LogHolt"}
        return fallback.get(dataset_label, "Naive")
    return str(df["Model"].iloc[0])


def concat_ts_tables() -> pd.DataFrame:
    frames = []
    if not gdp_ts_summary_df.empty:
        frames.append(gdp_ts_summary_df.assign(dataset_label="GDP"))
    if not life_ts_summary_df.empty:
        frames.append(life_ts_summary_df.assign(dataset_label="Life Expectancy"))
    if not pop_ts_summary_df.empty:
        frames.append(pop_ts_summary_df.assign(dataset_label="Population"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_heatmap_paths(indicator_label: str) -> tuple[Path, Path]:
    lookup = {
        "GDP per Capita": (GDP_HEATMAP_1960_PATH, GDP_HEATMAP_2023_PATH),
        "Population": (POP_HEATMAP_1960_PATH, POP_HEATMAP_2023_PATH),
        "Life Expectancy": (LIFE_HEATMAP_1960_PATH, LIFE_HEATMAP_2023_PATH),
    }
    return lookup[indicator_label]


# -------------------------------------------------------------------
# Global defaults and derived data
# -------------------------------------------------------------------
all_regions = sorted(panel_df["wb_region"].dropna().unique().tolist())
all_countries = sorted(panel_df["country_name"].dropna().unique().tolist())
default_country = "Viet Nam" if "Viet Nam" in all_countries else (all_countries[0] if all_countries else None)
available_years = sorted(panel_df["year"].dropna().astype(int).unique().tolist())
year_min = int(panel_df["year"].min())
year_max = int(panel_df["year"].max())
chart_template = "plotly_white"

latest_year = int(panel_df["year"].max())
n_rows = int(len(panel_df))
n_countries = int(panel_df["country_code"].nunique())
n_regions = int(panel_df["wb_region"].nunique())

best_main_row = get_best_main_model_row()
best_main_model_name = str(best_main_row["model"]) if best_main_row is not None else "N/A"
best_main_rmse = format_number(best_main_row["RMSE"], 2) if best_main_row is not None else "N/A"
best_main_mape = format_number(best_main_row["MAPE_pct"], 2) if best_main_row is not None else "N/A"
best_main_r2 = format_number(best_main_row["R_squared"], 4) if best_main_row is not None else "N/A"

core_coverage_cards = {
    "GDP per Capita": coverage_summary(panel_df, "gdp_per_capita_usd"),
    "Population": coverage_summary(panel_df, "population_total"),
    "Life Expectancy": coverage_summary(panel_df, "life_expectancy_years"),
}

extra_driver_cards = {
    "Inflation": coverage_summary(panel_df, "inflation_pct_clean"),
    "Unemployment": coverage_summary(panel_df, "unemployment_pct_clean"),
    "Internet Usage": coverage_summary(panel_df, "internet_users_pct_clean"),
}

main_model_ready_cols = [
    "gdp_per_capita_usd",
    "population_total",
    "life_expectancy_years",
    "inflation_pct_clean",
    "unemployment_pct_clean",
    "internet_users_pct_clean",
    "target_log_gdp_next_year",
]
main_model_panel_df = panel_df.dropna(subset=main_model_ready_cols).copy()

best_ts_lookup = {
    "GDP": get_best_ts_model("GDP"),
    "Life Expectancy": get_best_ts_model("Life Expectancy"),
    "Population": get_best_ts_model("Population"),
}


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">Final Project Workflow Dashboard</div>
                <div class="hero-title">GDP Intelligence Workflow</div>
                <div class="hero-copy">
                    This dashboard is now structured more like a final presentation board than a control panel.
                    It walks through the project in one clean sequence:
                    <strong>Overview</strong>, <strong>Data</strong>, <strong>EDA</strong>,
                    <strong>Modelling</strong>, and <strong>Forecasting</strong>.
                    The focus remains on GDP, while population and life expectancy support the modelling
                    and future-path forecasting layers.
                </div>
            </div>
            <div class="hero-side-card">
                <div class="hero-side-kicker">Best Main Model</div>
                <div class="hero-side-value">{best_main_model_name}</div>
                <div class="hero-side-copy">
                    Test RMSE {best_main_rmse} · Test MAPE {best_main_mape}% · Test R² {best_main_r2}
                </div>
                <div class="hero-stat-row">
                    <div class="hero-mini">
                        <div class="hero-mini-label">Coverage</div>
                        <div class="hero-mini-value">{n_countries} countries</div>
                        <div class="hero-mini-sub">{year_min} to {year_max}</div>
                    </div>
                    <div class="hero-mini">
                        <div class="hero-mini-label">Forecast Layer</div>
                        <div class="hero-mini-value">10-year window</div>
                        <div class="hero-mini-sub">Best TS model per indicator</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_chip_cloud(
    [
        "3 main GDP models",
        "3 time-series model families",
        "6 explanatory drivers",
        "5 global event blocks",
        "Interactive Streamlit dashboard",
    ]
)

overview_tab, data_tab, eda_tab, modelling_tab, forecasting_tab = st.tabs(
    ["Overview", "Data", "EDA", "Modelling", "Forecasting"]
)


# -------------------------------------------------------------------
# Overview tab
# -------------------------------------------------------------------
with overview_tab:
    render_section_header(
        "Project Story",
        "Overview",
        "This opening view is designed for a final demo. It surfaces the project objective, panel coverage, "
        "and the strongest modelling result immediately, so the audience can understand the direction before going deeper.",
    )

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        render_metric_card("Countries", f"{n_countries}", "Country-level panel coverage")
    with o2:
        render_metric_card("Regions", f"{n_regions}", "World Bank regional groups")
    with o3:
        render_metric_card("Year Window", f"{year_min}-{year_max}", "Historical coverage across the project")
    with o4:
        render_metric_card("Main Model Rows", f"{len(main_model_panel_df):,}", "Complete rows for GDP panel modelling")

    left, right = st.columns([1.08, 0.92])
    with left:
        render_insight_box(
            "Research objective",
            "The project aims to explain and forecast <strong>next-year GDP per capita</strong> using demographic, development, "
            "macroeconomic, and event-based information. GDP is the central target, while population and life expectancy play a dual role "
            "as explanatory indicators and as forecasted series in the future-path workflow.",
        )
        st.markdown("")
        render_insight_box(
            "Why the structure matters",
            "The analytical workflow is intentionally split into two layers. First, panel models explain GDP_(t+1) across countries. "
            "Second, separate time-series models forecast future paths for GDP, population, and life expectancy using rolling 10-year windows.",
        )

    with right:
        st.markdown(
            """
            <div class="soft-panel">
                <div class="soft-panel-title">What the final demo should make clear</div>
                <div class="soft-panel-copy">
                    1. The project is no longer only a population study; it is now a GDP-focused forecasting and explanation project.<br><br>
                    2. The data layer combines core indicators, richer macro drivers, and engineered global-shock features.<br><br>
                    3. The modelling layer compares multiple main GDP specifications and a separate time-series forecasting foundation.<br><br>
                    4. The dashboard turns notebooks and report outputs into a presentation-ready exploration workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### What is included in the project")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_metric_card("Core Datasets", "3", "GDP, Population, Life Expectancy")
    with p2:
        render_metric_card("Extra Drivers", "3", "Inflation, Unemployment, Internet")
    with p3:
        render_metric_card("Shock Blocks", "5", "Global event dummies for GDP context")

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            f"""
            <div class="formula-card best">
                <div class="formula-name">Best Main GDP Result</div>
                <div class="formula-badge">Current headline result</div>
                <div class="formula-text">
                    <strong>{best_main_model_name}</strong> currently gives the strongest balance of low forecast error
                    and stable level prediction in the rebuilt lag-based GDP framework.
                </div>
                <div class="formula-metric">
                    <strong>RMSE:</strong> {best_main_rmse}<br>
                    <strong>MAPE:</strong> {best_main_mape}%<br>
                    <strong>R²:</strong> {best_main_r2}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            f"""
            <div class="formula-card">
                <div class="formula-name">Best Time-Series Layer</div>
                <div class="formula-badge">Used for future paths</div>
                <div class="formula-metric">
                    <strong>GDP:</strong> {best_ts_lookup['GDP']}<br>
                    <strong>Life Expectancy:</strong> {best_ts_lookup['Life Expectancy']}<br>
                    <strong>Population:</strong> {best_ts_lookup['Population']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
# Data tab
# -------------------------------------------------------------------
with data_tab:
    render_section_header(
        "Dataset Architecture",
        "Data",
        "This section explains where the modelling evidence comes from, how broad the country-year coverage is, "
        "and how the merged panel was engineered before running GDP and time-series models.",
    )

    st.markdown(
        """
        <div class="soft-panel">
            <div class="soft-panel-title">Merged panel logic</div>
            <div class="soft-panel-copy">
                The final panel aligns country-year records across GDP, population, life expectancy, inflation,
                unemployment, internet usage, and World Bank region metadata. It also adds log transforms,
                next-year GDP targets, and event dummies so the same cleaned structure can support both EDA
                and modelling.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Core datasets")
    d1, d2, d3 = st.columns(3)
    for col, (label, summary) in zip([d1, d2, d3], core_coverage_cards.items()):
        with col:
            render_metric_card(
                label,
                f"{summary['observations']:,}",
                f"{summary['countries']} countries · {summary['year_min']}-{summary['year_max']}",
            )

    st.markdown("### Additional drivers")
    dd1, dd2, dd3 = st.columns(3)
    for col, (label, summary) in zip([dd1, dd2, dd3], extra_driver_cards.items()):
        with col:
            render_metric_card(
                label,
                f"{summary['observations']:,}",
                f"{summary['countries']} countries with usable values",
            )

    cover_rows = []
    for label in INDICATOR_CONFIG:
        summary = core_coverage_cards[label]
        cover_rows.append(
            {
                "Dataset": label,
                "Observations": summary["observations"],
                "Countries": summary["countries"],
                "Year range": f"{summary['year_min']}-{summary['year_max']}",
            }
        )
    for label, summary in extra_driver_cards.items():
        cover_rows.append(
            {
                "Dataset": label,
                "Observations": summary["observations"],
                "Countries": summary["countries"],
                "Year range": f"{summary['year_min']}-{summary['year_max']}" if summary["year_min"] is not None else "N/A",
            }
        )

    left, right = st.columns([1.0, 1.0])
    with left:
        st.dataframe(pd.DataFrame(cover_rows), use_container_width=True, hide_index=True)
    with right:
        render_insight_box(
            "Feature engineering",
            "<strong>log_gdp_per_capita</strong> stabilises scale; "
            "<strong>log_population_total</strong> captures demographic magnitude; "
            "<strong>target_log_gdp_next_year</strong> defines the GDP_(t+1) prediction task; "
            "<strong>inflation/unemployment/internet cleaned fields</strong> support the richer main models.",
        )
        st.markdown("")
        render_insight_box(
            "Panel subset for main models",
            f"After listwise completion across all required main-model variables, the modelling subset retains "
            f"<strong>{len(main_model_panel_df):,}</strong> rows from <strong>{main_model_panel_df['country_code'].nunique()}</strong> countries. "
            "This is the sample used for the rebuilt dynamic GDP models.",
        )

    st.markdown("### Event engineering")
    render_chip_cloud(GLOBAL_EVENT_LABELS)

    st.markdown("### Panel preview")
    preview_year = st.selectbox(
        "Preview a specific year",
        options=available_years,
        index=len(available_years) - 1,
        key="data_preview_year",
    )
    preview_df = panel_df[panel_df["year"] == preview_year].copy()
    preview_cols = [
        "country_name",
        "country_code",
        "wb_region",
        "year",
        "gdp_per_capita_usd",
        "life_expectancy_years",
        "population_total",
        "inflation_pct_clean",
        "unemployment_pct_clean",
        "internet_users_pct_clean",
    ]
    preview_cols = [col for col in preview_cols if col in preview_df.columns]
    st.dataframe(preview_df[preview_cols].head(30), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# EDA tab
# -------------------------------------------------------------------
with eda_tab:
    render_section_header(
        "Pattern Discovery",
        "EDA",
        "This part turns the merged panel into a visual story. It highlights long-run trends, country and regional gaps, "
        "spatial patterns, and relationships between GDP and the surrounding indicators.",
    )

    st.markdown('<div class="filter-shell"><div class="filter-title">EDA Controls</div></div>', unsafe_allow_html=True)
    ef1, ef2, ef3, ef4 = st.columns([1.65, 1.15, 1.0, 0.9])
    with ef1:
        selected_regions_eda = st.multiselect(
            "Regions",
            options=all_regions,
            default=all_regions,
            key="eda_regions",
        )
    eda_region_df = panel_df[panel_df["wb_region"].isin(selected_regions_eda)].copy() if selected_regions_eda else panel_df.copy()
    eda_country_options = sorted(eda_region_df["country_name"].dropna().unique().tolist())
    eda_default_country = (
        default_country if default_country in eda_country_options else (eda_country_options[0] if eda_country_options else None)
    )
    with ef2:
        selected_country_eda = st.selectbox(
            "Country",
            options=eda_country_options,
            index=eda_country_options.index(eda_default_country) if eda_default_country in eda_country_options else 0,
            key="eda_country",
        )
    with ef3:
        selected_indicator_eda = st.selectbox(
            "Indicator",
            options=list(INDICATOR_CONFIG.keys()),
            index=0,
            key="eda_indicator_new",
        )
    with ef4:
        comparison_year_eda = st.selectbox(
            "Comparison year",
            options=available_years,
            index=len(available_years) - 1,
            key="eda_comp_year",
        )

    eda_cfg = INDICATOR_CONFIG[selected_indicator_eda]
    latest_value = eda_region_df[eda_region_df["year"] == latest_year][eda_cfg["column"]].dropna()
    latest_stat = latest_value.sum() if eda_cfg["agg"] == "sum" else latest_value.mean()
    compare_value = eda_region_df[eda_region_df["year"] == comparison_year_eda][eda_cfg["column"]].dropna()
    compare_stat = compare_value.sum() if eda_cfg["agg"] == "sum" else compare_value.mean()
    delta_value = latest_stat - compare_stat if pd.notna(latest_stat) and pd.notna(compare_stat) else np.nan
    latest_region_df = eda_region_df[eda_region_df["year"] == comparison_year_eda].copy()

    em1, em2, em3 = st.columns(3)
    with em1:
        render_metric_card(
            f"Latest {selected_indicator_eda}",
            format_indicator_value(latest_stat, selected_indicator_eda),
            f"Across selected regions in {latest_year}",
        )
    with em2:
        render_metric_card(
            f"{comparison_year_eda} benchmark",
            format_indicator_value(compare_stat, selected_indicator_eda),
            "Reference year for comparison",
        )
    with em3:
        render_metric_card(
            "Change to latest",
            format_indicator_value(delta_value, selected_indicator_eda),
            f"Difference from {comparison_year_eda} to {latest_year}",
        )

    eda_trend_tab, eda_spatial_tab, eda_rel_tab = st.tabs(
        ["Trend and Spotlight", "Regional and Spatial", "Relationships"]
    )

    with eda_trend_tab:
        t1, t2 = st.columns(2)
        with t1:
            st.plotly_chart(
                build_indicator_trend(eda_region_df, selected_indicator_eda, chart_template),
                use_container_width=True,
            )
        with t2:
            st.plotly_chart(
                build_country_spotlight(eda_region_df, selected_country_eda, selected_indicator_eda, chart_template),
                use_container_width=True,
            )
        i1, i2 = st.columns(2)
        with i1:
            render_insight_box(
                "Reading the trend",
                f"The global path for <strong>{selected_indicator_eda}</strong> shows how the selected regions evolve over time, "
                f"while the country spotlight isolates the full time path for <strong>{selected_country_eda}</strong>.",
            )
        with i2:
            render_insight_box(
                "Why it matters",
                "These charts help separate smooth long-run structure from short-run noise, which is useful when later interpreting "
                "why some variables work better in time-series forecasting while others matter more in panel GDP models.",
            )

    with eda_spatial_tab:
        s1, s2 = st.columns([1.02, 0.98])
        with s1:
            st.plotly_chart(
                build_region_bar(eda_region_df, comparison_year_eda, selected_indicator_eda, chart_template),
                use_container_width=True,
            )
        with s2:
            render_insight_box(
                "Regional reading",
                f"This bar chart ranks World Bank regions for <strong>{selected_indicator_eda}</strong> in <strong>{comparison_year_eda}</strong>. "
                "It helps connect country-level outcomes to broader development blocs rather than reading countries in isolation.",
            )

        map_left, map_right = st.columns(2)
        with map_left:
            map_year_a = st.selectbox("Map year A", options=available_years, index=0, key="eda_map_a")
            st.plotly_chart(
                build_indicator_choropleth(
                    eda_region_df,
                    selected_indicator_eda,
                    map_year_a,
                    chart_template,
                    f"{selected_indicator_eda} map · {map_year_a}",
                ),
                use_container_width=True,
            )
        with map_right:
            map_year_b = st.selectbox("Map year B", options=available_years, index=len(available_years) - 1, key="eda_map_b")
            st.plotly_chart(
                build_indicator_choropleth(
                    eda_region_df,
                    selected_indicator_eda,
                    map_year_b,
                    chart_template,
                    f"{selected_indicator_eda} map · {map_year_b}",
                ),
                use_container_width=True,
            )

        heat_1960_path, heat_2023_path = get_heatmap_paths(selected_indicator_eda)
        hm1, hm2 = st.columns(2)
        with hm1:
            if heat_1960_path.exists():
                st.image(str(heat_1960_path), caption=f"{selected_indicator_eda} heatmap · 1960", use_container_width=True)
        with hm2:
            if heat_2023_path.exists():
                st.image(str(heat_2023_path), caption=f"{selected_indicator_eda} heatmap · 2023", use_container_width=True)

        if EVENT_TIMELINE_PATH.exists():
            st.image(str(EVENT_TIMELINE_PATH), caption="Global event dummies timeline", use_container_width=True)

    with eda_rel_tab:
        r1, r2 = st.columns(2)
        with r1:
            scatter_df = latest_region_df.dropna(
                subset=["gdp_per_capita_usd", "life_expectancy_years", "population_total"]
            ).copy()
            if not scatter_df.empty:
                fig = px.scatter(
                    scatter_df,
                    x="life_expectancy_years",
                    y="gdp_per_capita_usd",
                    color="wb_region",
                    size="population_total",
                    hover_name="country_name",
                    template=chart_template,
                    title=f"GDP vs Life Expectancy ({comparison_year_eda})",
                    labels={
                        "life_expectancy_years": "Life Expectancy (years)",
                        "gdp_per_capita_usd": "GDP per Capita (US$)",
                        "wb_region": "Region",
                    },
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with r2:
            scatter_df = latest_region_df.dropna(
                subset=["gdp_per_capita_usd", "internet_users_pct_clean", "population_total"]
            ).copy()
            if not scatter_df.empty:
                fig = px.scatter(
                    scatter_df,
                    x="internet_users_pct_clean",
                    y="gdp_per_capita_usd",
                    color="wb_region",
                    size="population_total",
                    hover_name="country_name",
                    template=chart_template,
                    title=f"GDP vs Internet Usage ({comparison_year_eda})",
                    labels={
                        "internet_users_pct_clean": "Internet Users (%)",
                        "gdp_per_capita_usd": "GDP per Capita (US$)",
                        "wb_region": "Region",
                    },
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

        corr_cols = [
            "gdp_per_capita_usd",
            "life_expectancy_years",
            "population_total",
            "inflation_pct_clean",
            "unemployment_pct_clean",
            "internet_users_pct_clean",
        ]
        corr_ready = eda_region_df[corr_cols].dropna().copy()
        if not corr_ready.empty:
            corr_matrix = corr_ready.corr(numeric_only=True).round(3)
            st.dataframe(corr_matrix, use_container_width=True)


# -------------------------------------------------------------------
# Modelling tab
# -------------------------------------------------------------------
with modelling_tab:
    render_section_header(
        "Model Comparison",
        "Modelling",
        "This section compares the three main GDP models and the time-series forecasting foundation. "
        "The layout is intentionally visual, so the audience can understand both the ranking and the modelling logic quickly.",
    )

    render_insight_box(
        "How to read this section",
        "The top block compares the three main GDP models used to predict GDP_(t+1). Below that, the time-series layer shows the "
        "best model family chosen separately for GDP, life expectancy, and population under rolling backtesting.",
    )
    st.markdown("")

    if main_metrics_df.empty:
        st.warning("Main model metrics file is missing.")
    else:
        test_level_df = main_metrics_df[
            (main_metrics_df["split"].astype(str).str.lower() == "test")
            & (main_metrics_df["scale"].astype(str).str.lower() == "level_gdp_usd")
        ].copy()
        bias_lookup = (
            main_bias_df.set_index("model").to_dict("index")
            if not main_bias_df.empty and "model" in main_bias_df.columns
            else {}
        )
        sorted_models = [
            "Model 1 - Baseline Dynamic",
            "Model 2 - Extended Dynamic",
            "Model 3 - Full Dynamic",
        ]

        model_cards = st.columns(3)
        for col, model_name in zip(model_cards, sorted_models):
            with col:
                row = test_level_df[test_level_df["model"] == model_name]
                row = row.iloc[0] if not row.empty else None
                bias = bias_lookup.get(model_name, {})
                best_class = "best" if model_name == best_main_model_name else ""
                rmse = format_number(row["RMSE"], 2) if row is not None else "N/A"
                mape = format_number(row["MAPE_pct"], 2) if row is not None else "N/A"
                r2 = format_number(row["R_squared"], 4) if row is not None else "N/A"
                ratio = format_number(bias.get("pred_to_actual_mean_ratio", np.nan), 4)
                st.markdown(
                    f"""
                    <div class="formula-card {best_class}">
                        <div class="formula-name">{model_name}</div>
                        <div class="formula-badge">{'Best current specification' if model_name == best_main_model_name else 'Compared specification'}</div>
                        <div class="formula-text"><strong>Formula:</strong> {MAIN_MODEL_FORMULAS[model_name]}</div>
                        <div class="formula-metric">
                            <strong>RMSE:</strong> {rmse}<br>
                            <strong>MAPE:</strong> {mape}%<br>
                            <strong>R²:</strong> {r2}<br>
                            <strong>Pred / Actual Ratio:</strong> {ratio}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        c_left, c_right = st.columns(2)
        compare_df = test_level_df.sort_values("RMSE")
        with c_left:
            fig = px.bar(
                compare_df,
                x="model",
                y="RMSE",
                color="model",
                template=chart_template,
                title="Main GDP models · test RMSE",
                text_auto=".2f",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c_right:
            fig = px.bar(
                compare_df,
                x="model",
                y="MAPE_pct",
                color="model",
                template=chart_template,
                title="Main GDP models · test MAPE (%)",
                text_auto=".2f",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        model_pick = st.selectbox(
            "Inspect a main model in detail",
            options=sorted_models,
            index=sorted_models.index(best_main_model_name) if best_main_model_name in sorted_models else 0,
            key="modelling_pick",
        )
        yearly_pick_df = main_yearly_df[main_yearly_df["model"] == model_pick].copy()
        pred_pick_df = main_test_pred_df[main_test_pred_df["model"] == model_pick].copy()

        m1, m2 = st.columns(2)
        with m1:
            if not yearly_pick_df.empty:
                line_fig = go.Figure()
                line_fig.add_trace(
                    go.Scatter(
                        x=yearly_pick_df["target_year"],
                        y=yearly_pick_df["actual_mean_gdp"],
                        mode="lines+markers",
                        name="Actual",
                        line=dict(color="#173f34", width=3),
                    )
                )
                line_fig.add_trace(
                    go.Scatter(
                        x=yearly_pick_df["target_year"],
                        y=yearly_pick_df["predicted_mean_gdp"],
                        mode="lines+markers",
                        name="Predicted",
                        line=dict(color="#cb7f18", width=3, dash="dash"),
                    )
                )
                line_fig.update_layout(
                    template=chart_template,
                    title=f"{model_pick} · mean actual vs predicted GDP",
                    height=430,
                    margin=dict(l=10, r=10, t=60, b=10),
                    xaxis_title="Target year",
                    yaxis_title="GDP per Capita (US$)",
                )
                st.plotly_chart(line_fig, use_container_width=True)
        with m2:
            if not pred_pick_df.empty:
                scatter_fig = px.scatter(
                    pred_pick_df,
                    x="actual_gdp_next_year",
                    y="predicted_gdp_next_year",
                    color="wb_region",
                    hover_name="country_name",
                    template=chart_template,
                    title=f"{model_pick} · actual vs predicted GDP",
                    labels={
                        "actual_gdp_next_year": "Actual GDP per Capita (US$)",
                        "predicted_gdp_next_year": "Predicted GDP per Capita (US$)",
                        "wb_region": "Region",
                    },
                )
                min_val = float(min(pred_pick_df["actual_gdp_next_year"].min(), pred_pick_df["predicted_gdp_next_year"].min()))
                max_val = float(max(pred_pick_df["actual_gdp_next_year"].max(), pred_pick_df["predicted_gdp_next_year"].max()))
                scatter_fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="45° reference",
                    )
                )
                scatter_fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("### Time-series foundation")
    ts_card_cols = st.columns(3)
    ts_best_rows = {
        "GDP": gdp_ts_best_df.iloc[0] if not gdp_ts_best_df.empty else None,
        "Life Expectancy": life_ts_best_df.iloc[0] if not life_ts_best_df.empty else None,
        "Population": pop_ts_best_df.iloc[0] if not pop_ts_best_df.empty else None,
    }
    for col, dataset_label in zip(ts_card_cols, ["GDP", "Life Expectancy", "Population"]):
        with col:
            row = ts_best_rows[dataset_label]
            model = str(row["Model"]) if row is not None else "N/A"
            rmse = format_number(row["RMSE"], 2) if row is not None else "N/A"
            mape = format_number(row["MAPE_pct"], 2) if row is not None else "N/A"
            render_metric_card(
                f"{dataset_label} best model",
                model,
                f"RMSE {rmse} · MAPE {mape}%",
            )

    ts_tab1, ts_tab2, ts_tab3 = st.tabs(["GDP", "Life Expectancy", "Population"])
    ts_views = [
        (ts_tab1, gdp_ts_summary_df, GDP_TS_FIG_PATH),
        (ts_tab2, life_ts_summary_df, LIFE_TS_FIG_PATH),
        (ts_tab3, pop_ts_summary_df, POP_TS_FIG_PATH),
    ]
    for tab, summary_df, fig_path in ts_views:
        with tab:
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
            if not summary_df.empty:
                st.dataframe(summary_df.round(4), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# Forecasting tab
# -------------------------------------------------------------------
with forecasting_tab:
    render_section_header(
        "Future Paths",
        "Forecasting",
        "This final section generates future values using the selected future-projection model for each indicator. "
        "It keeps the rolling backtest winner for evaluation, while allowing a different long-horizon projection model "
        "when that produces a more plausible future path.",
    )

    st.markdown('<div class="filter-shell"><div class="filter-title">Forecast Controls</div></div>', unsafe_allow_html=True)
    ff1, ff2 = st.columns([1.3, 0.7])
    with ff1:
        forecast_country = st.selectbox(
            "Country",
            options=all_countries,
            index=all_countries.index(default_country) if default_country in all_countries else 0,
            key="forecast_country_v2",
        )
    with ff2:
        forecast_target_year = st.number_input(
            "Target year",
            min_value=year_max + 1,
            max_value=year_max + 25,
            value=min(2040, year_max + 17),
            step=1,
            key="forecast_year_v2",
        )

    render_insight_box(
        "Forecast workflow",
        "For each dataset, the app takes the last 10 usable years for the selected country, identifies the rolling "
        "backtest winner, then applies the deployed future-projection model and rolls the prediction forward "
        "recursively until the target year is reached.",
    )

    forecast_country_df = panel_df[panel_df["country_name"] == forecast_country].copy().sort_values("year")
    if forecast_country_df.empty:
        st.warning("No data is available for the selected country.")
    else:
        forecast_results = {}
        forecast_errors = {}
        for dataset_label, cfg in TS_CONFIG.items():
            model_name = best_ts_lookup[dataset_label]
            try:
                forecast_results[dataset_label] = forecast_series_to_target_year(
                    forecast_country_df,
                    cfg["column"],
                    model_name,
                    int(forecast_target_year),
                    window_size=FORECAST_WINDOW_YEARS,
                )
            except Exception as exc:
                forecast_errors[dataset_label] = str(exc)

        fc1, fc2, fc3 = st.columns(3)
        forecast_summary_rows = []
        for col, dataset_label in zip([fc1, fc2, fc3], ["GDP", "Population", "Life Expectancy"]):
            with col:
                if dataset_label in forecast_results and not forecast_results[dataset_label].empty:
                    row = forecast_results[dataset_label].iloc[-1]
                    indicator_label = (
                        "GDP per Capita" if dataset_label == "GDP"
                        else "Population" if dataset_label == "Population"
                        else "Life Expectancy"
                    )
                    render_metric_card(
                        f"{dataset_label} · {forecast_target_year}",
                        format_indicator_value(row["predicted_value"], indicator_label),
                        f"{best_ts_lookup[dataset_label]} · window {int(row['window_start_year'])}-{int(row['window_end_year'])}",
                    )
                    forecast_summary_rows.append(
                        {
                            "Dataset": dataset_label,
                            "Best model": best_ts_lookup[dataset_label],
                            "Target year": int(forecast_target_year),
                            "Window used": f"{int(row['window_start_year'])}-{int(row['window_end_year'])}",
                            "Forecast value": format_indicator_value(row["predicted_value"], indicator_label),
                        }
                    )
                else:
                    render_metric_card(
                        dataset_label,
                        "N/A",
                        forecast_errors.get(dataset_label, "Forecast could not be generated."),
                    )

        path_tab1, path_tab2, path_tab3, path_tab4 = st.tabs(
            ["GDP Path", "Population Path", "Life Expectancy Path", "Forecast Table"]
        )
        for tab, dataset_label in zip([path_tab1, path_tab2, path_tab3], ["GDP", "Population", "Life Expectancy"]):
            with tab:
                cfg = TS_CONFIG[dataset_label]
                value_col = cfg["column"]
                history_df = (
                    forecast_country_df[["year", value_col]]
                    .dropna()
                    .drop_duplicates(subset=["year"])
                    .sort_values("year")
                    .copy()
                )
                if dataset_label in forecast_results and not forecast_results[dataset_label].empty:
                    fdf = forecast_results[dataset_label].copy()
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=history_df["year"],
                            y=history_df[value_col],
                            mode="lines+markers",
                            name="Historical",
                            line=dict(color="#173f34", width=3),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fdf["forecast_year"],
                            y=fdf["predicted_value"],
                            mode="lines+markers",
                            name=f"Forecast ({best_ts_lookup[dataset_label]})",
                            line=dict(color="#cb7f18", width=3, dash="dash"),
                        )
                    )
                    fig.update_layout(
                        template=chart_template,
                        title=f"{forecast_country} · {dataset_label} forecast path to {forecast_target_year}",
                        height=450,
                        margin=dict(l=10, r=10, t=60, b=10),
                        xaxis_title="Year",
                        yaxis_title=cfg["label"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(forecast_errors.get(dataset_label, "Forecast could not be generated."))

        with path_tab4:
            if forecast_summary_rows:
                st.dataframe(pd.DataFrame(forecast_summary_rows), use_container_width=True, hide_index=True)
            combined_rows = []
            for dataset_label, fdf in forecast_results.items():
                temp = fdf.copy()
                temp["dataset"] = dataset_label
                combined_rows.append(temp)
            if combined_rows:
                st.dataframe(pd.concat(combined_rows, ignore_index=True).round(4), use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="soft-panel">
                <div class="soft-panel-title">Interpretation note</div>
                <div class="soft-panel-copy">
                    This forecasting block is intentionally separated from the main GDP panel models. The main models explain how GDP behaves
                    as a function of other indicators, while the forecasting layer creates future paths using each indicator’s own historical pattern.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
