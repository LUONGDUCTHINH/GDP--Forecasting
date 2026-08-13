from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import build_missingness_chart
from src.components import (
    render_insight_box,
    render_metric_card,
    render_note_box,
    render_page_header,
    render_pipeline,
)
from src.dashboard_data import (
    load_cleaned_core,
    load_main_model_sample,
    load_panel,
    summarize_raw_source,
)
from src.formatting import format_integer, format_percent


def render() -> None:
    panel_df = load_panel()
    cleaned_core_df = load_cleaned_core()
    main_model_sample_df = load_main_model_sample()

    gdp_summary = summarize_raw_source("gdp")
    pop_summary = summarize_raw_source("population")
    life_summary = summarize_raw_source("life_expectancy")

    panel_year_min = int(panel_df["year"].min())
    panel_year_max = int(panel_df["year"].max())
    panel_countries = int(panel_df["country_code"].nunique())
    panel_regions = int(panel_df["wb_region"].nunique())

    model_year_min = (
        int(main_model_sample_df["year"].min())
        if not main_model_sample_df.empty and "year" in main_model_sample_df.columns
        else None
    )
    model_year_max = (
        int(main_model_sample_df["year"].max())
        if not main_model_sample_df.empty and "year" in main_model_sample_df.columns
        else None
    )
    model_countries = (
        int(main_model_sample_df["country_code"].nunique())
        if not main_model_sample_df.empty and "country_code" in main_model_sample_df.columns
        else 0
    )

    render_page_header(
        active_key="data",
        title="Data Workflow",
        question=(
            "How were the raw country-level datasets transformed into the final analytical panel, "
            "and how did missing data affect the modelling sample?"
        ),
        description=(
            "This page follows the methodology reported in the project: raw World Bank-style datasets "
            "were cleaned, reshaped from wide to long format, merged by country and year, enriched with "
            "regional metadata, and extended through feature engineering before modelling."
        ),
        chips=[
            f"Extended panel: {format_integer(len(panel_df))} rows",
            f"Countries: {format_integer(panel_countries)}",
            f"Regions: {format_integer(panel_regions)}",
            f"Coverage: {panel_year_min}-{panel_year_max}",
            f"Main-model sample: {format_integer(len(main_model_sample_df))} rows",
        ],
    )

    render_note_box(
        "Methodological role of this page",
        (
            "The report separates the full extended panel from the smaller listwise-complete sample used "
            "for the richer GDP models. This distinction is important because additional macroeconomic "
            "predictors improve explanatory richness but reduce data coverage."
        ),
    )

    st.markdown("### Data sources and analytical roles")
    source_df = pd.DataFrame([gdp_summary, pop_summary, life_summary])[
        [
            "display_name",
            "role",
            "indicator_name",
            "indicator_code",
            "countries",
            "valid_observations",
            "missing_values",
            "year_min",
            "year_max",
        ]
    ].rename(
        columns={
            "display_name": "Dataset",
            "role": "Analytical role",
            "indicator_name": "Indicator",
            "indicator_code": "Indicator code",
            "countries": "Countries",
            "valid_observations": "Valid observations",
            "missing_values": "Missing values",
            "year_min": "Minimum year",
            "year_max": "Maximum year",
        }
    )
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    render_note_box(
        "Indicator roles",
        (
            "GDP per capita is the primary economic target. Population provides demographic scale, "
            "while life expectancy acts as a contextual proxy for long-run human development. "
            "Inflation, unemployment, internet usage, regional structure, and global-event variables "
            "enter later in the main GDP modelling stage."
        ),
    )

    st.markdown("### Datasets and engineered variables")
    dataset_col, feature_col = st.columns(2, gap="large")

    with dataset_col:
        render_note_box(
            "Raw datasets and analytical purpose",
            (
                "• World Bank GDP per capita — Primary economic indicator and prediction target.\n\n"
                "• World Bank Population — Demographic scale and labour-market context.\n\n"
                "• World Bank Life Expectancy — Human-development and health proxy.\n\n"
                "• World Bank Inflation — Macroeconomic price-stability indicator.\n\n"
                "• World Bank Unemployment — Labour-market condition.\n\n"
                "• World Bank Internet Users — Digital-development indicator.\n\n"
                "• World Bank Country Metadata — Country classification and regional mapping."
            ),
        )

    with feature_col:
        render_note_box(
            "Engineered features and purpose",
            (
                "• Log GDP per capita — Reduce right-skewness and the influence of extreme values.\n\n"
                "• Log Population — Stabilise large differences in population scale across countries.\n\n"
                "• Population Growth Rate — Capture demographic change through time.\n\n"
                "• Target GDP (t+1) — Convert the task into next-year GDP per capita prediction.\n\n"
                "• World Bank Region Mapping — Support regional comparison and fixed effects.\n\n"
                "• Year Trend — Represent broad long-run temporal movement.\n\n"
                "• Global Event Dummies — Represent major international shocks that affect many countries simultaneously."
            ),
        )

    st.markdown("### Global event dummies")
    event_col, purpose_col = st.columns([1.0, 1.2], gap="large")

    with event_col:
        render_note_box(
            "Events included",
            (
                "• Asian Financial Crisis — 1997-1998\n\n"
                "• Global Financial Crisis — 2008-2009\n\n"
                "• COVID-19 shock — 2020\n\n"
                "• COVID-19 rebound — 2021\n\n"
                "• Ukraine-energy shock — 2022-2024"
            ),
        )

    with purpose_col:
        render_note_box(
            "Why these variables were engineered",
            (
                "Country-specific indicators alone may not fully explain sudden GDP changes caused by "
                "international crises. Binary event variables were therefore engineered to capture "
                "common macroeconomic shocks across countries. They support interpretation of structural "
                "breaks, distinguish normal trends from exceptional periods, and provide the full GDP "
                "specification with explicit global-shock information."
            ),
        )

    render_note_box(
        "How to interpret an event dummy",
        (
            "Each event variable equals 1 during the defined event period and 0 otherwise. "
            "Its estimated coefficient represents the average shift in next-year GDP associated "
            "with that event period after controlling for the other variables in the model. "
            "This remains an associational modelling device rather than proof of causality."
        ),
    )

    st.markdown("### Panel construction pipeline")
    render_pipeline(
        [
            "Raw annual country-level datasets",
            "Skip source metadata rows",
            "Remove unnamed and non-analytical columns",
            "Reshape wide year columns into long country-year format",
            "Convert values to numeric form",
            "Validate country-year uniqueness",
            "Merge datasets by country code and year",
            "Attach World Bank region metadata",
            "Engineer logs, next-year GDP targets, and event variables",
            "Create final extended panel and modelling subsets",
        ]
    )

    raw_valid_gdp = int(gdp_summary["valid_observations"])
    merge_retention_pct = (
        (len(panel_df) / raw_valid_gdp) * 100
        if raw_valid_gdp
        else float("nan")
    )
    rows_removed = raw_valid_gdp - len(panel_df)
    duplicate_rows = int(
        panel_df.duplicated(subset=["country_code", "year"]).sum()
    )
    main_sample_share = (
        (len(main_model_sample_df) / len(panel_df)) * 100
        if len(panel_df)
        else float("nan")
    )

    st.markdown("### Coverage retained through the workflow")
    coverage_cols = st.columns(5)
    with coverage_cols[0]:
        render_metric_card(
            "Valid raw GDP rows",
            format_integer(raw_valid_gdp),
            "Country-year GDP observations after reshaping",
        )
    with coverage_cols[1]:
        render_metric_card(
            "Final extended panel",
            format_integer(len(panel_df)),
            "Merged observations used for EDA and indicator analysis",
        )
    with coverage_cols[2]:
        render_metric_card(
            "Rows removed",
            format_integer(rows_removed),
            "Difference relative to valid raw GDP observations",
        )
    with coverage_cols[3]:
        render_metric_card(
            "Merge retention",
            format_percent(merge_retention_pct),
            "Final panel divided by valid raw GDP rows",
        )
    with coverage_cols[4]:
        render_metric_card(
            "Duplicate country-years",
            format_integer(duplicate_rows),
            "Expected to remain at zero after validation",
        )

    st.markdown("### Extended panel versus main-model sample")
    sample_cols = st.columns(4)
    with sample_cols[0]:
        render_metric_card(
            "Extended panel rows",
            format_integer(len(panel_df)),
            f"{panel_year_min}-{panel_year_max}",
        )
    with sample_cols[1]:
        render_metric_card(
            "Main-model rows",
            format_integer(len(main_model_sample_df)),
            (
                f"{model_year_min}-{model_year_max}"
                if model_year_min is not None and model_year_max is not None
                else "Year range unavailable"
            ),
        )
    with sample_cols[2]:
        render_metric_card(
            "Main-model countries",
            format_integer(model_countries),
            "Countries retained after listwise deletion",
        )
    with sample_cols[3]:
        render_metric_card(
            "Sample retained",
            format_percent(main_sample_share),
            "Main-model sample as a share of the extended panel",
        )

    render_insight_box(
        "Coverage interpretation",
        [
            (
                "The extended panel supports broad descriptive analysis because the three core indicators "
                "are complete after merging."
            ),
            (
                "The main GDP modelling sample is smaller because inflation, unemployment, internet usage, "
                "and next-year targets must all be available for the same country-year observation."
            ),
            (
                "This creates a direct trade-off between model richness and geographical or historical coverage."
            ),
        ],
    )

    render_note_box(
        "Final modelling panel",
        (
            "The final modelling panel integrates demographic, macroeconomic, regional, "
            "and engineered variables into a unified country-year dataset used consistently "
            "across the main GDP forecasting models.\n\n"
            f"• Observations: {format_integer(len(main_model_sample_df))}\n"
            f"• Countries: {format_integer(model_countries)}\n"
            f"• Coverage: {model_year_min}-{model_year_max}\n"
            "• Target: Next-year GDP per capita\n\n"
            "Using the same listwise-complete panel across the competing specifications "
            "supports a fair shared-holdout comparison."
        ),
    )

    st.markdown("### Missingness and data availability")
    core_missing = {
        "GDP per capita": int(panel_df["gdp_per_capita_usd"].isna().sum()),
        "Population": int(panel_df["population_total"].isna().sum()),
        "Life expectancy": int(panel_df["life_expectancy_years"].isna().sum()),
    }

    missing_cols = st.columns(4)
    with missing_cols[0]:
        render_metric_card(
            "Missing GDP values",
            format_integer(core_missing["GDP per capita"]),
            "Within the final extended panel",
        )
    with missing_cols[1]:
        render_metric_card(
            "Missing population values",
            format_integer(core_missing["Population"]),
            "Within the final extended panel",
        )
    with missing_cols[2]:
        render_metric_card(
            "Missing life expectancy values",
            format_integer(core_missing["Life expectancy"]),
            "Within the final extended panel",
        )
    with missing_cols[3]:
        render_metric_card(
            "Listwise-complete GDP sample",
            format_integer(len(main_model_sample_df)),
            "Rows available to the rebuilt main GDP models",
        )

    raw_missing_df = pd.DataFrame(
        [
            {
                "variable": "GDP per capita",
                "stage": "Raw transformed",
                "missing_count": int(gdp_summary["missing_values"]),
            },
            {
                "variable": "Population",
                "stage": "Raw transformed",
                "missing_count": int(pop_summary["missing_values"]),
            },
            {
                "variable": "Life expectancy",
                "stage": "Raw transformed",
                "missing_count": int(life_summary["missing_values"]),
            },
            {
                "variable": "GDP per capita",
                "stage": "Final panel",
                "missing_count": core_missing["GDP per capita"],
            },
            {
                "variable": "Population",
                "stage": "Final panel",
                "missing_count": core_missing["Population"],
            },
            {
                "variable": "Life expectancy",
                "stage": "Final panel",
                "missing_count": core_missing["Life expectancy"],
            },
        ]
    )

    st.plotly_chart(
        build_missingness_chart(raw_missing_df),
        use_container_width=True,
    )

    render_note_box(
        "Why the modelling sample becomes smaller",
        (
            "The richer GDP specifications require inflation, unemployment, internet usage, "
            "regional metadata, and the next-year GDP target in addition to the three core variables. "
            "Rows with missing required predictors are excluded through listwise deletion, so the "
            "main-model sample has narrower coverage than the extended panel used for EDA."
        ),
    )

    st.markdown("### Data inspection")
    with st.expander("Preview the cleaned analytical dataset", expanded=False):
        preview_df = cleaned_core_df if not cleaned_core_df.empty else panel_df
        st.dataframe(
            preview_df.head(15),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Preview the main GDP modelling sample", expanded=False):
        st.dataframe(
            main_model_sample_df.head(15),
            use_container_width=True,
            hide_index=True,
        )

    render_note_box(
        "Continue through the workflow",
        (
            "The next analytical pages use the extended panel for GDP trends, country comparisons, "
            "and relationships. The Forecasting page then uses the saved time-series results and the "
            "smaller shared-holdout GDP modelling sample described here."
        ),
    )