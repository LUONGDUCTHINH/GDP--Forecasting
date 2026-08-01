from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import build_missingness_chart
from src.components import render_metric_card, render_note_box, render_page_header, render_pipeline
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

    render_page_header(
        active_key="data",
        title="Data Workflow",
        question="How was the analytical dataset built, and how much coverage was retained after cleaning and merging?",
        description=(
            "This page tracks the project from raw World Bank-style source files through cleaning, "
            "wide-to-long transformation, merge validation, and final panel construction."
        ),
        chips=[
            f"Final panel rows: {format_integer(len(panel_df))}",
            f"Countries retained: {format_integer(panel_df['country_code'].nunique())}",
            f"Year range retained: {int(panel_df['year'].min())}-{int(panel_df['year'].max())}",
            f"Main-model sample: {format_integer(len(main_model_sample_df))} rows",
        ],
    )

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
            "role": "Role",
            "indicator_name": "Indicator",
            "indicator_code": "Indicator code",
            "countries": "Countries",
            "valid_observations": "Valid observations",
            "missing_values": "Missing values",
            "year_min": "Min year",
            "year_max": "Max year",
        }
    )
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.markdown("### Data pipeline")
    render_pipeline(
        [
            "Raw data sources",
            "Skip metadata rows",
            "Remove unnamed columns",
            "Wide-to-long melt",
            "Numeric conversion",
            "Country-year validation",
            "Dataset merge",
            "Event and feature engineering",
            "Final analytical panel",
        ]
    )

    raw_valid_gdp = int(gdp_summary["valid_observations"])
    merge_retention_pct = (len(panel_df) / raw_valid_gdp) * 100 if raw_valid_gdp else float("nan")
    rows_removed = raw_valid_gdp - len(panel_df)
    duplicate_rows = int(panel_df.duplicated(subset=["country_code", "year"]).sum())

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("Raw GDP rows", format_integer(raw_valid_gdp), "Valid country-year GDP observations after melt")
    with metric_cols[1]:
        render_metric_card("Final panel rows", format_integer(len(panel_df)), "Merged analytical observations")
    with metric_cols[2]:
        render_metric_card("Rows removed", format_integer(rows_removed), "Difference versus valid raw GDP rows")
    with metric_cols[3]:
        render_metric_card("Merge retention", format_percent(merge_retention_pct), "Final panel / valid raw GDP rows")
    with metric_cols[4]:
        render_metric_card("Duplicate country-years", format_integer(duplicate_rows), "Should remain at zero after validation")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card(
            "Missing GDP values",
            format_integer(panel_df["gdp_per_capita_usd"].isna().sum()),
            "Within the final analytical panel",
        )
    with metric_cols[1]:
        render_metric_card(
            "Missing population values",
            format_integer(panel_df["population_total"].isna().sum()),
            "Within the final analytical panel",
        )
    with metric_cols[2]:
        render_metric_card(
            "Missing life expectancy values",
            format_integer(panel_df["life_expectancy_years"].isna().sum()),
            "Within the final analytical panel",
        )
    with metric_cols[3]:
        render_metric_card(
            "Main-model rows",
            format_integer(len(main_model_sample_df)),
            "Listwise-complete sample for the rebuilt GDP models",
        )

    raw_missing_df = pd.DataFrame(
        [
            {"variable": "GDP per capita", "stage": "Raw transformed", "missing_count": int(gdp_summary["missing_values"])},
            {"variable": "Population", "stage": "Raw transformed", "missing_count": int(pop_summary["missing_values"])},
            {"variable": "Life expectancy", "stage": "Raw transformed", "missing_count": int(life_summary["missing_values"])},
            {"variable": "GDP per capita", "stage": "Final panel", "missing_count": int(panel_df["gdp_per_capita_usd"].isna().sum())},
            {"variable": "Population", "stage": "Final panel", "missing_count": int(panel_df["population_total"].isna().sum())},
            {"variable": "Life expectancy", "stage": "Final panel", "missing_count": int(panel_df["life_expectancy_years"].isna().sum())},
        ]
    )

    st.plotly_chart(build_missingness_chart(raw_missing_df), use_container_width=True)

    render_note_box(
        "Methodology note",
        "The final panel keeps GDP per capita as the primary analytical target. Population and life "
        "expectancy remain supporting indicators, while the listwise-complete main-model sample becomes "
        "smaller after adding inflation, unemployment, internet usage, and next-year GDP targets.",
    )

    with st.expander("View cleaned analytical dataset", expanded=False):
        preview_df = cleaned_core_df if not cleaned_core_df.empty else panel_df
        st.dataframe(preview_df.head(15), use_container_width=True)

    with st.expander("View final panel used by the main GDP models", expanded=False):
        st.dataframe(main_model_sample_df.head(15), use_container_width=True)
