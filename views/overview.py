from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.analytics import (
    aggregate_gdp_trend,
    generate_overview_insights,
    get_best_main_model,
    summarize_period_change,
)
from src.charts import build_ranking_bar, build_trend_chart
from src.components import (
    render_insight_box,
    render_metric_card,
    render_note_box,
    render_page_header,
    render_pipeline,
)
from src.dashboard_data import (
    DASHBOARD_SUBTITLE,
    DASHBOARD_TITLE,
    GDP_MAP_1960_PATH,
    GDP_MAP_2023_PATH,
    load_main_model_metrics,
    load_panel,
)
from src.formatting import format_currency, format_integer


WORKFLOW_IMAGE_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "project_workflow.png"
)


def render() -> None:
    panel_df = load_panel()
    main_metrics_df = load_main_model_metrics()

    available_years = sorted(
        panel_df["year"].dropna().astype(int).unique().tolist()
    )
    first_year = available_years[0]
    latest_year = available_years[-1]

    render_page_header(
        active_key="overview",
        title=DASHBOARD_TITLE,
        question=(
            "How does the project combine data preparation, exploratory analysis, "
            "time-series forecasting, and GDP prediction into one interpretable workflow?"
        ),
        description=DASHBOARD_SUBTITLE,
        chips=[
            f"Coverage: {first_year}-{latest_year}",
            f"Countries: {format_integer(panel_df['country_code'].nunique())}",
            f"Observations: {format_integer(len(panel_df))}",
            f"Variables: {format_integer(len(panel_df.columns))}",
            "Primary target: next-year GDP per capita",
        ],
    )

    render_note_box(
        "Project purpose",
        (
            "This project develops an interpretable country-level GDP per capita forecasting "
            "workflow using demographic and macroeconomic indicators. It combines exploratory "
            "analysis, rolling time-series evaluation, multivariate GDP prediction, model "
            "benchmarking, robustness checks, and interactive Streamlit reporting."
        ),
    )

    st.markdown("### Project profile")
    profile_cols = st.columns(4)
    with profile_cols[0]:
        render_metric_card(
            "Country-year observations",
            format_integer(len(panel_df)),
            "Final extended analytical panel",
        )
    with profile_cols[1]:
        render_metric_card(
            "Countries",
            format_integer(panel_df["country_code"].nunique()),
            "Countries retained after cleaning and merging",
        )
    with profile_cols[2]:
        render_metric_card(
            "Time coverage",
            f"{first_year}-{latest_year}",
            "Annual country-level observations",
        )
    with profile_cols[3]:
        render_metric_card(
            "Variables",
            format_integer(len(panel_df.columns)),
            "Core, macroeconomic, regional, and engineered fields",
        )

    st.markdown("### End-to-end analytical workflow")
    st.caption(
        "The dashboard is the final communication layer of the same workflow used in "
        "the report: raw data, cleaning, panel construction, exploratory analysis, "
        "indicator forecasting, main GDP modelling, evaluation, and future projection."
    )

    render_pipeline(
    [
        "Raw data",
        "Data cleaning",
        "Metadata merge and region mapping",
        "Panel construction and feature engineering",
        "Exploratory data analysis",
        "Indicator time-series forecasting",
        "Main GDP prediction models",
        "Benchmarking and robustness checks",
        "Future GDP forecasting",
        "Dashboard and report outputs",
    ]
)

    with st.expander("Research questions addressed by the dashboard", expanded=False):
        st.markdown(
            """
            1. What long-term patterns are visible in GDP per capita, population, and life expectancy across countries and regions?
            2. Which time-series models perform best for GDP, population, and life expectancy under rolling 10-year evaluation?
            3. How much predictive improvement is obtained when moving from a simple GDP baseline to broader macroeconomic and structural specifications?
            4. How can global event dummies and regional effects be incorporated into a practical GDP forecasting workflow?
            5. How effectively can the analytical outputs be communicated through an interactive dashboard?
            """
        )

    best_main = get_best_main_model(main_metrics_df)
    executive_points = [
        (
            f"The final panel contains {format_integer(len(panel_df))} observations "
            f"covering {format_integer(panel_df['country_code'].nunique())} countries "
            f"from {first_year} to {latest_year}."
        ),
        (
            "GDP per capita remains the central target, while population, life expectancy, "
            "inflation, unemployment, internet usage, regional structure, and global-event "
            "variables provide demographic and macroeconomic context."
        ),
        (
            "The modelling workflow separates indicator-level time-series forecasting from "
            "multivariate next-year GDP prediction so that the two analytical tasks are not conflated."
        ),
    ]
    if best_main is not None:
        executive_points.append(
            f"The strongest shared-holdout GDP configuration is {best_main['model']} "
            f"with a test RMSE of {best_main['RMSE']:.2f}."
        )

    render_insight_box("Executive summary", executive_points)

    st.markdown("### Economic snapshot")
    selected_year = st.selectbox(
        "Select the focus year for the overview",
        available_years,
        index=len(available_years) - 1,
        key="overview_year",
    )

    current_df = panel_df[
        (panel_df["year"] == int(selected_year))
        & panel_df["gdp_per_capita_usd"].notna()
        & panel_df["estimated_total_gdp_usd"].notna()
    ].copy()

    if current_df.empty:
        st.warning("No valid GDP observations are available for the selected overview year.")
        return

    trend_df = aggregate_gdp_trend(panel_df, "represented_total_gdp")
    trend_df = trend_df[trend_df["year"] <= int(selected_year)].copy()
    trend_summary = summarize_period_change(trend_df)

    highest_pc_row = current_df.sort_values(
        "gdp_per_capita_usd", ascending=False
    ).iloc[0]
    median_pc = current_df["gdp_per_capita_usd"].median()
    represented_total = current_df["estimated_total_gdp_usd"].sum()

    snapshot_cols = st.columns(4)
    with snapshot_cols[0]:
        render_metric_card(
            "GDP represented",
            format_currency(represented_total, compact=True),
            (
                "Derived from matched GDP per capita × population "
                f"observations in {selected_year}"
            ),
        )
    with snapshot_cols[1]:
        render_metric_card(
            "Countries represented",
            format_integer(current_df["country_code"].nunique()),
            f"Valid country observations in {selected_year}",
        )
    with snapshot_cols[2]:
        render_metric_card(
            "Highest GDP per capita",
            format_currency(highest_pc_row["gdp_per_capita_usd"]),
            f"{highest_pc_row['country_name']} in {selected_year}",
        )
    with snapshot_cols[3]:
        render_metric_card(
            "Median GDP per capita",
            format_currency(median_pc),
            f"Cross-country median in {selected_year}",
        )

    chart_cols = st.columns([1.25, 1.0], gap="large")
    with chart_cols[0]:
        st.plotly_chart(
            build_trend_chart(
                trend_df,
                title="Estimated GDP represented across covered countries over time",
                y_label="Estimated total GDP (current US$)",
            ),
            use_container_width=True,
        )
    with chart_cols[1]:
        top_rank_df = (
            current_df.nlargest(10, "estimated_total_gdp_usd")[
                [
                    "country_name",
                    "estimated_total_gdp_usd",
                    "gdp_per_capita_usd",
                    "population_total",
                ]
            ]
            .copy()
        )
        st.plotly_chart(
            build_ranking_bar(
                top_rank_df,
                value_col="estimated_total_gdp_usd",
                label_col="country_name",
                title=f"Top 10 countries by represented GDP in {selected_year}",
                x_label="Estimated total GDP (current US$)",
                color_scale="Blues",
            ),
            use_container_width=True,
        )

    st.caption(
        "The ranking is a cross-sectional snapshot for the selected year. "
        "Long-term and country-level patterns are explored in the following analytical pages."
    )

    insight_cols = st.columns([1.25, 1.0], gap="large")
    with insight_cols[0]:
        render_insight_box(
            "Overview insights",
            generate_overview_insights(panel_df, selected_year),
        )
    with insight_cols[1]:
        change_note = (
            f"From {trend_summary['start_year']} to {trend_summary['end_year']}, "
            "represented GDP across the covered panel changes by "
            f"{format_currency(trend_summary['absolute_change'], compact=True)}."
            if trend_summary
            else "No trend summary is available for the current data."
        )
        render_note_box(
            "Interpretation note",
            (
                change_note
                + " Represented GDP is a derived scale measure. GDP per capita remains "
                "the project's primary economic target for cross-country forecasting."
            ),
        )

    if GDP_MAP_1960_PATH.exists() and GDP_MAP_2023_PATH.exists():
        st.markdown("### Spatial GDP context")
        st.caption(
            "The two maps provide a high-level view of how GDP per capita distribution "
            "changed between the beginning and end of the study period."
        )
        map_cols = st.columns(2, gap="large")
        with map_cols[0]:
            st.image(
                str(GDP_MAP_1960_PATH),
                caption="GDP per capita map snapshot: 1960",
            )
        with map_cols[1]:
            st.image(
                str(GDP_MAP_2023_PATH),
                caption="GDP per capita map snapshot: 2023",
            )

    render_note_box(
        "Continue through the workflow",
        (
            "Use Data Workflow to review panel construction, then continue to GDP Trends, "
            "country comparison, relationships, forecasting, and final findings. "
            "The page order follows the analytical logic used in the report."
        ),
    )
