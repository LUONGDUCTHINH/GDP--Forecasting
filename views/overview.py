from __future__ import annotations

import streamlit as st

from src.analytics import aggregate_gdp_trend, calculate_correlations, generate_overview_insights, summarize_period_change
from src.charts import build_ranking_bar, build_trend_chart
from src.components import render_insight_box, render_metric_card, render_note_box, render_page_header
from src.dashboard_data import DASHBOARD_SUBTITLE, DASHBOARD_TITLE, GDP_MAP_1960_PATH, GDP_MAP_2023_PATH, load_panel
from src.formatting import format_currency, format_integer


def render() -> None:
    panel_df = load_panel()
    available_years = sorted(panel_df["year"].dropna().astype(int).unique().tolist())
    selected_year = st.selectbox(
        "Select the focus year for overview rankings",
        available_years,
        index=len(available_years) - 1,
        key="overview_year",
    )

    render_page_header(
        active_key="overview",
        title=DASHBOARD_TITLE,
        question="What does the project cover, and what is the scale of GDP represented in the analytical panel?",
        description=DASHBOARD_SUBTITLE,
        chips=[
            f"Coverage: {available_years[0]}-{available_years[-1]}",
            f"Countries: {format_integer(panel_df['country_code'].nunique())}",
            f"Observations: {format_integer(len(panel_df))}",
            "GDP indicator: GDP per capita (current US$)",
            f"Latest year: {selected_year}",
        ],
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

    highest_total_row = current_df.sort_values("estimated_total_gdp_usd", ascending=False).iloc[0]
    highest_pc_row = current_df.sort_values("gdp_per_capita_usd", ascending=False).iloc[0]
    median_pc = current_df["gdp_per_capita_usd"].median()
    represented_total = current_df["estimated_total_gdp_usd"].sum()

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card(
            "GDP Represented",
            format_currency(represented_total, compact=True),
            f"Estimated from matched GDP per capita x population in {selected_year}",
        )
    with metric_cols[1]:
        render_metric_card(
            "Countries Represented",
            format_integer(current_df["country_code"].nunique()),
            f"Valid GDP-country observations in {selected_year}",
        )
    with metric_cols[2]:
        render_metric_card(
            "Highest GDP per Capita",
            format_currency(highest_pc_row["gdp_per_capita_usd"]),
            f"{highest_pc_row['country_name']} in {selected_year}",
        )
    with metric_cols[3]:
        render_metric_card(
            "Median GDP per Capita",
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
                ["country_name", "estimated_total_gdp_usd", "gdp_per_capita_usd", "population_total"]
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

    insight_cols = st.columns([1.25, 1.0], gap="large")
    with insight_cols[0]:
        render_insight_box("Executive insights", generate_overview_insights(panel_df, selected_year))
    with insight_cols[1]:
        change_note = (
            f"From {trend_summary['start_year']} to {trend_summary['end_year']}, the panel-level "
            f"represented GDP changes by {format_currency(trend_summary['absolute_change'], compact=True)}."
            if trend_summary
            else "No trend summary is available for the current data."
        )
        render_note_box(
            "Interpretation note",
            change_note
            + " The dashboard keeps GDP per capita as the core indicator and only derives total GDP "
            "when GDP per capita and population refer to the same country-year observation.",
        )

    corr_df = calculate_correlations(current_df)
    if not corr_df.empty:
        st.dataframe(
            corr_df[["pair_label", "correlation", "direction", "strength", "n_obs"]].rename(
                columns={
                    "pair_label": "Relationship",
                    "correlation": "Correlation",
                    "direction": "Direction",
                    "strength": "Strength",
                    "n_obs": "Valid observations",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if GDP_MAP_1960_PATH.exists() and GDP_MAP_2023_PATH.exists():
        st.markdown("### Spatial GDP context")
        map_cols = st.columns(2, gap="large")
        with map_cols[0]:
            st.image(str(GDP_MAP_1960_PATH), caption="GDP per capita map snapshot: 1960")
        with map_cols[1]:
            st.image(str(GDP_MAP_2023_PATH), caption="GDP per capita map snapshot: 2023")
