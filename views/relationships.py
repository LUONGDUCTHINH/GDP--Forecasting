from __future__ import annotations

import streamlit as st

from src.analytics import build_relationship_matrix, calculate_correlations
from src.charts import (
    build_bubble_relationship_chart,
    build_correlation_heatmap,
)
from src.components import (
    render_insight_box,
    render_metric_card,
    render_note_box,
    render_page_header,
)
from src.dashboard_data import load_panel


def render() -> None:
    panel_df = load_panel()

    available_years = sorted(
        panel_df["year"].dropna().astype(int).unique().tolist()
    )
    regions = sorted(
        panel_df["wb_region"].dropna().unique().tolist()
    )

    render_page_header(
        active_key="relationships",
        title="GDP Relationships",
        question=(
            "How is GDP associated with population and life expectancy "
            "across countries, and how should these relationships be interpreted?"
        ),
        description=(
            "This section examines cross-country associations between GDP and the "
            "supporting demographic and development indicators used later in the "
            "forecasting models. The analysis is descriptive and does not imply causality."
        ),
        chips=[
            "Primary focus: GDP per capita and represented GDP",
            "Supporting indicators: population and life expectancy",
            "Correlation is reported with valid sample size",
            "Associations are not interpreted as causal effects",
        ],
    )

    render_note_box(
        "Purpose of this section",
        (
            "The exploratory and country-comparison pages show that economies differ "
            "substantially in scale and development. This page tests whether those "
            "differences are systematically associated with population and life expectancy, "
            "providing descriptive support for their inclusion in the later GDP models."
        ),
    )

    st.markdown("### Relationship controls")
    filter_cols = st.columns([0.75, 0.8, 1.85, 0.8], gap="large")

    with filter_cols[0]:
        selected_year = st.selectbox(
            "Relationship year",
            available_years,
            index=len(available_years) - 1,
            key="rel_year",
        )

    with filter_cols[1]:
        use_all_regions = st.toggle(
            "All regions",
            value=True,
            key="rel_all_regions",
        )

    with filter_cols[2]:
        if use_all_regions:
            selected_regions = regions
            st.caption(
                f"All {len(regions)} World Bank regions are included "
                "in the current sample."
            )
        else:
            selected_regions = st.multiselect(
                "Regions",
                regions,
                default=regions,
                key="rel_regions",
            )

    with filter_cols[3]:
        log_x = st.toggle(
            "Log-scale GDP axis",
            value=True,
            key="rel_log_x",
        )

    if not selected_regions:
        st.warning(
            "Please choose at least one region to calculate "
            "the filtered GDP relationships."
        )
        return

    current_df = panel_df[
        (panel_df["year"] == int(selected_year))
        & panel_df["wb_region"].isin(selected_regions)
    ].copy()

    corr_df = calculate_correlations(current_df)

    if corr_df.empty:
        st.warning(
            "No valid GDP relationship sample is available for "
            "the selected year and region filters."
        )
        return

    st.markdown("### Correlation summary")
    metric_cols = st.columns(4)

    for col, (_, row) in zip(metric_cols, corr_df.iterrows()):
        with col:
            render_metric_card(
                row["pair_label"],
                (
                    f"{row['correlation']:.2f}"
                    if row["n_obs"] >= 3
                    else "N/A"
                ),
                (
                    f"{row['direction']} | {row['strength']} | "
                    f"n = {int(row['n_obs'])}"
                ),
            )

    render_note_box(
        "How to read the correlation metrics",
        (
            "The correlation coefficient summarises the direction and strength of "
            "a linear association. Values closer to +1 indicate stronger positive "
            "association, values closer to -1 indicate stronger negative association, "
            "and values near zero indicate weak linear association. These results do "
            "not establish causal effects."
        ),
    )

    st.markdown("### Visual relationship analysis")
    chart_cols = st.columns([1.25, 1.0], gap="large")

    with chart_cols[0]:
        bubble_df = current_df.dropna(
            subset=[
                "estimated_total_gdp_usd",
                "life_expectancy_years",
                "population_total",
            ]
        ).copy()

        st.markdown(
            f"#### GDP, population and life expectancy in {selected_year}"
        )
        st.caption(
            "Bubble colour shows World Bank region. "
            "Bubble size shows population."
        )

        st.plotly_chart(
            build_bubble_relationship_chart(
                bubble_df,
                title=None,
                x_col="estimated_total_gdp_usd",
                y_col="life_expectancy_years",
                size_col="population_total",
                hover_cols=[
                    "estimated_total_gdp_usd",
                    "gdp_per_capita_usd",
                    "population_total",
                    "life_expectancy_years",
                ],
                log_x=log_x,
            ),
            use_container_width=True,
        )

        render_note_box(
            "Bubble chart interpretation",
            (
                "Countries further to the right have larger represented GDP, while countries higher on "
                "the chart have longer life expectancy. Bubble size reflects population, so the chart "
                "separates economic scale from living-standard outcomes. Regional colours also reveal "
                "whether countries at similar development levels form visible clusters."
            ),
        )

    with chart_cols[1]:
        heatmap_df = build_relationship_matrix(
            current_df,
            use_log_gdp=log_x,
        )

        st.markdown("#### Correlation heatmap")
        st.caption(
            "The matrix summarises the pairwise relationships "
            "for the filtered country sample."
        )

        st.plotly_chart(
            build_correlation_heatmap(
                heatmap_df,
                title=None,
            ),
            use_container_width=True,
        )

        render_note_box(
            "Heatmap interpretation",
            (
                "The heatmap summarises pairwise linear associations among GDP, population, life expectancy, "
                "inflation, unemployment, and internet usage. Strong positive values indicate that two variables "
                "tend to increase together, while negative values indicate an inverse pattern. It is used for "
                "exploratory analysis and multicollinearity screening, not for causal conclusions."
            ),
        )

    render_insight_box(
        "Key takeaway from the two charts",
        [
            (
                "The bubble chart combines economic scale, life expectancy, population, and region in one view."
            ),
            (
                "The heatmap provides a compact statistical summary of pairwise associations and highlights "
                "which variables move together strongly or weakly."
            ),
            (
                "Together, the charts provide descriptive motivation for later model inputs, while predictive "
                "usefulness is evaluated separately using out-of-sample results."
            ),
        ],
    )

    render_note_box(
        "Why a logarithmic GDP axis is useful",
        (
            "GDP values differ substantially across countries. A logarithmic scale "
            "compresses extreme values and makes the overall cross-country pattern "
            "easier to interpret without changing the ordering of observations."
        ),
    )

    st.markdown("### Statistical relationship table")
    display_corr_df = corr_df[
        [
            "pair_label",
            "correlation",
            "direction",
            "strength",
            "n_obs",
        ]
    ].rename(
        columns={
            "pair_label": "Relationship",
            "correlation": "Correlation",
            "direction": "Direction",
            "strength": "Strength",
            "n_obs": "Valid observations",
        }
    )

    st.dataframe(
        display_corr_df,
        use_container_width=True,
        hide_index=True,
    )

    strongest_row = corr_df.loc[
        corr_df["correlation"].abs().idxmax()
    ]

    render_insight_box(
        "Relationship summary",
        [
            (
                f"The strongest visible association in the current sample is "
                f"{strongest_row['pair_label']} with a correlation of "
                f"{strongest_row['correlation']:.2f}."
            ),
            (
                f"This is classified as a "
                f"{strongest_row['strength'].lower()} "
                f"{strongest_row['direction'].lower()} relationship."
            ),
            (
                "Population reflects demographic scale, while life expectancy "
                "provides broader human-development context."
            ),
            (
                "The observed associations support the use of these variables "
                "as explanatory inputs, but they do not prove that changes in one "
                "variable directly cause changes in GDP."
            ),
        ],
    )

    render_note_box(
        "Connection to the forecasting models",
        (
            "The main GDP specifications combine demographic, macroeconomic, "
            "regional, and temporal information rather than relying on any single "
            "correlation. The relationship analysis therefore provides descriptive "
            "motivation for model inputs, while predictive usefulness is assessed "
            "separately through out-of-sample evaluation."
        ),
    )

    render_note_box(
        "Next stage",
        (
            "The exploratory, comparison, and relationship analyses establish the "
            "empirical context for modelling. The next page evaluates the separate "
            "indicator time-series models and the main next-year GDP prediction "
            "specifications on their respective test designs."
        ),
    )