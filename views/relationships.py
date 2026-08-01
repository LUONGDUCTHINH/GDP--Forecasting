from __future__ import annotations

import streamlit as st

from src.analytics import build_relationship_matrix, calculate_correlations
from src.charts import build_bubble_relationship_chart, build_correlation_heatmap
from src.components import render_metric_card, render_note_box, render_page_header
from src.dashboard_data import load_panel
from src.formatting import format_percent


def render() -> None:
    panel_df = load_panel()
    available_years = sorted(panel_df["year"].dropna().astype(int).unique().tolist())
    regions = sorted(panel_df["wb_region"].dropna().unique().tolist())

    render_page_header(
        active_key="relationships",
        title="GDP Relationships",
        question="How is GDP associated with population and life expectancy across countries?",
        description=(
            "GDP remains the central variable on this page. Population and life expectancy are shown as contextual "
            "or explanatory indicators, and every correlation is presented as association rather than causation."
        ),
        chips=[
            "Primary focus: GDP",
            "Context variables: population and life expectancy",
            "Correlation is reported with valid sample size",
        ],
    )

    filter_cols = st.columns([0.75, 0.8, 1.85, 0.8], gap="large")
    with filter_cols[0]:
        selected_year = st.selectbox(
            "Relationship year",
            available_years,
            index=len(available_years) - 1,
            key="rel_year",
        )
    with filter_cols[1]:
        use_all_regions = st.toggle("All regions", value=True, key="rel_all_regions")
    with filter_cols[2]:
        if use_all_regions:
            selected_regions = regions
            st.caption(f"All {len(regions)} World Bank regions are included in the current sample.")
        else:
            selected_regions = st.multiselect(
                "Regions",
                regions,
                default=regions,
                key="rel_regions",
            )
    with filter_cols[3]:
        log_x = st.toggle("Log-scale GDP axis", value=True, key="rel_log_x")

    if not selected_regions:
        st.warning("Please choose at least one region to calculate the filtered GDP relationships.")
        return

    current_df = panel_df[
        (panel_df["year"] == int(selected_year))
        & (panel_df["wb_region"].isin(selected_regions))
    ].copy()

    corr_df = calculate_correlations(current_df)
    if corr_df.empty:
        st.warning("No valid GDP relationship sample is available for the selected year and region filters.")
        return

    metric_cols = st.columns(4)
    for col, (_, row) in zip(metric_cols, corr_df.iterrows()):
        with col:
            render_metric_card(
                row["pair_label"],
                f"{row['correlation']:.2f}" if row["n_obs"] >= 3 else "N/A",
                f"{row['direction']} | {row['strength']} | n = {int(row['n_obs'])}",
            )

    chart_cols = st.columns([1.25, 1.0], gap="large")
    with chart_cols[0]:
        bubble_df = current_df.dropna(
            subset=["estimated_total_gdp_usd", "life_expectancy_years", "population_total"]
        ).copy()
        st.markdown(f"### GDP, population and life expectancy in {selected_year}")
        st.caption("Bubble colour shows World Bank region. Bubble size shows population.")
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
    with chart_cols[1]:
        heatmap_df = build_relationship_matrix(current_df, use_log_gdp=log_x)
        st.markdown("### Compact relationship heatmap")
        st.caption("Correlation matrix for the filtered GDP sample and supporting indicators.")
        st.plotly_chart(
            build_correlation_heatmap(
                heatmap_df,
                title=None,
            ),
            use_container_width=True,
        )

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

    strongest_row = corr_df.iloc[corr_df["correlation"].abs().argsort()[::-1]].iloc[0]
    render_note_box(
        "Statistical interpretation",
        f"The strongest visible association in the current filtered sample is {strongest_row['pair_label']} "
        f"with a correlation of {strongest_row['correlation']:.2f}. This is a {strongest_row['strength'].lower()} "
        f"{strongest_row['direction'].lower()} relationship, not a causal claim.",
    )
