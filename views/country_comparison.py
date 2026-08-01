from __future__ import annotations

import streamlit as st

from src.analytics import build_country_growth_frame, filter_panel
from src.charts import build_country_history_chart, build_growth_scatter, build_ranking_bar
from src.components import render_insight_box, render_metric_card, render_note_box, render_page_header
from src.dashboard_data import load_panel
from src.formatting import format_currency, format_percent


def render() -> None:
    panel_df = load_panel()
    years = sorted(panel_df["year"].dropna().astype(int).unique().tolist())
    regions = sorted(panel_df["wb_region"].dropna().unique().tolist())

    default_start = years[-11] if len(years) > 10 else years[0]
    default_end = years[-1]

    render_page_header(
        active_key="comparison",
        title="GDP Growth and Country Comparison",
        question="Which countries have the largest represented economies, and which countries are growing fastest?",
        description=(
            "This comparison keeps economic size and economic growth separate. Size uses represented GDP derived "
            "from matched GDP-per-capita and population observations, while growth tracks the change across the selected period."
        ),
        chips=[
            f"Default comparison window: {default_start}-{default_end}",
            "Economic size and growth are shown separately",
            "Population and life expectancy remain contextual indicators",
        ],
    )

    filter_cols = st.columns([0.8, 0.8, 1.3, 0.8], gap="large")
    with filter_cols[0]:
        start_year = st.selectbox("Start year", years[:-1], index=max(0, years[:-1].index(default_start)), key="compare_start")
    with filter_cols[1]:
        end_year = st.selectbox("End year", years[1:], index=years[1:].index(default_end), key="compare_end")
    with filter_cols[2]:
        selected_regions = st.multiselect("Regions", regions, default=regions, key="compare_regions")
    with filter_cols[3]:
        top_n = st.slider("Top N countries", min_value=5, max_value=20, value=10, step=1, key="compare_top_n")

    if end_year <= start_year:
        st.warning("End year must be later than start year for GDP growth comparison.")
        return

    growth_df = build_country_growth_frame(panel_df, start_year, end_year, selected_regions)
    if growth_df.empty:
        st.warning("The selected period and region filters do not leave enough valid country pairs to compare.")
        return

    largest_row = growth_df.sort_values("end_total_gdp", ascending=False).iloc[0]
    fastest_row = growth_df.sort_values("cagr_pct", ascending=False).iloc[0]
    absolute_row = growth_df.sort_values("absolute_change_gdp", ascending=False).iloc[0]
    decline_df = growth_df[growth_df["absolute_change_gdp"] < 0].copy()

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card(
            "Largest economy",
            format_currency(largest_row["end_total_gdp"], compact=True),
            f"{largest_row['country_name']} in {end_year}",
        )
    with metric_cols[1]:
        render_metric_card(
            "Fastest GDP CAGR",
            format_percent(fastest_row["cagr_pct"]),
            f"{fastest_row['country_name']} from {start_year} to {end_year}",
        )
    with metric_cols[2]:
        render_metric_card(
            "Largest absolute increase",
            format_currency(absolute_row["absolute_change_gdp"], compact=True),
            f"{absolute_row['country_name']} over the selected period",
        )
    with metric_cols[3]:
        if decline_df.empty:
            render_metric_card("Largest decline", "None", "No negative GDP change in the current filter")
        else:
            decline_row = decline_df.sort_values("absolute_change_gdp").iloc[0]
            render_metric_card(
                "Largest decline",
                format_currency(decline_row["absolute_change_gdp"], compact=True),
                f"{decline_row['country_name']} from {start_year} to {end_year}",
            )

    row_one = st.columns(2, gap="large")
    with row_one[0]:
        st.plotly_chart(
            build_ranking_bar(
                growth_df.nlargest(top_n, "end_total_gdp")[["country_name", "end_total_gdp"]],
                value_col="end_total_gdp",
                label_col="country_name",
                title=f"Top {top_n} countries by represented GDP in {end_year}",
                x_label="Estimated total GDP (current US$)",
                color_scale="Blues",
            ),
            use_container_width=True,
        )
    with row_one[1]:
        st.plotly_chart(
            build_ranking_bar(
                growth_df.nlargest(top_n, "cagr_pct")[["country_name", "cagr_pct"]],
                value_col="cagr_pct",
                label_col="country_name",
                title=f"Top {top_n} countries by GDP CAGR ({start_year}-{end_year})",
                x_label="GDP CAGR (%)",
                color_scale="Greens",
            ),
            use_container_width=True,
        )

    row_two = st.columns(2, gap="large")
    with row_two[0]:
        st.markdown("### Top countries by absolute GDP increase")
        st.plotly_chart(
            build_ranking_bar(
                growth_df.nlargest(top_n, "absolute_change_gdp")[["country_name", "absolute_change_gdp"]],
                value_col="absolute_change_gdp",
                label_col="country_name",
                title=None,
                x_label="Absolute GDP change (current US$)",
                color_scale="Oranges",
            ),
            use_container_width=True,
        )
    with row_two[1]:
        st.markdown("### Economic size versus GDP growth")
        st.caption("Bubble colour shows World Bank region. Bubble size shows population.")
        st.plotly_chart(
            build_growth_scatter(
                growth_df,
                title=None,
                x_col="end_total_gdp",
                y_col="cagr_pct",
                size_col="end_population_total",
                hover_cols=[
                    "end_total_gdp",
                    "cagr_pct",
                    "end_population_total",
                    "end_life_expectancy_years",
                    "end_gdp_per_capita",
                ],
            ),
            use_container_width=True,
        )

    highlighted_country = st.selectbox(
        "Select a country profile",
        sorted(growth_df["country_name"].unique().tolist()),
        index=0,
        key="compare_country_profile",
    )
    profile_row = growth_df[growth_df["country_name"] == highlighted_country].iloc[0]
    history_df = filter_panel(panel_df, year_range=(start_year, end_year), countries=[highlighted_country])

    st.markdown("### Country profile")
    profile_cols = st.columns(4)
    with profile_cols[0]:
        render_metric_card(
            "Represented GDP",
            format_currency(profile_row["end_total_gdp"], compact=True),
            f"{highlighted_country} in {end_year}",
        )
    with profile_cols[1]:
        render_metric_card(
            "GDP CAGR",
            format_percent(profile_row["cagr_pct"]),
            f"From {start_year} to {end_year}",
        )
    with profile_cols[2]:
        render_metric_card(
            "GDP per capita",
            format_currency(profile_row["end_gdp_per_capita"]),
            f"Latest in-window value ({end_year})",
        )
    with profile_cols[3]:
        render_metric_card(
            "Life expectancy",
            f"{profile_row['end_life_expectancy_years']:.1f} years",
            f"Latest in-window value ({end_year})",
        )

    st.plotly_chart(
        build_country_history_chart(
            history_df.dropna(subset=["estimated_total_gdp_usd"]).copy(),
            value_col="estimated_total_gdp_usd",
            title=f"{highlighted_country}: represented GDP history",
            y_label="Estimated total GDP (current US$)",
        ),
        use_container_width=True,
    )

    render_note_box(
        "Interpretation note",
        "Country rankings here use derived represented GDP because the raw project indicator is GDP per capita. "
        "This preserves internal consistency while keeping the primary forecasting target unchanged.",
    )

    render_insight_box(
        "Comparison insights",
        [
            f"{largest_row['country_name']} leads the selected sample by represented GDP in {end_year}.",
            f"{fastest_row['country_name']} has the strongest GDP CAGR from {start_year} to {end_year}.",
            f"{absolute_row['country_name']} adds the largest absolute GDP amount over the same period.",
        ],
    )
