from __future__ import annotations

import streamlit as st

from src.analytics import build_country_growth_frame, filter_panel
from src.charts import (
    build_country_history_chart,
    build_growth_scatter,
    build_ranking_bar,
)
from src.components import (
    render_insight_box,
    render_metric_card,
    render_note_box,
    render_page_header,
)
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
        title="Country Comparison",
        question=(
            "How do countries differ in terms of economic size, GDP per capita, "
            "and long-term growth?"
        ),
        description=(
            "This section compares countries from both cross-sectional and longitudinal "
            "perspectives. It highlights economic diversity before the dashboard moves "
            "to statistical relationships and forecasting models."
        ),
        chips=[
            f"Default comparison window: {default_start}-{default_end}",
            "Economic size and growth are analysed separately",
            "GDP per capita remains the primary economic indicator",
            "Population and life expectancy provide contextual information",
        ],
    )

    render_note_box(
        "Purpose of this section",
        (
            "Country comparison complements the exploratory analysis by showing how "
            "economic size, GDP per capita, and growth differ across countries. "
            "These differences provide empirical motivation for the multivariate GDP "
            "forecasting models developed later in the project."
        ),
    )

    st.markdown("### Comparison controls")
    filter_cols = st.columns([0.8, 0.8, 1.3, 0.8], gap="large")

    with filter_cols[0]:
        start_year = st.selectbox(
            "Start year",
            years[:-1],
            index=max(0, years[:-1].index(default_start)),
            key="compare_start",
        )

    with filter_cols[1]:
        end_year = st.selectbox(
            "End year",
            years[1:],
            index=years[1:].index(default_end),
            key="compare_end",
        )

    with filter_cols[2]:
        selected_regions = st.multiselect(
            "Regions",
            regions,
            default=regions,
            key="compare_regions",
        )

    with filter_cols[3]:
        top_n = st.slider(
            "Top N countries",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key="compare_top_n",
        )

    if end_year <= start_year:
        st.warning(
            "End year must be later than start year for GDP growth comparison."
        )
        return

    if not selected_regions:
        st.warning("Please select at least one region for country comparison.")
        return

    growth_df = build_country_growth_frame(
        panel_df,
        start_year,
        end_year,
        selected_regions,
    )

    if growth_df.empty:
        st.warning(
            "The selected period and region filters do not leave enough valid "
            "country pairs to compare."
        )
        return

    largest_row = growth_df.sort_values(
        "end_total_gdp",
        ascending=False,
    ).iloc[0]

    fastest_row = growth_df.sort_values(
        "cagr_pct",
        ascending=False,
    ).iloc[0]

    absolute_row = growth_df.sort_values(
        "absolute_change_gdp",
        ascending=False,
    ).iloc[0]

    decline_df = growth_df[
        growth_df["absolute_change_gdp"] < 0
    ].copy()

    st.markdown("### Cross-country snapshot")
    metric_cols = st.columns(4)

    with metric_cols[0]:
        render_metric_card(
            "Largest represented economy",
            format_currency(
                largest_row["end_total_gdp"],
                compact=True,
            ),
            f"{largest_row['country_name']} in {end_year}",
        )

    with metric_cols[1]:
        render_metric_card(
            "Fastest GDP CAGR",
            format_percent(fastest_row["cagr_pct"]),
            (
                f"{fastest_row['country_name']} "
                f"from {start_year} to {end_year}"
            ),
        )

    with metric_cols[2]:
        render_metric_card(
            "Largest absolute increase",
            format_currency(
                absolute_row["absolute_change_gdp"],
                compact=True,
            ),
            (
                f"{absolute_row['country_name']} "
                "over the selected period"
            ),
        )

    with metric_cols[3]:
        if decline_df.empty:
            render_metric_card(
                "Largest decline",
                "None",
                "No negative represented-GDP change in the current filter",
            )
        else:
            decline_row = decline_df.sort_values(
                "absolute_change_gdp"
            ).iloc[0]

            render_metric_card(
                "Largest decline",
                format_currency(
                    decline_row["absolute_change_gdp"],
                    compact=True,
                ),
                (
                    f"{decline_row['country_name']} "
                    f"from {start_year} to {end_year}"
                ),
            )

    st.markdown("### Cross-country economic ranking")
    row_one = st.columns(2, gap="large")

    with row_one[0]:
        st.plotly_chart(
            build_ranking_bar(
                growth_df.nlargest(
                    top_n,
                    "end_total_gdp",
                )[["country_name", "end_total_gdp"]],
                value_col="end_total_gdp",
                label_col="country_name",
                title=f"Economic size in {end_year}",
                x_label="Estimated total GDP (current US$)",
                color_scale="Blues",
            ),
            use_container_width=True,
        )

    with row_one[1]:
        st.plotly_chart(
            build_ranking_bar(
                growth_df.nlargest(
                    top_n,
                    "cagr_pct",
                )[["country_name", "cagr_pct"]],
                value_col="cagr_pct",
                label_col="country_name",
                title=f"GDP growth from {start_year} to {end_year}",
                x_label="GDP CAGR (%)",
                color_scale="Greens",
            ),
            use_container_width=True,
        )

    row_two = st.columns(2, gap="large")

    with row_two[0]:
        st.markdown("#### Largest absolute GDP increase")
        st.plotly_chart(
            build_ranking_bar(
                growth_df.nlargest(
                    top_n,
                    "absolute_change_gdp",
                )[["country_name", "absolute_change_gdp"]],
                value_col="absolute_change_gdp",
                label_col="country_name",
                title=None,
                x_label="Absolute GDP change (current US$)",
                color_scale="Oranges",
            ),
            use_container_width=True,
        )

    with row_two[1]:
        st.markdown("#### Economic size versus growth")
        st.caption(
            "Bubble colour shows World Bank region. "
            "Bubble size shows population."
        )
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

    render_note_box(
        "How to interpret the comparison",
        (
            "Economic size and economic growth should not be interpreted as the same "
            "concept. A large economy may grow slowly, while a smaller economy may "
            "record a high growth rate. GDP per capita also remains distinct from total "
            "economic scale because it reflects average output per person."
        ),
    )

    st.markdown("### Country profile")
    st.caption(
        "The selected country is examined in greater detail to illustrate "
        "its position and long-term economic trajectory within the comparison window."
    )

    highlighted_country = st.selectbox(
        "Select a country profile",
        sorted(growth_df["country_name"].unique().tolist()),
        index=0,
        key="compare_country_profile",
    )

    profile_row = growth_df[
        growth_df["country_name"] == highlighted_country
    ].iloc[0]

    history_df = filter_panel(
        panel_df,
        year_range=(start_year, end_year),
        countries=[highlighted_country],
    )

    profile_cols = st.columns(4)

    with profile_cols[0]:
        render_metric_card(
            "Represented GDP",
            format_currency(
                profile_row["end_total_gdp"],
                compact=True,
            ),
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
            history_df.dropna(
                subset=["estimated_total_gdp_usd"]
            ).copy(),
            value_col="estimated_total_gdp_usd",
            title=f"{highlighted_country}: represented GDP history",
            y_label="Estimated total GDP (current US$)",
        ),
        use_container_width=True,
    )

    selected_change = float(profile_row["absolute_change_gdp"])
    selected_cagr = float(profile_row["cagr_pct"])

    direction = (
        "increased"
        if selected_change > 0
        else "decreased"
        if selected_change < 0
        else "remained unchanged"
    )

    render_note_box(
        "Historical interpretation",
        (
            f"{highlighted_country}'s represented GDP {direction} over the selected "
            f"period, with a CAGR of {format_percent(selected_cagr)}. "
            "The chart provides a longitudinal view that complements the "
            "cross-sectional rankings above."
        ),
    )

    render_note_box(
        "Measurement note",
        (
            "Country rankings on this page use represented GDP derived from matched "
            "GDP per capita and population observations. GDP per capita remains the "
            "primary forecasting target, while represented GDP is used only to compare "
            "economic scale consistently within the project dataset."
        ),
    )

    render_insight_box(
        "Comparison summary",
        [
            (
                f"{largest_row['country_name']} leads the selected sample by "
                f"represented GDP in {end_year}."
            ),
            (
                f"{fastest_row['country_name']} records the strongest GDP CAGR "
                f"from {start_year} to {end_year}."
            ),
            (
                f"{absolute_row['country_name']} records the largest absolute "
                "increase in represented GDP over the selected period."
            ),
            (
                "The results show that economic size, GDP per capita, and growth "
                "capture different dimensions of country performance."
            ),
            (
                "Substantial cross-country heterogeneity supports the use of "
                "demographic, macroeconomic, regional, and temporal predictors "
                "in the later GDP models."
            ),
        ],
    )

    render_note_box(
        "Next stage",
        (
            "Country comparison demonstrates substantial variation across economies. "
            "The next section investigates whether these differences are systematically "
            "associated with population and life expectancy before the forecasting "
            "models are evaluated."
        ),
    )