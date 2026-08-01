from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics import aggregate_gdp_trend, build_country_growth_frame, filter_panel, generate_trend_insights, summarize_period_change
from src.charts import build_country_history_chart, build_ranking_bar, build_trend_chart
from src.components import render_insight_box, render_metric_card, render_note_box, render_page_header
from src.dashboard_data import GDP_MAP_1960_PATH, GDP_MAP_2023_PATH, load_panel
from src.formatting import format_currency, format_percent


TREND_MODE_OPTIONS = {
    "Estimated GDP represented across covered countries": "represented_total_gdp",
    "Mean GDP per capita": "mean_gdp_per_capita",
    "Median GDP per capita": "median_gdp_per_capita",
    "Indexed represented GDP (base year = 100)": "indexed_represented_total_gdp",
}


def _build_animated_gdp_map(panel_df):
    map_df = (
        panel_df[
            ["country_code", "country_name", "wb_region", "year", "gdp_per_capita_usd"]
        ]
        .dropna(subset=["country_code", "year", "gdp_per_capita_usd"])
        .copy()
    )
    map_df["year"] = map_df["year"].astype(int)
    map_df = map_df.sort_values(["year", "country_name"])

    fig = px.choropleth(
        map_df,
        locations="country_code",
        locationmode="ISO-3",
        color="gdp_per_capita_usd",
        hover_name="country_name",
        hover_data={
            "wb_region": True,
            "year": True,
            "gdp_per_capita_usd": ":,.2f",
            "country_code": False,
        },
        animation_frame="year",
        animation_group="country_code",
        color_continuous_scale="YlGnBu",
        projection="natural earth",
        labels={
            "gdp_per_capita_usd": "GDP per capita (current US$)",
            "wb_region": "World Bank region",
            "year": "Year",
        },
        title="Animated GDP per capita heatmap by year",
    )
    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=70, b=20),
        coloraxis_colorbar=dict(
            title="GDP per capita",
            tickformat=",.0f",
            len=0.72,
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(24,33,29,0.28)",
            showland=True,
            landcolor="rgba(255,255,255,0.72)",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_traces(
        marker_line_color="rgba(255,255,255,0.62)",
        marker_line_width=0.35,
    )

    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
        fig.layout.updatemenus[0].x = 0.02
        fig.layout.updatemenus[0].y = 0.02

    if fig.layout.sliders:
        fig.layout.sliders[0]["currentvalue"]["prefix"] = "Year: "
        fig.layout.sliders[0]["x"] = 0.12
        fig.layout.sliders[0]["y"] = 0.01
        fig.layout.sliders[0]["len"] = 0.82

    return fig


def render() -> None:
    panel_df = load_panel()
    available_years = sorted(panel_df["year"].dropna().astype(int).unique().tolist())
    region_options = sorted(panel_df["wb_region"].dropna().unique().tolist())
    country_options = sorted(panel_df["country_name"].dropna().unique().tolist())

    render_page_header(
        active_key="trends",
        title="Global GDP Trends",
        question="How has GDP changed across countries and over time within the covered dataset?",
        description=(
            "This page separates long-run GDP movement from country ranking snapshots. "
            "The main analytical lens remains GDP per capita, while represented total GDP is derived "
            "only from matched GDP-per-capita and population observations."
        ),
        chips=[
            f"Coverage: {available_years[0]}-{available_years[-1]}",
            "Primary variable: GDP per capita (current US$)",
            "Derived size metric: GDP per capita x population",
        ],
    )

    filter_cols = st.columns([1.2, 1.0, 1.2], gap="large")
    with filter_cols[0]:
        year_range = st.slider(
            "Select year range",
            min_value=available_years[0],
            max_value=available_years[-1],
            value=(available_years[0], available_years[-1]),
            key="trend_year_range",
        )
    with filter_cols[1]:
        selected_regions = st.multiselect(
            "Filter by World Bank region",
            region_options,
            default=region_options,
            key="trend_regions",
        )
    with filter_cols[2]:
        selected_countries = st.multiselect(
            "Optional country spotlight",
            country_options,
            default=[],
            max_selections=5,
            key="trend_countries",
        )

    selected_mode_label = st.radio(
        "Trend view",
        list(TREND_MODE_OPTIONS.keys()),
        horizontal=False,
        key="trend_mode_label",
    )

    filtered_df = filter_panel(
        panel_df,
        year_range=year_range,
        regions=selected_regions,
        countries=selected_countries if selected_countries else None,
    )

    trend_df = aggregate_gdp_trend(filtered_df, TREND_MODE_OPTIONS[selected_mode_label])
    summary = summarize_period_change(trend_df)

    if not summary:
        st.warning("The current filters leave no valid GDP data to aggregate.")
        return

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card(
            "Start value",
            format_currency(summary["start_value"], compact=True)
            if "current US$" in summary["unit_label"]
            else format_percent(summary["start_value"]),
            f"{summary['metric_label']} in {summary['start_year']}",
        )
    with metric_cols[1]:
        render_metric_card(
            "End value",
            format_currency(summary["end_value"], compact=True)
            if "current US$" in summary["unit_label"]
            else f"{summary['end_value']:.2f}",
            f"{summary['metric_label']} in {summary['end_year']}",
        )
    with metric_cols[2]:
        render_metric_card(
            "Absolute change",
            format_currency(summary["absolute_change"], compact=True)
            if "current US$" in summary["unit_label"]
            else f"{summary['absolute_change']:.2f}",
            "Difference across the selected period",
        )
    with metric_cols[3]:
        render_metric_card(
            "Period growth",
            format_percent(summary["pct_change"]),
            f"CAGR: {format_percent(summary['cagr_pct'])}",
        )

    chart_cols = st.columns([1.3, 1.0], gap="large")
    with chart_cols[0]:
        y_axis_label = (
            "Estimated total GDP (current US$)"
            if "current US$" in summary["unit_label"]
            else summary["unit_label"]
        )
        st.plotly_chart(
            build_trend_chart(
                trend_df,
                title=selected_mode_label,
                y_label=y_axis_label,
            ),
            use_container_width=True,
        )
    with chart_cols[1]:
        latest_year = int(year_range[1])
        ranking_df = (
            filtered_df[filtered_df["year"] == latest_year]
            .dropna(subset=["estimated_total_gdp_usd"])
            .nlargest(10, "estimated_total_gdp_usd")[["country_name", "estimated_total_gdp_usd"]]
            .copy()
        )
        st.plotly_chart(
            build_ranking_bar(
                ranking_df,
                value_col="estimated_total_gdp_usd",
                label_col="country_name",
                title=f"Top countries by represented GDP in {latest_year}",
                x_label="Estimated total GDP (current US$)",
                color_scale="Teal",
            ),
            use_container_width=True,
        )

    insight_df = build_country_growth_frame(panel_df, year_range[0], year_range[1], regions=selected_regions)
    render_insight_box(
        "Trend interpretation",
        generate_trend_insights(insight_df, year_range[0], year_range[1]),
    )

    if selected_countries:
        st.markdown("### Country spotlight")
        country_cols = st.columns(len(selected_countries))
        for col, country_name in zip(country_cols, selected_countries):
            history_df = (
                filtered_df[filtered_df["country_name"] == country_name]
                .dropna(subset=["gdp_per_capita_usd"])
                .copy()
            )
            with col:
                st.plotly_chart(
                    build_country_history_chart(
                        history_df,
                        value_col="gdp_per_capita_usd",
                        title=country_name,
                        y_label="GDP per Capita (current US$)",
                    ),
                    use_container_width=True,
                )

    render_note_box(
        "Coverage reminder",
        "This trend page does not present the values as complete world GDP. Every aggregate is tied to the "
        "countries and years that are genuinely covered by the analytical panel after cleaning and merging.",
    )

    st.markdown("### Animated spatial GDP heatmap")
    render_note_box(
        "Spatial view",
        "This animated map reuses the same GDP-per-capita geography explored during the EDA stage, "
        "but now lets you scrub year by year to see how the global spatial pattern changes over time.",
    )
    st.plotly_chart(_build_animated_gdp_map(panel_df), use_container_width=True)

    if GDP_MAP_1960_PATH.exists() and GDP_MAP_2023_PATH.exists():
        with st.expander("Open the original EDA snapshot pair (1960 vs 2023)"):
            map_cols = st.columns(2, gap="large")
            with map_cols[0]:
                st.image(str(GDP_MAP_1960_PATH), caption="Original GDP per capita snapshot: 1960")
            with map_cols[1]:
                st.image(str(GDP_MAP_2023_PATH), caption="Original GDP per capita snapshot: 2023")
