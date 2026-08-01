from __future__ import annotations

import streamlit as st

from src.analytics import generate_project_findings, get_best_main_model
from src.components import render_insight_box, render_metric_card, render_note_box, render_page_header
from src.dashboard_data import load_main_model_metrics, load_main_model_sample, load_panel, load_ts_summary
from src.formatting import format_integer


def render() -> None:
    panel_df = load_panel()
    main_metrics_df = load_main_model_metrics()
    main_model_sample_df = load_main_model_sample()
    gdp_ts_summary_df = load_ts_summary("gdp")

    best_main = get_best_main_model(main_metrics_df)
    best_ts_row = (
        gdp_ts_summary_df.sort_values(["RMSE", "MAPE_pct", "MAE"]).iloc[0]
        if not gdp_ts_summary_df.empty
        else None
    )

    render_page_header(
        active_key="conclusions",
        title="Findings and Limitations",
        question="What are the main academic takeaways of the project, and where do the current data and models remain limited?",
        description=(
            "This closing page summarises the most defensible GDP findings from the repository and keeps the "
            "limitations visible for reporting and demo discussion."
        ),
        chips=[
            f"Final panel: {format_integer(len(panel_df))} rows",
            f"Main-model sample: {format_integer(len(main_model_sample_df))} rows",
            "GDP indicator: GDP per capita (current US$)",
            "No fake confidence intervals are displayed",
        ],
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card(
            "Countries in final panel",
            format_integer(panel_df["country_code"].nunique()),
            f"Years: {int(panel_df['year'].min())}-{int(panel_df['year'].max())}",
        )
    with metric_cols[1]:
        render_metric_card(
            "Main-model sample",
            format_integer(len(main_model_sample_df)),
            "Listwise-complete rows after adding macro and target variables",
        )
    with metric_cols[2]:
        if best_main is None:
            render_metric_card("Best main model", "N/A", "No test metrics file available")
        else:
            render_metric_card(
                "Best main model",
                str(best_main["model"]),
                f"Test RMSE: {best_main['RMSE']:.2f}",
            )
    with metric_cols[3]:
        if best_ts_row is None:
            render_metric_card("GDP backtest winner", "N/A", "No summary file available")
        else:
            render_metric_card(
                "GDP backtest winner",
                str(best_ts_row["Model"]),
                f"Rolling one-step RMSE: {best_ts_row['RMSE']:.2f}",
            )

    render_note_box(
        "Backtest winner versus deployed projection",
        "The time-series benchmark winner is selected from rolling one-step historical backtesting, but the future projection model may differ when the benchmark winner would produce an implausibly flat long-horizon path. In the current dashboard, GDP keeps Naive as the backtest winner but uses ARIMA for future projection; life expectancy keeps Naive as the backtest winner but uses Holt for future projection; population uses LogHolt in both stages.",
    )

    render_insight_box(
        "Key findings",
        generate_project_findings(panel_df, main_metrics_df, gdp_ts_summary_df),
    )

    st.markdown("### Current limitations")
    st.markdown(
        """
        - The GDP indicator used in the project is **GDP per capita (current US$)**, so it is affected by current-price inflation and exchange-rate movement.
        - Coverage falls after merging and again after listwise deletion for the rebuilt main GDP models, which reduces the number of usable country-year observations.
        - Represented total GDP is a derived value based on matched GDP per capita and population, not a directly downloaded total-GDP series.
        - Correlation results describe association only and do not establish causal effects between GDP, population, and life expectancy.
        - Forecast uncertainty naturally increases with horizon length, and the repository does not provide saved confidence intervals for the dashboard to display.
        - Country-level GDP and life expectancy indicators cannot capture within-country inequality or distributional differences.
        """
    )

    st.markdown("### Future work")
    st.markdown(
        """
        - Add a dedicated constant-price or inflation-adjusted GDP series to complement the current-price GDP-per-capita indicator.
        - Extend the benchmark layer with additional explainable machine-learning comparisons while preserving the core academic methodology.
        - Add automated data refresh logic for future repository updates.
        - Explore forecast uncertainty bands only when the underlying notebook pipeline saves them explicitly.
        - Expand the dashboard with regional or income-group drilldowns if those classifications are maintained consistently in future cleaned datasets.
        """
    )

    render_note_box(
        "Project framing",
        "The dashboard keeps GDP at the centre of the analytical story. Population and life expectancy are included as supporting variables because they help explain economic scale, development context, and the structure of the predictive models without replacing GDP as the core research focus.",
    )
