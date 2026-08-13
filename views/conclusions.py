from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components import (
    render_insight_box,
    render_metric_card,
    render_note_box,
    render_page_header,
)
from src.dashboard_data import (
    load_main_model_metrics,
    load_main_model_sample,
    load_panel,
    load_ts_summary,
)
from src.formatting import format_currency, format_integer, format_percent


REPORT_MAIN_MODEL_RESULTS = pd.DataFrame(
    [
        {
            "Specification": "Model 1 - Baseline Dynamic",
            "Best algorithm": "OLS",
            "Test MAE": 1498.12,
            "Test RMSE": 3477.15,
            "Test MAPE (%)": 9.16,
            "Test R²": 0.9766,
        },
        {
            "Specification": "Model 2 - Extended Dynamic",
            "Best algorithm": "Elastic Net",
            "Test MAE": 1486.91,
            "Test RMSE": 3468.62,
            "Test MAPE (%)": 9.16,
            "Test R²": 0.9767,
        },
        {
            "Specification": "Model 3 - Full Dynamic",
            "Best algorithm": "OLS",
            "Test MAE": 1468.19,
            "Test RMSE": 3472.99,
            "Test MAPE (%)": 8.90,
            "Test R²": 0.9766,
        },
    ]
)


def _normalise(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _find_metric_row(
    metrics_df: pd.DataFrame,
    model_terms: list[str],
) -> pd.Series | None:
    if metrics_df.empty or "model" not in metrics_df.columns:
        return None

    normalised_models = metrics_df["model"].astype(str).map(_normalise)
    mask = pd.Series(True, index=metrics_df.index)

    for term in model_terms:
        mask &= normalised_models.str.contains(
            _normalise(term),
            regex=False,
        )

    matches = metrics_df[mask].copy()
    if matches.empty:
        return None

    if "split" in matches.columns:
        test_rows = matches[
            matches["split"].astype(str).str.lower() == "test"
        ]
        if not test_rows.empty:
            matches = test_rows

    if "RMSE" in matches.columns:
        matches = matches.sort_values("RMSE")

    return matches.iloc[0]


def _best_ts_row(summary_df: pd.DataFrame) -> pd.Series | None:
    if summary_df.empty or "Model" not in summary_df.columns:
        return None

    sort_cols = [
        col
        for col in ["RMSE", "MAPE_pct", "MAE"]
        if col in summary_df.columns
    ]

    if sort_cols:
        return summary_df.sort_values(sort_cols).iloc[0]

    return summary_df.iloc[0]


def render() -> None:
    panel_df = load_panel()
    main_metrics_df = load_main_model_metrics()
    main_model_sample_df = load_main_model_sample()

    gdp_ts_summary_df = load_ts_summary("gdp")
    life_ts_summary_df = load_ts_summary("life_expectancy")
    pop_ts_summary_df = load_ts_summary("population")

    gdp_best = _best_ts_row(gdp_ts_summary_df)
    life_best = _best_ts_row(life_ts_summary_df)
    pop_best = _best_ts_row(pop_ts_summary_df)

    predictive_row = _find_metric_row(
        main_metrics_df,
        ["model 2", "elastic net"],
    )
    explanatory_row = _find_metric_row(
        main_metrics_df,
        ["model 3", "ols"],
    )

    render_page_header(
        active_key="conclusions",
        title="Findings & Conclusions",
        question=(
            "What are the final answers to the research questions, "
            "which model results are most defensible, and what limitations remain?"
        ),
        description=(
            "This final page consolidates the exploratory evidence, time-series "
            "results, shared-holdout GDP benchmark, interpretation, limitations, "
            "and future work using the same conclusions reported in the final document."
        ),
        chips=[
            f"Extended panel: {format_integer(len(panel_df))} rows",
            f"Main-model sample: {format_integer(len(main_model_sample_df))} rows",
            "Predictive benchmark: Model 2 + Elastic Net",
            "Preferred explanatory model: Model 3 + OLS",
        ],
    )

    st.markdown("### Final project profile")
    profile_cols = st.columns(4)

    with profile_cols[0]:
        render_metric_card(
            "Extended panel",
            format_integer(len(panel_df)),
            (
                f"{panel_df['country_code'].nunique()} countries | "
                f"{int(panel_df['year'].min())}-{int(panel_df['year'].max())}"
            ),
        )

    with profile_cols[1]:
        render_metric_card(
            "Main-model sample",
            format_integer(len(main_model_sample_df)),
            (
                f"{main_model_sample_df['country_code'].nunique()} countries | "
                f"{int(main_model_sample_df['year'].min())}-"
                f"{int(main_model_sample_df['year'].max())}"
            ),
        )

    with profile_cols[2]:
        if predictive_row is None:
            render_metric_card(
                "Best RMSE configuration",
                "Model 2 + Elastic Net",
                "Test RMSE: 3,468.62",
            )
        else:
            render_metric_card(
                "Best RMSE configuration",
                str(predictive_row["model"]),
                f"Test RMSE: {predictive_row['RMSE']:.2f}",
            )

    with profile_cols[3]:
        if explanatory_row is None:
            render_metric_card(
                "Preferred explanatory model",
                "Model 3 + OLS",
                "MAE: 1,468.19 | MAPE: 8.90%",
            )
        else:
            render_metric_card(
                "Preferred explanatory model",
                str(explanatory_row["model"]),
                (
                    f"MAE: {explanatory_row['MAE']:.2f} | "
                    f"MAPE: {explanatory_row['MAPE_pct']:.2f}%"
                ),
            )

    st.markdown("### Research-question conclusions")
    render_insight_box(
        "Final answers",
        [
            (
                "Long-run GDP per capita and life expectancy generally rise over "
                "the study period, but large and persistent cross-country and "
                "regional differences remain."
            ),
            (
                "Rolling 10-year backtesting selects Naive for GDP per capita, "
                "Naive for life expectancy, and LogHolt for population. "
                "Different indicators therefore favour different temporal structures."
            ),
            (
                "Adding macroeconomic and technological indicators improves GDP "
                "prediction modestly relative to the baseline. Model 2 + Elastic Net "
                "achieves the lowest holdout RMSE and highest R²."
            ),
            (
                "Model 3 + OLS achieves the lowest MAE and MAPE while retaining "
                "direct interpretation of regional effects, event dummies, and "
                "structural heterogeneity. It is therefore the preferred substantive model."
            ),
            (
                "The Streamlit application communicates the complete workflow from "
                "data preparation and EDA to forecasting, model comparison, future "
                "scenarios, limitations, and conclusions."
            ),
        ],
    )

    st.markdown("### Indicator-level time-series conclusions")
    ts_cols = st.columns(3)

    for col, label, best_row, report_winner in [
        (ts_cols[0], "GDP per capita", gdp_best, "Naive"),
        (ts_cols[1], "Life expectancy", life_best, "Naive"),
        (ts_cols[2], "Population", pop_best, "LogHolt"),
    ]:
        with col:
            if best_row is None:
                render_metric_card(
                    label,
                    report_winner,
                    "Winner reported in the final report",
                )
            else:
                detail_parts = []
                if "RMSE" in best_row.index:
                    detail_parts.append(
                        f"RMSE: {best_row['RMSE']:.3f}"
                    )
                if "MAPE_pct" in best_row.index:
                    detail_parts.append(
                        f"MAPE: {best_row['MAPE_pct']:.3f}%"
                    )

                render_metric_card(
                    label,
                    str(best_row["Model"]),
                    " | ".join(detail_parts),
                )

    render_note_box(
        "Backtest winner versus projection model",
        (
            "The report's formal time-series conclusions are based on rolling "
            "one-step-ahead backtesting. The dashboard may use ARIMA for future GDP "
            "projection and Holt for future life-expectancy projection when a Naive "
            "winner would produce a flat recursive long-horizon path. Population uses "
            "LogHolt in both evaluation and projection. This deployment choice does "
            "not replace the historical benchmark result."
        ),
    )

    st.markdown("### Main GDP benchmark")
    display_results = REPORT_MAIN_MODEL_RESULTS.copy()
    display_results["Test MAE"] = display_results["Test MAE"].apply(
        format_currency
    )
    display_results["Test RMSE"] = display_results["Test RMSE"].apply(
        format_currency
    )
    display_results["Test MAPE (%)"] = display_results[
        "Test MAPE (%)"
    ].apply(format_percent)
    display_results["Test R²"] = display_results["Test R²"].map(
        lambda value: f"{value:.4f}"
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    render_note_box(
        "Why the project keeps two final model roles",
        (
            "Model 2 + Elastic Net is the strongest pure predictive configuration "
            "under RMSE and R². Model 3 + OLS is the preferred explanatory model "
            "because it remains highly competitive while supporting direct interpretation "
            "of structural, regional, and event-related terms. Predictive and explanatory "
            "selection are therefore reported separately."
        ),
    )

    st.markdown("### Methodological strengths")
    st.markdown(
        """
        - The project connects data preparation, EDA, time-series forecasting,
          multivariate GDP modelling, benchmarking, and dashboard communication
          in one reproducible workflow.
        - It clearly separates indicator-level temporal forecasting from
          next-year multivariate GDP prediction.
        - All fifteen GDP configurations are evaluated on the same chronological
          holdout sample.
        - Interpretability is preserved through progressive specifications and
          Model 3 + OLS for substantive analysis.
        - The dashboard uses the same cleaned data and saved analytical outputs
          as the modelling workflow.
        """
    )

    st.markdown("### Limitations")
    st.markdown(
        """
        - GDP per capita is measured in **current US dollars**, so inflation,
          exchange-rate movement, and nominal-price changes affect comparisons.
        - Missing inflation, unemployment, and internet-usage values reduce the
          extended panel from 11,373 observations to a 4,722-row main-model sample.
        - Annual data cannot capture short-run within-year macroeconomic dynamics.
        - The project focuses on association and forecasting rather than strict
          causal identification.
        - Common event dummies cannot capture different shock intensity in every country.
        - Recursive long-horizon forecasts accumulate uncertainty, while validated
          confidence intervals are not available for every dashboard scenario.
        - Country-level averages do not represent within-country inequality.
        - Represented total GDP is derived from GDP per capita and population,
          not downloaded as a separate total-GDP series.
        """
    )

    st.markdown("### Why black-box models were not the core choice")
    render_note_box(
        "Model discipline",
        (
            "Random Forest and XGBoost were retained as predictive benchmarks, "
            "but they did not dominate the shared holdout. The strongest results "
            "were concentrated among OLS, Ridge, and Elastic Net. Given the annual "
            "panel structure and the need for academic explanation, the project "
            "prioritises transparent and regularised models rather than complexity "
            "for its own sake."
        ),
    )

    st.markdown("### Future improvements")
    st.markdown(
        """
        - Add constant-price or purchasing-power-adjusted GDP indicators.
        - Improve missing-data treatment for macroeconomic predictors.
        - Compare pooled models with country-specific or region-specific extensions.
        - Evaluate additional time-series and panel-forecasting methods using the
          same chronological validation design.
        - Save and validate forecast intervals before displaying uncertainty bands.
        - Expand structural-shock modelling when consistent country-level exposure
          data are available.
        - Add automated data refresh and model retraining.
        - Extend dashboard drilldowns to income groups and other country classifications.
        """
    )

    render_note_box(
        "Overall conclusion",
        (
            "The project demonstrates that a coherent and interpretable GDP per "
            "capita forecasting workflow can combine open international data, "
            "descriptive analysis, indicator-level time-series models, progressive "
            "multivariate specifications, algorithm benchmarking, and interactive "
            "reporting. Its main contribution is methodological integration and "
            "transparent evaluation rather than the invention of a new algorithm."
        ),
    )