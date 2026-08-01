from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.formatting import correlation_direction, correlation_strength, safe_ratio


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """Return any required columns that are missing from a dataframe."""
    return [col for col in required_cols if col not in df.columns]


def filter_panel(
    df: pd.DataFrame,
    year_range: tuple[int, int] | None = None,
    regions: list[str] | None = None,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """Apply shared year, region, and country filters."""
    out = df.copy()

    if year_range is not None:
        start_year, end_year = year_range
        out = out[(out["year"] >= int(start_year)) & (out["year"] <= int(end_year))]

    if regions:
        out = out[out["wb_region"].isin(regions)]

    if countries:
        out = out[out["country_name"].isin(countries)]

    return out.copy()


def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """Calculate compound annual growth rate."""
    if pd.isna(start_value) or pd.isna(end_value):
        return math.nan
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return math.nan
    return ((end_value / start_value) ** (1 / periods) - 1) * 100


def aggregate_gdp_trend(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Aggregate annual GDP metrics using an explicit and labeled mode."""
    clean = df.copy()

    if mode == "represented_total_gdp":
        trend_df = (
            clean.dropna(subset=["estimated_total_gdp_usd"])
            .groupby("year", as_index=False)["estimated_total_gdp_usd"]
            .sum()
            .rename(columns={"estimated_total_gdp_usd": "value"})
        )
        trend_df["metric_label"] = "Estimated GDP represented across covered countries"
        trend_df["unit_label"] = "current US$"
        return trend_df

    if mode == "mean_gdp_per_capita":
        trend_df = (
            clean.dropna(subset=["gdp_per_capita_usd"])
            .groupby("year", as_index=False)["gdp_per_capita_usd"]
            .mean()
            .rename(columns={"gdp_per_capita_usd": "value"})
        )
        trend_df["metric_label"] = "Mean GDP per Capita"
        trend_df["unit_label"] = "current US$"
        return trend_df

    if mode == "median_gdp_per_capita":
        trend_df = (
            clean.dropna(subset=["gdp_per_capita_usd"])
            .groupby("year", as_index=False)["gdp_per_capita_usd"]
            .median()
            .rename(columns={"gdp_per_capita_usd": "value"})
        )
        trend_df["metric_label"] = "Median GDP per Capita"
        trend_df["unit_label"] = "current US$"
        return trend_df

    if mode == "indexed_represented_total_gdp":
        trend_df = aggregate_gdp_trend(clean, "represented_total_gdp")
        if not trend_df.empty:
            base_value = trend_df["value"].iloc[0]
            trend_df["value"] = (trend_df["value"] / base_value) * 100
        trend_df["metric_label"] = "Indexed represented GDP"
        trend_df["unit_label"] = "Index (base year = 100)"
        return trend_df

    raise ValueError(f"Unsupported aggregation mode: {mode}")


def summarize_period_change(trend_df: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute start, end, absolute change, percentage change, and CAGR."""
    if trend_df.empty:
        return {}

    start_row = trend_df.iloc[0]
    end_row = trend_df.iloc[-1]
    start_value = float(start_row["value"])
    end_value = float(end_row["value"])
    periods = int(end_row["year"]) - int(start_row["year"])

    pct_change = safe_ratio(end_value - start_value, start_value)
    if not pd.isna(pct_change):
        pct_change *= 100

    return {
        "start_year": int(start_row["year"]),
        "end_year": int(end_row["year"]),
        "start_value": start_value,
        "end_value": end_value,
        "absolute_change": end_value - start_value,
        "pct_change": pct_change,
        "cagr_pct": calculate_cagr(start_value, end_value, periods),
        "metric_label": str(start_row.get("metric_label", "Value")),
        "unit_label": str(start_row.get("unit_label", "")),
    }


def build_country_growth_frame(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """Build a country comparison frame using derived total GDP from matched country-years."""
    if end_year <= start_year:
        return pd.DataFrame()

    filtered = df.copy()
    if regions:
        filtered = filtered[filtered["wb_region"].isin(regions)].copy()

    start_df = (
        filtered[filtered["year"] == int(start_year)]
        .dropna(subset=["estimated_total_gdp_usd"])
        .copy()
        .rename(
            columns={
                "estimated_total_gdp_usd": "start_total_gdp",
                "gdp_per_capita_usd": "start_gdp_per_capita",
            }
        )
    )
    end_df = (
        filtered[filtered["year"] == int(end_year)]
        .dropna(subset=["estimated_total_gdp_usd"])
        .copy()
        .rename(
            columns={
                "estimated_total_gdp_usd": "end_total_gdp",
                "gdp_per_capita_usd": "end_gdp_per_capita",
                "population_total": "end_population_total",
                "life_expectancy_years": "end_life_expectancy_years",
            }
        )
    )

    join_cols = ["country_name", "country_code", "wb_region"]
    merged = start_df[join_cols + ["start_total_gdp", "start_gdp_per_capita"]].merge(
        end_df[
            join_cols
            + [
                "end_total_gdp",
                "end_gdp_per_capita",
                "end_population_total",
                "end_life_expectancy_years",
            ]
        ],
        on=join_cols,
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    periods = int(end_year) - int(start_year)
    merged["absolute_change_gdp"] = merged["end_total_gdp"] - merged["start_total_gdp"]
    merged["percentage_change_gdp"] = (
        (merged["absolute_change_gdp"] / merged["start_total_gdp"]) * 100
    )
    merged["cagr_pct"] = merged.apply(
        lambda row: calculate_cagr(row["start_total_gdp"], row["end_total_gdp"], periods),
        axis=1,
    )
    merged["gdp_per_capita_change_pct"] = (
        (merged["end_gdp_per_capita"] - merged["start_gdp_per_capita"])
        / merged["start_gdp_per_capita"]
    ) * 100

    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.dropna(subset=["start_total_gdp", "end_total_gdp"]).copy()


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate selected GDP-focused correlations for the current filtered sample."""
    specs = [
        ("Estimated total GDP", "Population", "estimated_total_gdp_usd", "population_total"),
        ("Estimated total GDP", "Life expectancy", "estimated_total_gdp_usd", "life_expectancy_years"),
        ("GDP per capita", "Life expectancy", "gdp_per_capita_usd", "life_expectancy_years"),
        ("Population", "Life expectancy", "population_total", "life_expectancy_years"),
    ]
    rows: list[dict[str, object]] = []

    for left_label, right_label, left_col, right_col in specs:
        temp = df[[left_col, right_col]].dropna().copy()
        corr_value = temp[left_col].corr(temp[right_col]) if len(temp) >= 3 else math.nan
        rows.append(
            {
                "left_label": left_label,
                "right_label": right_label,
                "pair_label": f"{left_label} vs {right_label}",
                "correlation": corr_value,
                "n_obs": int(len(temp)),
                "direction": correlation_direction(corr_value),
                "strength": correlation_strength(corr_value),
            }
        )
    return pd.DataFrame(rows)


def build_relationship_matrix(df: pd.DataFrame, use_log_gdp: bool = False) -> pd.DataFrame:
    """Return a compact correlation matrix for the relationships page."""
    if use_log_gdp:
        numeric_df = df[
            [
                "log_gdp_per_capita",
                "log_population_total",
                "life_expectancy_years",
                "inflation_pct_clean",
                "unemployment_pct_clean",
                "internet_users_pct_clean",
            ]
        ].copy()
        rename_map = {
            "log_gdp_per_capita": "Log GDP per Capita",
            "log_population_total": "Log Population",
            "life_expectancy_years": "Life Expectancy",
            "inflation_pct_clean": "Inflation",
            "unemployment_pct_clean": "Unemployment",
            "internet_users_pct_clean": "Internet Users",
        }
    else:
        numeric_df = df[
            [
                "estimated_total_gdp_usd",
                "gdp_per_capita_usd",
                "population_total",
                "life_expectancy_years",
            ]
        ].copy()
        rename_map = {
            "estimated_total_gdp_usd": "Estimated Total GDP",
            "gdp_per_capita_usd": "GDP per Capita",
            "population_total": "Population",
            "life_expectancy_years": "Life Expectancy",
        }

    numeric_df = numeric_df.rename(columns=rename_map)
    return numeric_df.corr(numeric_only=True)


def get_best_main_model(metrics_df: pd.DataFrame) -> pd.Series | None:
    """Select the best rebuilt main GDP model using test RMSE, then MAPE, then MAE."""
    if metrics_df.empty:
        return None

    temp = metrics_df[
        (metrics_df["split"].astype(str).str.lower() == "test")
        & (metrics_df["scale"].astype(str).str.lower() == "level_gdp_usd")
    ].copy()
    if temp.empty:
        return None
    return temp.sort_values(["RMSE", "MAPE_pct", "MAE"]).iloc[0]


def build_model_test_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return the core test metrics table for the three rebuilt main models."""
    if metrics_df.empty:
        return pd.DataFrame()
    temp = metrics_df[
        (metrics_df["split"].astype(str).str.lower() == "test")
        & (metrics_df["scale"].astype(str).str.lower() == "level_gdp_usd")
    ].copy()
    return temp.sort_values(["RMSE", "MAPE_pct", "MAE"]).reset_index(drop=True)


def generate_overview_insights(panel_df: pd.DataFrame, focus_year: int) -> list[str]:
    """Generate concise, data-driven overview findings."""
    insights: list[str] = []

    current_df = panel_df[panel_df["year"] == int(focus_year)].dropna(
        subset=["gdp_per_capita_usd", "estimated_total_gdp_usd"]
    )
    if current_df.empty:
        return ["No valid GDP observations are available for the selected year."]

    top_total = current_df.sort_values("estimated_total_gdp_usd", ascending=False).iloc[0]
    insights.append(
        f"{top_total['country_name']} has the largest represented GDP in {focus_year} "
        f"within the covered sample."
    )

    top_pc = current_df.sort_values("gdp_per_capita_usd", ascending=False).iloc[0]
    insights.append(
        f"{top_pc['country_name']} records the highest GDP per capita in {focus_year}."
    )

    first_year = int(panel_df["year"].min())
    trend_df = aggregate_gdp_trend(panel_df, "represented_total_gdp")
    summary = summarize_period_change(trend_df)
    if summary:
        insights.append(
            f"Represented GDP changes from {first_year} to {summary['end_year']} across "
            f"the countries covered in the panel."
        )

    corr_df = calculate_correlations(current_df)
    gdp_life_row = corr_df[corr_df["pair_label"] == "GDP per capita vs Life expectancy"]
    if not gdp_life_row.empty:
        row = gdp_life_row.iloc[0]
        insights.append(
            f"GDP per capita and life expectancy show a {str(row['strength']).lower()} "
            f"{str(row['direction']).lower()} association in {focus_year}."
        )

    return insights[:4]


def generate_trend_insights(growth_df: pd.DataFrame, start_year: int, end_year: int) -> list[str]:
    """Generate insights for the comparison page from the growth table."""
    if growth_df.empty:
        return ["The selected filters do not leave enough valid country-year pairs to compare."]

    fastest_row = growth_df.sort_values("cagr_pct", ascending=False).iloc[0]
    largest_row = growth_df.sort_values("end_total_gdp", ascending=False).iloc[0]
    increase_row = growth_df.sort_values("absolute_change_gdp", ascending=False).iloc[0]

    insights = [
        f"{largest_row['country_name']} has the largest represented GDP in {end_year}.",
        f"{fastest_row['country_name']} has the strongest GDP CAGR from {start_year} to {end_year}.",
        f"{increase_row['country_name']} contributes the largest absolute GDP increase over the selected period.",
    ]

    decline_df = growth_df[growth_df["absolute_change_gdp"] < 0].copy()
    if not decline_df.empty:
        decline_row = decline_df.sort_values("absolute_change_gdp").iloc[0]
        insights.append(
            f"{decline_row['country_name']} shows the largest decline in represented GDP over the same period."
        )

    return insights


def generate_project_findings(
    panel_df: pd.DataFrame,
    main_metrics_df: pd.DataFrame,
    gdp_ts_summary_df: pd.DataFrame,
) -> list[str]:
    """Generate project-level findings for the conclusions page."""
    findings: list[str] = []

    trend_df = aggregate_gdp_trend(panel_df, "represented_total_gdp")
    summary = summarize_period_change(trend_df)
    if summary:
        findings.append(
            "Represented GDP across covered countries increases over the long historical window "
            f"from {summary['start_year']} to {summary['end_year']}."
        )

    latest_year = int(panel_df["year"].max())
    latest_df = panel_df[panel_df["year"] == latest_year].dropna(subset=["gdp_per_capita_usd"])
    if not latest_df.empty:
        top_pc = latest_df.sort_values("gdp_per_capita_usd", ascending=False).iloc[0]
        findings.append(
            f"{top_pc['country_name']} records the highest GDP per capita in the latest available year ({latest_year})."
        )

    best_main = get_best_main_model(main_metrics_df)
    if best_main is not None:
        findings.append(
                    f"{best_main['model']} gives the strongest test-set performance among the rebuilt main GDP models."
        )

    if not gdp_ts_summary_df.empty:
        best_ts = gdp_ts_summary_df.sort_values(["RMSE", "MAPE_pct", "MAE"]).iloc[0]
        findings.append(
            f"{best_ts['Model']} is the best GDP time-series benchmark in the rolling 10-year backtest."
        )

    return findings[:4]
