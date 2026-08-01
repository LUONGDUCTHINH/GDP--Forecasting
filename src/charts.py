from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PRIMARY_BLUE = "#2563EB"
POSITIVE_GREEN = "#059669"
NEGATIVE_RED = "#DC2626"
NAVY = "#0F172A"
SLATE = "#64748B"
BORDER = "#E2E8F0"


def empty_figure(message: str, height: int = 420) -> go.Figure:
    """Return a styled empty-state figure."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=58, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=NAVY),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color=SLATE),
            )
        ],
    )
    return fig


def apply_standard_layout(
    fig: go.Figure,
    height: int = 420,
    title: str | None = None,
    legend_title: str | None = None,
) -> go.Figure:
    """Apply a consistent Plotly layout for the dashboard."""
    top_margin = 64 if title else 20
    layout_kwargs = dict(
        template="plotly_white",
        height=height,
        margin=dict(l=14, r=14, t=top_margin, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=NAVY),
        legend_title_text=legend_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=NAVY),
    )
    if title:
        layout_kwargs["title"] = title
        layout_kwargs["title_font"] = dict(size=19, color=NAVY)

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor="#EEF2F7", linecolor=BORDER)
    return fig


def build_trend_chart(
    trend_df: pd.DataFrame,
    title: str,
    y_label: str,
    color: str = PRIMARY_BLUE,
) -> go.Figure:
    """Build a clean annual trend line."""
    if trend_df.empty:
        return empty_figure("No data is available for the selected trend.")

    fig = px.line(
        trend_df,
        x="year",
        y="value",
        markers=True,
        color_discrete_sequence=[color],
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    fig.update_yaxes(title=y_label)
    fig.update_xaxes(title="Year")
    return apply_standard_layout(fig, title=title)


def build_ranking_bar(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str | None,
    x_label: str,
    color_col: str | None = None,
    color_scale: str = "Blues",
) -> go.Figure:
    """Build a horizontal ranking bar chart."""
    if df.empty:
        return empty_figure("No ranking data is available for the selected filters.")

    plot_df = df.copy().sort_values(value_col, ascending=True)
    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        color=color_col or value_col,
        color_continuous_scale=color_scale,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="")
    return apply_standard_layout(fig, title=title)


def build_growth_scatter(
    df: pd.DataFrame,
    title: str | None,
    x_col: str,
    y_col: str,
    size_col: str,
    hover_cols: list[str],
    log_x: bool = True,
) -> go.Figure:
    """Build a size-versus-growth bubble chart."""
    if df.empty:
        return empty_figure("No country-level growth comparison is available.")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color="wb_region",
        hover_name="country_name",
        hover_data=hover_cols,
        size_max=42,
    )
    fig.update_xaxes(type="log" if log_x else "linear", title="Estimated GDP represented (current US$)")
    fig.update_yaxes(title="GDP CAGR (%)")
    fig = apply_standard_layout(fig, title=title, legend_title=None, height=500)
    fig.update_layout(
        margin=dict(l=14, r=14, t=16 if not title else 56, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
            font=dict(size=10),
            itemwidth=70,
            itemsizing="constant",
        ),
    )
    return fig


def build_country_history_chart(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    color: str = PRIMARY_BLUE,
) -> go.Figure:
    """Build a single-country history chart."""
    if df.empty:
        return empty_figure("No country history is available for this selection.")
    plot_df = df.dropna(subset=[value_col]).sort_values("year").copy()
    fig = px.line(plot_df, x="year", y=value_col, markers=True, color_discrete_sequence=[color])
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title=y_label)
    return apply_standard_layout(fig, title=title)


def build_bubble_relationship_chart(
    df: pd.DataFrame,
    title: str | None,
    x_col: str,
    y_col: str,
    size_col: str,
    hover_cols: list[str],
    log_x: bool = True,
) -> go.Figure:
    """Build the GDP-population-life expectancy relationship bubble chart."""
    if df.empty:
        return empty_figure("No valid observations are available for the selected year.")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color="wb_region",
        hover_name="country_name",
        hover_data=hover_cols,
        size_max=52,
    )
    fig.update_xaxes(
        type="log" if log_x else "linear",
        title="Estimated GDP represented (current US$)",
    )
    fig.update_yaxes(title="Life Expectancy (years)")
    fig = apply_standard_layout(fig, title=title, legend_title=None, height=500)
    fig.update_layout(
        margin=dict(l=14, r=14, t=16 if not title else 56, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
            font=dict(size=10),
            itemwidth=70,
            itemsizing="constant",
        ),
    )
    return fig


def build_correlation_heatmap(corr_df: pd.DataFrame, title: str | None) -> go.Figure:
    """Build a compact correlation heatmap."""
    if corr_df.empty:
        return empty_figure("No valid correlation matrix is available.")

    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    return apply_standard_layout(fig, title=title, height=460)


def build_missingness_chart(missing_df: pd.DataFrame) -> go.Figure:
    """Build a grouped missing-data chart."""
    if missing_df.empty:
        return empty_figure("No missing-data summary is available.", height=340)

    fig = px.bar(
        missing_df,
        x="variable",
        y="missing_count",
        color="stage",
        barmode="group",
        color_discrete_map={"Raw transformed": PRIMARY_BLUE, "Final panel": POSITIVE_GREEN},
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Missing values")
    return apply_standard_layout(fig, title="Missing values before and after cleaning", height=360)


def build_main_model_yearly_chart(yearly_df: pd.DataFrame) -> go.Figure:
    """Plot actual versus predicted mean GDP by target year for the main models."""
    if yearly_df.empty:
        return empty_figure("No yearly main-model summary is available.")

    fig = go.Figure()
    actual_df = yearly_df.drop_duplicates(subset=["target_year"]).sort_values("target_year")
    fig.add_trace(
        go.Scatter(
            x=actual_df["target_year"],
            y=actual_df["actual_mean_gdp"],
            mode="lines+markers",
            name="Actual mean GDP per capita",
            line=dict(color=NAVY, width=3),
        )
    )

    palette = ["#2563EB", "#0F766E", "#9333EA", "#DC2626"]
    for color, (model_name, model_df) in zip(
        palette,
        yearly_df.groupby("model", sort=False),
    ):
        ordered_df = model_df.sort_values("target_year")
        fig.add_trace(
            go.Scatter(
                x=ordered_df["target_year"],
                y=ordered_df["predicted_mean_gdp"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=color, width=2.6, dash="dash"),
            )
        )

    fig.update_xaxes(title="Target year")
    fig.update_yaxes(title="GDP per Capita (current US$)")
    return apply_standard_layout(fig, title="Main GDP model comparison on the shared test years", height=440)


def build_forecast_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    forecast_label: str,
) -> go.Figure:
    """Build a forecast path chart with actual and future segments."""
    history = (
        history_df[["year", value_col]]
        .dropna()
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .copy()
    )
    if history.empty:
        return empty_figure("No historical series is available for this country.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["year"],
            y=history[value_col],
            mode="lines+markers",
            name="Historical actual values",
            line=dict(color=NAVY, width=3),
        )
    )

    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["forecast_year"],
                y=forecast_df["predicted_value"],
                mode="lines+markers",
                name=forecast_label,
                line=dict(color=PRIMARY_BLUE, width=3, dash="dash"),
            )
        )
        fig.add_vline(
            x=int(history["year"].max()) + 0.5,
            line_dash="dot",
            line_color=SLATE,
            annotation_text="Forecast start",
            annotation_position="top left",
        )

    fig.update_xaxes(title="Year")
    fig.update_yaxes(title=y_label)
    return apply_standard_layout(fig, title=title, height=460)


def build_multi_forecast_chart(
    history_df: pd.DataFrame,
    forecast_map: dict[str, pd.DataFrame],
    value_col: str,
    title: str,
    y_label: str,
    best_model_name: str | None = None,
) -> go.Figure:
    """Build a chart with one historical series and multiple future forecast paths."""
    history = (
        history_df[["year", value_col]]
        .dropna()
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .copy()
    )
    if history.empty:
        return empty_figure("No historical series is available for this country.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["year"],
            y=history[value_col],
            mode="lines+markers",
            name="Historical actual values",
            line=dict(color=NAVY, width=3),
        )
    )

    palette = ["#2563EB", "#0F766E", "#9333EA", "#DC2626", "#D97706"]
    for idx, model_name in enumerate(forecast_map.keys()):
        forecast_df = forecast_map[model_name]
        if forecast_df.empty:
            continue

        color = palette[idx % len(palette)]
        line_width = 3.2 if model_name == best_model_name else 2.5
        dash_style = "dash" if model_name == best_model_name else "dot"
        trace_name = (
            f"Forecast ({model_name}) [Best]"
            if model_name == best_model_name
            else f"Forecast ({model_name})"
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["forecast_year"],
                y=forecast_df["predicted_value"],
                mode="lines+markers",
                name=trace_name,
                line=dict(color=color, width=line_width, dash=dash_style),
            )
        )

    fig.add_vline(
        x=int(history["year"].max()) + 0.5,
        line_dash="dot",
        line_color=SLATE,
        annotation_text="Forecast start",
        annotation_position="top left",
    )

    fig.update_xaxes(title="Year")
    fig.update_yaxes(title=y_label)
    fig = apply_standard_layout(fig, title=title, height=500)
    fig.update_layout(
        margin=dict(l=14, r=14, t=64, b=92),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0.0,
            font=dict(size=10),
        ),
    )
    return fig
