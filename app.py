from __future__ import annotations

import streamlit as st

from src.components import inject_global_styles
from src.dashboard_data import DASHBOARD_TITLE, PROJECT_TITLE
from views.conclusions import render as render_conclusions
from views.country_comparison import render as render_country_comparison
from views.data_workflow import render as render_data_workflow
from views.forecasting import render as render_forecasting
from views.gdp_trends import render as render_gdp_trends
from views.overview import render as render_overview
from views.relationships import render as render_relationships


st.set_page_config(
    page_title=DASHBOARD_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()

PAGE_MAP = {
    "Executive Overview": render_overview,
    "Data Workflow": render_data_workflow,
    "Global GDP Trends": render_gdp_trends,
    "GDP Growth and Country Comparison": render_country_comparison,
    "GDP Relationships": render_relationships,
    "GDP Forecasting": render_forecasting,
    "Findings and Limitations": render_conclusions,
}


def render_sidebar_title() -> None:
    """Render a concise sidebar introduction."""
    st.sidebar.markdown(f"## {DASHBOARD_TITLE}")
    st.sidebar.caption(PROJECT_TITLE)
    st.sidebar.caption(
        "GDP-first workflow dashboard built from the real cleaned panel, forecasting outputs, and saved evaluation files in this repository."
    )


def run_navigation() -> None:
    """Use Streamlit navigation when available, otherwise fall back to a radio menu."""
    render_sidebar_title()

    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = [
            st.Page(render_overview, title="Executive Overview", url_path="overview"),
            st.Page(render_data_workflow, title="Data Workflow", url_path="data-workflow"),
            st.Page(render_gdp_trends, title="Global GDP Trends", url_path="gdp-trends"),
            st.Page(
                render_country_comparison,
                title="GDP Growth and Country Comparison",
                url_path="country-comparison",
            ),
            st.Page(render_relationships, title="GDP Relationships", url_path="relationships"),
            st.Page(render_forecasting, title="GDP Forecasting", url_path="forecasting"),
            st.Page(
                render_conclusions,
                title="Findings and Limitations",
                url_path="findings-and-limitations",
            ),
        ]
        navigation = st.navigation(pages, position="sidebar")
        navigation.run()
        return

    selected_page = st.sidebar.radio("Workflow pages", list(PAGE_MAP.keys()))
    PAGE_MAP[selected_page]()


run_navigation()
