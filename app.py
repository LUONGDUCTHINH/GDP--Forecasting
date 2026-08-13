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
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()


PAGE_CONFIG = [
    {
        "key": "overview",
        "title": "Project Overview",
        "icon": "🏠",
        "url_path": "overview",
        "render": render_overview,
    },
    {
        "key": "data",
        "title": "Data Workflow",
        "icon": "📂",
        "url_path": "data-workflow",
        "render": render_data_workflow,
    },
    {
        "key": "eda",
        "title": "Exploratory Analysis",
        "icon": "📊",
        "url_path": "exploratory-analysis",
        "render": render_gdp_trends,
    },
    {
        "key": "comparison",
        "title": "Country Comparison",
        "icon": "🌍",
        "url_path": "country-comparison",
        "render": render_country_comparison,
    },
    {
        "key": "relationships",
        "title": "GDP Relationships",
        "icon": "🔗",
        "url_path": "gdp-relationships",
        "render": render_relationships,
    },
    {
        "key": "forecasting",
        "title": "GDP Forecasting Models",
        "icon": "📈",
        "url_path": "gdp-forecasting-models",
        "render": render_forecasting,
    },
    {
        "key": "conclusions",
        "title": "Findings & Conclusions",
        "icon": "📑",
        "url_path": "findings-and-conclusions",
        "render": render_conclusions,
    },
]


def render_sidebar_header() -> None:
    """Render the project identity and research workflow in the sidebar."""
    st.sidebar.markdown(f"## 🌍 {DASHBOARD_TITLE}")
    st.sidebar.caption(PROJECT_TITLE)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Analytical workflow")
    st.sidebar.caption(
        "Project Overview → Data Workflow → Exploratory Analysis → "
        "Country Comparison → GDP Relationships → GDP Forecasting Models → "
        "Findings & Conclusions"
    )

    st.sidebar.markdown("---")


def render_sidebar_footer() -> None:
    """Render concise project metadata below the navigation."""
    st.sidebar.markdown("---")

    st.sidebar.caption(
        """
Final Year Project

BSc (Hons) Computing

University of Greenwich

Luong Duc Thinh · 2026
"""
    )


def run_navigation() -> None:
    """Render the multipage dashboard with a compatible fallback menu."""
    render_sidebar_header()

    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = [
            st.Page(
                page["render"],
                title=f"{page['icon']} {page['title']}",
                url_path=page["url_path"],
            )
            for page in PAGE_CONFIG
        ]

        navigation = st.navigation(pages, position="sidebar")
        render_sidebar_footer()
        navigation.run()
        return

    page_labels = [
        f"{page['icon']} {page['title']}"
        for page in PAGE_CONFIG
    ]
    selected_label = st.sidebar.radio(
        "Dashboard sections",
        page_labels,
        index=0,
    )

    selected_index = page_labels.index(selected_label)
    render_sidebar_footer()
    PAGE_CONFIG[selected_index]["render"]()


run_navigation()
