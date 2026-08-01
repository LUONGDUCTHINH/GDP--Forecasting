from __future__ import annotations

from html import escape

import streamlit as st


WORKFLOW_STEPS = [
    ("overview", "01", "Overview"),
    ("data", "02", "Data"),
    ("trends", "03", "Explore"),
    ("comparison", "04", "Compare"),
    ("relationships", "05", "Analyse"),
    ("forecasting", "06", "Forecast"),
    ("conclusions", "07", "Conclude"),
]


def inject_global_styles() -> None:
    """Inject one shared CSS block for the dashboard."""
    st.markdown(
        """
        <style>
            .stApp {
                background: #F8FAFC;
                color: #0F172A;
            }

            .block-container {
                max-width: 1380px;
                padding-top: 1.4rem;
                padding-bottom: 2.2rem;
            }

            .page-hero {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 24px;
                padding: 1.45rem 1.55rem 1.25rem 1.55rem;
                margin-bottom: 1rem;
            }

            .page-kicker {
                color: #2563EB;
                font-size: 0.77rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .page-title {
                color: #0F172A;
                font-size: 2rem;
                line-height: 1.15;
                font-weight: 800;
                margin-bottom: 0.35rem;
            }

            .page-question {
                color: #0F172A;
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }

            .page-copy {
                color: #475569;
                font-size: 0.98rem;
                line-height: 1.65;
                margin-bottom: 0.8rem;
            }

            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.4rem;
            }

            .chip {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 999px;
                color: #334155;
                font-size: 0.84rem;
                font-weight: 600;
                padding: 0.32rem 0.65rem;
            }

            .workflow-strip {
                display: grid;
                grid-template-columns: repeat(7, minmax(0, 1fr));
                gap: 0.65rem;
                margin-bottom: 0.9rem;
            }

            .workflow-step {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 18px;
                padding: 0.7rem 0.8rem;
                min-height: 72px;
            }

            .workflow-step.active {
                border-color: #2563EB;
                box-shadow: inset 0 0 0 1px #2563EB;
                background: #EFF6FF;
            }

            .workflow-num {
                color: #64748B;
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.22rem;
            }

            .workflow-step.active .workflow-num {
                color: #2563EB;
            }

            .workflow-label {
                color: #0F172A;
                font-size: 0.95rem;
                font-weight: 700;
                line-height: 1.3;
            }

            .metric-card {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 22px;
                padding: 1rem 1.05rem 0.95rem 1.05rem;
                height: 100%;
            }

            .metric-label {
                color: #64748B;
                font-size: 0.84rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }

            .metric-value {
                color: #0F172A;
                font-size: 1.8rem;
                line-height: 1.05;
                font-weight: 800;
                margin-bottom: 0.3rem;
            }

            .metric-delta {
                font-size: 0.9rem;
                font-weight: 700;
                margin-bottom: 0.22rem;
            }

            .metric-delta.positive {
                color: #059669;
            }

            .metric-delta.negative {
                color: #DC2626;
            }

            .metric-delta.neutral {
                color: #2563EB;
            }

            .metric-note {
                color: #475569;
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .insight-box, .note-box {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 22px;
                padding: 1rem 1.05rem;
            }

            .insight-title, .note-title {
                color: #0F172A;
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }

            .insight-list {
                margin: 0;
                padding-left: 1rem;
                color: #334155;
                line-height: 1.7;
            }

            .note-copy {
                color: #475569;
                line-height: 1.65;
                font-size: 0.95rem;
            }

            .pipeline-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
                align-items: center;
            }

            .pipeline-step {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 18px;
                padding: 0.8rem 0.95rem;
                font-weight: 700;
                color: #0F172A;
                font-size: 0.9rem;
            }

            .pipeline-arrow {
                color: #2563EB;
                font-size: 1.05rem;
                font-weight: 800;
            }

            @media (max-width: 1100px) {
                .workflow-strip {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_indicator(active_key: str) -> None:
    """Render the workflow strip with the active stage highlighted."""
    cols = st.columns(len(WORKFLOW_STEPS), gap="small")
    for col, (key, number, label) in zip(cols, WORKFLOW_STEPS):
        active_class = "workflow-step active" if key == active_key else "workflow-step"
        with col:
            st.markdown(
                f"""
                <div class="{active_class}">
                    <div class="workflow-num">{number}</div>
                    <div class="workflow-label">{escape(label)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_page_header(
    active_key: str,
    title: str,
    question: str,
    description: str,
    chips: list[str] | None = None,
) -> None:
    """Render the standard page header."""
    render_workflow_indicator(active_key)
    chip_html = ""
    if chips:
        chip_html = "<div class='chip-row'>" + "".join(
            f"<span class='chip'>{escape(chip)}</span>" for chip in chips
        ) + "</div>"

    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-kicker">GDP Workflow Dashboard</div>
            <div class="page-title">{escape(title)}</div>
            <div class="page-question">{escape(question)}</div>
            <div class="page-copy">{escape(description)}</div>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    note: str,
    delta: str | None = None,
    tone: str = "neutral",
) -> None:
    """Render a dashboard metric card."""
    delta_html = ""
    if delta:
        delta_html = f"<div class='metric-delta {escape(tone)}'>{escape(delta)}</div>"

    card_html = (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{escape(label)}</div>"
        f"<div class='metric-value'>{escape(value)}</div>"
        f"{delta_html}"
        f"<div class='metric-note'>{escape(note)}</div>"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_insight_box(title: str, insights: list[str]) -> None:
    """Render a concise insight box."""
    if not insights:
        insights = ["No automatic insight is available for the current selection."]
    items = "".join(f"<li>{escape(item)}</li>" for item in insights)
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">{escape(title)}</div>
            <ul class="insight-list">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note_box(title: str, body: str) -> None:
    """Render a methodological note."""
    st.markdown(
        f"""
        <div class="note-box">
            <div class="note-title">{escape(title)}</div>
            <div class="note-copy">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline(steps: list[str]) -> None:
    """Render a simple workflow pipeline."""
    chunks: list[str] = []
    for index, step in enumerate(steps):
        chunks.append(f"<div class='pipeline-step'>{escape(step)}</div>")
        if index < len(steps) - 1:
            chunks.append("<div class='pipeline-arrow'>&rarr;</div>")

    st.markdown(
        f"<div class='pipeline-row'>{''.join(chunks)}</div>",
        unsafe_allow_html=True,
    )
