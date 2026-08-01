from __future__ import annotations

import math

import pandas as pd


def format_compact_number(value: float | int | None, digits: int = 2) -> str:
    """Format a number with compact K/M/B/T suffixes."""
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)
    sign = "-" if value < 0 else ""
    abs_value = abs(value)

    if abs_value >= 1_000_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000_000:.{digits}f}T"
    if abs_value >= 1_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000:.{digits}f}B"
    if abs_value >= 1_000_000:
        return f"{sign}{abs_value / 1_000_000:.{digits}f}M"
    if abs_value >= 1_000:
        return f"{sign}{abs_value / 1_000:.{digits}f}K"
    if abs_value >= 100:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def format_currency(
    value: float | int | None,
    digits: int = 2,
    symbol: str = "$",
    compact: bool = False,
) -> str:
    """Format a monetary value with a configurable symbol."""
    if value is None or pd.isna(value):
        return "N/A"

    if compact:
        return f"{symbol}{format_compact_number(value, digits=digits)}"
    return f"{symbol}{float(value):,.{digits}f}"


def format_percent(value: float | int | None, digits: int = 2) -> str:
    """Format a percentage value."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}%"


def format_years(value: float | int | None, digits: int = 1) -> str:
    """Format a life expectancy style value in years."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f} years"


def format_integer(value: float | int | None) -> str:
    """Format an integer-like value with thousand separators."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}"


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float:
    """Return a safe division result or NaN."""
    if numerator is None or denominator is None:
        return math.nan
    if pd.isna(numerator) or pd.isna(denominator):
        return math.nan
    denominator = float(denominator)
    if denominator == 0:
        return math.nan
    return float(numerator) / denominator


def correlation_strength(value: float | int | None) -> str:
    """Return a qualitative description for a correlation coefficient."""
    if value is None or pd.isna(value):
        return "Unavailable"

    abs_value = abs(float(value))
    if abs_value < 0.20:
        return "Very weak"
    if abs_value < 0.40:
        return "Weak"
    if abs_value < 0.60:
        return "Moderate"
    if abs_value < 0.80:
        return "Strong"
    return "Very strong"


def correlation_direction(value: float | int | None) -> str:
    """Return the correlation direction label."""
    if value is None or pd.isna(value):
        return "Unavailable"
    return "Positive" if float(value) >= 0 else "Negative"
