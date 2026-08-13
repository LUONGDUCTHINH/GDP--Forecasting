"""
Evaluate the three main GDP models on a common train/test split.

This script focuses on the forecastable versions of the three main models.

Why this matters:
    - Raw year fixed effects, C(year), cannot predict an unseen future year
    - Pure global event dummies for 2020, 2021, and 2022+ cannot be learned
      when the training window ends in 2017
    - For that reason, the backtest below keeps Model 3 in a forecast-ready form:
      extra drivers + trainable pre-2018 event dummies + region effects + year trend

The full fixed-effects robustness version of Model 3 should be estimated
separately on the full sample rather than inside this out-of-time backtest.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def resolve_base_dir() -> Path:
    if "__file__" in globals():
        for candidate in Path(__file__).resolve().parents:
            if (candidate / "Data").exists() and (candidate / "app.py").exists():
                return candidate
    return Path("/Users/tonytony/Final Project")


BASE_DIR = resolve_base_dir()

DATA_PATH = BASE_DIR / "Data" / "Cleaned" / "panel_with_event_dummies_and_extra_drivers.csv"
OUTPUT_DIR = BASE_DIR / "Data" / "Cleaned"

TRAIN_END_YEAR = 2017
TEST_START_YEAR = 2018
TEST_END_YEAR = 2022

MODEL_SPECS = [
    {
        "model": "Model 1 - Baseline",
        "family": "Pooled OLS forecast model",
        "formula": (
            "target_log_gdp_next_year ~ log_population_total + life_expectancy_years"
        ),
        "note": "Uses population and life expectancy only.",
    },
    {
        "model": "Model 2 - Extended",
        "family": "Pooled OLS forecast model",
        "formula": (
            "target_log_gdp_next_year ~ log_population_total + life_expectancy_years "
            "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean"
        ),
        "note": "Adds inflation, unemployment, and internet usage.",
    },
    {
        "model": "Model 3 - Full Forecasting",
        "family": "Pooled OLS forecast model",
        "formula": (
            "target_log_gdp_next_year ~ log_population_total + life_expectancy_years "
            "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean "
            "+ asian_financial_crisis_9798 + global_financial_crisis_0809 "
            "+ C(wb_region) + year_trend"
        ),
        "note": (
            "Forecast-ready Model 3. Uses region effects and a linear year trend. "
            "Post-2017 global event dummies are excluded from the backtest because "
            "they do not appear in the training window."
        ),
    },
]


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    nonzero_mask = actual != 0
    if nonzero_mask.any():
        mape = np.mean(
            np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])
        ) * 100
    else:
        mape = np.nan

    sst = np.sum((actual - actual.mean()) ** 2)
    sse = np.sum((actual - predicted) ** 2)
    r2 = 1 - (sse / sst) if sst > 0 else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_pct": mape,
        "R_squared": r2,
    }


def build_prediction_frame(model, df, model_name, family, note):
    pred_df = df.copy()
    pred_df["model"] = model_name
    pred_df["model_family"] = family
    pred_df["model_note"] = note
    pred_df["predicted_log_gdp_next_year"] = model.predict(pred_df)
    pred_df["actual_log_gdp_next_year"] = pred_df["target_log_gdp_next_year"]
    pred_df["predicted_gdp_next_year"] = np.exp(pred_df["predicted_log_gdp_next_year"])
    pred_df["actual_gdp_next_year"] = np.exp(pred_df["actual_log_gdp_next_year"])
    pred_df["feature_year"] = pred_df["year"].astype(int)
    pred_df["target_year"] = pred_df["feature_year"] + 1
    pred_df["absolute_error_usd"] = np.abs(
        pred_df["actual_gdp_next_year"] - pred_df["predicted_gdp_next_year"]
    )
    pred_df["absolute_percentage_error_pct"] = (
        pred_df["absolute_error_usd"] / pred_df["actual_gdp_next_year"]
    ) * 100
    return pred_df


def build_region_metrics(pred_df):
    rows = []
    for (model_name, family, region_name), region_df in pred_df.groupby(
        ["model", "model_family", "wb_region"]
    ):
        metrics = regression_metrics(
            actual=region_df["actual_gdp_next_year"],
            predicted=region_df["predicted_gdp_next_year"],
        )
        rows.append(
            {
                "model": model_name,
                "model_family": family,
                "wb_region": region_name,
                "n_obs": len(region_df),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "RMSE"]).reset_index(drop=True)


def build_metrics_rows(model, model_name, family, note, train_df, test_df):
    rows = []
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        pred_df = build_prediction_frame(model, split_df, model_name, family, note)

        level_metrics = regression_metrics(
            actual=pred_df["actual_gdp_next_year"],
            predicted=pred_df["predicted_gdp_next_year"],
        )
        log_metrics = regression_metrics(
            actual=pred_df["actual_log_gdp_next_year"],
            predicted=pred_df["predicted_log_gdp_next_year"],
        )

        rows.append(
            {
                "model": model_name,
                "model_family": family,
                "model_note": note,
                "split": split_name,
                "scale": "level_gdp_usd",
                "n_obs": len(pred_df),
                **level_metrics,
            }
        )
        rows.append(
            {
                "model": model_name,
                "model_family": family,
                "model_note": note,
                "split": split_name,
                "scale": "log_gdp",
                "n_obs": len(pred_df),
                **log_metrics,
            }
        )
    return rows


model_cols = [
    "country_name",
    "country_code",
    "wb_region",
    "year",
    "gdp_per_capita_usd",
    "population_total",
    "log_population_total",
    "life_expectancy_years",
    "inflation_pct_clean",
    "unemployment_pct_clean",
    "internet_users_pct_clean",
    "target_log_gdp_next_year",
    "asian_financial_crisis_9798",
    "global_financial_crisis_0809",
]

panel_df = pd.read_csv(DATA_PATH, usecols=model_cols)
model_df = panel_df.dropna().copy()
model_df["year"] = pd.to_numeric(model_df["year"], errors="coerce").astype(int)
model_df["year_trend"] = model_df["year"] - model_df["year"].min()

train_df = model_df[model_df["year"] <= TRAIN_END_YEAR].copy()
test_df = model_df[
    (model_df["year"] >= TEST_START_YEAR) & (model_df["year"] <= TEST_END_YEAR)
].copy()

print("Shared model sample:", model_df.shape)
print("Train rows:", train_df.shape[0])
print("Test rows:", test_df.shape[0])

metrics_rows = []
coef_rows = []
test_predictions = []

for spec in MODEL_SPECS:
    model_name = spec["model"]
    formula = spec["formula"]
    family = spec["family"]
    note = spec["note"]

    fitted = smf.ols(formula=formula, data=train_df).fit(cov_type="HC3")
    print(f"\n{model_name}")
    print("Formula:", formula)
    print("Train R-squared:", round(fitted.rsquared, 4))

    metrics_rows.extend(
        build_metrics_rows(fitted, model_name, family, note, train_df, test_df)
    )

    coef_ci = fitted.conf_int()
    coef_table = pd.DataFrame(
        {
            "model": model_name,
            "model_family": family,
            "model_note": note,
            "term": fitted.params.index,
            "coefficient": fitted.params.values,
            "std_error": fitted.bse.values,
            "t_value": fitted.tvalues.values,
            "p_value": fitted.pvalues.values,
            "ci_lower": coef_ci[0].values,
            "ci_upper": coef_ci[1].values,
        }
    )
    coef_rows.append(coef_table)

    test_pred_df = build_prediction_frame(fitted, test_df, model_name, family, note)
    test_predictions.append(
        test_pred_df[
            [
                "model",
                "model_family",
                "model_note",
                "country_name",
                "country_code",
                "wb_region",
                "feature_year",
                "target_year",
                "actual_gdp_next_year",
                "predicted_gdp_next_year",
                "actual_log_gdp_next_year",
                "predicted_log_gdp_next_year",
                "absolute_error_usd",
                "absolute_percentage_error_pct",
            ]
        ]
    )

metrics_df = pd.DataFrame(metrics_rows).round(4)
coef_df = pd.concat(coef_rows, ignore_index=True).round(6)
test_predictions_df = pd.concat(test_predictions, ignore_index=True).round(4)
region_metrics_df = build_region_metrics(test_predictions_df).round(4)
spec_df = pd.DataFrame(MODEL_SPECS)

metrics_path = OUTPUT_DIR / "gdp_main_models_train_test_metrics.csv"
coef_path = OUTPUT_DIR / "gdp_main_models_coefficients.csv"
test_pred_path = OUTPUT_DIR / "gdp_main_models_test_predictions.csv"
region_metrics_path = OUTPUT_DIR / "gdp_main_models_region_metrics.csv"
spec_path = OUTPUT_DIR / "gdp_main_models_specifications.csv"

metrics_df.to_csv(metrics_path, index=False)
coef_df.to_csv(coef_path, index=False)
test_predictions_df.to_csv(test_pred_path, index=False)
region_metrics_df.to_csv(region_metrics_path, index=False)
spec_df.to_csv(spec_path, index=False)

print("\nSaved:")
print("-", metrics_path)
print("-", coef_path)
print("-", test_pred_path)
print("-", region_metrics_path)
print("-", spec_path)
