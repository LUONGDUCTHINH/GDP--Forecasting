# %%
"""
Model 3 - Full GDP modeling

This file keeps two connected versions of Model 3:

1. Forecasting specification
   - Used for train/test prediction
   - Adds extra drivers, trainable event dummies, region effects, and year trend

2. Full fixed-effects robustness specification
   - Used on the full sample for interpretation
   - Adds region effects, year fixed effects, and exposure-based event dummies

Why split Model 3 this way?
   - Raw year fixed effects cannot predict an unseen future year
   - Pure global event dummies are collinear with year fixed effects
   - Post-2017 global shocks do not appear in the 2017 training window
"""

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


pd.options.display.float_format = "{:,.4f}".format
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


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

FORECAST_MODEL_NAME = "Model 3 - Full Forecasting"
FORECAST_FORMULA = (
    "target_log_gdp_next_year ~ log_population_total + life_expectancy_years "
    "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean "
    "+ asian_financial_crisis_9798 + global_financial_crisis_0809 "
    "+ C(wb_region) + year_trend"
)

ROBUSTNESS_MODEL_NAME = "Model 3 - Full Fixed Effects"
ROBUSTNESS_FORMULA = (
    "target_log_gdp_next_year ~ log_population_total + life_expectancy_years "
    "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean "
    "+ asia_crisis_exposed_9798 + energy_shock_exposed_2022_2024 "
    "+ C(wb_region) + C(year)"
)

print("Base dir:", BASE_DIR)
print("Data path:", DATA_PATH)
print("Output dir:", OUTPUT_DIR)


# %%
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


def build_prediction_frame(model, df):
    pred_df = df.copy()
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


def build_region_metrics(df):
    rows = []
    for region_name, region_df in df.groupby("wb_region"):
        metrics = regression_metrics(
            actual=region_df["actual_gdp_next_year"],
            predicted=region_df["predicted_gdp_next_year"],
        )
        rows.append(
            {
                "wb_region": region_name,
                "n_obs": len(region_df),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def coefficient_table(model, model_name):
    coef_ci = model.conf_int()
    return pd.DataFrame(
        {
            "model": model_name,
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower": coef_ci[0].values,
            "ci_upper": coef_ci[1].values,
        }
    ).round(6)


def show_or_close_plot():
    if "ipykernel" in sys.modules:
        plt.show()
    else:
        plt.close()


print("Helper functions ready.")


# %%
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
    "asia_crisis_exposed_9798",
    "energy_shock_exposed_2022_2024",
]

panel_df = pd.read_csv(DATA_PATH, usecols=model_cols)
model_df = panel_df.dropna().copy()

numeric_cols = [
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
    "asia_crisis_exposed_9798",
    "energy_shock_exposed_2022_2024",
]
for col in numeric_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df = model_df.dropna().copy()
model_df["year"] = model_df["year"].astype(int)
model_df["year_trend"] = model_df["year"] - model_df["year"].min()

print("Model 3 sample shape:", model_df.shape)
print("Feature-year range:", int(model_df["year"].min()), "-", int(model_df["year"].max()))


# %%
train_df = model_df[model_df["year"] <= TRAIN_END_YEAR].copy()
test_df = model_df[
    (model_df["year"] >= TEST_START_YEAR) & (model_df["year"] <= TEST_END_YEAR)
].copy()

print("Train rows:", train_df.shape[0])
print("Test rows:", test_df.shape[0])


# %%
forecast_model = smf.ols(formula=FORECAST_FORMULA, data=train_df).fit(cov_type="HC3")

print(FORECAST_MODEL_NAME)
print(forecast_model.summary())


# %%
forecast_train_df = build_prediction_frame(model=forecast_model, df=train_df)
forecast_test_df = build_prediction_frame(model=forecast_model, df=test_df)

forecast_metrics_df = pd.DataFrame(
    [
        {
            "split": "train",
            "scale": "level_gdp_usd",
            "model": FORECAST_MODEL_NAME,
            "n_obs": len(forecast_train_df),
            **regression_metrics(
                actual=forecast_train_df["actual_gdp_next_year"],
                predicted=forecast_train_df["predicted_gdp_next_year"],
            ),
        },
        {
            "split": "train",
            "scale": "log_gdp",
            "model": FORECAST_MODEL_NAME,
            "n_obs": len(forecast_train_df),
            **regression_metrics(
                actual=forecast_train_df["actual_log_gdp_next_year"],
                predicted=forecast_train_df["predicted_log_gdp_next_year"],
            ),
        },
        {
            "split": "test",
            "scale": "level_gdp_usd",
            "model": FORECAST_MODEL_NAME,
            "n_obs": len(forecast_test_df),
            **regression_metrics(
                actual=forecast_test_df["actual_gdp_next_year"],
                predicted=forecast_test_df["predicted_gdp_next_year"],
            ),
        },
        {
            "split": "test",
            "scale": "log_gdp",
            "model": FORECAST_MODEL_NAME,
            "n_obs": len(forecast_test_df),
            **regression_metrics(
                actual=forecast_test_df["actual_log_gdp_next_year"],
                predicted=forecast_test_df["predicted_log_gdp_next_year"],
            ),
        },
    ]
).round(4)

forecast_metrics_df


# %%
forecast_region_metrics_df = build_region_metrics(forecast_test_df).round(4)
forecast_region_metrics_df


# %%
country_plot_df = (
    forecast_test_df.groupby("target_year", as_index=False)[
        ["actual_gdp_next_year", "predicted_gdp_next_year"]
    ]
    .mean()
    .sort_values("target_year")
)

plt.plot(
    country_plot_df["target_year"],
    country_plot_df["actual_gdp_next_year"],
    marker="o",
    linewidth=2.5,
    label="Actual GDP per Capita",
)
plt.plot(
    country_plot_df["target_year"],
    country_plot_df["predicted_gdp_next_year"],
    marker="o",
    linewidth=2.5,
    linestyle="--",
    label="Predicted GDP per Capita",
)
plt.title("Model 3 Forecasting Spec: Actual vs Predicted GDP per Capita")
plt.xlabel("Target Year")
plt.ylabel("GDP per Capita (US$)")
plt.legend()
plt.tight_layout()
show_or_close_plot()


# %%
forecast_coef_df = coefficient_table(forecast_model, FORECAST_MODEL_NAME)

forecast_metrics_path = OUTPUT_DIR / "gdp_model_3_full_forecast_metrics.csv"
forecast_coef_path = OUTPUT_DIR / "gdp_model_3_full_forecast_coefficients.csv"
forecast_pred_path = OUTPUT_DIR / "gdp_model_3_full_forecast_test_predictions.csv"
forecast_region_metrics_path = OUTPUT_DIR / "gdp_model_3_full_forecast_region_metrics.csv"

forecast_metrics_df.to_csv(forecast_metrics_path, index=False)
forecast_coef_df.to_csv(forecast_coef_path, index=False)
forecast_test_df.to_csv(forecast_pred_path, index=False)
forecast_region_metrics_df.to_csv(forecast_region_metrics_path, index=False)

print("Saved forecasting outputs:")
print("-", forecast_metrics_path)
print("-", forecast_coef_path)
print("-", forecast_pred_path)
print("-", forecast_region_metrics_path)


# %%
robustness_model = smf.ols(formula=ROBUSTNESS_FORMULA, data=model_df).fit(cov_type="HC3")

print(ROBUSTNESS_MODEL_NAME)
print(robustness_model.summary())


# %%
robustness_fitted_df = build_prediction_frame(model=robustness_model, df=model_df)
robustness_coef_df = coefficient_table(robustness_model, ROBUSTNESS_MODEL_NAME)

robustness_metrics_df = pd.DataFrame(
    [
        {
            "split": "full_sample",
            "scale": "level_gdp_usd",
            "model": ROBUSTNESS_MODEL_NAME,
            "n_obs": len(robustness_fitted_df),
            **regression_metrics(
                actual=robustness_fitted_df["actual_gdp_next_year"],
                predicted=robustness_fitted_df["predicted_gdp_next_year"],
            ),
        },
        {
            "split": "full_sample",
            "scale": "log_gdp",
            "model": ROBUSTNESS_MODEL_NAME,
            "n_obs": len(robustness_fitted_df),
            **regression_metrics(
                actual=robustness_fitted_df["actual_log_gdp_next_year"],
                predicted=robustness_fitted_df["predicted_log_gdp_next_year"],
            ),
        },
    ]
).round(4)

robustness_metrics_df


# %%
robustness_metrics_path = OUTPUT_DIR / "gdp_model_3_full_fe_metrics.csv"
robustness_coef_path = OUTPUT_DIR / "gdp_model_3_full_fe_coefficients.csv"
robustness_fitted_path = OUTPUT_DIR / "gdp_model_3_full_fe_fitted_values.csv"

robustness_metrics_df.to_csv(robustness_metrics_path, index=False)
robustness_coef_df.to_csv(robustness_coef_path, index=False)
robustness_fitted_df.to_csv(robustness_fitted_path, index=False)

print("Saved fixed-effects robustness outputs:")
print("-", robustness_metrics_path)
print("-", robustness_coef_path)
print("-", robustness_fitted_path)
