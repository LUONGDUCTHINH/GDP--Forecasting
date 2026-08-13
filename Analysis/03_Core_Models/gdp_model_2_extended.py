# %%
"""
Model 2 - Extended GDP prediction

Equation:
    GDP_(t+1) ~ Population_t + LifeExpectancy_t
               + Inflation_t + Unemployment_t + InternetUsage_t

Implementation choice:
    - Target is modeled in log scale
    - Estimator: Pooled OLS with HC3 robust standard errors
    - Split: train feature years <= 2017, test feature years 2018 to 2022
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

MODEL_NAME = "Model 2 - Extended Pooled OLS"
FORMULA = (
    "target_log_gdp_next_year ~ log_population_total + life_expectancy_years "
    "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean"
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
]
for col in numeric_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df = model_df.dropna().copy()
model_df["wb_region"] = model_df["wb_region"].astype(str)
model_df["country_name"] = model_df["country_name"].astype(str)
model_df["country_code"] = model_df["country_code"].astype(str)
model_df["year"] = model_df["year"].astype(int)

print("Model 2 sample shape:", model_df.shape)
print("Feature-year range:", int(model_df["year"].min()), "-", int(model_df["year"].max()))
print(
    "Target-year range:",
    int((model_df["year"] + 1).min()),
    "-",
    int((model_df["year"] + 1).max()),
)

model_df.head()


# %%
train_df = model_df[model_df["year"] <= TRAIN_END_YEAR].copy()
test_df = model_df[
    (model_df["year"] >= TEST_START_YEAR) & (model_df["year"] <= TEST_END_YEAR)
].copy()

print("Train rows:", train_df.shape[0])
print("Test rows:", test_df.shape[0])
print(
    "Train feature years:",
    int(train_df["year"].min()),
    "-",
    int(train_df["year"].max()),
)
print(
    "Test feature years:",
    int(test_df["year"].min()),
    "-",
    int(test_df["year"].max()),
)


# %%
model_2 = smf.ols(formula=FORMULA, data=train_df).fit(cov_type="HC3")

print(MODEL_NAME)
print(model_2.summary())


# %%
coef_ci = model_2.conf_int()
coef_table = pd.DataFrame(
    {
        "term": model_2.params.index,
        "coefficient": model_2.params.values,
        "std_error": model_2.bse.values,
        "t_value": model_2.tvalues.values,
        "p_value": model_2.pvalues.values,
        "ci_lower": coef_ci[0].values,
        "ci_upper": coef_ci[1].values,
    }
).round(6)

coef_table


# %%
train_predictions_df = build_prediction_frame(model=model_2, df=train_df)
test_predictions_df = build_prediction_frame(model=model_2, df=test_df)

metrics_df = pd.DataFrame(
    [
        {
            "split": "train",
            "scale": "level_gdp_usd",
            "model": MODEL_NAME,
            "n_obs": len(train_predictions_df),
            **regression_metrics(
                actual=train_predictions_df["actual_gdp_next_year"],
                predicted=train_predictions_df["predicted_gdp_next_year"],
            ),
        },
        {
            "split": "train",
            "scale": "log_gdp",
            "model": MODEL_NAME,
            "n_obs": len(train_predictions_df),
            **regression_metrics(
                actual=train_predictions_df["actual_log_gdp_next_year"],
                predicted=train_predictions_df["predicted_log_gdp_next_year"],
            ),
        },
        {
            "split": "test",
            "scale": "level_gdp_usd",
            "model": MODEL_NAME,
            "n_obs": len(test_predictions_df),
            **regression_metrics(
                actual=test_predictions_df["actual_gdp_next_year"],
                predicted=test_predictions_df["predicted_gdp_next_year"],
            ),
        },
        {
            "split": "test",
            "scale": "log_gdp",
            "model": MODEL_NAME,
            "n_obs": len(test_predictions_df),
            **regression_metrics(
                actual=test_predictions_df["actual_log_gdp_next_year"],
                predicted=test_predictions_df["predicted_log_gdp_next_year"],
            ),
        },
    ]
).round(4)

metrics_df


# %%
region_metrics_df = build_region_metrics(test_predictions_df).round(4)
region_metrics_df


# %%
country_plot_df = (
    test_predictions_df.groupby("target_year", as_index=False)[
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
plt.title("Model 2: Actual vs Predicted GDP per Capita")
plt.xlabel("Target Year")
plt.ylabel("GDP per Capita (US$)")
plt.legend()
plt.tight_layout()
show_or_close_plot()


# %%
metrics_path = OUTPUT_DIR / "gdp_model_2_extended_metrics.csv"
coef_path = OUTPUT_DIR / "gdp_model_2_extended_coefficients.csv"
test_pred_path = OUTPUT_DIR / "gdp_model_2_extended_test_predictions.csv"
region_metrics_path = OUTPUT_DIR / "gdp_model_2_extended_region_metrics.csv"

metrics_df.to_csv(metrics_path, index=False)
coef_table.to_csv(coef_path, index=False)
test_predictions_df.to_csv(test_pred_path, index=False)
region_metrics_df.to_csv(region_metrics_path, index=False)

print("Saved:")
print("-", metrics_path)
print("-", coef_path)
print("-", test_pred_path)
print("-", region_metrics_path)
