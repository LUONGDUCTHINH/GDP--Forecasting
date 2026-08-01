# %%
"""
Model 1 - Baseline GDP prediction

Equation:
    GDP_(t+1) ~ Population_t + LifeExpectancy_t

Implementation choice:
    - Target is modeled in log scale:
      target_log_gdp_next_year ~ log_population_total + life_expectancy_years
    - Estimator: Pooled OLS with HC3 robust standard errors
    - Split: time-based train/test split

This file is written with `# %%` cell markers so you can run it like a notebook
inside VS Code, or run it as a normal Python script.
"""

# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


pd.options.display.float_format = "{:,.4f}".format
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parents[1]
else:
    BASE_DIR = Path("/Users/tonytony/Final Project")

DATA_PATH = BASE_DIR / "Data" / "Cleaned" / "panel_with_event_dummies_and_extra_drivers.csv"
OUTPUT_DIR = BASE_DIR / "Data" / "Cleaned"

TRAIN_END_YEAR = 2017
TEST_START_YEAR = 2018
TEST_END_YEAR = 2022

MODEL_NAME = "Model 1 - Baseline Pooled OLS"
FORMULA = "target_log_gdp_next_year ~ log_population_total + life_expectancy_years"

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
    "log_gdp_per_capita",
    "population_total",
    "log_population_total",
    "life_expectancy_years",
    "target_log_gdp_next_year",
]

panel_df = pd.read_csv(DATA_PATH, usecols=model_cols)
model_df = panel_df.dropna().copy()

numeric_cols = [
    "year",
    "gdp_per_capita_usd",
    "log_gdp_per_capita",
    "population_total",
    "log_population_total",
    "life_expectancy_years",
    "target_log_gdp_next_year",
]
for col in numeric_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df = model_df.dropna().copy()
model_df["wb_region"] = model_df["wb_region"].astype(str)
model_df["country_name"] = model_df["country_name"].astype(str)
model_df["country_code"] = model_df["country_code"].astype(str)
model_df["year"] = model_df["year"].astype(int)

print("Model 1 sample shape:", model_df.shape)
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
print(
    "Test target years:",
    int((test_df["year"] + 1).min()),
    "-",
    int((test_df["year"] + 1).max()),
)


# %%
model_1 = smf.ols(formula=FORMULA, data=train_df).fit(cov_type="HC3")

print(MODEL_NAME)
print(model_1.summary())


# %%
coef_ci = model_1.conf_int()
coef_table = pd.DataFrame(
    {
        "term": model_1.params.index,
        "coefficient": model_1.params.values,
        "std_error": model_1.bse.values,
        "t_value": model_1.tvalues.values,
        "p_value": model_1.pvalues.values,
        "ci_lower": coef_ci[0].values,
        "ci_upper": coef_ci[1].values,
    }
).round(6)

coef_table


# %%
train_predictions_df = build_prediction_frame(model=model_1, df=train_df)
test_predictions_df = build_prediction_frame(model=model_1, df=test_df)

train_metrics_level = regression_metrics(
    actual=train_predictions_df["actual_gdp_next_year"],
    predicted=train_predictions_df["predicted_gdp_next_year"],
)
test_metrics_level = regression_metrics(
    actual=test_predictions_df["actual_gdp_next_year"],
    predicted=test_predictions_df["predicted_gdp_next_year"],
)
train_metrics_log = regression_metrics(
    actual=train_predictions_df["actual_log_gdp_next_year"],
    predicted=train_predictions_df["predicted_log_gdp_next_year"],
)
test_metrics_log = regression_metrics(
    actual=test_predictions_df["actual_log_gdp_next_year"],
    predicted=test_predictions_df["predicted_log_gdp_next_year"],
)

metrics_df = pd.DataFrame(
    [
        {
            "split": "train",
            "scale": "level_gdp_usd",
            "model": MODEL_NAME,
            "n_obs": len(train_predictions_df),
            **train_metrics_level,
        },
        {
            "split": "train",
            "scale": "log_gdp",
            "model": MODEL_NAME,
            "n_obs": len(train_predictions_df),
            **train_metrics_log,
        },
        {
            "split": "test",
            "scale": "level_gdp_usd",
            "model": MODEL_NAME,
            "n_obs": len(test_predictions_df),
            **test_metrics_level,
        },
        {
            "split": "test",
            "scale": "log_gdp",
            "model": MODEL_NAME,
            "n_obs": len(test_predictions_df),
            **test_metrics_log,
        },
    ]
).round(4)

metrics_df


# %%
region_metrics_df = build_region_metrics(test_predictions_df).round(4)
region_metrics_df.head(15)


# %%
test_preview_cols = [
    "country_name",
    "country_code",
    "wb_region",
    "feature_year",
    "target_year",
    "actual_gdp_next_year",
    "predicted_gdp_next_year",
    "absolute_error_usd",
    "absolute_percentage_error_pct",
]

test_predictions_df[test_preview_cols].sort_values(
    ["country_name", "feature_year"]
).head(20)


# %%
yearly_plot_df = (
    test_predictions_df.groupby("target_year", as_index=False)[
        ["actual_gdp_next_year", "predicted_gdp_next_year"]
    ]
    .mean()
    .sort_values("target_year")
)

plt.figure(figsize=(12, 6))
plt.plot(
    yearly_plot_df["target_year"],
    yearly_plot_df["actual_gdp_next_year"],
    marker="o",
    linewidth=2.2,
    label="Actual GDP per Capita",
)
plt.plot(
    yearly_plot_df["target_year"],
    yearly_plot_df["predicted_gdp_next_year"],
    marker="o",
    linewidth=2.2,
    linestyle="--",
    label="Predicted GDP per Capita",
)
plt.title("Model 1 Test Performance: Average Actual vs Predicted GDP per Capita")
plt.xlabel("Target Year")
plt.ylabel("GDP per Capita (US$)")
plt.legend()
plt.tight_layout()
yearly_plot_path = OUTPUT_DIR / "gdp_model_1_baseline_yearly_test_plot.png"
plt.savefig(yearly_plot_path, dpi=200, bbox_inches="tight")
show_or_close_plot()


# %%
scatter_sample_df = test_predictions_df.sample(
    n=min(800, len(test_predictions_df)),
    random_state=42,
).copy()

plt.figure(figsize=(8, 8))
plt.scatter(
    scatter_sample_df["actual_gdp_next_year"],
    scatter_sample_df["predicted_gdp_next_year"],
    alpha=0.55,
)

min_val = min(
    scatter_sample_df["actual_gdp_next_year"].min(),
    scatter_sample_df["predicted_gdp_next_year"].min(),
)
max_val = max(
    scatter_sample_df["actual_gdp_next_year"].max(),
    scatter_sample_df["predicted_gdp_next_year"].max(),
)
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
plt.title("Model 1 Test Scatter: Actual vs Predicted GDP per Capita")
plt.xlabel("Actual GDP per Capita (US$)")
plt.ylabel("Predicted GDP per Capita (US$)")
plt.tight_layout()
scatter_plot_path = OUTPUT_DIR / "gdp_model_1_baseline_scatter_test_plot.png"
plt.savefig(scatter_plot_path, dpi=200, bbox_inches="tight")
show_or_close_plot()


# %%
metrics_path = OUTPUT_DIR / "gdp_model_1_baseline_metrics.csv"
coef_path = OUTPUT_DIR / "gdp_model_1_baseline_coefficients.csv"
test_pred_path = OUTPUT_DIR / "gdp_model_1_baseline_test_predictions.csv"
region_metrics_path = OUTPUT_DIR / "gdp_model_1_baseline_region_metrics.csv"

metrics_df.to_csv(metrics_path, index=False)
coef_table.to_csv(coef_path, index=False)
test_predictions_df[test_preview_cols].round(4).to_csv(test_pred_path, index=False)
region_metrics_df.to_csv(region_metrics_path, index=False)

print("Saved:")
print("-", metrics_path)
print("-", coef_path)
print("-", test_pred_path)
print("-", region_metrics_path)
print("-", yearly_plot_path)
print("-", scatter_plot_path)


# %%
# Important interpretation note:
# Model 1 predicts next-year GDP from same-year population and life expectancy.
# For a truly future target year such as 2040, we would first need:
#   1. Population_2039
#   2. LifeExpectancy_2039
# Those inputs can come from the separate time-series forecasting models you built.
