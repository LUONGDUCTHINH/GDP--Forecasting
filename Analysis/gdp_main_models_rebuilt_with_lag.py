# %%
"""
Rebuilt 3 main GDP models with lag GDP as a core predictor.

Why this rebuild?
    - The previous pooled OLS versions showed large bias in level predictions
    - GDP per capita is highly persistent over time, so GDP_t should help predict GDP_(t+1)
    - The rebuilt models keep the original idea of 3 progressive specifications,
      but turn them into dynamic forecasting models

Rebuilt models:
    Model 1:
        GDP_(t+1) ~ GDP_t + Population_t + LifeExpectancy_t

    Model 2:
        GDP_(t+1) ~ GDP_t + Population_t + LifeExpectancy_t
                   + Inflation_t + Unemployment_t + Internet_t

    Model 3:
        GDP_(t+1) ~ GDP_t + Population_t + LifeExpectancy_t
                   + Inflation_t + Unemployment_t + Internet_t
                   + Event Dummies + Region Effects + Year Trend

Implementation:
    - Target is modeled in log scale:
      target_log_gdp_next_year
    - Estimator: pooled OLS with HC3 robust standard errors
    - Backtest split:
        train feature years <= 2017
        test feature years 2018 to 2022
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


if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parents[1]
else:
    BASE_DIR = Path("/Users/tonytony/Final Project")

DATA_PATH = BASE_DIR / "Data" / "Cleaned" / "panel_with_event_dummies_and_extra_drivers.csv"
OUTPUT_DIR = BASE_DIR / "Data" / "Cleaned"

TRAIN_END_YEAR = 2017
TEST_START_YEAR = 2018
TEST_END_YEAR = 2022

MODEL_SPECS = [
    {
        "model": "Model 1 - Baseline Dynamic",
        "family": "Dynamic pooled OLS",
        "formula": (
            "target_log_gdp_next_year ~ "
            "log_gdp_per_capita + log_population_total + life_expectancy_years"
        ),
        "note": "Adds lag GDP to the baseline specification.",
    },
    {
        "model": "Model 2 - Extended Dynamic",
        "family": "Dynamic pooled OLS",
        "formula": (
            "target_log_gdp_next_year ~ "
            "log_gdp_per_capita + log_population_total + life_expectancy_years "
            "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean"
        ),
        "note": "Adds macro and digital indicators on top of lag GDP.",
    },
    {
        "model": "Model 3 - Full Dynamic",
        "family": "Dynamic pooled OLS",
        "formula": (
            "target_log_gdp_next_year ~ "
            "log_gdp_per_capita + log_population_total + life_expectancy_years "
            "+ inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean "
            "+ asian_financial_crisis_9798 + global_financial_crisis_0809 "
            "+ C(wb_region) + year_trend"
        ),
        "note": (
            "Full predictive model with lag GDP, extra drivers, trainable event dummies, "
            "region effects, and a linear time trend."
        ),
    },
]

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
    pred_df["signed_percentage_error_pct"] = (
        (pred_df["predicted_gdp_next_year"] - pred_df["actual_gdp_next_year"])
        / pred_df["actual_gdp_next_year"]
    ) * 100

    return pred_df


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


def build_region_metrics(pred_df):
    rows = []
    for (model_name, region_name), region_df in pred_df.groupby(["model", "wb_region"]):
        metrics = regression_metrics(
            actual=region_df["actual_gdp_next_year"],
            predicted=region_df["predicted_gdp_next_year"],
        )
        rows.append(
            {
                "model": model_name,
                "wb_region": region_name,
                "n_obs": len(region_df),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "RMSE"]).reset_index(drop=True)


def build_yearly_summary(pred_df):
    rows = []
    for (model_name, target_year), year_df in pred_df.groupby(["model", "target_year"]):
        rows.append(
            {
                "model": model_name,
                "target_year": int(target_year),
                "n_obs": len(year_df),
                "actual_mean_gdp": year_df["actual_gdp_next_year"].mean(),
                "predicted_mean_gdp": year_df["predicted_gdp_next_year"].mean(),
                "pred_to_actual_ratio": (
                    year_df["predicted_gdp_next_year"].mean()
                    / year_df["actual_gdp_next_year"].mean()
                ),
                "mean_absolute_percentage_error_pct": year_df[
                    "absolute_percentage_error_pct"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "target_year"]).reset_index(drop=True)


def build_decile_bias_table(pred_df):
    rows = []
    for model_name, model_df in pred_df.groupby("model"):
        work_df = model_df.copy()
        work_df["actual_decile"] = pd.qcut(
            work_df["actual_gdp_next_year"],
            q=10,
            duplicates="drop",
        )

        for decile_label, decile_df in work_df.groupby("actual_decile", observed=False):
            rows.append(
                {
                    "model": model_name,
                    "actual_decile": str(decile_label),
                    "n_obs": len(decile_df),
                    "actual_mean_gdp": decile_df["actual_gdp_next_year"].mean(),
                    "predicted_mean_gdp": decile_df["predicted_gdp_next_year"].mean(),
                    "pred_to_actual_ratio": (
                        decile_df["predicted_gdp_next_year"].mean()
                        / decile_df["actual_gdp_next_year"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows).reset_index(drop=True)


def coefficient_table(fitted_model, model_name, family, note):
    coef_ci = fitted_model.conf_int()
    return pd.DataFrame(
        {
            "model": model_name,
            "model_family": family,
            "model_note": note,
            "term": fitted_model.params.index,
            "coefficient": fitted_model.params.values,
            "std_error": fitted_model.bse.values,
            "t_value": fitted_model.tvalues.values,
            "p_value": fitted_model.pvalues.values,
            "ci_lower": coef_ci[0].values,
            "ci_upper": coef_ci[1].values,
        }
    )


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
    "inflation_pct_clean",
    "unemployment_pct_clean",
    "internet_users_pct_clean",
    "target_log_gdp_next_year",
    "asian_financial_crisis_9798",
    "global_financial_crisis_0809",
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
    "inflation_pct_clean",
    "unemployment_pct_clean",
    "internet_users_pct_clean",
    "target_log_gdp_next_year",
    "asian_financial_crisis_9798",
    "global_financial_crisis_0809",
]
for col in numeric_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df = model_df.dropna().copy()
model_df["year"] = model_df["year"].astype(int)
model_df["country_name"] = model_df["country_name"].astype(str)
model_df["country_code"] = model_df["country_code"].astype(str)
model_df["wb_region"] = model_df["wb_region"].astype(str)
model_df["year_trend"] = model_df["year"] - model_df["year"].min()

print("Model sample shape:", model_df.shape)
print("Countries:", model_df["country_code"].nunique())
print("Regions:", model_df["wb_region"].nunique())
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
print("Train countries:", train_df["country_code"].nunique())
print("Test countries:", test_df["country_code"].nunique())
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
fitted_models = {}
metrics_rows = []
coef_tables = []
train_prediction_tables = []
test_prediction_tables = []

for spec in MODEL_SPECS:
    model_name = spec["model"]
    family = spec["family"]
    formula = spec["formula"]
    note = spec["note"]

    fitted = smf.ols(formula=formula, data=train_df).fit(cov_type="HC3")
    fitted_models[model_name] = fitted

    print("\n" + "=" * 90)
    print(model_name)
    print("Family :", family)
    print("Formula:", formula)
    print("Train R-squared:", round(fitted.rsquared, 4))

    metrics_rows.extend(
        build_metrics_rows(fitted, model_name, family, note, train_df, test_df)
    )
    coef_tables.append(coefficient_table(fitted, model_name, family, note))
    train_prediction_tables.append(
        build_prediction_frame(fitted, train_df, model_name, family, note)
    )
    test_prediction_tables.append(
        build_prediction_frame(fitted, test_df, model_name, family, note)
    )


# %%
metrics_df = pd.DataFrame(metrics_rows).round(4)
coef_df = pd.concat(coef_tables, ignore_index=True).round(6)
train_predictions_df = pd.concat(train_prediction_tables, ignore_index=True).round(4)
test_predictions_df = pd.concat(test_prediction_tables, ignore_index=True).round(4)
region_metrics_df = build_region_metrics(test_predictions_df).round(4)
yearly_summary_df = build_yearly_summary(test_predictions_df).round(4)
decile_bias_df = build_decile_bias_table(test_predictions_df).round(4)
spec_df = pd.DataFrame(MODEL_SPECS)

print("Metrics shape:", metrics_df.shape)
print("Coefficient rows:", coef_df.shape[0])
print("Train prediction rows:", train_predictions_df.shape[0])
print("Test prediction rows:", test_predictions_df.shape[0])

metrics_df


# %%
metrics_pivot = metrics_df.pivot_table(
    index=["model", "split"],
    columns="scale",
    values=["RMSE", "MAPE_pct", "R_squared"],
)
metrics_pivot.round(4)


# %%
coef_df.head(30)


# %%
test_predictions_df.head(20)


# %%
bias_summary_df = (
    test_predictions_df.groupby("model", as_index=False)
    .agg(
        n_obs=("actual_gdp_next_year", "size"),
        actual_mean_gdp=("actual_gdp_next_year", "mean"),
        predicted_mean_gdp=("predicted_gdp_next_year", "mean"),
        actual_median_gdp=("actual_gdp_next_year", "median"),
        predicted_median_gdp=("predicted_gdp_next_year", "median"),
        mean_absolute_percentage_error_pct=("absolute_percentage_error_pct", "mean"),
        mean_signed_percentage_error_pct=("signed_percentage_error_pct", "mean"),
        median_signed_percentage_error_pct=("signed_percentage_error_pct", "median"),
    )
)
bias_summary_df["pred_to_actual_mean_ratio"] = (
    bias_summary_df["predicted_mean_gdp"] / bias_summary_df["actual_mean_gdp"]
)
bias_summary_df = bias_summary_df.round(4)

bias_summary_df


# %%
yearly_summary_df


# %%
decile_bias_df


# %%
year_plot_df = (
    test_predictions_df.groupby(["model", "target_year"], as_index=False)[
        ["actual_gdp_next_year", "predicted_gdp_next_year"]
    ]
    .mean()
    .sort_values(["model", "target_year"])
)

plt.figure(figsize=(14, 7))
sns.lineplot(
    data=year_plot_df,
    x="target_year",
    y="actual_gdp_next_year",
    hue="model",
    style="model",
    linewidth=2.5,
    legend=True,
)
sns.lineplot(
    data=year_plot_df,
    x="target_year",
    y="predicted_gdp_next_year",
    hue="model",
    style="model",
    linewidth=2.5,
    dashes=True,
    legend=False,
)
plt.title("Rebuilt 3 Main GDP Models: Mean Actual vs Predicted GDP per Capita")
plt.xlabel("Target Year")
plt.ylabel("GDP per Capita (US$)")
plt.tight_layout()
show_or_close_plot()


# %%
best_model_name = (
    metrics_df[
        (metrics_df["split"] == "test")
        & (metrics_df["scale"] == "level_gdp_usd")
    ]
    .sort_values(["RMSE", "MAPE_pct"])
    .iloc[0]["model"]
)

best_model_pred_df = test_predictions_df[
    test_predictions_df["model"] == best_model_name
].copy()

plt.figure(figsize=(8, 8))
plt.scatter(
    best_model_pred_df["actual_gdp_next_year"],
    best_model_pred_df["predicted_gdp_next_year"],
    alpha=0.6,
)

diag_min = min(
    best_model_pred_df["actual_gdp_next_year"].min(),
    best_model_pred_df["predicted_gdp_next_year"].min(),
)
diag_max = max(
    best_model_pred_df["actual_gdp_next_year"].max(),
    best_model_pred_df["predicted_gdp_next_year"].max(),
)
plt.plot([diag_min, diag_max], [diag_min, diag_max], color="red", linestyle="--")
plt.title(f"Best Rebuilt Model: Actual vs Predicted GDP per Capita\n{best_model_name}")
plt.xlabel("Actual GDP per Capita (US$)")
plt.ylabel("Predicted GDP per Capita (US$)")
plt.tight_layout()
show_or_close_plot()

print("Best rebuilt model on test level metrics:", best_model_name)


# %%
output_files = {
    "metrics": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_metrics.csv",
    "coefficients": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_coefficients.csv",
    "train_predictions": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_train_predictions.csv",
    "test_predictions": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_test_predictions.csv",
    "region_metrics": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_region_metrics.csv",
    "yearly_summary": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_yearly_summary.csv",
    "decile_bias": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_decile_bias.csv",
    "bias_summary": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_bias_summary.csv",
    "specifications": OUTPUT_DIR / "gdp_main_models_rebuilt_with_lag_specifications.csv",
}

metrics_df.to_csv(output_files["metrics"], index=False)
coef_df.to_csv(output_files["coefficients"], index=False)
train_predictions_df.to_csv(output_files["train_predictions"], index=False)
test_predictions_df.to_csv(output_files["test_predictions"], index=False)
region_metrics_df.to_csv(output_files["region_metrics"], index=False)
yearly_summary_df.to_csv(output_files["yearly_summary"], index=False)
decile_bias_df.to_csv(output_files["decile_bias"], index=False)
bias_summary_df.to_csv(output_files["bias_summary"], index=False)
spec_df.to_csv(output_files["specifications"], index=False)

for label, file_path in output_files.items():
    print(f"{label}: {file_path}")
