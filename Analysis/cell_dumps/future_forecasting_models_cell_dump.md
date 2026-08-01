# NOTEBOOK DUMP: future_forecasting_models.ipynb

## CELL 1 [CODE]

```python
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

GDP_PATH = "/Users/tonytony/Final Project/Data/Raw/gdp.csv"
LIFE_PATH = "/Users/tonytony/Final Project/Data/Raw/life_expectancy.csv"
POP_PATH = "/Users/tonytony/Final Project/Data/Raw/population.csv"
META_PATH = "/Users/tonytony/Final Project/Data/Raw/world_bank_country_metadata.csv"
```

--------------------------------------------------------------------------------

## CELL 2 [CODE]

```python
meta_df = pd.read_csv(META_PATH)

region_map = meta_df[[
    "Country Code",
    "World Bank Region"
]].copy().rename(columns={
    "Country Code": "country_code",
    "World Bank Region": "wb_region"
})

region_map["country_type"] = np.where(
    region_map["wb_region"].astype(str).str.strip().eq("Aggregates"),
    "Aggregate",
    "Country/Territory"
)

def load_wdi(path):
    df = pd.read_csv(path, skiprows=4)
    df = df.drop(
        columns=[col for col in df.columns if str(col).startswith("Unnamed")],
        errors="ignore"
    )
    year_cols = [col for col in df.columns if str(col).isdigit()]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")
    return df

def to_long(df, value_name):
    long_df = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        value_vars=[col for col in df.columns if str(col).isdigit()],
        var_name="year",
        value_name=value_name
    ).rename(columns={
        "Country Name": "country_name",
        "Country Code": "country_code",
        "Indicator Name": "indicator_name",
        "Indicator Code": "indicator_code"
    })

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("int64")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    return long_df

def add_region_info(long_df):
    merged = long_df.merge(region_map, on="country_code", how="left")
    merged = merged[merged["country_type"] == "Country/Territory"].copy()
    return merged

print("Helper functions are ready")
```

--------------------------------------------------------------------------------

## CELL 3 [CODE]

```python
gdp_long = add_region_info(to_long(load_wdi(GDP_PATH), "gdp_per_capita_usd"))
life_long = add_region_info(to_long(load_wdi(LIFE_PATH), "life_expectancy_years"))
pop_long = add_region_info(to_long(load_wdi(POP_PATH), "population_total"))

gdp_long = gdp_long[(gdp_long["gdp_per_capita_usd"].notna()) & (gdp_long["gdp_per_capita_usd"] > 0)].copy()
life_long = life_long[life_long["life_expectancy_years"].notna()].copy()
pop_long = pop_long[(pop_long["population_total"].notna()) & (pop_long["population_total"] > 0)].copy()

print("GDP latest year:", int(gdp_long["year"].max()))
print("Life latest year:", int(life_long["year"].max()))
print("Population latest year:", int(pop_long["year"].max()))
```

--------------------------------------------------------------------------------

## CELL 4 [CODE]

```python
forecast_plan = pd.DataFrame({
    "Dataset": ["GDP", "Life Expectancy", "Population"],
    "Baseline Model": ["Naive", "Naive", "Naive"],
    "Main Model": ["ARIMA(1,1,1)", "Holt Linear Trend", "Holt Trend on log(population)"],
    "Optional Model": ["AutoReg(lags=3)", "AutoReg(lags=3)", "AutoReg(lags=3)"],
    "Why": [
        "GDP is more volatile and shock-sensitive, so ARIMA is the main forecasting model.",
        "Life expectancy follows a relatively smooth long-run upward trend, so Holt trend is appropriate.",
        "Population is highly persistent and smooth, so trend-based forecasting on log scale is appropriate."
    ]
})

forecast_plan
```

--------------------------------------------------------------------------------

## CELL 5 [CODE]

```python
forecast_horizon = 5

print("Future forecast horizon:", forecast_horizon, "years")
```

--------------------------------------------------------------------------------

## CELL 6 [CODE]

```python
def forecast_country_series(series, model_name, forecast_horizon, lags=3):
    series = pd.Series(series).dropna().astype(float)

    if model_name == "Naive":
        last_value = series.iloc[-1]
        return np.repeat(last_value, forecast_horizon)

    elif model_name == "AutoReg":
        fit = AutoReg(series, lags=lags, old_names=False, trend="c").fit()
        fc = fit.predict(start=len(series), end=len(series) + forecast_horizon - 1)
        return np.asarray(fc, dtype=float)

    elif model_name == "ARIMA":
        fit = ARIMA(
            series,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        fc = fit.forecast(steps=forecast_horizon)
        return np.asarray(fc, dtype=float)

    elif model_name == "Holt":
        fit = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            damped_trend=False
        ).fit()
        fc = fit.forecast(forecast_horizon)
        return np.asarray(fc, dtype=float)

    elif model_name == "LogHolt":
        log_series = np.log(series)
        fit = ExponentialSmoothing(
            log_series,
            trend="add",
            seasonal=None,
            damped_trend=False
        ).fit()
        fc = fit.forecast(forecast_horizon)
        return np.exp(np.asarray(fc, dtype=float))

    else:
        raise ValueError(f"Unknown model: {model_name}")

def forecast_dataset(
    long_df,
    value_col,
    dataset_name,
    models_to_use,
    forecast_horizon,
    min_obs=20
):
    output_rows = []
    skipped_rows = []

    for country_code, group in long_df.groupby("country_code"):
        group = group.sort_values("year").copy()
        country_name = group["country_name"].iloc[0]
        wb_region = group["wb_region"].iloc[0]

        series = group[value_col].dropna().astype(float)

        if len(series) < min_obs:
            skipped_rows.append({
                "Dataset": dataset_name,
                "country_code": country_code,
                "country_name": country_name,
                "reason": "too_few_observations"
            })
            continue

        last_year = int(group["year"].max())
        future_years = list(range(last_year + 1, last_year + forecast_horizon + 1))

        for model_name in models_to_use:
            try:
                fc_values = forecast_country_series(
                    series=series,
                    model_name=model_name,
                    forecast_horizon=forecast_horizon
                )

                temp = pd.DataFrame({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "country_code": country_code,
                    "country_name": country_name,
                    "wb_region": wb_region,
                    "forecast_year": future_years,
                    "predicted_value": fc_values
                })

                output_rows.append(temp)

            except Exception as e:
                skipped_rows.append({
                    "Dataset": dataset_name,
                    "country_code": country_code,
                    "country_name": country_name,
                    "reason": f"{model_name}: {str(e)}"
                })

    forecast_df = pd.concat(output_rows, ignore_index=True) if output_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows)

    return forecast_df, skipped_df

print("Forecast functions are ready")
```

--------------------------------------------------------------------------------

## CELL 7 [CODE]

```python
gdp_future_forecast, gdp_future_skipped = forecast_dataset(
    long_df=gdp_long,
    value_col="gdp_per_capita_usd",
    dataset_name="GDP",
    models_to_use=["Naive", "AutoReg", "ARIMA"],
    forecast_horizon=forecast_horizon,
    min_obs=20
)

print("GDP future forecast shape:", gdp_future_forecast.shape)
print("GDP skipped rows:", gdp_future_skipped.shape[0])

gdp_future_forecast.head(15)
```

--------------------------------------------------------------------------------

## CELL 8 [CODE]

```python
life_future_forecast, life_future_skipped = forecast_dataset(
    long_df=life_long,
    value_col="life_expectancy_years",
    dataset_name="Life Expectancy",
    models_to_use=["Naive", "AutoReg", "Holt"],
    forecast_horizon=forecast_horizon,
    min_obs=20
)

print("Life future forecast shape:", life_future_forecast.shape)
print("Life skipped rows:", life_future_skipped.shape[0])

life_future_forecast.head(15)
```

--------------------------------------------------------------------------------

## CELL 9 [CODE]

```python
pop_future_forecast, pop_future_skipped = forecast_dataset(
    long_df=pop_long,
    value_col="population_total",
    dataset_name="Population",
    models_to_use=["Naive", "AutoReg", "LogHolt"],
    forecast_horizon=forecast_horizon,
    min_obs=20
)

print("Population future forecast shape:", pop_future_forecast.shape)
print("Population skipped rows:", pop_future_skipped.shape[0])

pop_future_forecast.head(15)
```

--------------------------------------------------------------------------------

## CELL 10 [CODE]

```python
all_future_forecasts = pd.concat([
    gdp_future_forecast,
    life_future_forecast,
    pop_future_forecast
], ignore_index=True)

all_future_forecasts.head(20)
```

--------------------------------------------------------------------------------

## CELL 11 [CODE]

```python
def plot_future_forecast_examples(long_df, forecast_df, value_col, dataset_name, model_to_show, top_n=3):
    sample_countries = (
        long_df.groupby(["country_code", "country_name"])[value_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    for _, row in sample_countries.iterrows():
        code = row["country_code"]
        name = row["country_name"]

        hist = long_df[long_df["country_code"] == code].sort_values("year")
        fc = forecast_df[
            (forecast_df["country_code"] == code) &
            (forecast_df["Model"] == model_to_show)
        ].sort_values("forecast_year")

        plt.figure(figsize=(10, 5))
        plt.plot(hist["year"], hist[value_col], marker="o", linewidth=2.5, label="Historical")
        plt.plot(fc["forecast_year"], fc["predicted_value"], marker="o", linewidth=2.5, label=f"{model_to_show} Forecast")

        plt.title(f"{dataset_name} Future Forecast - {name}", fontsize=13, fontweight="bold")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_future_forecast_examples(gdp_long, gdp_future_forecast, "gdp_per_capita_usd", "GDP", "ARIMA", top_n=3)
plot_future_forecast_examples(life_long, life_future_forecast, "life_expectancy_years", "Life Expectancy", "Holt", top_n=3)
plot_future_forecast_examples(pop_long, pop_future_forecast, "population_total", "Population", "LogHolt", top_n=3)
```

--------------------------------------------------------------------------------

## CELL 12 [CODE]

```python

```

--------------------------------------------------------------------------------
