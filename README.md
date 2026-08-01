# Global GDP per Capita Forecasting and Economic Drivers

## Dashboard Purpose

This repository contains a Final Year Project focused on **GDP-first analysis and forecasting** using World Bank-style country-year data. The new Streamlit entrypoint (`app.py`) presents the project as a clean workflow dashboard:

1. Executive Overview
2. Data Workflow
3. Global GDP Trends
4. GDP Growth and Country Comparison
5. GDP Relationships
6. GDP Forecasting
7. Findings and Limitations

The dashboard interface title is:

**Global GDP Trends and Forecasting Dashboard**

while the formal academic project title remains:

**Global GDP per Capita Forecasting and Economic Drivers**

## GDP-First Analytical Direction

The repository now presents GDP as the central analytical variable. In this project:

- **GDP per capita (current US$)** is the actual downloaded GDP indicator.
- **Population** is a supporting variable used to explain economic scale and derive represented total GDP when country-year observations match.
- **Life expectancy** is a supporting variable used to contextualise development and enrich the GDP modelling story.

The dashboard does **not** treat population as the main topic.

## Data Sources

Core raw sources are stored in [`Data/Raw`](/Users/tonytony/Final Project/Data/Raw):

- [`gdp.csv`](/Users/tonytony/Final Project/Data/Raw/gdp.csv)
- [`population.csv`](/Users/tonytony/Final Project/Data/Raw/population.csv)
- [`life_expectancy.csv`](/Users/tonytony/Final Project/Data/Raw/life_expectancy.csv)
- [`world_bank_country_metadata.csv`](/Users/tonytony/Final Project/Data/Raw/world_bank_country_metadata.csv)

Additional macro and digital indicators used in the extended GDP models are also present in the same raw-data folder:

- [`inflation.csv`](/Users/tonytony/Final Project/Data/Raw/inflation.csv)
- [`unemployment.csv`](/Users/tonytony/Final Project/Data/Raw/unemployment.csv)
- [`individuals_using_ the_Internet.csv`](/Users/tonytony/Final Project/Data/Raw/individuals_using_%20the_Internet.csv)

The main cleaned analytical panel used by the dashboard is:

- [`panel_with_event_dummies_and_extra_drivers.csv`](/Users/tonytony/Final Project/Data/Cleaned/panel_with_event_dummies_and_extra_drivers.csv)

## Indicator Definitions

- **GDP indicator:** GDP per capita (current US$)
- **Indicator code:** `NY.GDP.PCAP.CD`
- **Population indicator:** total population
- **Life expectancy indicator:** life expectancy at birth, total (years)

Because the primary GDP variable is per capita rather than total GDP, any total-GDP view in the dashboard is clearly labelled as:

**Estimated GDP represented = GDP per capita x population**

and is only calculated when the GDP and population values refer to the same country-year observation and population is positive.

## Data-Cleaning Workflow

The dashboard follows the real preprocessing logic already present in the notebooks:

1. Read raw World Bank-style CSV files
2. Skip metadata rows
3. Remove unnamed columns
4. Convert wide year columns into long format with `pandas.melt`
5. Convert year and indicator values into numeric types
6. Validate country-year keys
7. Merge datasets across matched country-year observations
8. Add region information and extra engineered features
9. Save the cleaned analytical panel

## Dashboard Structure

The dashboard is implemented through:

- [`app.py`](/Users/tonytony/Final Project/app.py)
- [`src/dashboard_data.py`](/Users/tonytony/Final Project/src/dashboard_data.py)
- [`src/analytics.py`](/Users/tonytony/Final Project/src/analytics.py)
- [`src/charts.py`](/Users/tonytony/Final Project/src/charts.py)
- [`src/forecasting_utils.py`](/Users/tonytony/Final Project/src/forecasting_utils.py)
- [`src/components.py`](/Users/tonytony/Final Project/src/components.py)
- [`src/formatting.py`](/Users/tonytony/Final Project/src/formatting.py)
- [`views/overview.py`](/Users/tonytony/Final Project/views/overview.py)
- [`views/data_workflow.py`](/Users/tonytony/Final Project/views/data_workflow.py)
- [`views/gdp_trends.py`](/Users/tonytony/Final Project/views/gdp_trends.py)
- [`views/country_comparison.py`](/Users/tonytony/Final Project/views/country_comparison.py)
- [`views/relationships.py`](/Users/tonytony/Final Project/views/relationships.py)
- [`views/forecasting.py`](/Users/tonytony/Final Project/views/forecasting.py)
- [`views/conclusions.py`](/Users/tonytony/Final Project/views/conclusions.py)

## Forecasting Methodology

The repository contains two forecasting layers:

### 1. Indicator-level time-series forecasting

Used to forecast future annual paths for GDP, population, and life expectancy by country.

- GDP best benchmark: Naive
- Life expectancy best benchmark: Naive
- Population best benchmark: LogHolt
- Evaluation approach: rolling 10-year backtesting

### 2. Main GDP predictive models

These are the academically central GDP models stored in the cleaned outputs:

- Model 1: Baseline Dynamic
- Model 2: Extended Dynamic
- Model 3: Full Dynamic

The rebuilt specifications are stored in:

- [`gdp_main_models_rebuilt_with_lag_specifications.csv`](/Users/tonytony/Final Project/Data/Cleaned/gdp_main_models_rebuilt_with_lag_specifications.csv)

and the holdout metrics are stored in:

- [`gdp_main_models_rebuilt_with_lag_metrics.csv`](/Users/tonytony/Final Project/Data/Cleaned/gdp_main_models_rebuilt_with_lag_metrics.csv)

The dashboard reads these real saved outputs rather than fabricating metrics or confidence intervals.

## Installation

Create or activate the local virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## Run Command

Launch the dashboard with:

```bash
streamlit run app.py
```

## Folder Structure

```text
Final Project/
├── app.py
├── .streamlit/
├── src/
├── views/
├── Data/
│   ├── Raw/
│   └── Cleaned/
├── Analysis/
└── output/
```

## Limitations

- The core GDP indicator is **current-price GDP per capita**, so it is affected by inflation and exchange-rate movement.
- The full cleaned panel retains more rows than the stricter main-model sample, which becomes smaller after listwise deletion.
- Correlation outputs show association only and do not establish causation.
- Forecast uncertainty rises with longer horizons.
- The dashboard does not display confidence bands because the repository does not provide saved confidence-interval outputs for the current workflow.

## Optional Benchmark Note

Some notebooks in [`Analysis`](/Users/tonytony/Final Project/Analysis) include Random Forest and XGBoost benchmark work. The dashboard keeps those as secondary repository outputs and preserves the dynamic pooled OLS models as the main academic modelling layer.
