# Report Evidence Map

This file maps the current project artifacts to the report chapters so the report can be written from evidence rather than memory.

## Core data files

- Extended panel dataset:
  [`Data/Cleaned/panel_with_event_dummies_and_extra_drivers.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/panel_with_event_dummies_and_extra_drivers.csv)
- GDP with region:
  [`Data/Cleaned/gdp_with_wb_region.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_with_wb_region.csv)
- Life expectancy with region:
  [`Data/Cleaned/life_with_wb_region.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/life_with_wb_region.csv)
- Population with region:
  [`Data/Cleaned/pop_with_wb_region.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/pop_with_wb_region.csv)

## Main preparation notebooks

- Event engineering and merged panel construction:
  [`Analysis/gdp_with_global_events.ipynb`](/Users/tonytony/Final%20Project/Analysis/gdp_with_global_events.ipynb)
- Main GDP forecasting notebook:
  [`Analysis/future_gdp_forecast_mainmodel.ipynb`](/Users/tonytony/Final%20Project/Analysis/future_gdp_forecast_mainmodel.ipynb)

## Separate time-series modelling

- GDP:
  [`Analysis/gdp_time_series_models.ipynb`](/Users/tonytony/Final%20Project/Analysis/gdp_time_series_models.ipynb)
- Population:
  [`Analysis/population_time_series_models.ipynb`](/Users/tonytony/Final%20Project/Analysis/population_time_series_models.ipynb)
- Life expectancy:
  [`Analysis/life_expectancy_time_series_models.ipynb`](/Users/tonytony/Final%20Project/Analysis/life_expectancy_time_series_models.ipynb)

## Time-series result tables

- GDP summary:
  [`Data/Cleaned/gdp_time_series_model_selection_summary_10y.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_time_series_model_selection_summary_10y.csv)
- GDP best:
  [`Data/Cleaned/gdp_time_series_best_model_10y.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_time_series_best_model_10y.csv)
- Life expectancy summary:
  [`Data/Cleaned/life_time_series_model_selection_summary_10y.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/life_time_series_model_selection_summary_10y.csv)
- Life expectancy best:
  [`Data/Cleaned/life_time_series_best_model_10y.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/life_time_series_best_model_10y.csv)
- Population summary:
  [`Data/Cleaned/population_time_series_model_selection_summary_10y.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/population_time_series_model_selection_summary_10y.csv)
- Population best:
  [`Data/Cleaned/population_time_series_best_model_10y.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/population_time_series_best_model_10y.csv)

## GDP baseline evidence

- Baseline script:
  [`Analysis/gdp_model_1_baseline.py`](/Users/tonytony/Final%20Project/Analysis/gdp_model_1_baseline.py)
- Baseline metrics:
  [`Data/Cleaned/gdp_model_1_baseline_metrics.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_model_1_baseline_metrics.csv)
- Baseline coefficients:
  [`Data/Cleaned/gdp_model_1_baseline_coefficients.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_model_1_baseline_coefficients.csv)
- Baseline region metrics:
  [`Data/Cleaned/gdp_model_1_baseline_region_metrics.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_model_1_baseline_region_metrics.csv)
- Baseline test predictions:
  [`Data/Cleaned/gdp_model_1_baseline_test_predictions.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_model_1_baseline_test_predictions.csv)

## GDP three-model evaluation evidence

- Shared evaluation script:
  [`Analysis/gdp_main_models_evaluation.py`](/Users/tonytony/Final%20Project/Analysis/gdp_main_models_evaluation.py)
- Shared train/test metrics:
  [`Data/Cleaned/gdp_main_models_train_test_metrics.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_main_models_train_test_metrics.csv)
- Shared coefficients:
  [`Data/Cleaned/gdp_main_models_coefficients.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_main_models_coefficients.csv)
- Shared test predictions:
  [`Data/Cleaned/gdp_main_models_test_predictions.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_main_models_test_predictions.csv)
- Shared region metrics:
  [`Data/Cleaned/gdp_main_models_region_metrics.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_main_models_region_metrics.csv)

## Dashboard implementation

- Streamlit app:
  [`app.py`](/Users/tonytony/Final%20Project/app.py)
- Trial presentation:
  [`trial_demo_presentation.html`](/Users/tonytony/Final%20Project/trial_demo_presentation.html)

## Suggested report mapping

- Chapter 1 Introduction:
  Use proposal scope, aim, objectives, and GDP-per-capita motivation.
- Chapter 2 Literature Review:
  Use forecasting references plus justification for interpretability and event dummies.
- Chapter 3 Methodology:
  Use event-engineering notebook, panel construction, formulas, and evaluation logic.
- Chapter 4 EDA and descriptive findings:
  Use indicator notebooks, regional summaries, and dashboard trend plots.
- Chapter 5 Time-series modelling:
  Use the three 10-year summary CSV files and best-model tables.
- Chapter 6 Main GDP models:
  Use formulas and summary table from the main GDP notebook, plus baseline metrics CSV and the shared three-model train/test evaluation outputs.
- Chapter 7 Dashboard implementation:
  Use `app.py` tabs and features.
- Chapter 8 Evaluation and discussion:
  Compare interpretability, data coverage, missing-data trade-offs, and forecast behaviour.
- Chapter 9 Conclusion:
  Summarise the contribution and future improvements.
