# Data Folder Guide

This folder is organized to keep the dashboard inputs easy to find while grouping secondary outputs into clearer subfolders.

## High-level layout

- `Raw/`
  - Original source datasets used across notebooks and preprocessing steps.
  - Kept mostly flat on purpose so older notebooks with direct `Data/Raw/...` paths remain easy to follow.

- `Cleaned/`
  - Root-level files here are the main cleaned inputs still used directly by the dashboard and core analysis workflow.
  - Additional benchmark, forecast, comparison, legacy, and robustness outputs are grouped into subfolders.

## Cleaned folder meaning

- Root of `Cleaned/`
  - Core panel files
  - region-mapped cleaned files
  - rebuilt main-model files that the dashboard reads directly
  - time-series summary and best-model files that the dashboard reads directly
  - benchmark metrics still referenced by the dashboard

- `02_Time_Series_Details/`
  - large time-series prediction exports

- `03_Main_Model_Details/`
  - detailed final-model summary tables and rebuilt-model detail outputs

- `04_Benchmark_Models/`
  - algorithm-specific outputs for Elastic Net, Ridge, Random Forest, and XGBoost

- `05_Model_Comparisons/`
  - cross-model comparison tables and shared evaluation samples

- `06_Future_Forecasts/`
  - saved future forecast outputs and comparison tables

- `07_Legacy_Models/`
  - earlier model-generation outputs retained for completeness

- `08_Robustness_Checks/`
  - saved robustness-check outputs

No files were deleted during this cleanup.
