# Analysis Folder Guide

This folder is organized to make the project easier to review without changing the underlying analysis files.

Recommended reading order:

1. `01_EDA/`
2. `02_Data_Preparation/`
3. `03_Core_Models/`
4. `05_Forecasting/`
5. `04_Benchmarks/` for additional model comparisons
6. `99_Archive/` for supporting or older exploratory materials

Folder summary:

- `01_EDA/`: exploratory notebooks for GDP, population, life expectancy, and regional views
- `02_Data_Preparation/`: preprocessing, region mapping, event-feature construction, and pipeline figure scripts
- `03_Core_Models/`: main GDP model scripts and primary dynamic-model notebooks
- `04_Benchmarks/`: ridge, elastic net, random forest, XGBoost, and comparison notebooks
- `05_Forecasting/`: time-series forecasting and future forecast notebooks
- `99_Archive/`: demo or supporting artifacts retained for completeness

All original analysis files are preserved; this is only a structural cleanup for readability.
