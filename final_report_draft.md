# Global GDP per Capita Forecasting and Economic Drivers

## A Data Science Approach Using Demographic and Macroeconomic Indicators

Author: Luong Duc Thinh  
Programme: Bachelor of Science with Honours in Computing  
Institution: University of Greenwich  

---

## Abstract

Forecasting economic development indicators remains an important problem in data science, economics, and public policy. Among alternative macroeconomic indicators, GDP per capita is particularly useful because it reflects average output relative to population size and provides a more interpretable proxy for living standards than total GDP alone. However, GDP per capita is influenced by multiple interacting demographic, social, and macroeconomic factors, including population dynamics, life expectancy, inflation, unemployment, digital access, and exposure to major global shocks. This project investigates whether these indicators can be combined within an interpretable forecasting framework to model future GDP per capita at the country level.

The study uses open World Bank country-level datasets covering GDP per capita, population, life expectancy, inflation, unemployment, internet usage, and country metadata. These datasets were cleaned, standardised, and merged into a country-year panel. The final extended panel dataset contains 11,373 observations, 25 variables, 214 countries, 7 World Bank regions, and the period 1960 to 2023. In addition to the merged panel, separate univariate time-series models were developed for GDP, population, and life expectancy to support indicator-level forecasting and model comparison.

Three main GDP models were implemented using `statsmodels` in an econometric framework. Model 1 uses population and life expectancy as baseline predictors. Model 2 extends the baseline by adding inflation, unemployment, and internet usage. Model 3 further incorporates global event dummies, regional structure, and a year trend for future-oriented forecasting. For the supporting time-series analysis, rolling 10-year backtesting was used to compare alternative models. The best-performing models were Naive for GDP, Naive for life expectancy, and LogHolt for population. For the main GDP models, the in-sample fit improved progressively from Model 1 (`R^2 = 0.6950`) to Model 2 (`R^2 = 0.7484`) and Model 3 (`R^2 = 0.7983`), indicating that broader macroeconomic and structural variables improved explanatory power.

The analytical outputs were integrated into a Streamlit dashboard that supports multi-indicator exploration, model comparison, rolling backtest inspection, and future forecast visualisation. Overall, the project demonstrates how interpretable data science methods can be used to combine exploratory analysis, time-series forecasting, panel-style regression, and interactive reporting in a single GDP-focused decision-support workflow.

Keywords: GDP per capita, forecasting, panel data, time series, ARIMA, AutoReg, Holt, regression, Streamlit, World Bank.

---

## Chapter 1. Introduction

### 1.1 Background

Economic performance varies substantially across countries and regions due to differences in demographic structure, health outcomes, labour productivity, macroeconomic stability, and technological development. GDP per capita is one of the most frequently used indicators in economic analysis because it offers a concise view of average output relative to population size. Compared with total GDP, it is often easier to interpret in relation to living standards, productivity, and long-term development outcomes.

At the same time, GDP per capita does not evolve in isolation. Population growth affects labour supply and dependency ratios. Life expectancy reflects broader health and human development conditions. Inflation and unemployment capture macroeconomic instability and labour market performance. Internet usage acts as a proxy for digital adoption and broader socio-economic modernisation. Major global events such as financial crises, pandemic shocks, or energy disruptions can also introduce structural breaks in country-level GDP trajectories.

From a data science perspective, this makes GDP forecasting a suitable final-year project problem because it combines multiple competencies: data acquisition, cleaning, exploratory analysis, time-series modelling, statistical reasoning, performance evaluation, and interactive visualisation. The project also offers a useful balance between interpretability and predictive modelling, which is important in academic work where both methodological transparency and analytical justification are required.

### 1.2 Problem Statement

Although GDP forecasting has been widely studied, many simple academic demonstrations either focus on a single variable time series or use black-box models without sufficient interpretability. In this project, the challenge is to construct a workflow that remains academically rigorous while still being understandable and defendable in a final-year report. More specifically, the project addresses three practical problems:

1. How can historical GDP per capita patterns be explored across countries and regions in a consistent and interpretable way?
2. To what extent do population, life expectancy, inflation, unemployment, internet usage, and global events help explain next-year GDP per capita?
3. How can analytical and forecasting outputs be deployed into a usable interface that supports comparison, interpretation, and demonstration?

The resulting report therefore needs to distinguish clearly between:

- indicator-level forecasting, where GDP, population, and life expectancy are each analysed as separate time series; and
- GDP prediction modelling, where demographic and macroeconomic indicators are used jointly to estimate future GDP per capita.

### 1.3 Aim

The aim of this project is to analyse global GDP per capita trends, investigate the relationship between GDP per capita and selected demographic and macroeconomic indicators, and develop interpretable forecasting models that predict future GDP per capita using open international data.

### 1.4 Objectives

The project objectives are:

1. To collect and prepare open country-level datasets for GDP per capita, population, life expectancy, inflation, unemployment, internet usage, and country metadata.
2. To conduct exploratory data analysis on GDP and its related indicators across countries, regions, and years.
3. To engineer a country-year panel dataset with additional global event dummy variables.
4. To build and compare separate time-series models for GDP, population, and life expectancy using rolling 10-year backtesting.
5. To construct three interpretable GDP models of increasing complexity, from a simple baseline to a fuller macroeconomic and structural specification.
6. To evaluate model behaviour using standard accuracy or fit metrics such as MAE, RMSE, MAPE, and `R^2`.
7. To present the results through an interactive Streamlit dashboard for exploration and demonstration.

### 1.5 Research Questions

This report is organised around the following research questions:

1. What long-term patterns are visible in GDP per capita, population, and life expectancy across the available countries and regions?
2. Which indicator-level time-series models perform best for GDP, population, and life expectancy when evaluated using a rolling 10-year window?
3. How much explanatory improvement is obtained by extending the GDP model from demographic variables only to a broader set of macroeconomic and structural variables?
4. How can global event dummies and region effects be incorporated into a practical GDP forecasting workflow?
5. How effectively can the resulting analysis and forecasts be communicated through a data dashboard?

### 1.6 Scope of the Study

The project focuses on country-level annual data derived from open World Bank style datasets. The study is limited to:

- GDP per capita as the primary economic target;
- annual country-level observations;
- interpretable forecasting and regression approaches implemented in Python; and
- dashboard deployment through Streamlit.

The project does not attempt to establish strict causal claims. Instead, it focuses on association, modelling usefulness, and forecasting-oriented interpretation. In addition, although machine learning methods such as Random Forest or XGBoost could be incorporated in future work, the present implementation prioritises transparency over black-box predictive performance.

### 1.7 Report Structure

The remainder of the report is organised as follows:

- Chapter 2 reviews relevant literature on GDP forecasting, time-series methods, regression-based modelling, and macroeconomic event effects.
- Chapter 3 explains the methodology, including data collection, preprocessing, feature engineering, modelling design, and evaluation strategy.
- Chapter 4 presents the implementation workflow and exploratory analysis.
- Chapter 5 reports the time-series modelling results.
- Chapter 6 reports the three main GDP models and compares their behaviour.
- Chapter 7 discusses the Streamlit dashboard implementation.
- Chapter 8 evaluates the project critically, summarises limitations, and outlines future improvements.
- Chapter 9 concludes the report.

---

## Chapter 2. Literature Review

### 2.1 GDP per Capita as a Forecasting Target

GDP forecasting is important in economics, public policy, and development analysis because it informs planning, budgeting, and performance assessment. Among candidate targets, GDP per capita is often more analytically useful than total GDP because it adjusts for population size and therefore gives a clearer indication of average economic output. In the context of this project, using GDP per capita also helps align the target variable with the broader socio-economic indicators used as predictors, especially life expectancy and internet usage.

The literature suggests that GDP per capita is associated with multiple underlying processes rather than a single direct driver. Population affects the size and composition of the economy, but GDP per capita also depends on productivity, institutional quality, macroeconomic stability, and broader development conditions. This justifies a multivariate modelling strategy rather than a purely univariate one.

### 2.2 Time-Series Forecasting in Economic and Demographic Data

Classical time-series forecasting approaches remain widely used because they are transparent, computationally feasible, and relatively easy to interpret. According to Hyndman and Athanasopoulos (2021), models such as Naive forecasting, Autoregressive models (AutoReg), ARIMA, and exponential smoothing families are well suited to series that exhibit persistence, long-run trend, or moderate non-stationary behaviour.

For this project, separate time-series forecasting was important for two reasons. First, it allowed GDP, population, and life expectancy to be analysed independently before combining them into broader GDP models. Second, it created a practical mechanism for forecasting future input variables when generating multi-year GDP scenarios. The literature supports this staged approach because macroeconomic and demographic variables often differ in smoothness, volatility, and temporal memory.

### 2.3 Regression and Panel-Style Modelling for GDP

Regression-based modelling remains a central approach in applied econometrics because it offers both predictive structure and coefficient-level interpretability. In GDP-related studies, demographic and socio-economic indicators are commonly used as explanatory variables, with log transformations often applied to reduce scale effects and improve model stability.

In this project, the main target variable was defined as next-year log GDP per capita:

`target_log_gdp_next_year`

This decision has three advantages. First, it aligns the target with forecasting logic by explicitly predicting the next year rather than the current year. Second, the log transformation reduces the impact of extreme GDP values across countries. Third, it allows coefficients to be interpreted in a more stable and academically defensible way.

The baseline model follows a standard multiple regression form:

`log GDP_(t+1) ~ log Population_t + Life Expectancy_t`

This is then extended with additional macroeconomic variables and event structure. Such progressive model expansion is consistent with the literature on interpretable model comparison, where a simple reference model is used first and then compared with richer specifications.

### 2.4 Macroeconomic Drivers and Global Event Dummies

Inflation and unemployment are widely used as macroeconomic indicators because they capture price instability and labour market weakness. Internet usage reflects broader technological adoption and digital access, which can be linked indirectly to productivity, connectivity, and inclusion. While none of these variables alone fully determines GDP, together they provide a richer proxy for structural economic conditions.

The literature also shows that major global events can create structural breaks that are difficult to capture with smooth trend models alone. For this reason, the project includes hand-engineered event dummies for:

- the Asian Financial Crisis (1997-1998);
- the Global Financial Crisis (2008-2009);
- the COVID shock (2020);
- the COVID rebound (2021); and
- the Ukraine and energy shock period (2022-2024).

These variables do not claim causality in a strict econometric sense, but they improve the ability of the model to represent historically unusual periods.

### 2.5 Interpretability Versus Predictive Complexity

Recent studies increasingly apply machine learning methods such as Random Forest, Gradient Boosting, or neural networks to macroeconomic forecasting. These approaches may improve predictive performance when relationships are highly non-linear, but they often reduce transparency and make academic interpretation more difficult. For an undergraduate final-year project, interpretability is particularly important because the student must justify design choices, explain features, and critically evaluate results.

For this reason, this project prioritises interpretable models as the core methodology. Statistical clarity, transparent preprocessing, clear formulas, and understandable model comparisons are treated as more important than maximising accuracy through black-box methods.

### 2.6 Research Gap and Project Positioning

The main gap addressed by this project is not the invention of a new forecasting algorithm, but the integration of several academic elements into one coherent workflow:

- separate time-series modelling for core indicators;
- progressive GDP model building from baseline to fuller specification;
- incorporation of global event dummies;
- explicit comparison of model fit or forecast quality; and
- deployment through an interactive Streamlit dashboard.

This positions the study as an applied, interpretable, and system-oriented data science project rather than a purely theoretical econometric exercise.

---

## Chapter 3. Methodology

### 3.1 Research Design

The project follows an applied quantitative design. It combines exploratory data analysis, time-series forecasting, regression-based GDP modelling, and dashboard deployment. The methodology was intentionally organised in stages so that each part supports the next:

1. collect and clean the datasets;
2. merge them into a country-year panel;
3. analyse the indicators descriptively;
4. evaluate separate time-series models;
5. build the three main GDP models; and
6. integrate the outputs into a Streamlit dashboard.

This sequential design supports both methodological transparency and reproducibility.

### 3.2 Data Sources

The raw datasets currently used in the project are:

- [`Data/Raw/gdp.csv`](/Users/tonytony/Final%20Project/Data/Raw/gdp.csv)
- [`Data/Raw/population.csv`](/Users/tonytony/Final%20Project/Data/Raw/population.csv)
- [`Data/Raw/life_expectancy.csv`](/Users/tonytony/Final%20Project/Data/Raw/life_expectancy.csv)
- [`Data/Raw/inflation.csv`](/Users/tonytony/Final%20Project/Data/Raw/inflation.csv)
- [`Data/Raw/unemployment.csv`](/Users/tonytony/Final%20Project/Data/Raw/unemployment.csv)
- [`Data/Raw/individuals_using_ the_Internet.csv`](/Users/tonytony/Final%20Project/Data/Raw/individuals_using_%20the_Internet.csv)
- [`Data/Raw/world_bank_country_metadata.csv`](/Users/tonytony/Final%20Project/Data/Raw/world_bank_country_metadata.csv)

These files were used to derive both individual indicator datasets and the merged panel used in the main GDP models.

### 3.3 Data Preparation and Panel Construction

The data preparation process was implemented mainly in:

- [`Analysis/gdp_with_global_events.ipynb`](/Users/tonytony/Final%20Project/Analysis/gdp_with_global_events.ipynb)
- [`Analysis/future_gdp_forecast_mainmodel.ipynb`](/Users/tonytony/Final%20Project/Analysis/future_gdp_forecast_mainmodel.ipynb)

The final merged dataset saved for modelling is:

- [`Data/Cleaned/panel_with_event_dummies_and_extra_drivers.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/panel_with_event_dummies_and_extra_drivers.csv)

This dataset contains:

- 11,373 rows;
- 25 columns;
- 214 countries;
- 7 World Bank regions; and
- the year range 1960 to 2023.

Key transformations include:

- converting wide-format indicator tables to long country-year format;
- merging indicators by `country_code` and `year`;
- attaching World Bank regional metadata;
- creating log variables such as `log_gdp_per_capita` and `log_population_total`;
- creating the one-step-ahead target `target_log_gdp_next_year`; and
- engineering event dummies and exposure variables for major global shocks.

The main GDP modelling subset is smaller than the full panel because not all macroeconomic variables are available for all countries and years. After listwise deletion for the variables required by the full model, the main-model sample contains 4,722 observations over the years 1991 to 2022.

### 3.4 Feature Engineering

The project includes both direct indicators and engineered variables. Important engineered features are:

- `target_log_gdp_next_year`
- `log_population_total`
- `asian_financial_crisis_9798`
- `global_financial_crisis_0809`
- `covid_shock_2020`
- `covid_rebound_2021`
- `ukraine_energy_shock_2022_2024`
- `asia_crisis_exposed_9798`
- `energy_shock_exposed_2022_2024`
- `inflation_pct_clean`
- `unemployment_pct_clean`
- `internet_users_pct_clean`

The cleaned macroeconomic variables have uneven coverage. Across the full extended panel, there are 8,653 non-null observations for inflation, 6,023 for unemployment, and 5,943 for internet usage. This has an important methodological consequence: richer GDP models have stronger explanatory scope but use fewer valid country-year observations.

### 3.5 Exploratory Analysis Strategy

Before modelling, the indicators were explored individually and comparatively. The analysis notebooks in the project include GDP, population, life expectancy, and regional views, while the final dashboard also supports direct interactive exploration. The EDA stage was used to:

- inspect long-term trends and missingness;
- compare country and region patterns;
- observe the scale gap between high-income and low-income countries;
- identify shock years such as 2008-2009 and 2020; and
- confirm that some variables are more suitable for smooth trend modelling than others.

### 3.6 Separate Time-Series Modelling

Three separate notebooks were used for time-series model evaluation:

- [`Analysis/gdp_time_series_models.ipynb`](/Users/tonytony/Final%20Project/Analysis/gdp_time_series_models.ipynb)
- [`Analysis/population_time_series_models.ipynb`](/Users/tonytony/Final%20Project/Analysis/population_time_series_models.ipynb)
- [`Analysis/life_expectancy_time_series_models.ipynb`](/Users/tonytony/Final%20Project/Analysis/life_expectancy_time_series_models.ipynb)

The rolling evaluation logic uses a 10-year window and performs one-step-ahead backtesting for each country where enough historical data exist. Candidate models were selected based on the characteristics of each dataset.

#### GDP Time-Series Candidates

- Naive
- ARIMA
- AutoReg

Best result:

- Naive
- MAE = 962.8993
- RMSE = 2472.1792
- MAPE = 15.3457%
- `R^2 = 0.9853`
- 9,437 predictions
- 212 countries modelled

#### Life Expectancy Time-Series Candidates

- Naive
- Holt
- AutoReg

Best result:

- Naive
- MAE = 0.5428
- RMSE = 1.4984
- MAPE = 0.9950%
- `R^2 = 0.9797`
- 11,684 predictions
- 217 countries modelled

#### Population Time-Series Candidates

- LogHolt
- Naive
- AutoReg

Best result:

- LogHolt
- MAE = 70,265.2708
- RMSE = 316,547.7427
- MAPE = 0.5801%
- `R^2 = 0.999992`
- 11,905 predictions
- 217 countries modelled

These results indicate that smooth and strongly persistent variables such as population benefit from trend-based exponential smoothing, while GDP and life expectancy often perform competitively even under simple persistence-based baselines.

### 3.7 Main GDP Modelling

The principal GDP modelling workflow was implemented in:

- [`Analysis/future_gdp_forecast_mainmodel.ipynb`](/Users/tonytony/Final%20Project/Analysis/future_gdp_forecast_mainmodel.ipynb)
- [`Analysis/gdp_model_1_baseline.py`](/Users/tonytony/Final%20Project/Analysis/gdp_model_1_baseline.py)

All three models use `statsmodels.formula.api.ols` and robust `HC3` covariance estimation.

#### Model 1: Baseline

`target_log_gdp_next_year ~ log_population_total + life_expectancy_years`

This is the simplest interpretable GDP model and tests whether demographic scale and health conditions alone can explain next-year GDP per capita.

#### Model 2: Extended

`target_log_gdp_next_year ~ log_population_total + life_expectancy_years + inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean`

This model extends the baseline by incorporating additional macroeconomic and development indicators.

#### Model 3: Full Forecasting Model

`target_log_gdp_next_year ~ log_population_total + life_expectancy_years + inflation_pct_clean + unemployment_pct_clean + internet_users_pct_clean + asian_financial_crisis_9798 + global_financial_crisis_0809 + covid_shock_2020 + covid_rebound_2021 + ukraine_energy_shock_2022_2024 + C(wb_region) + year_trend`

This specification is designed to be more realistic for future-oriented forecasting because it retains regional structure and event information while replacing strict year fixed effects with a numeric year trend.

### 3.8 GDP Model Comparison

The in-sample summary comparison extracted from the notebook is shown below.

| Model | R_squared | Adj_R_squared | AIC | BIC |
|---|---:|---:|---:|---:|
| Model 1 - Baseline | 0.6950 | 0.6948 | 11927.8938 | 11947.2738 |
| Model 2 - Extended | 0.7484 | 0.7481 | 11024.2695 | 11063.0294 |
| Model 3 - Full Forecasting Model | 0.7983 | 0.7976 | 10003.7829 | 10120.0627 |

The progression suggests that broader macroeconomic and structural information improves model fit substantially. However, the report must distinguish clearly between:

- explanatory in-sample fit; and
- true out-of-sample forecast performance.

At present, the strongest out-of-sample GDP evidence available in the workspace is the baseline Model 1 evaluation saved in:

- [`Data/Cleaned/gdp_model_1_baseline_metrics.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_model_1_baseline_metrics.csv)

For the test split, the baseline model achieved:

- level-scale MAE = 12,277.6584
- level-scale RMSE = 25,904.9196
- level-scale MAPE = 60.9438%
- log-scale `R^2 = 0.6699`

These results show that Model 1 is analytically useful as a baseline but not sufficient as a strong forecast model by itself.

### 3.9 Dashboard Implementation

The final dashboard is implemented in:

- [`app.py`](/Users/tonytony/Final%20Project/app.py)

It includes the following functional areas:

- Project Overview
- Time-Series Models
- Indicator Trends
- Comparison
- Forecast Explorer

The dashboard supports:

- country and region filtering;
- multi-indicator visualisation for GDP per capita, population, and life expectancy;
- model selection and backtest comparison;
- best-model summaries from rolling 10-year evaluation;
- live target-year forecasting; and
- saved output inspection for forecast tables.

This deployment stage is important academically because it demonstrates not only modelling, but also communication, reproducibility, and decision-support usability.

### 3.10 Evaluation Criteria

The evaluation strategy uses:

- MAE for average absolute forecast error;
- RMSE for error magnitude with stronger penalty on large misses;
- MAPE for relative percentage error;
- `R^2` for goodness of fit or explained variance, where appropriate;
- AIC and BIC for comparing in-sample econometric model parsimony; and
- rolling 10-year backtesting for the separate indicator time-series models.

This combination allows the project to compare models from both explanatory and predictive perspectives.

---

## Chapter 4. Exploratory Analysis and Descriptive Findings

### 4.1 Dataset Profile

The final extended panel used in this project contains 11,373 country-year observations, 214 countries, 7 World Bank regions, and the period 1960 to 2023. However, the modelling sample is not identical across all tasks. For the richer GDP models that require inflation, unemployment, and internet usage together with demographic variables, the valid sample falls to 4,722 observations, 177 countries, and the period 1991 to 2022 after listwise deletion.

This difference is important because it means that stronger model specification is achieved at the cost of reduced data coverage. From an academic perspective, this trade-off should be stated explicitly rather than hidden, because it affects how the model comparisons are interpreted.

### 4.2 Missingness and Data Availability

The core variables GDP per capita, population, and life expectancy are complete in the final merged panel. By contrast, the extra macroeconomic drivers are much less complete:

| Variable | Non-null observations | Missing share |
|---|---:|---:|
| `inflation_pct_clean` | 8,653 | 23.92% |
| `unemployment_pct_clean` | 6,023 | 47.04% |
| `internet_users_pct_clean` | 5,943 | 47.74% |

This pattern explains why the simpler models can use a much broader sample than the extended models. It also helps justify why multiple model tiers were retained instead of forcing a single complex model across all available years and countries.

### 4.3 Regional Differences in 2023

The 2023 regional averages show a strong global development gradient. Mean GDP per capita was highest in North America at approximately US$89,285.68, followed by Europe and Central Asia at US$42,351.41. At the other end, Sub-Saharan Africa had a mean GDP per capita of only US$2,596.16. Life expectancy followed a similar pattern, with North America averaging 80.78 years compared with 64.61 years in Sub-Saharan Africa.

| Region | Mean GDP per Capita 2023 (US$) | Mean Life Expectancy 2023 | Mean Population 2023 |
|---|---:|---:|---:|
| North America | 89,285.68 | 80.78 | 125,651,500 |
| Europe & Central Asia | 42,351.41 | 78.95 | 16,511,620 |
| Middle East, North Africa, Afghanistan & Pakistan | 18,824.53 | 75.94 | 34,991,240 |
| East Asia & Pacific | 18,339.33 | 73.79 | 70,739,120 |
| Latin America & Caribbean | 17,369.66 | 74.99 | 17,011,190 |
| South Asia | 4,446.88 | 74.75 | 277,096,800 |
| Sub-Saharan Africa | 2,596.16 | 64.61 | 27,063,830 |

These differences support the inclusion of region structure in the full GDP model. A single pooled relationship without regional controls would ignore large persistent differences in development level, economic structure, and historical trajectory.

### 4.4 Country-Level Extremes

At the country level, the 2023 data also show very wide dispersion. The highest recorded GDP per capita values in the panel were Monaco, Liechtenstein, Luxembourg, Bermuda, and Ireland. The lowest values were observed in Burundi, Afghanistan, the Central African Republic, Madagascar, and Somalia. This scale gap is one of the reasons that the project models `target_log_gdp_next_year` rather than the raw GDP level directly.

### 4.5 Cross-Indicator Relationships

The correlation matrix provides a first descriptive view of how the indicators move together across the extended panel:

| Pair | Correlation |
|---|---:|
| GDP per capita and internet usage | 0.583 |
| GDP per capita and life expectancy | 0.529 |
| Life expectancy and internet usage | 0.621 |
| GDP per capita and inflation | -0.160 |
| GDP per capita and unemployment | -0.133 |
| GDP per capita and population | -0.042 |

Several observations follow from these results. First, internet usage and life expectancy are both moderately positively associated with GDP per capita. Second, inflation and unemployment show mild negative associations, but not strong enough to justify causal interpretation on their own. Third, total population by itself has almost no simple linear relationship with GDP per capita in the pooled cross-country data, which is reasonable because large population does not imply high per-capita output.

### 4.6 Event-Dummy Coverage

The engineered event variables were mapped to the following years in the final panel:

- `asian_financial_crisis_9798`: 1997 and 1998
- `global_financial_crisis_0809`: 2008 and 2009
- `covid_shock_2020`: 2020
- `covid_rebound_2021`: 2021
- `ukraine_energy_shock_2022_2024`: 2022 and 2023 in the current dataset
- `high_global_rates_2023_2024`: 2023 in the current dataset

These dummies were not intended to prove causal impact in a strict econometric sense. Instead, they were designed to help the panel capture historically unusual periods that would otherwise be difficult to represent using only smooth trend variables.

---

## Chapter 5. Time-Series Modelling Results

### 5.1 Evaluation Design

The three separate time-series notebooks use a rolling 10-year window to forecast one year ahead for each country where sufficient history exists. This makes the evaluation more realistic than fitting a model once on the full history and testing it on the same observations. It also matches the intended workflow of using the most recent historical window to predict the next value of an indicator.

### 5.2 Summary of Candidate Models

| Dataset | Candidate Models | Best Model | MAE | RMSE | MAPE (%) | `R^2` | Predictions | Countries |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GDP | Naive, ARIMA, AutoReg | Naive | 962.8993 | 2472.1792 | 15.3457 | 0.9853 | 9,437 | 212 |
| Life Expectancy | Naive, Holt, AutoReg | Naive | 0.5428 | 1.4984 | 0.9950 | 0.9797 | 11,684 | 217 |
| Population | LogHolt, Naive, AutoReg | LogHolt | 70,265.2708 | 316,547.7427 | 0.5801 | 0.999992 | 11,905 | 217 |

### 5.3 GDP Time-Series Interpretation

For GDP per capita, the Naive model outperformed both ARIMA and AutoReg on the shared rolling evaluation. This suggests that in the current dataset, year-to-year persistence is a very strong predictor, and the more complex time-series structures do not consistently improve one-step-ahead accuracy across countries. ARIMA remained competitive but produced worse RMSE and substantially worse MAPE than Naive, while AutoReg performed much more poorly overall.

From an academic standpoint, this is a defensible result rather than a weak one. In macroeconomic annual data, especially with limited country-specific sample length, a persistence baseline often performs surprisingly well.

### 5.4 Life Expectancy Time-Series Interpretation

Life expectancy showed a similar pattern. The Naive model slightly outperformed Holt and substantially outperformed AutoReg. This is consistent with the fact that life expectancy tends to change gradually from one year to the next, often following a slow-moving development path rather than abrupt cyclical fluctuation. Holt remained a reasonable alternative, especially for smoother forward projection, but the backtesting evidence supports Naive as the best empirical choice in the current implementation.

### 5.5 Population Time-Series Interpretation

Population behaved differently from GDP and life expectancy. The best performer was LogHolt, which combines smoothing with growth in log scale and is well suited to long-run demographic trend data. It achieved an extremely low MAPE and near-perfect `R^2`, while the Naive and AutoReg alternatives were clearly weaker. This result is methodologically intuitive because population usually grows in a gradual compounding pattern rather than a volatile annual cycle.

### 5.6 Role of Time-Series Models in the Full Project

These separate models are not isolated side experiments. They serve two important functions in the overall project:

1. They provide independent evidence on how each core indicator behaves over time.
2. They provide a practical mechanism for generating future input variables when forecasting GDP to a user-selected target year.

This link between separate indicator forecasting and multivariate GDP forecasting is one of the main strengths of the project design.

---

## Chapter 6. Main GDP Model Results

### 6.1 Full-Sample Explanatory Comparison

The main notebook first compares the three GDP models using the full available modelling sample.

| Model | R_squared | Adj_R_squared | AIC | BIC |
|---|---:|---:|---:|---:|
| Model 1 - Baseline | 0.6950 | 0.6948 | 11927.8938 | 11947.2738 |
| Model 2 - Extended | 0.7484 | 0.7481 | 11024.2695 | 11063.0294 |
| Model 3 - Full Forecasting Model | 0.7983 | 0.7976 | 10003.7829 | 10120.0627 |

The pattern is consistent and meaningful. As the model expands from demographics only to a broader macroeconomic and structural specification, the fit improves and both AIC and BIC decrease. In econometric model comparison, this combination suggests that the added variables improve explanatory power while remaining justified after penalising specification complexity (Wooldridge, 2010; Baltagi, 2021).

### 6.2 Shared Train/Test Comparison

To make the comparison more rigorous, a second evaluation was run on a common sample and a common time-based split. This type of chronological holdout is methodologically appropriate for predictive assessment because it respects temporal ordering rather than mixing past and future observations during evaluation (Hyndman and Athanasopoulos, 2021). The split used:

- train feature years up to 2017; and
- test feature years from 2018 to 2022.

The shared evaluation script is:

- [`Analysis/gdp_main_models_evaluation.py`](/Users/tonytony/Final%20Project/Analysis/gdp_main_models_evaluation.py)

The resulting metrics are saved in:

- [`Data/Cleaned/gdp_main_models_train_test_metrics.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_main_models_train_test_metrics.csv)

The common modelling subset contains 4,722 observations, with 3,909 rows in train and 813 rows in test.

| Model | Test MAE (level) | Test RMSE (level) | Test MAPE (level %) | Test `R^2` (level) | Test MAE (log) | Test RMSE (log) | Test `R^2` (log) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model 1 - Baseline | 8,831.5736 | 17,383.9719 | 65.1570 | 0.4139 | 0.6169 | 0.7649 | 0.7077 |
| Model 2 - Extended | 7,902.4876 | 12,185.6670 | 106.1816 | 0.7120 | 0.6302 | 0.7924 | 0.6864 |
| Model 3 - Full | 6,271.8990 | 11,599.7431 | 59.7330 | 0.7390 | 0.4656 | 0.6168 | 0.8100 |

These results support three conclusions.

First, Model 1 is the weakest of the three in predictive terms, even though it remains useful as an interpretable benchmark, which is one of the main strengths of linear regression in applied economic modelling (Wooldridge, 2010). Second, Model 2 improves level-scale RMSE substantially relative to Model 1, showing that inflation, unemployment, and internet usage add useful predictive signal. This is also consistent with the broader macroeconomic literature, where inflation is linked to growth stability, labour-market conditions are closely connected to output performance, and digital connectivity can support productivity and development (Fischer, 1993; Okun, 1962; Czernich et al., 2011). Third, Model 3 is the strongest overall on the shared test split, with the lowest MAE and RMSE and the highest `R^2` in both level and log scale.

The unusually high test MAPE for Model 2 should be interpreted carefully. Percentage-based metrics can become unstable when actual GDP per capita values are very small, because even moderate absolute deviations can produce very large relative errors. For this reason, RMSE, MAE, and log-scale performance are more reliable than MAPE alone when judging cross-country GDP models with both high-income and low-income economies (Hyndman and Athanasopoulos, 2021).

### 6.3 Coefficient Interpretation

The coefficient tables were exported to:

- [`Data/Cleaned/gdp_main_models_coefficients.csv`](/Users/tonytony/Final%20Project/Data/Cleaned/gdp_main_models_coefficients.csv)

Several patterns are consistent across the models. These coefficients should be interpreted as conditional associations rather than direct causal effects, because pooled cross-country regressions may still reflect omitted structure, measurement differences, and cross-sectional heterogeneity even after additional controls are introduced (Wooldridge, 2010; Baltagi, 2021).

- `life_expectancy_years` remains positive and highly significant, suggesting that countries with stronger health and development conditions tend to have higher next-year GDP per capita. This is consistent with the health-and-growth literature, which treats life expectancy as a meaningful proxy for human capital and labour productivity conditions (Bloom, Canning and Sevilla, 2004; Bloom et al., 2024).
- `log_population_total` is negative once the target is GDP per capita rather than total GDP. This does not mean that population is economically harmful. Rather, it shows that larger population size alone does not imply higher output per person after controlling for the other included variables.
- `internet_users_pct_clean` is positive and strongly significant in Models 2 and 3, which is consistent with the descriptive evidence that digital access is associated with higher development level and with prior evidence linking digital infrastructure to stronger economic performance (Czernich et al., 2011).
- `inflation_pct_clean` is negative in Models 2 and 3, which aligns with the argument that macroeconomic instability is associated with weaker per-capita output performance (Fischer, 1993).
- `unemployment_pct_clean` is positive in the fitted models, which is counterintuitive if read causally. This reinforces the need to avoid causal claims. In a pooled multivariate setting, sign reversals can arise from omitted structure, multicollinearity, regional composition, or interactions with development stage (Wooldridge, 2010).

The regional terms in Model 3 are also informative. Some regions remain significantly above or below the reference group after controlling for the included variables, which supports the decision to preserve region effects in the fuller specification. This is in line with panel-data reasoning, where persistent unobserved group structure often needs to be represented explicitly rather than absorbed into a single pooled slope pattern (Baltagi, 2021).

### 6.4 Event Dummy Interpretation

The event terms require careful interpretation. In the full-sample notebook, the event dummies contribute to improved fit because they help distinguish crisis and shock periods from ordinary years. However, in the shared train/test evaluation with training ending in 2017, the variables for `covid_shock_2020`, `covid_rebound_2021`, and `ukraine_energy_shock_2022_2024` cannot be learned from the training sample because those events have not yet occurred in the training period. This reflects a standard forecasting problem: structural shocks that are outside the historical training window cannot be recovered as ordinary learned patterns unless they are encoded as explicit scenario assumptions (Hyndman and Athanasopoulos, 2021; Wooldridge, 2010).

This creates an important academic distinction:

- in the full-sample analysis, event dummies act as explanatory historical controls; but
- in strict out-of-sample forecasting, future event effects must be treated as scenario assumptions unless similar shocks already exist in the training data.

This is a strength rather than a flaw when it is explained clearly. It shows that the project understands the difference between retrospective modelling and genuine future uncertainty, which is especially important in macroeconomic panel settings affected by rare external shocks (Baltagi, 2021).

### 6.5 Regional Test Behaviour

Regional test performance also varies considerably. In the shared test evaluation, Model 3 performed especially well relative to the other models in South Asia, East Asia and Pacific, and the Middle East, North Africa, Afghanistan and Pakistan group. However, performance remained weaker in North America and parts of Sub-Saharan Africa, where either the regional sample is small or the economies are highly heterogeneous. This suggests that a single pooled model still cannot perfectly capture every regional structure, even after adding event and region effects, which is a familiar limitation in applied panel modelling when countries differ in structural regime, volatility, and development path (Baltagi, 2021; Wooldridge, 2010).

### 6.6 Overall Main-Model Conclusion

Taken together, the evidence suggests the following hierarchy:

1. Model 1 is appropriate as a simple baseline and for demonstrating interpretability.
2. Model 2 adds useful macroeconomic information and clearly improves level-scale prediction.
3. Model 3 is the best overall specification in the current project because it combines broader drivers, regional structure, and shock awareness.

This makes Model 3 the strongest candidate for the final report and demo, while Models 1 and 2 remain important for progressive comparison and justification. In other words, the final specification is strongest not simply because it is larger, but because it balances broader economic information, structural controls, and out-of-sample predictive performance in a way that remains interpretable for academic discussion (Wooldridge, 2010; Baltagi, 2021).

---

## Chapter 7. Dashboard Implementation

### 7.1 Purpose of the Dashboard

The Streamlit dashboard was developed to translate analytical outputs into an accessible decision-support tool. Its purpose is not only visual presentation, but also interactive inspection of model logic, country-level behaviour, and forecast paths. This is important for a final-year project because it demonstrates that the work goes beyond offline notebook analysis and can be presented to non-technical users.

### 7.2 Dashboard Structure

The dashboard is implemented in:

- [`app.py`](/Users/tonytony/Final%20Project/app.py)

It contains five main tabs:

1. `Project Overview`
2. `Time-Series Models`
3. `Indicator Trends`
4. `Comparison`
5. `Forecast Explorer`

The sidebar provides filters for region, country, year range, indicator, chart theme, and comparison year.

### 7.3 Functional Coverage

The dashboard currently supports:

- exploration of GDP per capita, population, and life expectancy from the merged panel;
- comparison of indicator behaviour across countries and regions;
- display of best time-series model summaries for GDP, life expectancy, and population;
- country-level rolling backtest inspection across alternative time-series models;
- future forecasting to a selected target year using the chosen indicator model logic; and
- integration of project outputs into a visual narrative suitable for trial demo presentation.

### 7.4 Academic Value of the Dashboard

The dashboard strengthens the project in several ways. First, it improves interpretability by letting the user move between descriptive data and model outputs. Second, it increases reproducibility because the same cleaned files and summary outputs used in the report are also used in the interface. Third, it adds practical value by showing how analytical work can be delivered in a usable format rather than remaining inside code cells only.

---

## Chapter 8. Critical Evaluation, Limitations, and Future Work

### 8.1 Strengths of the Project

The project has several clear strengths.

- It integrates data engineering, EDA, time-series modelling, regression modelling, and dashboard deployment in one coherent workflow.
- It distinguishes between separate indicator forecasting and multivariate GDP prediction instead of mixing them without explanation.
- It keeps interpretability at the centre of the modelling choices, which is especially appropriate for an academic final report.
- It incorporates global event dummies and region effects, which gives the modelling framework more economic realism than a purely mechanical regression.

### 8.2 Limitations

Despite these strengths, the project also has important limitations.

- Missingness in unemployment and internet usage substantially reduces the usable sample for the richer models.
- The main GDP models are pooled OLS specifications rather than full country fixed-effects or dynamic panel models.
- Event dummies are manually engineered from known global periods and therefore depend on researcher judgement.
- MAPE can be unstable for low-income countries with very small GDP per capita values.
- Some future shocks cannot be learned directly in a strict pre-shock training sample and must be handled as scenarios.

These limitations do not invalidate the project, but they do define the boundaries of what can be claimed.

### 8.3 Why More Complex Machine Learning Was Not the Core Choice

Methods such as Random Forest, XGBoost, Prophet, or LSTM could have been explored as advanced alternatives. However, the current project intentionally prioritises models that are easier to explain, defend, and connect to the economic meaning of the variables. For a final-year academic project, this is a reasonable methodological decision because strong interpretability and critical discussion are often more valuable than black-box accuracy alone.

### 8.4 Future Improvements

Several future extensions would strengthen the work further.

- Add country fixed effects or dynamic panel methods to better separate cross-country structure from within-country change.
- Evaluate ARIMAX or SARIMAX variants when exogenous predictors are introduced into the time-series stage.
- Compare the current interpretable models with tree-based machine learning benchmarks such as Random Forest or XGBoost.
- Develop formal scenario settings for future global shocks rather than treating all future years as normal continuation.
- Add uncertainty intervals to future forecasts so the dashboard reflects prediction uncertainty rather than point values only.
- Improve missing-data handling through principled imputation or restricted balanced panels.

### 8.5 Critical Reflection

The most important lesson from the project is that forecasting quality depends not only on model complexity, but also on variable definition, data availability, scale choice, and the difference between historical fit and genuine future prediction. In that sense, the project is academically valuable because it demonstrates not just modelling success, but methodological awareness.

---

## Chapter 9. Conclusion

This project developed an interpretable GDP per capita forecasting workflow using demographic, macroeconomic, and event-based information from open international datasets. The work combined three main strands: exploratory analysis of country-year indicators, separate rolling time-series modelling for GDP, population, and life expectancy, and progressive GDP regression models ranging from a simple baseline to a fuller forecasting-oriented specification.

The results show that separate indicator forecasting is feasible and informative, with Naive performing best for GDP and life expectancy, and LogHolt performing best for population under the current 10-year rolling backtest design. For the main GDP models, the evidence consistently shows that Model 3 provides the strongest overall performance when compared with Models 1 and 2, especially under a shared train/test evaluation. At the same time, the project also identifies important caveats around missingness, pooled modelling assumptions, and the treatment of future shock events.

Overall, the project meets its aim of building a transparent, academically defensible, and practically demonstrable GDP forecasting system. It also provides a strong foundation for further development through richer econometric structure, scenario analysis, and expanded model benchmarking.

---

## References for Current Draft

Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*. 3rd edn. Melbourne: OTexts.

Our World in Data (2024) *GDP per capita*. Available at: https://ourworldindata.org

Seabold, S. and Perktold, J. (2010) ‘Statsmodels: Econometric and statistical modeling with Python’, in *Proceedings of the 9th Python in Science Conference*. Austin, Texas, pp. 57-61.

Streamlit Inc. (2024) *Streamlit Documentation*. Available at: https://docs.streamlit.io

United Nations Department of Economic and Social Affairs (UN DESA) (2023) *World Population Prospects 2022*. Available at: https://population.un.org

World Bank (2024) *World Development Indicators*. Available at: https://data.worldbank.org
