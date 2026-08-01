# Report Reference Bank (Harvard Style)

This file is the canonical citation and reference bank for the final report:
`Global GDP per Capita Forecasting and Economic Drivers`.

Use this file as the single source of truth for:
- in-text citations;
- the final Harvard-style reference list;
- chapter-level citation mapping;
- consistency checks before final submission.

Last updated: 28 July 2026

## 1. Citation Rules for This Report

Use citations for:
- theory and conceptual framing;
- dataset provenance and indicator definitions;
- forecasting and modelling methods;
- prior empirical findings from the literature;
- software or framework references when they are explicitly discussed.

Do not add external citations for:
- your own charts, tables, metrics, and model outputs;
- your own cleaning decisions, feature engineering, or dashboard layout choices, unless they directly follow a published method;
- straightforward descriptions of what your code does, when the point is procedural rather than theoretical.

Practical rule:
- if a sentence says what the literature, a dataset provider, or a method source claims, cite it;
- if a sentence says what your own analysis found, cite your own table or figure instead.

## 2. Chapter-by-Chapter Citation Map

## Chapter 1. Introduction

Recommended sources:
- Solow (1956) for long-run growth framing.
- World Bank (2026a) for the importance and coverage of WDI.
- Bloom, Canning and Sevilla (2004) or Bloom et al. (2024) for health-growth motivation.

Example uses:
- economic growth and development framing;
- why GDP per capita is a meaningful outcome variable;
- why demographic and development factors matter.

## Chapter 2. Literature Review

Recommended sources:
- Solow (1956)
- Bloom, Canning and Sevilla (2004)
- Bloom et al. (2024)
- Fischer (1993)
- Czernich et al. (2011)
- Hyndman and Athanasopoulos (2021)
- Hyndman and Khandakar (2008)
- Wooldridge (2010)
- Baltagi (2021)

Example uses:
- GDP per capita as a development-oriented target;
- time-series forecasting logic;
- dynamic regression and panel-style modelling;
- inflation, internet usage, and unemployment as macroeconomic drivers;
- interpretability versus predictive complexity.

## Chapter 3. Methodology

Recommended sources:
- World Bank (2026a)
- World Bank (n.d.-a) to World Bank (n.d.-f)
- Seabold and Perktold (2010)
- Streamlit Inc. (n.d.)
- Hyndman and Athanasopoulos (2021)
- Wooldridge (2010)

Example uses:
- data source descriptions;
- indicator definitions;
- model implementation environment;
- rolling backtest and forecasting workflow;
- dashboard deployment description.

## Chapter 4. Exploratory Analysis and Descriptive Findings

Recommended sources:
- World Bank (2026a)
- World Bank (n.d.-a) to World Bank (n.d.-f)

Example uses:
- indicator definition notes;
- interpretation of what each variable measures;
- caution notes around modelled unemployment or population estimates.

## Chapter 5. Time-Series Modelling Results

Recommended sources:
- Hyndman and Athanasopoulos (2021)
- Hyndman and Khandakar (2008)
- Seabold and Perktold (2010)

Example uses:
- why Naive is a valid baseline;
- why ARIMA/AutoReg/Holt are appropriate benchmark models;
- why rolling one-step-ahead backtesting is defensible.

## Chapter 6. Main GDP Model Benchmarking and Interpretation

Recommended sources:
- Wooldridge (2010)
- Baltagi (2021)
- Hoerl and Kennard (1970)
- Zou and Hastie (2005)
- Breiman (2001)
- Chen and Guestrin (2016)
- Bloom, Canning and Sevilla (2004)
- Bloom et al. (2024)
- Fischer (1993)
- Czernich et al. (2011)

Example uses:
- OLS and panel-style regression interpretation;
- Ridge and Elastic Net as regularised linear benchmarks;
- Random Forest and XGBoost as nonlinear ML benchmarks;
- life expectancy, inflation, internet, and structural shock interpretation.

## Chapter 7. Dashboard Implementation

Recommended sources:
- Streamlit Inc. (n.d.)
- World Bank (2026a)

Example uses:
- why Streamlit was chosen;
- reproducibility between report outputs and dashboard views;
- data app deployment for decision support.

## Chapter 8. Critical Evaluation, Limitations, and Future Work

Recommended sources:
- Arellano and Bond (1991)
- Baltagi (2021)
- Wooldridge (2010)
- Hyndman and Athanasopoulos (2021)

Example uses:
- why pooled dynamic regression has limits;
- why dynamic panel methods are a future improvement;
- why scenario-based forecasting and robustness checks matter.

## Chapter 9. Conclusion

Recommended sources:
- Usually very light citation only.
- If needed, cite only broad framing sources already used earlier, not a large new reference set.

## 3. Ready-to-Use In-Text Citation Patterns

You can reuse these patterns directly in the report.

- Economic growth framing:
  - `Long-run growth analysis is commonly rooted in the neoclassical growth framework (Solow, 1956).`

- Health and development:
  - `A substantial literature links health improvement and life expectancy to stronger economic performance (Bloom, Canning and Sevilla, 2004; Bloom et al., 2024).`

- Inflation and growth:
  - `Macroeconomic instability, especially inflation, has often been associated with weaker growth performance (Fischer, 1993).`

- Internet and growth:
  - `Digital connectivity can support economic growth through productivity and infrastructure effects (Czernich et al., 2011).`

- Time-series methods:
  - `Time-series forecasting methods such as Naive, ARIMA, and exponential smoothing remain standard benchmark tools in applied forecasting (Hyndman and Athanasopoulos, 2021).`

- Auto-ARIMA logic:
  - `Automatic ARIMA model selection is commonly motivated by the forecasting framework described by Hyndman and Khandakar (2008).`

- Econometric interpretation:
  - `OLS-based dynamic regression remains attractive where interpretability is a central project objective (Wooldridge, 2010).`

- Panel-data limitation / future work:
  - `Future work could extend the analysis toward more formal dynamic panel methods (Arellano and Bond, 1991; Baltagi, 2021).`

- Machine learning benchmark:
  - `Random Forest and XGBoost were included as predictive benchmarks rather than as the primary explanatory framework (Breiman, 2001; Chen and Guestrin, 2016).`

- Software implementation:
  - `The statistical models were implemented in Python using statsmodels (Seabold and Perktold, 2010), while the interactive dashboard was built with Streamlit (Streamlit Inc., n.d.).`

## 4. Canonical Harvard Reference List

Arellano, M. and Bond, S. (1991) 'Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations', The Review of Economic Studies, 58(2), pp. 277-297. Available at: https://doi.org/10.2307/2297968

Baltagi, B.H. (2021) Econometric Analysis of Panel Data. Cham: Springer. Available at: https://doi.org/10.1007/978-3-030-53953-5

Bloom, D.E., Canning, D. and Sevilla, J. (2004) 'The Effect of Health on Economic Growth: A Production Function Approach', World Development, 32(1), pp. 1-13. Available at: https://doi.org/10.1016/j.worlddev.2003.07.002

Bloom, D.E., Canning, D., Kotschy, R., Prettner, K. and Schuenemann, J. (2024) 'Health and economic growth: Reconciling the micro and macro evidence', World Development, 178, 106575. Available at: https://doi.org/10.1016/j.worlddev.2024.106575

Breiman, L. (2001) 'Random Forests', Machine Learning, 45(1), pp. 5-32. Available at: https://doi.org/10.1023/A:1010933404324

Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM, pp. 785-794. Available at: https://doi.org/10.1145/2939672.2939785

Czernich, N., Falck, O., Kretschmer, T. and Woessmann, L. (2011) 'Broadband Infrastructure and Economic Growth', The Economic Journal, 121(552), pp. 505-532. Available at: https://doi.org/10.1111/j.1468-0297.2011.02420.x

Fischer, S. (1993) 'The Role of Macroeconomic Factors in Growth', Journal of Monetary Economics, 32(3), pp. 485-512. Available at: https://doi.org/10.3386/w4565

Hoerl, A.E. and Kennard, R.W. (1970) 'Ridge Regression: Biased Estimation for Nonorthogonal Problems', Technometrics, 12(1), pp. 55-67. Available at: https://doi.org/10.1080/00401706.1970.10488634

Hyndman, R.J. and Athanasopoulos, G. (2021) Forecasting: Principles and Practice. 3rd edn. Melbourne: OTexts. Available at: https://otexts.com/fpp3/ (Accessed: 28 July 2026).

Hyndman, R.J. and Khandakar, Y. (2008) 'Automatic Time Series Forecasting: The forecast Package for R', Journal of Statistical Software, 27(3), pp. 1-22. Available at: https://doi.org/10.18637/jss.v027.i03

Okun, A.M. (1962) 'Potential GNP: Its Measurement and Significance', in Proceedings of the Business and Economic Statistics Section of the American Statistical Association. Washington, DC: American Statistical Association, pp. 98-104.

Seabold, S. and Perktold, J. (2010) 'Statsmodels: Econometric and statistical modeling with Python', in van der Walt, S. and Millman, J. (eds.) Proceedings of the 9th Python in Science Conference. Austin, TX, pp. 92-96. Available at: https://conference.scipy.org/proceedings/scipy2010/seabold.html (Accessed: 28 July 2026).

Solow, R.M. (1956) 'A Contribution to the Theory of Economic Growth', The Quarterly Journal of Economics, 70(1), pp. 65-94. Available at: https://doi.org/10.2307/1884513

Streamlit Inc. (n.d.) Streamlit Documentation. Available at: https://docs.streamlit.io/ (Accessed: 28 July 2026).

Wooldridge, J.M. (2010) Econometric Analysis of Cross Section and Panel Data. 2nd edn. Cambridge, MA: MIT Press. Available at: https://mitpress.mit.edu/9780262232586/econometric-analysis-of-cross-section-and-panel-data/ (Accessed: 28 July 2026).

World Bank (2026a) World Development Indicators. Available at: https://datacatalog.worldbank.org/search/dataset/0037712/world-development-indicators (Accessed: 28 July 2026).

World Bank (n.d.-a) GDP per capita (current US$). Available at: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD (Accessed: 28 July 2026).

World Bank (n.d.-b) Population, total. Available at: https://data.worldbank.org/indicator/SP.POP.TOTL (Accessed: 28 July 2026).

World Bank (n.d.-c) Life expectancy at birth, total (years). Available at: https://databank.worldbank.org/metadataglossary/world-development-indicators/series/SP.DYN.LE00.IN (Accessed: 28 July 2026).

World Bank (n.d.-d) Inflation, consumer prices (annual %). Available at: https://databank.worldbank.org/metadataglossary/world-development-indicators/series/FP.CPI.TOTL.ZG (Accessed: 28 July 2026).

World Bank (n.d.-e) Unemployment, total (% of total labor force) (modeled ILO estimate). Available at: https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS (Accessed: 28 July 2026).

World Bank (n.d.-f) Individuals using the Internet (% of population). Available at: https://data.worldbank.org/indicator/IT.NET.USER.ZS (Accessed: 28 July 2026).

Zou, H. and Hastie, T. (2005) 'Regularization and Variable Selection via the Elastic Net', Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2), pp. 301-320. Available at: https://doi.org/10.1111/j.1467-9868.2005.00503.x

## 5. High-Priority References If You Need to Cut the List Down

If the final report should use a shorter reference list, keep these first:
- World Bank (2026a)
- World Bank (n.d.-a) to World Bank (n.d.-f)
- Hyndman and Athanasopoulos (2021)
- Hyndman and Khandakar (2008)
- Seabold and Perktold (2010)
- Wooldridge (2010)
- Bloom, Canning and Sevilla (2004)
- Fischer (1993)
- Czernich et al. (2011)
- Hoerl and Kennard (1970)
- Zou and Hastie (2005)
- Breiman (2001)
- Chen and Guestrin (2016)

## 6. Notes for the Final Cleanup Pass

Before final submission, check the following:
- every literature-based claim in Chapters 1-3 and 5-8 has at least one citation;
- every dataset definition uses a World Bank source;
- every algorithm discussed in Chapter 6 has its own method citation;
- the final reference list is alphabetized exactly as in this file;
- `n.d.` website references include an access date;
- your own figures and tables are referenced by figure/table number, not by external citations.
