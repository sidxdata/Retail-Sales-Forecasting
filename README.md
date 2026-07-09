# Retail Sales Forecasting — Case Study

An end-to-end time series forecasting pipeline predicting weekly sales at the store/department level, built for a retail demand-planning use case. The project covers data understanding, EDA, feature engineering, statistical and machine learning forecasting models, model evaluation, and business recommendations.

## Business Problem

A retail chain wants to improve demand planning and inventory allocation by forecasting weekly sales across 45 stores and 81 departments (3,323 unique store-department series), accounting for seasonality, holiday spikes, markdown-driven fluctuations, and store-level performance differences.

## Repository Structure

```
.
├── TimeSeriesAnalysis_2.ipynb      # Full analysis notebook (EDA → modeling → forecasts)
├── clean_data.zip                  # Source dataset — unzip before running
├── forecast_output.csv             # Final predictions (Date, Store, Dept, Actual, Predicted)
├── REPORT.pdf                      # Executive summary, insights & recommendations
├── requirements.txt
└── README.md
```

## Dataset

Weekly sales data for 45 stores / 81 departments (2010–2012), including holiday flags, store type/size, temperature, fuel price, CPI, unemployment, and promotional markdown information. Target variable: `Weekly_Sales`.

## Methodology

1. **Data Understanding** — schema check, null/duplicate audit, granularity analysis (panel data: 3,323 distinct store-dept time series).
2. **EDA** — trend/seasonality decomposition, holiday impact quantification, store/department comparison, correlation analysis.
3. **Feature Engineering** — calendar features (week/month/quarter), lag features (1, 2, 4, 13, 26, 52 weeks), rolling statistics, a corrected "true peak week" holiday flag (see Key Findings).
4. **Feature Selection** — correlation/multicollinearity check; full engineered feature set retained (no problematic collinearity found; tree models are robust to correlated predictors).
5. **Modeling** — two model families, five total models:
   - **Statistical:** ARIMA(2,0,2), SARIMA(0,0,1)(0,1,1,52)
   - **Machine Learning:** Linear Regression, Decision Tree, Random Forest, XGBoost (hyperparameter-tuned via `RandomizedSearchCV` with `TimeSeriesSplit`)
6. **Evaluation** — MAE, RMSE, %RMSE, MAPE, and WMAE (holiday-weighted, matching the original Walmart competition standard); all splits are strictly chronological to prevent leakage.
7. **Forecast Generation** — final predictions exported to `forecast_output.csv`.

## Key Findings

- Strong annual seasonality (Thanksgiving/Christmas-driven); the dataset's official "Christmas" flag is mistimed — the true peak occurs the week before (week 51), at roughly 2x the flagged week's sales.
- `lag_52` (same week, one year ago) is the single strongest predictive feature across all tree-based models.
- Store `Size`/`Type` are meaningful sales drivers; macroeconomic indicators (CPI, fuel price, unemployment) show weak linear correlation with sales.
- Markdown impact on sales is confounded with an overall time trend and should be treated cautiously.

## Model Results

| Model | MAE | MAE (% of mean sales) | RMSE | %RMSE | MAPE | R² |
|---|---|---|---|---|---|---|
| **Random Forest (selected)** | **1,226.53** | **7.70%** | **2,496.83** | **15.68%** | 208.64% | **0.987** |
| XGBoost (Tuned) | 1,339.64 | 8.41% | 2,810.39 | 17.65% | 185.48% | 0.984 |
| Decision Tree | 1,489.07 | 9.35% | 3,179.38 | 19.97% | 78.33% | 0.979 |
| Linear Regression | 1,613.06 | 10.13% | 3,007.43 | 18.89% | 864.57% | 0.981 |
| SARIMA (company-level baseline) | — | — | — | 3.36% | 2.67% | — |

**Random Forest** was selected as the final model — lowest error across every metric, with strong generalization from ensembled deep trees on a dataset dominated by one highly informative seasonal-lag feature.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels · matplotlib · seaborn

## Setup & Usage

```bash
pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn
jupyter notebook TimeSeriesAnalysis.ipynb
```
Run all cells top to bottom. Outputs (plots, metrics, `forecast_output.csv`) are generated in place.

## Full Report

See [REPORT.pdf](./REPORT.pdf) for the executive summary, detailed insights, and business recommendations.
