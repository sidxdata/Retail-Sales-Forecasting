# Retail Sales Forecasting using Time Series Analysis & Machine Learning

An end-to-end retail demand forecasting project that predicts weekly sales across **45 stores**, **81 departments**, and **420K+ records** using statistical time-series models and machine learning. The project combines business analytics, feature engineering, predictive modelling, and model evaluation to support inventory planning and demand forecasting.

---

## Project Highlights

- 📊 **420K+** weekly sales records
- 🏬 **45 Stores | 81 Departments**
- 📅 **3 Years** of historical sales data
- 🤖 Compared **6 forecasting models**
- 🏆 Best Model: **Random Forest (R² = 0.987)**
- 📈 End-to-end Time Series & Machine Learning pipeline
- 📦 Business use case: Retail Demand Forecasting & Inventory Planning

---

## Business Problem

Retail businesses must accurately forecast product demand to optimize inventory, workforce planning, and supply chain operations. This project predicts weekly sales across multiple stores and departments while accounting for seasonality, holiday effects, promotional markdowns, and store-specific characteristics to improve business decision-making.

---

## Repository Structure

```text
.
├── TimeSeriesAnalysis_2.ipynb      # Complete analysis notebook
├── clean_data.zip                  # Dataset (extract before running)
├── forecast_output.csv             # Final predictions
├── Retail_Sales_Forecasting_Report.pdf
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the Walmart Retail Sales dataset containing weekly sales information from **2010–2012**.

### Dataset Summary

| Attribute | Description |
|-----------|-------------|
| Stores | 45 |
| Departments | 81 |
| Store–Department Series | 3,323 |
| Total Records | 420K+ |
| Time Period | 2010–2012 |
| Target Variable | Weekly_Sales |

Additional variables include:

- Holiday Flag
- Store Type
- Store Size
- Temperature
- Fuel Price
- CPI
- Unemployment
- Markdown Features (MarkDown1–5)

---

## Methodology

The forecasting pipeline consists of the following stages:

### 1. Data Understanding

- Dataset exploration
- Missing value analysis
- Duplicate detection
- Panel data identification
- Data quality validation

### 2. Exploratory Data Analysis (EDA)

- Weekly sales trend analysis
- Seasonal decomposition
- Holiday impact analysis
- Store and department comparison
- Correlation analysis
- Distribution analysis

### 3. Feature Engineering

Engineered predictive features including:

- Calendar Features (Week, Month, Quarter)
- Lag Features (1, 2, 4, 13, 26, 52 weeks)
- Rolling Mean & Rolling Standard Deviation
- Holiday Features
- Store Characteristics
- Economic Indicators

### 4. Feature Selection

- Correlation analysis
- Multicollinearity assessment
- Feature importance analysis

The complete engineered feature set was retained since no significant multicollinearity affecting tree-based models was observed.

### 5. Forecasting Models

#### Statistical Models

- ARIMA (2,0,2)
- SARIMA (0,0,1)(0,1,1,52)

#### Machine Learning Models

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost (RandomizedSearchCV + TimeSeriesSplit)

### 6. Model Evaluation

Models were evaluated using chronological train-test splits to avoid data leakage.

Evaluation metrics:

- MAE
- RMSE
- MAPE
- WMAE
- R² Score

### 7. Forecast Generation

The final model was used to generate weekly sales forecasts for every Store–Department combination.

---

## Model Performance

### 🏆 Final Selected Model: **Random Forest**

| Model | MAE | MAE (% of Mean Sales) | RMSE | %RMSE | MAPE | R² |
|:------|----:|----------------------:|-----:|------:|-----:|---:|
| 🏆 **Random Forest** | **1,226.53** | **7.70%** | **2,496.83** | **15.68%** | 208.64% | **0.987** |
| XGBoost *(Tuned)* | 1,339.64 | 8.41% | 2,810.39 | 17.65% | 185.48% | 0.984 |
| Decision Tree | 1,489.07 | 9.35% | 3,179.38 | 19.97% | 78.33% | 0.979 |
| Linear Regression | 1,613.06 | 10.13% | 3,007.43 | 18.89% | 864.57% | 0.981 |
| SARIMA *(Statistical Baseline)* | — | — | — | **3.36%** | **2.67%** | — |

> **Evaluation Metrics**
>
> - **MAE:** Mean Absolute Error
> - **RMSE:** Root Mean Squared Error
> - **%RMSE:** RMSE expressed as a percentage of mean sales
> - **MAPE:** Mean Absolute Percentage Error
> - **R²:** Coefficient of Determination
>
> **Random Forest** achieved the lowest forecasting error and the highest predictive performance among all evaluated machine learning models, making it the final model selected for demand forecasting. SARIMA served as the statistical forecasting baseline for comparison.

---

## Key Business Insights

- Holiday periods generate the highest sales demand, particularly during Thanksgiving and the week preceding Christmas.
- **lag_52** (sales from the same week in the previous year) is the most influential forecasting feature.
- Store Size and Store Type contribute significantly more to forecasting accuracy than macroeconomic indicators.
- Promotional markdown effects are partially confounded with long-term sales trends and should be interpreted cautiously.
- Random Forest consistently outperformed all other statistical and machine learning models.

---

## Business Impact

The developed forecasting pipeline can help retailers:

- Improve demand planning
- Optimize inventory allocation
- Reduce stock-outs and excess inventory
- Support staffing and workforce planning
- Improve promotional planning
- Enable data-driven operational decision-making

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Time Series | Statsmodels |
| Visualization | Matplotlib, Seaborn |
| Development | Jupyter Notebook |

---

## Results

The project produces:

- Weekly sales forecasts
- Forecast performance metrics
- Feature importance analysis
- Business insights
- Forecast output file (`forecast_output.csv`)
- Executive report with recommendations

---

## Setup & Usage

Clone the repository:

```bash
git clone https://github.com/your-username/Retail-Sales-Forecasting.git
cd Retail-Sales-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract the dataset:

```text
clean_data.zip
```

Run the notebook:

```bash
jupyter notebook TimeSeriesAnalysis_2.ipynb
```

---

## Future Improvements

- Deploy the forecasting pipeline using Streamlit
- Build an interactive Power BI dashboard
- Compare with Prophet and LightGBM
- Explore deep learning models (LSTM)
- Automate periodic model retraining

---

## Report

The complete project report containing methodology, visualizations, model evaluation, business insights, and recommendations is available in:

**📄 Retail_Sales_Forecasting_Report.pdf**

---

## Author

**Siddharth Gupta**

B.Tech | Netaji Subhas University of Technology (NSUT)

- LinkedIn: *(https://www.linkedin.com/in/siddharth-gupta-16b829305/)*

---

⭐ If you found this project useful, consider giving the repository a star.
