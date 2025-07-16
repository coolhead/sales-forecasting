# 📊 Sales Forecasting with XGBoost | Capstone Project

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python%203.10-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/github/last-commit/coolhead/sales-forecasting?style=flat-square" />
  <img src="https://img.shields.io/github/repo-size/coolhead/sales-forecasting?style=flat-square" />
  <img src="https://img.shields.io/github/workflow/status/coolhead/sales-forecasting/CI?label=Build&style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<br>

This project forecasts 6-week future sales using a custom ML pipeline built with **XGBoost**, along with detailed data preprocessing, time-aware train-test split, feature engineering, model evaluation, and future prediction.

---
## Table of Contents
- [Project Overview](#project-overview)
- [Feature Importance - XGBoost](#feature-importance---xgboost)
- [How to Run](#how-to-run)
- [.gitignore Guidelines](#make-sure-your-gitignore-includes)
- [Acknowledgements](#-acknowledgements)

---

## Project Structure
```
sales_forecasting/
├── data/ # Raw data (train.csv, store.csv)
├── notebooks/
│ ├── sales_forecasting_RaghavendraSiddappa.ipynb 
│ ├── xgboost_predictions.csv  # Evaluation results
│ ├── xgboost_future_forecast.csv # 6-week forecast
│ ├── final_xgboost_model.pkl # Trained model
├── scripts/
│ └── preprocessing.py # reusable preprocessing
├── config/
│ └── config.yaml # Reserved for future config
├── requirements.txt # Python packages
└── README.md # This file
```
---

## Problem Statement

To forecast daily **sales** for each Rossmann store using historical sales and store-related data, while ensuring:

- 💡 Realistic time-aware validation
- 🧮 Model performance with R², RMSE, and MAPE
- 📈 Future 6-week forecasting
- 🎯 Promo impact analysis

---

## Steps Performed

1. **Data Merging & Cleaning**
   - Merged `train.csv` and `store.csv`
   - Removed closed stores and zero-sales days
   - Handled categorical variables via one-hot encoding

2. **Feature Engineering**
   - Extracted `DayOfWeek`, `Promo`, `SchoolHoliday`, and store-specific features
   - Handled skewness using `log1p(Sales)`

3. **Time-aware Train/Test Split**
   - Used last **6 weeks** for testing (not random split)
   - Ensured no data leakage across time

4. **Model Training & Evaluation**
   - Trained models: `XGBoost`, `LightGBM`, `CatBoost`
   - Metrics used: `RMSE`, `R²`, `MAPE`

5. **Best Model**
   - ✅ **XGBoost** gave:
     - R² = **0.9079**
     - RMSE ≈ **926.88**
     - MAPE ≈ **8.89%**

6. **Promo Analysis**
   - Visualized sales distributions with and without `Promo`
   - Observed higher average sales during promotions

7. **Future Forecasting**
   - Predicted sales for the next **42 days** (6 weeks)
   - Saved results to `xgboost_future_forecast.csv`

---

## Sample Output

| Store | Date       | Predicted_Sales |
|-------|------------|-----------------|
| 1     | 2015-07-01 | 5567.23         |
| 1     | 2015-07-02 | 5478.14         |
| ...   | ...        | ...             |

---

## Visual Insights

### Sales Distribution (Skewness)
![Original Sales Distribution](assets/original_sales_distribution.jpg)

### Correlation Heatmap
![Top Sales Correlations](assets/top_sales_correlations_heatmap.jpg)

### Promo Impact on Sales
![Promo vs Sales](assets/promo_vs_sales_boxplot.jpg)

### Model Prediction Fit
![Prediction Scatter Plot](assets/prediction_scatter_plot.jpg)

### Feature Importance - XGBoost
![Feature Importance](assets/feature_importance_xgboost.jpg)

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/coolhead/sales-forecasting.git
   cd sales-forecasting
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate         # On Windows use: venv\Scripts\activate
    ```
3. **Install dependencies:**

    pip install -r requirements.txt
    
5. **Launch the Jupyter notebook:**
   ```bash
   jupyter lab   # if you're already inside the notebooks folder else run the below command
   jupyter notebook notebooks/sales_forecasting_RaghavendraSiddappa.ipynb

   ```
---

### Make sure your .gitignore includes:
```
__pycache__/
*.pkl
*.ipynb_checkpoints
.env
.venv
```
---

## 🙌 Acknowledgements

- This project is part of the **IIIT-B & UpGrad Executive PG Program in AI/ML (ML C65)**
- Dataset: [Rossmann Store Sales - Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/)
- Author: Raghavendra Siddappa

---

