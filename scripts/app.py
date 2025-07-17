# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from datetime import datetime

# Load the trained XGBoost model
model = joblib.load("../models/xgb_best_model.pkl")  # update path as needed

# Load store encodings and scaler if any
X_columns = joblib.load("../models/x_columns.pkl")  # list of feature columns used

st.set_page_config(page_title="Rossmann Sales Predictor", layout="centered")
st.title("📊 Rossmann Store Sales Predictor")

# User input UI
store_id = st.selectbox("Select Store ID", [1, 2, 5, 7, 10, 15, 20, 22, 30])
day_of_week = st.slider("Day of Week (1=Monday)", 1, 7, 5)
promo = st.selectbox("Is Promo Active?", [0, 1])
school_holiday = st.selectbox("School Holiday?", [0, 1])
state_holiday = st.selectbox("State Holiday (0=No, a/b/c=Yes)", ['0', 'a', 'b', 'c'])

# Date picker
date = st.date_input("Date of Prediction", datetime.today())
year = date.year
month = date.month
day = date.day

# Feature Engineering (dummy example)
input_data = pd.DataFrame([{
    'Store': store_id,
    'DayOfWeek': day_of_week,
    'Promo': promo,
    'SchoolHoliday': school_holiday,
    'StateHoliday': state_holiday,
    'Year': year,
    'Month': month,
    'Day': day
}])

# If you encoded StateHoliday as 0=0, a=1, b=2, c=3 during training:
state_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
input_data['StateHoliday'] = input_data['StateHoliday'].map(state_map)

# Align features
for col in X_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing dummy features if any

input_data = input_data[X_columns]  # reorder

# Predict
if st.button("Predict Sales"):
    log_pred = model.predict(input_data)
    prediction = np.expm1(log_pred)[0]
    st.success(f"💰 Predicted Sales: €{prediction:,.2f}")

