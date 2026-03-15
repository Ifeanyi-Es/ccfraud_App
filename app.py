import os
import streamlit as st
import pandas as pd
import joblib
import shap 
import numpy as np
from datetime import date, datetime, time
from src.mlp_model import build_mlp , mlp_clf , SafeKerasClassifier # needed for MLP unpickling
from src.feature_engineering_utils import DateTimeTransformer, DynamicNumericTransformer
from src.model_insights import model_insights
# ML models / encoders for unpickling
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
import category_encoders as ce
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# load model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "Hybrid_Stack_on_SMOTE.pkl")

model = load_model(model_path)


# page configuration
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Transaction")

# input form
with st.form("transaction_form"):

    st.subheader("Transaction Details")

    # Categorical features with default values from known fraud sample
    merchant = st.text_input("Merchant", value="Heathcote, Yost and Kertzmann")  # default fraud sample

    category = st.text_input("Category", value="shopping_net")  # default fraud sample
    gender = st.selectbox("Gender", ["M", "F"])
    city = st.text_input("City", value="Springville") # default fraud sample
    state = st.text_input("State", value="LA")  # default fraud sample
    job = st.text_input("Occupation", value="Librarian")  # default fraud sample

    # Continuous / numeric are represented with sliders
    amt = st.slider("Transaction Amount ($)", 0.0, 2000.0, 50.0, 0.01)
    lat = st.slider("Customer Latitude", 20.0, 50.0, 43.0, 0.0001)
    long = st.slider("Customer Longitude", -130.0, -70.0, -80.0, 0.0001)
    zip_code = st.text_input("ZIP Code (4-5 digits)", max_chars=5, value="14141")  # default fraud sample
    city_pop = st.slider("City Population", 100, 2_000_000, 500_000)
    merch_lat = st.slider("Merchant Latitude", 20.0, 50.0, 35.0, 0.0001)
    merch_long = st.slider("Merchant Longitude", -130.0, -60.0, -90.0, 0.0001)
    
    # input Date of birth
    dob = st.date_input(
        "Date of Birth",
        min_value=pd.to_datetime("1920-01-01"),
        max_value=pd.to_datetime("2020-12-31")
    )

    # input transaction date & time
    st.subheader("Transaction Date & Time")

    trans_date = st.date_input(
        "Transaction Date",
        value=date(2020, 6, 21)  # default sample
    )
    trans_time = st.time_input(
        "Transaction Time",
        value=time(22, 38)  # default sample
    )

    # Combine into single datetime
    trans_date_trans_time = datetime.combine(trans_date, trans_time)
    st.text(f"Transaction Date & Time (combined): {trans_date_trans_time.strftime('%d/%m/%Y %H:%M:%S')}")

        # Combine
    trans_date_trans_time = datetime.combine(trans_date, trans_time)
    unix_time = int(trans_date_trans_time.timestamp())

    st.info(f"Selected: {trans_date_trans_time.strftime('%d/%m/%Y %H:%M:%S')} (unix: {unix_time})")
    

    submit = st.form_submit_button("Predict")

# dataframe for prediction
if submit:

    input_df = pd.DataFrame([{ 
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "gender": gender,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "dob": pd.to_datetime(dob),
        "trans_date_trans_time": trans_date_trans_time,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long
    }])

    # Duplicate row for stacking
    input_df_dup = pd.concat([input_df, input_df], ignore_index=True)

    prediction_array = model.predict(input_df_dup)
    probability_array = model.predict_proba(input_df_dup)[:, 1]

    # Take only first row
    prediction = prediction_array[0]
    probability = probability_array[0]

    if prediction == 1:
        st.error("🚨 Fraud Detected") #display result
    else:
        st.success("✅ Legitimate Transaction") # display result

    st.write(f"Fraud Probability: {probability:.2%}")

    result = model_insights(model, input_df)   # Call the model_insights function

    st.write(result["feature_contributions"])
