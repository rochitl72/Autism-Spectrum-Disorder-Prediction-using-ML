# frontend/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load("../model/asd_model.joblib")
scaler = joblib.load("../model/scaler.joblib")
label_encoders = joblib.load("../model/label_encoders.joblib")

st.set_page_config(page_title="ASD Prediction App", layout="centered")
st.title("🧠 Autism Spectrum Disorder Prediction")

st.markdown("Fill the form below to check for possible ASD traits:")

# Define input fields (based on original dataset)
inputs = {
    'A1_Score': st.selectbox("A1_Score", [0, 1]),
    'A2_Score': st.selectbox("A2_Score", [0, 1]),
    'A3_Score': st.selectbox("A3_Score", [0, 1]),
    'A4_Score': st.selectbox("A4_Score", [0, 1]),
    'A5_Score': st.selectbox("A5_Score", [0, 1]),
    'A6_Score': st.selectbox("A6_Score", [0, 1]),
    'A7_Score': st.selectbox("A7_Score", [0, 1]),
    'A8_Score': st.selectbox("A8_Score", [0, 1]),
    'A9_Score': st.selectbox("A9_Score", [0, 1]),
    'A10_Score': st.selectbox("A10_Score", [0, 1]),
    'age': st.slider("Age", 3, 100, 25),
    'gender': st.selectbox("Gender", ["m", "f"]),
    'ethnicity': st.selectbox("Ethnicity", ["White-European", "Latino", "Others"]),
    'jundice': st.selectbox("Jaundice at birth?", ["yes", "no"]),
    'austim': st.selectbox("Diagnosed with autism?", ["yes", "no"]),
    'contry_of_res': st.selectbox("Country of Residence", ["United States", "India", "Others"]),
    'used_app_before': st.selectbox("Used App Before?", ["yes", "no"]),
}


# Convert input to DataFrame
input_df = pd.DataFrame([inputs])

# Label encode categorical fields
for column in input_df.columns:
    if column in label_encoders:
        le = label_encoders[column]
        input_df[column] = le.transform(input_df[column].astype(str))

# Scale numeric inputs
input_scaled = scaler.transform(input_df)

# Predict
if st.button("🔍 Predict ASD"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ Likely ASD Detected")
    else:
        st.success("✅ No ASD Traits Detected")

import datetime
import os

# Log prediction
log_data = input_df.copy()
log_data["prediction"] = prediction
log_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log_path = "../logs/prediction_log.csv"
if not os.path.exists(log_path):
    log_data.to_csv(log_path, index=False)
else:
    log_data.to_csv(log_path, mode='a', header=False, index=False)
