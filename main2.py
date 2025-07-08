import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline (model + encoder inside)
model = joblib.load("model.pkl")

st.set_page_config(page_title="Workforce Distribution AI")
st.title("🌟 Workforce Distribution AI")
st.subheader("Predict if an employee will leave or stay.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.slider("Age", min_value=18, max_value=60, value=30)
experience = st.slider("Experience in Current Domain", min_value=0, max_value=20, value=5)

# Convert input into DataFrame
input_data = pd.DataFrame({
    "Gender": [gender],
    "EverBenched": [ever_benched],
    "City": [city],
    "Education": [education],
    "JoiningYear": [joining_year],
    "PaymentTier": [payment_tier],
    "Age": [age],
    "ExperienceInCurrentDomain": [experience]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Will Leave" if prediction == 1 else "Will Stay"
    st.success(f"Prediction: {result}")
