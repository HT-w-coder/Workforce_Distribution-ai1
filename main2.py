import streamlit as st
import pandas as pd
import joblib

# Load model (entire pipeline)
model = joblib.load("model.pkl")

st.title("🌟 Workforce Distribution AI")
st.write("Predict if an employee will leave or stay.")

# User input
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.slider("Joining Year", 2012, 2018)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.slider("Age", 20, 60)
experience = st.slider("Experience in Current Domain", 0, 10)

# Prepare input data as DataFrame
input_data = pd.DataFrame([{
    "Gender": gender,
    "EverBenched": ever_benched,
    "City": city,
    "Education": education,
    "JoiningYear": joining_year,
    "PaymentTier": payment_tier,
    "Age": age,
    "ExperienceInCurrentDomain": experience
}])

# Predict using the full pipeline
prediction = model.predict(input_data)[0]

# Output
st.subheader("Prediction")
st.write("Employee will **stay** ✅" if prediction == 0 else "Employee will **leave** ❌")
