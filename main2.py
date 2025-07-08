import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# App UI
st.set_page_config(page_title="Workforce Distribution AI", layout="centered")
st.title("🌟 Workforce Distribution AI")
st.markdown("Predict employee retention")

# Inputs
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
    city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
    education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
    joining_year = st.slider("Joining Year", 2012, 2018)
    payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
    age = st.slider("Age", 20, 60)
    experience = st.slider("Experience", 0, 10)
    
    if st.form_submit_button("Predict"):
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
        
        try:
            prediction = model.predict(input_data)[0]
            st.subheader("Prediction")
            st.error("⚠️ Likely to leave") if prediction == 1 else st.success("✅ Likely to stay")
        except Exception as e:
            st.error(f"Error: {str(e)}")
