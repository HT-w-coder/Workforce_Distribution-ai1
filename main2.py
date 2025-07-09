import streamlit as st
import pandas as pd
import joblib

# Load pipeline model
model = joblib.load("model.pkl")

st.title("📊 Workforce Distribution AI")
st.write("Predict whether an employee will leave or stay based on numeric features.")

# Inputs
JoiningYear = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
PaymentTier = st.selectbox("Payment Tier", [1, 2, 3])
Age = st.slider("Age", 20, 60, 30)
ExperienceInCurrentDomain = st.slider("Experience in Current Domain", 0, 10, 2)

if st.button("Predict"):
    try:
        input_data = pd.DataFrame([{
            "JoiningYear": JoiningYear,
            "PaymentTier": PaymentTier,
            "Age": Age,
            "ExperienceInCurrentDomain": ExperienceInCurrentDomain
        }])
        
        prediction = model.predict(input_data)[0]
        result = "❌ Will Leave" if prediction == 1 else "✅ Will Stay"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
