import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline (includes preprocessing)
model = joblib.load("model.pkl")

st.title("🌟 Workforce Distribution AI")
st.subheader("Predict if an employee will leave or stay.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.slider("Joining Year", 2000, 2022, 2015)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.slider("Age", 20, 60, 30)
experience = st.slider("Experience in Current Domain", 0, 10, 2)

# Create input DataFrame
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

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        result = "Leave" if prediction == 1 else "Stay"
        st.success(f"Prediction: The employee will likely **{result}**.")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
