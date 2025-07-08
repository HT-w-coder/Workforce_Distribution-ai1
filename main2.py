import streamlit as st
import pandas as pd
import joblib

# Load model (Pipeline: Preprocessing + Classifier)
model = joblib.load("model.pkl")

st.set_page_config(page_title="Workforce Distribution AI", layout="centered")
st.title("🌟 Workforce Distribution AI")
st.markdown("Predict whether an employee will **leave** or **stay**.")

# Collect inputs from user
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.slider("Joining Year", 2012, 2018)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.slider("Age", 20, 60)
experience = st.slider("Experience in Current Domain", 0, 10)

# Format into DataFrame
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

# Predict when button clicked
if st.button("Predict"):
    try:
        # Ensure input_data is a DataFrame and has the correct structure
        prediction = model.predict(input_data)  # This should work if the model is a Pipeline
        st.subheader("Prediction")
        if prediction[0] == 1:
            st.error("⚠️ The employee is likely to **leave**.")
        else:
            st.success("✅ The employee is likely to **stay**.")
    except Exception as e:
        st.error(f"❌ Error during prediction:\n{str(e)}")
