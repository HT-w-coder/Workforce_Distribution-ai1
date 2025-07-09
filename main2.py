import streamlit as st
import pandas as pd
import joblib

# Load saved model, encoder, and column list
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
categorical_cols = joblib.load("categorical_cols.pkl")

st.set_page_config(page_title="Workforce Distribution AI", layout="centered")
st.title("🌟 Workforce Distribution AI")
st.write("Predict if an employee will **leave or stay** based on their profile.")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.slider("Age", min_value=18, max_value=60, value=30)
experience = st.slider("Experience in Current Domain", min_value=0, max_value=10, value=3)

# Construct input DataFrame
input_dict = {
    "Gender": [gender],
    "EverBenched": [ever_benched],
    "City": [city],
    "Education": [education],
    "JoiningYear": [joining_year],
    "PaymentTier": [payment_tier],
    "Age": [age],
    "ExperienceInCurrentDomain": [experience]
}
input_df = pd.DataFrame(input_dict)

# Encode categorical features
encoded_input = encoder.transform(input_df[categorical_cols])
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))

# Combine encoded and numerical columns
numerical_cols = [col for col in input_df.columns if col not in categorical_cols]
final_input = pd.concat([input_df[numerical_cols].reset_index(drop=True), encoded_df], axis=1)

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(final_input)[0]
        st.subheader("Prediction")
        if prediction == 1:
            st.success("✅ The employee is likely to **STAY**.")
        else:
            st.warning("⚠️ The employee is likely to **LEAVE**.")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
