import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Workforce Distribution AI", layout="centered")
st.title("🌟 Workforce Distribution AI")
st.markdown("Predict if an employee will leave or stay.")

# Define the input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
EverBenched = st.selectbox("Ever Benched", ["Yes", "No"])
City = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
Education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
JoiningYear = st.selectbox("Joining Year", list(range(2012, 2022)))
PaymentTier = st.selectbox("Payment Tier", [1, 2, 3])
Age = st.slider("Age", 20, 60, 30)
ExperienceInCurrentDomain = st.slider("Experience in Current Domain", 0, 10, 3)

# When the user clicks Predict
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "EverBenched": EverBenched,
        "City": City,
        "Education": Education,
        "JoiningYear": JoiningYear,
        "PaymentTier": PaymentTier,
        "Age": Age,
        "ExperienceInCurrentDomain": ExperienceInCurrentDomain
    }])

    # Encode categorical variables
    categorical_cols = ["Gender", "EverBenched", "City", "Education"]
    encoded_input = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))
    input_numeric = input_data.drop(columns=categorical_cols).reset_index(drop=True)
    final_input = pd.concat([input_numeric, encoded_df], axis=1)

    # Make prediction
    prediction = model.predict(final_input)[0]
    if prediction == 1:
        st.success("✅ The employee is likely to stay.")
    else:
        st.warning("⚠️ The employee is likely to leave.")
