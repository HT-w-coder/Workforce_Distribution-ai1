import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("🌟 Workforce Distribution AI")
st.write("Predict if an employee will leave or stay (only numerical features).")

# Input form
with st.form("prediction_form"):
    JoiningYear = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
    PaymentTier = st.selectbox("Payment Tier", [1, 2, 3])
    Age = st.slider("Age", 20, 60, 30)
    Experience = st.slider("Experience in Current Domain", 0, 10, 2)
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    try:
        input_df = pd.DataFrame([{
            "JoiningYear": JoiningYear,
            "PaymentTier": PaymentTier,
            "Age": Age,
            "ExperienceInCurrentDomain": Experience
        }])

        prediction = model.predict(input_df)[0]
        result = "✅ Will Stay" if prediction == 0 else "❌ Will Leave"

        st.subheader("Prediction Result")
        st.success(result)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
