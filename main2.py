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

# Preprocess input data if necessary
# Example: Convert categorical variables to numerical if your model requires it
input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
input_data['EverBenched'] = input_data['EverBenched'].map({'Yes': 1, 'No': 0})
input_data['City'] = input_data['City'].map({'Bangalore': 0, 'Pune': 1, 'New Delhi': 2})
input_data['Education'] = input_data['Education'].map({'Bachelors': 0, 'Masters': 1, 'PHD': 2})

# Predict when button clicked
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction")
        if prediction == 1:
            st.error("⚠️ The employee is likely to **leave**.")
        else:
            st.success("✅ The employee is likely to **stay**.")
    except Exception as e:
        st.error(f"❌ Error during prediction:\n{str(e)}")
