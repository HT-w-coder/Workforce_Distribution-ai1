import streamlit as st
import pandas as pd
import joblib
import sklearn
import streamlit as st
st.write("Scikit-learn version:", sklearn.__version__)
s

# Load the model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoder.pkl")

st.title("Workforce Distribution AI")

# Input fields
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2018)
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.number_input("Age", min_value=18, max_value=60, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
experience = st.number_input("Experience in Current Domain", min_value=0, max_value=20, value=3)

# When user clicks Predict
if st.button("Predict"):
    try:
        # Prepare input
        input_data = {
            'Education': encoders['Education'].transform([education])[0],
            'JoiningYear': joining_year,
            'City': encoders['City'].transform([city])[0],
            'PaymentTier': payment_tier,
            'Age': age,
            'Gender': encoders['Gender'].transform([gender])[0],
            'EverBenched': encoders['EverBenched'].transform([ever_benched])[0],
            'ExperienceInCurrentDomain': experience
        }

        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]

        # Show result
        st.success("Prediction: Will Leave" if prediction == 1 else "Prediction: Will Stay")

    except Exception as e:
        st.error(f"Error: {str(e)}")
