import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
accuracy = joblib.load("accuracy.pkl")

st.title("Workforce Distribution AI Predictor")

# Form input
education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2030, step=1)
city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.number_input("Age", min_value=18, max_value=70)
gender = st.selectbox("Gender", ["Male", "Female"])
ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
experience = st.slider("Experience in Current Domain (Years)", 0, 30)

if st.button("Predict"):
    try:
        input_data = [
            encoders['Education'].transform([education])[0],
            joining_year,
            encoders['City'].transform([city])[0],
            payment_tier,
            age,
            encoders['Gender'].transform([gender])[0],
            encoders['EverBenched'].transform([ever_benched])[0],
            experience
        ]
        
        prediction = model.predict([input_data])[0]

        base_salary = payment_tier * 50000
        salary_growth = experience * 0.05 * base_salary
        overall_wage = base_salary + salary_growth

        st.success(f"Prediction: {'Will Leave' if prediction else 'Will Stay'}")
        st.metric("Expected Salary Growth", f"₹{salary_growth:,.2f}")
        st.metric("Overall Wage", f"₹{overall_wage:,.2f}")
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
