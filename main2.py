import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load("model.pkl")

# App UI
st.set_page_config(page_title="Employee Retention Predictor", layout="centered")
st.title("📊 Employee Retention Prediction")

# Input fields - must match EXACTLY with training data columns
with st.form("prediction_form"):
    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_benched = st.selectbox("Ever Benched", ["Yes", "No"])
    city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"])
    education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
    
    # Numerical inputs
    joining_year = st.slider("Joining Year", min_value=2012, max_value=2018)
    payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
    age = st.slider("Age", min_value=20, max_value=60)
    experience = st.slider("Experience", min_value=0, max_value=10)
    
    # Prediction button
    if st.form_submit_button("Predict Retention"):
        # Create input DataFrame with EXACTLY same column names as training data
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
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Show result
            if prediction == 1:
                st.error("🔴 High risk of leaving")
            else:
                st.success("🟢 Likely to stay")
                
            # Show probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)[0]
                st.write(f"Confidence: {max(proba)*100:.1f}%")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
