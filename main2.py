import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.title("Workforce Distribution AI - Leave Prediction")

# Input fields
satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.7)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_monthly_hours = st.number_input("Average Monthly Hours", 90, 310, 160)
time_spent = st.number_input("Time Spent at Company (Years)", 1, 10, 3)

city = st.selectbox("City", ['Bangalore', 'Pune', 'New Delhi'])

# Prepare numerical and categorical inputs
numerical_data = pd.DataFrame([{
    "satisfaction_level": satisfaction,
    "last_evaluation": evaluation,
    "number_project": number_project,
    "average_montly_hours": average_monthly_hours,
    "time_spend_company": time_spent
}])

categorical_data = pd.DataFrame([[city]], columns=["City"])
cat_encoded = encoder.transform(categorical_data)
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(["City"]))

# Combine
final_input = pd.concat([numerical_data, cat_encoded_df], axis=1)

# Prediction
if st.button("Predict Leave or Not"):
    prediction = model.predict(final_input)[0]
    st.success(f"Prediction: {'Will Leave' if prediction == 1 else 'Will Stay'}")
