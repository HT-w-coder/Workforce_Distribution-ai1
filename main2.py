import streamlit as st
import pandas as pd
import joblib

# Load encoder and model
encoder = joblib.load("encoder.pkl")  # ensure this is a fitted OneHotEncoder
model = joblib.load("model.pkl")

# Upload CSV or manual input
st.title("Workforce Prediction App")

st.write("Enter employee details:")

# Collect input
department = st.selectbox("Department", ["Analytics", "Finance", "HR", "Legal", "Operations", "Procurement", "R&D", "Sales", "Technology"])
education = st.selectbox("Education", ["Bachelor's", "Master's", "PhD"])
gender = st.selectbox("Gender", ["Male", "Female"])
recruitment_channel = st.selectbox("Recruitment Channel", ["sourcing", "other", "referred"])
no_of_trainings = st.number_input("Number of Trainings", min_value=0, max_value=20, step=1)
age = st.number_input("Age", min_value=18, max_value=60, step=1)
length_of_service = st.number_input("Length of Service (years)", min_value=0, max_value=40, step=1)
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, step=1)

# Create input DataFrame
input_df = pd.DataFrame([{
    "Department": department,
    "Education": education,
    "Gender": gender,
    "Recruitment_Channel": recruitment_channel,
    "No_of_Trainings": no_of_trainings,
    "Age": age,
    "Length_of_Service": length_of_service,
    "Avg_Training_Score": avg_training_score
}])

# Apply encoder on categorical columns
categorical_cols = ["Department", "Education", "Gender", "Recruitment_Channel"]
numerical_cols = ["No_of_Trainings", "Age", "Length_of_Service", "Avg_Training_Score"]

encoded_cat = encoder.transform(input_df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

# Combine with numeric columns
final_input = pd.concat([encoded_cat_df, input_df[numerical_cols].reset_index(drop=True)], axis=1)

# Predict
if st.button("Predict"):
    prediction = model.predict(final_input)
    st.success(f"Prediction (Leave or Not): {prediction[0]}")
