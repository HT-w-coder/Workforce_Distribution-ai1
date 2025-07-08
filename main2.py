# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("Employee.csv")

# Define target and features
X = df.drop(columns=["LeaveOrNot"])
y = df["LeaveOrNot"]

# Define categorical and numerical columns
categorical_cols = ["Gender", "EverBenched", "City", "Education"]
numerical_cols = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="passthrough"
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save full model pipeline
joblib.dump(pipeline, "model.pkl")