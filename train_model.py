# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load your dataset
df = pd.read_csv("Employee.csv")  # Replace with your actual dataset

# Encoding categorical columns
categorical_cols = ['Education', 'City', 'Gender', 'EverBenched']
encoders = {}

for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# Features and target
X = df[['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']]
y = df['LeaveOrNot']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoder.pkl")
joblib.dump(accuracy, "accuracy.pkl")
