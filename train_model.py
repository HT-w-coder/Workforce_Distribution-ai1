import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Employee.csv")

# Separate features and target
X = df.drop(columns=["LeaveOrNot"])
y = df["LeaveOrNot"]

# Select numerical + categorical features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = ['City']  # You can add more if needed

# Encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded_cat = encoder.fit_transform(X[categorical_cols])
X_encoded_cat_df = pd.DataFrame(X_encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

# Combine numerical and encoded categorical
X_final = pd.concat([X[numerical_cols].reset_index(drop=True), X_encoded_cat_df.reset_index(drop=True)], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
