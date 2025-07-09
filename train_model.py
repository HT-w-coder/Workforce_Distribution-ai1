import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load dataset
df = pd.read_csv("Employee.csv")

# Define target and features
if "LeaveOrNot" not in df.columns:
    raise ValueError("Target column 'LeaveOrNot' not found!")

X = df.select_dtypes(include=[np.number])  # Only numerical columns
y = df["LeaveOrNot"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Dummy encoder (for compatibility)
joblib.dump(None, "encoder.pkl")

# Save model
joblib.dump(model, "model.pkl")
