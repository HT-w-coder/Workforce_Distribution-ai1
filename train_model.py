import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("Employee.csv")

# Check target
if "LeaveOrNot" not in df.columns:
    raise ValueError("Target column 'LeaveOrNot' not found in dataset!")

# Only numeric features
features = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]
X = df[features]
y = df["LeaveOrNot"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Scaling + Model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Save model pipeline
joblib.dump(pipeline, "model.pkl")

# Dummy encoder (not used, but expected by older main2 versions)
joblib.dump(None, "encoder.pkl")

print("✅ Model saved as model.pkl")
