import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("Employee.csv")

# Define only the input features
features = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]

# Ensure no target leakage
X = df[features].copy()
y = df["LeaveOrNot"].copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save model only
joblib.dump(pipeline, "model.pkl")

print("✅ Model trained and saved as model.pkl")
