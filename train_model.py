import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("Employee.csv")

# Drop any unnecessary columns
if 'Employee ID' in df.columns:
    df.drop("Employee ID", axis=1, inplace=True)

# Define target and features
X = df.drop("LeaveOrNot", axis=1)
y = df["LeaveOrNot"]

# Identify categorical and numerical columns
categorical_cols = ["Gender", "EverBenched", "City", "Education"]
numerical_cols = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols),
    ]
)

# Full pipeline: preprocessing + model
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
clf.fit(X_train, y_train)

# Save entire pipeline to disk
joblib.dump(clf, "model.pkl")
print("✅ Model pipeline trained and saved as model.pkl")