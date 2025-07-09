import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv("Employee.csv")

# Ensure target exists
if "LeaveOrNot" not in data.columns:
    raise ValueError("Target column 'LeaveOrNot' not found in dataset!")

# Separate features and target
X = data.drop("LeaveOrNot", axis=1)
y = data["LeaveOrNot"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])

# Create DataFrame with encoded features
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

# Concatenate encoded categorical and numerical features
X_final = pd.concat([X[numerical_cols].reset_index(drop=True), X_encoded_df], axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")

print("✅ Model, encoder, and column names saved successfully.")
