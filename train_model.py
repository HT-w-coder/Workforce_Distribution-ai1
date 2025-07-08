import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv("Employee.csv")

# Separate features and target
if "LeaveOrNot" not in data.columns:
    raise ValueError("Target column 'LeaveOrNot' not found in dataset!")

X = data.drop("LeaveOrNot", axis=1)
y = data["LeaveOrNot"]

# Identify categorical columns for one-hot encoding
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# OneHotEncode the categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = encoder.fit_transform(X[categorical_cols])

# Create a DataFrame with encoded features and drop original categorical columns
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
X_remaining = X.drop(columns=categorical_cols).reset_index(drop=True)
X_final = pd.concat([X_remaining, X_encoded_df], axis=1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model and encoder saved successfully.")
