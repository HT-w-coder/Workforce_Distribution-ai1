import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
fromtarget
X = df.drop("LeaveOrNot", axis=1)
y = df["LeaveOrNot"]
 sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Employee.csv")

# Define features and 
# One-hot encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Combine with numerical columns
X_numeric = X.drop(columns=categorical_cols).reset_index(drop=True)
X_final = pd.concat([X_numeric, X_encoded_df], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
