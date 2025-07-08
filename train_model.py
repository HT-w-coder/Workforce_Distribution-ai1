import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Employee.csv")

# Define features and target
X = df.drop("LeaveOrNot", axis=1)
y = df["LeaveOrNot"]

# Ensure target variable is binary
y = y.map({'Yes': 1, 'No': 0})  # Adjust this mapping based on your actual data

# Define categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep the numerical columns as they are
)

# Create a pipeline that first transforms the data and then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the entire pipeline
joblib.dump(pipeline, "model_pipeline.pkl")
