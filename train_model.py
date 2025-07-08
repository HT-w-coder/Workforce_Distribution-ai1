import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load and prepare data
df = pd.read_csv("Employee.csv")

# Convert target to binary (if not already)
df['LeaveOrNot'] = df['LeaveOrNot'].map({'Yes': 1, 'No': 0})

# 2. Define features and target
X = df.drop("LeaveOrNot", axis=1)
y = df["LeaveOrNot"]

# 3. Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# 4. Create preprocessing pipeline
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    remainder='passthrough'
)

# 5. Create full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Save the complete pipeline
joblib.dump(model, "model.pkl")
