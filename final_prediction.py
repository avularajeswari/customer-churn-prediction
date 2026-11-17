
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Remove customer ID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Separate input and output
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    include=["number"]
).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),
        ("num", SimpleImputer(strategy="median"),
         numerical_cols)
    ]
)

# Create complete ML pipeline
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
best_model.fit(X_train, y_train)

print("Model trained successfully!")

# Evaluate model
accuracy = best_model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
model_path = BASE_DIR / "customer_churn_model.pkl"

joblib.dump(best_model, model_path)

print("Model saved successfully!")

# Load saved model
loaded_model = joblib.load(model_path)

print("Model verified successfully!")
print(loaded_model)
