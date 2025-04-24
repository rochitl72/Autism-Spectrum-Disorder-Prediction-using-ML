# backend/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Step 1: Load dataset
df = pd.read_csv("data/autism_data.csv")

# Step 2: Drop non-useful columns if any
df.drop(columns=["result", "age_desc", "relation"], inplace=True, errors='ignore')

# Step 3: Encode categorical features
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le


# Step 4: Separate features and target
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
print("[DEBUG] Columns used in training:")
print(X.columns.tolist())
# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Save model & scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/asd_model.joblib")
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(label_encoders, "model/label_encoders.joblib")

print("✅ Model and scalers saved in /model/")
