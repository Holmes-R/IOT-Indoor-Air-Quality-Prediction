import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
from collections import Counter
from sklearn.metrics import accuracy_score

# Air Quality Classification Using XGBoost

# Load Processed Dataset

# Standardize Column Names

# Feature Selection

# Encode Air Quality Labels

# Handle Class Imbalance

# Train-Test Split --> Splits dataset into 80% training and 20% testing

# Feature Scaling

# Train XGBoost Model

# Evaluate Model

# Save Model and Encoders

file_path = "data/cleaned_indoor_air_pollution.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("❌ Processed dataset not found! Run preprocess.py first.")

df = pd.read_csv(file_path)
print("📌 Columns in Training Data:", df.columns.tolist())

df.columns = df.columns.str.strip().str.replace(" ", "_")

expected_features = ["NH3", "NO2", "Carbon_Monoxide", "PM2_5", "Temperature", "Pressure", "Humidity", "O3"]
X = df[expected_features]
y = df["Air_Quality"]

le = LabelEncoder()
df["Air_Quality_Label"] = le.fit_transform(df["Air_Quality"])
y = df["Air_Quality_Label"]

print("📌 Original Class Distribution:", Counter(y))

df_poor = df[df["Air_Quality"] == "Poor"].sample(n=30_000, random_state=42)
df_hazardous = df[df["Air_Quality"] == "Hazardous"]
df_moderate = df[df["Air_Quality"] == "Moderate"]
df_good = df[df["Air_Quality"] == "Good"]
df_very_poor = df[df["Air_Quality"] == "Very Poor"]

df_balanced = pd.concat([df_poor, df_hazardous, df_moderate, df_good, df_very_poor])
X = df_balanced[expected_features]
y = df_balanced["Air_Quality_Label"]

smote = SMOTE(sampling_strategy="not majority", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("📌 New Class Distribution After SMOTE:", Counter(y_resampled))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgb_air_quality_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(le, "models/label_encoder.joblib")

print("✅ XGBoost Model Trained with SMOTE & Saved Successfully!")
