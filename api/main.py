from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from models.dtmc_model import predict_next_state
from models.ekf_filter import apply_ekf
from pydantic import BaseModel
from api.thingsboard_api import send_data_to_thingsboard  # ✅ Import the function
from api.iaq_calculations import calculate_aqi, calculate_vr, calculate_ppd, calculate_sia


app = FastAPI()

# ✅ Load Models
xgb_model = joblib.load("models/xgb_air_quality_model.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib") 

'''xgb_model = joblib.load("models/xgb_air_quality_model.joblib")
scaler = joblib.load("models/scaler.joblib")
print("Scaler was trained on:", scaler.feature_names_in_)
# Test Input Example
test_input = np.array([[1.0, 1.5, 0.8, 40, 25, 1008, 55, 0.5]])  # Example "Moderate" case

# Scale input
test_scaled = scaler.transform(test_input)

# Predict
prediction = xgb_model.predict(test_scaled)
print("Model Raw Prediction:", prediction)

# Load label encoder
le = joblib.load("models/label_encoder.joblib")
predicted_category = le.inverse_transform(prediction)[0]
print("Predicted Category:", predicted_category)'''


# ✅ Define Input Schema for Air Quality Prediction
class AirQualityInput(BaseModel):
    NH3: float
    NO2: float
    Carbon_Monoxide: float
    PM2_5: float
    Temperature: float
    Pressure: float
    Humidity: float
    O3: float  

@app.post("/predict/")
def predict_air_quality(input_data: AirQualityInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    # ✅ Ensure correct column order
    expected_features = scaler.feature_names_in_
    input_df = input_df[expected_features]

    # ✅ Scale the input
    input_scaled = scaler.transform(input_df)

    # ✅ Apply EKF noise reduction
    filtered_data = apply_ekf(input_scaled.flatten())

    # ✅ Predict air quality category
    prediction_label = xgb_model.predict([filtered_data])[0]
    predicted_category = label_encoder.inverse_transform([prediction_label])[0]

    # ✅ Predict future state using DTMC
    future_state = predict_next_state(predicted_category)

    # ✅ Compute IAQ Metrics
    aqi = calculate_aqi(input_data.PM2_5, input_data.NO2, input_data.O3)
    vr = calculate_vr(input_data.Pressure)  # Assuming Pressure relates to VR
    ppd = calculate_ppd(input_data.Temperature, input_data.Humidity)
    sia = calculate_sia(aqi, vr, ppd)

    sensor_data = input_data.dict()
    sensor_data.update({
        "Future_Air_Quality": future_state,
        "AQI": aqi,
        "PPD": ppd,
        "SIA": sia
    })

    send_data_to_thingsboard(sensor_data)

    return {"future_quality": future_state}  