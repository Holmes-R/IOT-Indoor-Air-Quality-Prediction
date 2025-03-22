from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from models.dtmc_model import predict_next_state
from models.ekf_filter import apply_ekf
from pydantic import BaseModel
from api.thingsboard_api import send_data_to_thingsboard  # ✅ Import the function
from iaq_calculations import calculate_aqi, calculate_vr, calculate_ppd, calculate_sia


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

    # ✅ Prepare data for ThingsBoard
    thingsboard_data = {
        "NH3": input_data.NH3,
        "NO2": input_data.NO2,
        "Carbon_Monoxide": input_data.Carbon_Monoxide,
        "PM2_5": input_data.PM2_5,
        "Temperature": input_data.Temperature,
        "Pressure": input_data.Pressure,
        "Humidity": input_data.Humidity,
        "O3": input_data.O3,
        "Future_Air_Quality": future_state  # ✅ Include future quality
    }

    # ✅ Send data to ThingsBoard
    send_data_to_thingsboard(thingsboard_data)

    return {"future_quality": future_state}  # ✅ Only return future quality


# ✅ Define Input Schema for IAQ Metrics Calculation
class IAQInput(BaseModel):
    CO2: float
    PM2_5: float
    NO2: float
    O3: float
    Temperature: float
    Humidity: float

@app.post("/compute_iaq/")
def compute_iaq(data: IAQInput):
    """
    API Endpoint to compute AQI, VR, PPD, SIA, and send data to ThingsBoard.
    """
    # ✅ Compute IAQ Metrics
    aqi = calculate_aqi(data.PM2_5, data.NO2, data.O3)
    vr = calculate_vr(data.CO2)
    ppd = calculate_ppd(data.Temperature, data.Humidity)
    sia = calculate_sia(aqi, vr, ppd)

    # ✅ Prepare sensor data for ThingsBoard
    sensor_data = data.dict()
    sensor_data.update({"AQI": aqi, "VR": vr, "PPD": ppd, "SIA": sia})

    # ✅ Send data to ThingsBoard
    send_data_to_thingsboard(sensor_data)

    return {"AQI": aqi, "VR": vr, "PPD": ppd, "SIA": sia, "message": "Data sent to ThingsBoard"}