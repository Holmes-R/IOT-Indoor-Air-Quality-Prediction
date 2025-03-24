from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from models.dtmc_model import predict_next_state
from models.ekf_filter import apply_ekf
from pydantic import BaseModel
from api.thingsboard_api import send_data_to_thingsboard  # ✅ Import the function
from api.iaq_calculations import calculate_aqi, calculate_vr, calculate_ppd, calculate_sia
import requests 

app = FastAPI()

# ✅ Load Models
xgb_model = joblib.load("models/xgb_air_quality_model.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib") 
AQI_API_URL = "https://api.waqi.info/feed/here/?token=8c006b8c7de3b31dbe8fe035a51ade51c787982b"



def get_aqi_data():
    response = requests.get(AQI_API_URL)
    if response.status_code == 200:
        aqi_data = response.json()
        if "data" in aqi_data and "aqi" in aqi_data["data"]:
            return aqi_data["data"]
    return None

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

    aqi_data = get_aqi_data()
    
    # ✅ Extract key parameters from AQI API
    if aqi_data:
        aqi = aqi_data.get("aqi", input_data.PM2_5)  # Use API AQI or fallback to local PM2.5
        api_no2 = aqi_data["iaqi"].get("no2", {}).get("v", input_data.NO2)
        api_o3 = aqi_data["iaqi"].get("o3", {}).get("v", input_data.O3)
    else:
        aqi = input_data.PM2_5  # Fallback to PM2.5 if API fails
        api_no2, api_o3 = input_data.NO2, input_data.O3 

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
         "API_AQI": aqi,
        "API_NO2": api_no2,
        "API_O3": api_o3,
        "Future_Air_Quality": future_state,
        "AQI": aqi,
        "PPD": ppd,
        "SIA": sia
    })

    send_data_to_thingsboard(sensor_data)

    return {"future_quality": future_state}  