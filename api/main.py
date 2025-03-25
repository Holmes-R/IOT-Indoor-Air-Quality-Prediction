from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from models.dtmc_model import predict_next_state
from models.ekf_filter import apply_ekf
from api.thingsboard_api import send_data_to_thingsboard
from api.iaq_calculations import calculate_aqi, calculate_vr, calculate_ppd, calculate_sia, classify_aqi_health
import requests
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/sensor")
def serve_sensor():
    return FileResponse("frontend/sensor.html")


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

user_context = {}

class UserContextInput(BaseModel):
    location_type: str
    user_health: str
    environment: str
    daily_activity: str

class AirQualityInput(BaseModel):
    NH3: float
    NO2: float
    Carbon_Monoxide: float
    PM2_5: float
    Temperature: float
    Pressure: float
    Humidity: float
    O3: float  
 
@app.post("/context/")
def save_user_context(context: UserContextInput):
    global user_context
    user_context = context.dict()
    return {"message": "User context saved successfully!"}

@app.post("/predict/")
def predict_air_quality(input_data: AirQualityInput):
    if not user_context:
        raise HTTPException(status_code=400, detail="❌ User context not provided. Call /context/ first.")

    input_dict = input_data.dict()

    # ✅ Fetch Real-Time AQI Data
    aqi_data = get_aqi_data()
    if aqi_data:
        api_aqi = aqi_data.get("aqi", input_data.PM2_5)  # Use API AQI or fallback to local PM2.5
        api_no2 = aqi_data["iaqi"].get("no2", {}).get("v", input_data.NO2)
        api_o3 = aqi_data["iaqi"].get("o3", {}).get("v", input_data.O3)
    else:
        api_aqi = input_data.PM2_5  # Fallback if API fails
        api_no2, api_o3 = input_data.NO2, input_data.O3

    # ✅ Ensure correct column order for ML model
    expected_features = scaler.feature_names_in_
    input_df = pd.DataFrame([input_dict], columns=expected_features)
    input_scaled = scaler.transform(input_df)

    # ✅ Apply EKF noise reduction
    filtered_data = apply_ekf(input_scaled.flatten())

    # ✅ Predict air quality category
    prediction_label = xgb_model.predict([filtered_data])[0]
    predicted_category = label_encoder.inverse_transform([prediction_label])[0]

    # ✅ Predict future state using DTMC
    future_state = predict_next_state(predicted_category)

    aqi = calculate_aqi(api_aqi, api_no2, api_o3)
    vr = calculate_vr(input_data.Pressure)
    ppd = calculate_ppd(input_data.Temperature, input_data.Humidity)
    sia = calculate_sia(aqi, vr, ppd)

    # ✅ Classify based on user health & lifestyle
    health_response = classify_aqi_health(aqi, user_context["user_health"], user_context["daily_activity"])

    sensor_data = input_dict.copy()
    sensor_data.update({
        "User_Context": user_context,
        "Future_Air_Quality": future_state,
        "AQI": aqi,
        "PPD": ppd,
        "SIA": sia,
        "Health_Impact": health_response["impact"],
        "Health_Advice": health_response["health_advice"],
        "Lifestyle_Guidance": health_response["lifestyle_guidance"],
        "API_AQI": api_aqi,
        "API_NO2": api_no2,
        "API_O3": api_o3,
    })

    send_data_to_thingsboard(sensor_data)

    return {
        "future_quality": future_state,
        "classified_aqi": aqi,
        "api_aqi": api_aqi,
        "health_impact": health_response["impact"],
        "health_advice": health_response["health_advice"],
        "lifestyle_guidance": health_response["lifestyle_guidance"]
    }
