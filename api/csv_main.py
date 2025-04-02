from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import time
import threading
from api.thingsboard_api import send_data_to_thingsboard
import joblib
import xgboost as xgb
import pickle
from pydantic import BaseModel
from models.dtmc_model import predict_next_state
from models.ekf_filter import apply_ekf
from sklearn.preprocessing import StandardScaler

app = FastAPI()

CSV_FILE_PATH = "data/Indoor Air Pollution Data.csv"    
sensor_data_df = pd.read_csv(CSV_FILE_PATH)

selected_columns = ["NH3", "NO2", "CO", "PM2.5", "Temp", "Pressure", "Humidity", "O3"]
sensor_data_df = sensor_data_df[selected_columns]


xgb_model = joblib.load("models/xgb_air_quality_model.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

current_index = 0  
user_confirmation = False  

def calculate_aqi(pm25, no2, o3):
    pm25_breakpoints = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]
    no2_breakpoints = [0, 53, 100, 360, 649, 1249, 2049]
    o3_breakpoints = [0, 54, 70, 85, 105, 200, 300]
    
    aqi_pm25 = np.interp(pm25, pm25_breakpoints, [0, 50, 100, 150, 200, 300, 500])
    aqi_no2 = np.interp(no2, no2_breakpoints, [0, 50, 100, 150, 200, 300, 500])
    aqi_o3 = np.interp(o3, o3_breakpoints, [0, 50, 100, 150, 200, 300, 500])
    
    return max(aqi_pm25, aqi_no2, aqi_o3)

def calculate_vr(co2, air_exchange_rate=0.5):
    base_co2 = 400
    return round(air_exchange_rate * (base_co2 / co2) if co2 > base_co2 else air_exchange_rate, 2)

def calculate_ppd(temperature, humidity):
    ideal_temp, ideal_humidity = 22, 50
    temp_diff = abs(temperature - ideal_temp)
    humidity_diff = abs(humidity - ideal_humidity)
    return round(min(100, (temp_diff * 2) + (humidity_diff * 1.5)), 2)

def calculate_sia(aqi, vr, ppd):
    W_AQI, W_VR, W_PPD = 0.5, 0.3, 0.2
    return round((W_AQI * aqi) + (W_VR * vr) + (W_PPD * (100 - ppd)), 2)

CSV_FILE_PATH_2 = "data/cleaned_indoor_air_pollution.csv"
sensor_data_df_2 = pd.read_csv(CSV_FILE_PATH_2)

selected_columns_2 = ["Air_Quality"]
sensor_data_df_2 = sensor_data_df_2[selected_columns_2]

@app.get("/get_next_data/")
def get_next_data():
    global current_index
    if current_index >= len(sensor_data_df) or current_index >= len(sensor_data_df_2):
        return JSONResponse(content={"message": "No more sensor data available."}, status_code=400)

    # Fetch the same row from both datasets
    sensor_data = sensor_data_df.iloc[current_index].to_dict()
    air_quality_data = sensor_data_df_2.iloc[current_index].to_dict()

    # Compute AQI, VR, PPD, and SIA
    aqi = calculate_aqi(sensor_data["PM2.5"], sensor_data["NO2"], sensor_data["O3"])
    vr = calculate_vr(sensor_data["CO"])
    ppd = calculate_ppd(sensor_data["Temp"], sensor_data["Humidity"])
    sia = calculate_sia(aqi, vr, ppd)

    sensor_data.update({"AQI": aqi, "VR": vr, "PPD": ppd, "SIA": sia})
    sensor_data.update(air_quality_data)  # Add Air_Quality column

    return {"sensor_data": sensor_data, "message": "Confirm to send this data."}





@app.post("/confirm_send/")
def confirm_send():
    global user_confirmation
    user_confirmation = True
    return {"message": "Data confirmed. Sending to ThingsBoard."}

def send_sensor_data_loop():
    global current_index, user_confirmation

    while current_index < len(sensor_data_df):
        if user_confirmation:
            sensor_data = sensor_data_df.iloc[current_index].to_dict()
            aqi = calculate_aqi(sensor_data["PM2.5"], sensor_data["NO2"], sensor_data["O3"])
            vr = calculate_vr(sensor_data["CO"])
            ppd = calculate_ppd(sensor_data["Temp"], sensor_data["Humidity"])
            sia = calculate_sia(aqi, vr, ppd)
            
            input_dict = sensor_data.copy()
            input_dict.update({"AQI": aqi, "VR": vr, "PPD": ppd, "SIA": sia})
            input_df = pd.DataFrame([input_dict], columns=scaler.feature_names_in_)
            input_scaled = scaler.transform(input_df)
            
            filtered_data = apply_ekf(input_scaled.flatten())
            prediction_label = xgb_model.predict([filtered_data])[0]
            predicted_category = label_encoder.inverse_transform([prediction_label])[0]
            future_state = predict_next_state(predicted_category)
            
            sensor_data.update({
                "AQI": aqi, "VR": vr, "PPD": ppd, "SIA": sia,
                "Predicted_Air_Quality": predicted_category,
                "Future_Air_Quality_Trend": future_state
            })
            
            send_data_to_thingsboard(sensor_data)
            print(f"Sent Data: {sensor_data}")
            
            user_confirmation = False
            current_index += 1
        time.sleep(2)

@app.on_event("startup")
def start_background_task():
    thread = threading.Thread(target=send_sensor_data_loop, daemon=True)
    thread.start()

