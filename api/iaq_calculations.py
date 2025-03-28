import numpy as np

"""Indoor Air Quality (IAQ) Assessment Functions"""

'''This module provides functions to estimate various indoor air quality (IAQ) parameters such as AQI, ventilation rate, predicted percentage dissatisfied (PPD), and the overall State of Indoor Air (SIA) Index. 
It also includes a function to classify AQI health impacts.'''

           
def calculate_aqi(pm25, no2, o3):
    """
    Approximate AQI Calculation using major pollutants.
    Formula based on simplified AQI standards.
    """
    pm25_breakpoints = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]
    no2_breakpoints = [0, 53, 100, 360, 649, 1249, 2049]
    o3_breakpoints = [0, 54, 70, 85, 105, 200, 300]

    aqi_pm25 = np.interp(pm25, pm25_breakpoints, [0, 50, 100, 150, 200, 300, 500])
    aqi_no2 = np.interp(no2, no2_breakpoints, [0, 50, 100, 150, 200, 300, 500])
    aqi_o3 = np.interp(o3, o3_breakpoints, [0, 50, 100, 150, 200, 300, 500])

    return max(aqi_pm25, aqi_no2, aqi_o3)

def calculate_vr(co2, air_exchange_rate=0.5):
    """
    Ventilation Rate (VR) estimation based on CO2 levels.
    Higher VR means better air circulation.
    """
    base_co2 = 400  
    if co2 > base_co2:
        vr = air_exchange_rate * (base_co2 / co2)
    else:
        vr = air_exchange_rate
    return round(vr, 2)

def calculate_ppd(temperature, humidity):
    """
    Compute PPD based on thermal comfort model.
    Assumes PPD is related to deviation from ideal conditions.
    """
    ideal_temp = 22  
    ideal_humidity = 50  

    temp_diff = abs(temperature - ideal_temp)
    humidity_diff = abs(humidity - ideal_humidity)

    ppd = min(100, (temp_diff * 2) + (humidity_diff * 1.5))
    return round(ppd, 2)

def calculate_sia(aqi, vr, ppd):
    """
    Compute State of Indoor Air (SIA) Index.
    Formula: SIA = (W_AQI * AQI) + (W_VR * VR) + (W_PPD * (100 - PPD))
    """
    W_AQI, W_VR, W_PPD = 0.5, 0.3, 0.2
    sia = (W_AQI * aqi) + (W_VR * vr) + (W_PPD * (100 - ppd))
    return round(sia, 2)

def classify_aqi_health(aqi, user_health, daily_activity):
    if aqi <= 50:
        impact = "Good ✅ - No health risks. Perfect air for all activities."
    elif aqi <= 100:
        impact = "Moderate 🟡 - Acceptable, but sensitive individuals (asthma, elderly) should take caution."
    elif aqi <= 150:
        impact = "Unhealthy for Sensitive Groups 🟠 - Asthma patients, children, elderly may have breathing issues."
    elif aqi <= 200:
        impact = "Unhealthy 🔴 - Everyone may feel discomfort, high risk for sensitive groups."
    elif aqi <= 300:
        impact = "Very Unhealthy 🟣 - Serious breathing issues. Avoid outdoor activities."
    else:
        impact = "Hazardous ☠️ - Severe health risk! Stay indoors with air purifiers."

    health_warnings = {
        "asthma": "❗ AQI above 100 may trigger breathing issues. Avoid polluted areas.",
        "athlete": "🏃‍♂️ Outdoor exercise is risky above AQI 150. Prefer indoor workouts.",
        "child": "👶 Children's lungs are sensitive. Keep them indoors if AQI > 120.",
    }

    lifestyle_advice = {
        "exercise": "🏋️ High AQI can reduce lung capacity. Try gym workouts instead of outdoor runs.",
        "commute": "🚗 High AQI = traffic pollution. Keep windows closed, use AC filters.",
    }

    return {
        "impact": impact,
        "health_advice": health_warnings.get(user_health, "Check AQI regularly."),
        "lifestyle_guidance": lifestyle_advice.get(daily_activity, "Stay safe and follow AQI updates.")
    }
