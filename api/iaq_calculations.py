import numpy as np

# AQI Calculation Based on PM2.5, NO2, O3
def calculate_aqi(pm25, no2, o3):
    """
    Approximate AQI Calculation using major pollutants.
    Formula based on simplified AQI standards.
    """
    # AQI breakpoints for PM2.5, NO2, and O3
    pm25_breakpoints = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]
    no2_breakpoints = [0, 53, 100, 360, 649, 1249, 2049]
    o3_breakpoints = [0, 54, 70, 85, 105, 200, 300]

    # Convert pollutant concentration to AQI scale
    aqi_pm25 = np.interp(pm25, pm25_breakpoints, [0, 50, 100, 150, 200, 300, 500])
    aqi_no2 = np.interp(no2, no2_breakpoints, [0, 50, 100, 150, 200, 300, 500])
    aqi_o3 = np.interp(o3, o3_breakpoints, [0, 50, 100, 150, 200, 300, 500])

    # Return the highest AQI value
    return max(aqi_pm25, aqi_no2, aqi_o3)

# Ventilation Rate Calculation (VR)
def calculate_vr(co2, air_exchange_rate=0.5):
    """
    Ventilation Rate (VR) estimation based on CO2 levels.
    Higher VR means better air circulation.
    """
    base_co2 = 400  # Outdoor CO2 baseline in ppm
    if co2 > base_co2:
        vr = air_exchange_rate * (base_co2 / co2)
    else:
        vr = air_exchange_rate
    return round(vr, 2)

# Predicted Percentage of Dissatisfied (PPD)
def calculate_ppd(temperature, humidity):
    """
    Compute PPD based on thermal comfort model.
    Assumes PPD is related to deviation from ideal conditions.
    """
    ideal_temp = 22  # Ideal temperature in °C
    ideal_humidity = 50  # Ideal humidity in %

    temp_diff = abs(temperature - ideal_temp)
    humidity_diff = abs(humidity - ideal_humidity)

    # Estimate PPD: Higher deviation = more dissatisfaction
    ppd = min(100, (temp_diff * 2) + (humidity_diff * 1.5))
    return round(ppd, 2)

# Compute State of Indoor Air (SIA)
def calculate_sia(aqi, vr, ppd):
    """
    Compute State of Indoor Air (SIA) Index.
    Formula: SIA = (W_AQI * AQI) + (W_VR * VR) + (W_PPD * (100 - PPD))
    """
    W_AQI, W_VR, W_PPD = 0.5, 0.3, 0.2  # Adjustable weights
    sia = (W_AQI * aqi) + (W_VR * vr) + (W_PPD * (100 - ppd))
    return round(sia, 2)
