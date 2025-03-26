import requests
import json

THINGSBOARD_HOST = "http://demo.thingsboard.io"
ACCESS_TOKEN = "LUlOt3MN2GbA4jQm5nDx" 
THINGSBOARD_URL = f"{THINGSBOARD_HOST}/api/v1/{ACCESS_TOKEN}/telemetry"


def send_data_to_thingsboard(sensor_data):
    """
    Sends sensor data, IAQ metrics, and future air quality predictions to ThingsBoard.
    """
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(THINGSBOARD_URL, data=json.dumps(sensor_data), headers=headers)

        if response.status_code == 200:
            print(" Data sent to ThingsBoard successfully!")
        else:
            print(f"Failed to send data. Status Code: {response.status_code}, Response: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f" Error sending data to ThingsBoard: {e}")