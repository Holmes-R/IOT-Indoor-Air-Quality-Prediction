import requests

# ✅ ThingsBoard API details (Replace with your actual token)
THINGSBOARD_HOST = "http://demo.thingsboard.io"
ACCESS_TOKEN = "LUlOt3MN2GbA4jQm5nDx"  # ✅ Replace with your real access token
THINGSBOARD_URL = f"{THINGSBOARD_HOST}/api/v1/{ACCESS_TOKEN}/telemetry"

def send_data_to_thingsboard(data):
    """ Sends air quality prediction and sensor data to ThingsBoard via HTTP API in the correct format. """
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(THINGSBOARD_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        print("✅ Data sent to ThingsBoard successfully!")
    else:
        print(f"❌ Failed to send data. Status Code: {response.status_code}, Response: {response.text}")
