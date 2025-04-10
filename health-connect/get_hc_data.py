import requests
import pandas as pd
import os
import json

BASE_URL = "https://api.hcgateway.shuchir.dev"
# USERNAME = "someshbgd3"
# PASSWORD = "Hc@SPB75895"

USERNAME = "gaurav_surtani"
PASSWORD = "Sjsu2024!"


# List of available health metrics
METRICS_str = "activeCaloriesBurned, basalBodyTemperature, basalMetabolicRate, bloodGlucose, bloodPressure, bodyFat, bodyTemperature, boneMass, cervicalMucus, distance, exerciseSession, elevationGained, floorsClimbed, heartRate, height, hydration, leanBodyMass, menstruationFlow, menstruationPeriod, nutrition, ovulationTest, oxygenSaturation, power, respiratoryRate, restingHeartRate, sleepSession, speed, steps, stepsCadence, totalCaloriesBurned, vo2Max, weight, wheelchairPushes"

METRICS = METRICS_str.split(", ")

def get_auth_token():
    url = f"{BASE_URL}/api/v2/login"
    payload = {"username": USERNAME, "password": PASSWORD}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers)
    print(response, response.text)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        return data["token"]
    else:
        print(f"❌ Login failed: {response.text}")
        return None

def fetch_health_data(token, metric):
    url = f"{BASE_URL}/api/v2/fetch/{metric}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json={}, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to fetch {metric}: {response.text}")
        return None

def save_json_to_csv(username, metric, json_data):

    try:
        df = pd.DataFrame(json_data)

        os.makedirs(f"Data/{username}", exist_ok=True)
        filename = os.path.join(f"Data/{username}", f"{username}_{metric}.csv")
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"📁 Successfully saved {metric} data as {filename}")
    except Exception as e:
        print(f"❌ Error saving {metric} to CSV: {e}")


print("🔄 Logging in to get API token...")
token = get_auth_token()

if not token:
    print("❌ Exiting: Failed to authenticate.")
    exit()

print("✅ Authentication successful!")
for metric in METRICS:
    print(f"🔍 Fetching data for {metric}...")
    data = fetch_health_data(token, metric)

    if data:
        save_json_to_csv(USERNAME, metric, data)
        print(f"✅ {metric} data saved successfully.")
    else:
        print(f"⚠️ No data available for {metric}.")