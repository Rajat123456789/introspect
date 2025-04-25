import requests
import pandas as pd
import os
import json
from pathlib import Path
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class HealthConnectAPI:
    def __init__(self, base_url, username, password, max_retries=3):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token = None
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def authenticate(self):
        """Authenticate and get token with retry logic."""
        url = f"{self.base_url}/api/v2/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"Content-Type": "application/json"}
        
        try:
            response = self.session.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            self.token = data["token"]
            print("✅ Authentication successful!")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Authentication failed: {str(e)}")
            return False

    def fetch_health_data(self, metric):
        """Fetch health data with retry logic and validation."""
        if not self.token:
            print("❌ No authentication token available")
            return None

        url = f"{self.base_url}/api/v2/fetch/{metric}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            response = self.session.post(url, json={}, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Validate response data
            if not isinstance(data, list):
                print(f"❌ Invalid data format for {metric}: expected list")
                return None
                
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch {metric}: {str(e)}")
            return None

    def save_json_to_csv(self, metric, json_data):
        """Save JSON data to CSV with validation and error handling."""
        if not json_data:
            print(f"⚠️ No data to save for {metric}")
            return False

        try:
            # Convert list of dictionaries to Pandas DataFrame
            df = pd.DataFrame(json_data)
            if df.empty:
                print(f"⚠️ Empty dataset for {metric}")
                return False

            # Ensure directory exists
            output_dir = Path("Data") / self.username / "Uncleaned"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save DataFrame as CSV
            output_file = output_dir / f"{metric}_{self.username}.csv"
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"📁 Successfully saved {metric} data as {output_file}")
            return True

        except Exception as e:
            print(f"❌ Error saving {metric} to CSV: {str(e)}")
            return False

def main():
    # Configuration
    BASE_URL = "https://api.hcgateway.shuchir.dev"
    USERNAME = "someshbgd3"  # Consider loading from environment variables
    PASSWORD = "Hc@SPB75895"  # Consider loading from environment variables
    # USERNAME = "gaurav_surtani"
    # PASSWORD = "Sjsu2024!"


    # Create Data directory if it doesn't exist
    data_dir = Path("Data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # List of health metrics
    METRICS = [
        "activeCaloriesBurned", "basalBodyTemperature", "basalMetabolicRate",
        "bloodGlucose", "bloodPressure", "bodyFat", "bodyTemperature",
        "boneMass", "cervicalMucus", "distance", "exerciseSession",
        "elevationGained", "floorsClimbed", "heartRate", "height",
        "hydration", "leanBodyMass", "menstruationFlow", "menstruationPeriod",
        "nutrition", "ovulationTest", "oxygenSaturation", "power",
        "respiratoryRate", "restingHeartRate", "sleepSession", "speed",
        "steps", "stepsCadence", "totalCaloriesBurned", "vo2Max",
        "weight", "wheelchairPushes"
    ]

    # Initialize API client
    api = HealthConnectAPI(BASE_URL, USERNAME, PASSWORD)
    
    print("🔄 Authenticating...")
    if not api.authenticate():
        print("❌ Exiting: Failed to authenticate.")
        return

    # Process each metric
    for metric in METRICS:
        print(f"\n🔍 Fetching data for {metric}...")
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            data = api.fetch_health_data(metric)
            if data is not None:
                if api.save_json_to_csv(metric, data):
                    print(f"✅ {metric} data processed successfully.")
                break
            
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"⚠️ Retrying {metric} in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to process {metric} after {max_retries} attempts.")

if __name__ == "__main__":
    main()