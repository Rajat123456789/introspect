import pandas as pd
import os

# List of available health metrics
METRICS_str = "activeCaloriesBurned, basalBodyTemperature, basalMetabolicRate, bloodGlucose, bloodPressure, bodyFat, bodyTemperature, boneMass, cervicalMucus, distance, exerciseSession, elevationGained, floorsClimbed, heartRate, height, hydration, leanBodyMass, menstruationFlow, menstruationPeriod, nutrition, ovulationTest, oxygenSaturation, power, respiratoryRate, restingHeartRate, sleepSession, speed, steps, stepsCadence, totalCaloriesBurned, vo2Max, weight, wheelchairPushes"
METRICS = METRICS_str.split(", ")

# Directory containing the CSV files
data_dir = "Data"
username = "someshbgd3"

def read_csv(username, metric):
    # try:
    file_path = os.path.join(data_dir, f"{username}_{metric}.csv")
    print(file_path)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(df.head())
        return df
    else:
        print(f"❌ {metric} CSV file does not exist")
        return None
    
# for metric in METRICS:
read_csv(username, METRICS[2])
    