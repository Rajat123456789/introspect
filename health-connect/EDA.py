import pandas as pd
import os

# List of available health metrics
METRICS_str = "activeCaloriesBurned, basalBodyTemperature, basalMetabolicRate, bloodGlucose, bloodPressure, bodyFat, bodyTemperature, boneMass, cervicalMucus, distance, exerciseSession, elevationGained, floorsClimbed, heartRate, height, hydration, leanBodyMass, menstruationFlow, menstruationPeriod, nutrition, ovulationTest, oxygenSaturation, power, respiratoryRate, restingHeartRate, sleepSession, speed, steps, stepsCadence, totalCaloriesBurned, vo2Max, weight, wheelchairPushes"
METRICS = METRICS_str.split(", ")

# Directory containing the CSV files
data_dir = "Data"
username = "someshbgd3"

# Initialize an empty DataFrame for merging
merged_df = pd.DataFrame()

# Function to round timestamps to the nearest minute
def round_to_minute(df, timestamp_col):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='ISO8601').dt.floor('min')
    return df

# Read and merge each CSV file
for metric in METRICS:
    file_path = os.path.join(data_dir, f"{username}_{metric}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'start' in df.columns:
            df = round_to_minute(df, 'start')
        if 'end' in df.columns:
            df = round_to_minute(df, 'end')
        
        # Keep only the required columns
        df = df[['start', 'end', 'app', 'data']]
        
        # Prefix column names with the metric name, except for 'start'
        df = df.add_prefix(f"{metric}_")
        df = df.rename(columns={f"{metric}_start": "start"})
        
        # Merge on the 'start' column
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='start', how='outer')

# Save the merged DataFrame to a new CSV file
merged_file_path = os.path.join(data_dir, f"{username}_merged_data.csv")
merged_df.to_csv(merged_file_path, index=False, encoding="utf-8-sig")

print(f"Merged data saved to {merged_file_path}")