import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta
from pathlib import Path
import os

class HealthDataProcessor:
    def __init__(self, data_dir="Data"):
        self.data_dir = data_dir
        self.metrics_dict = {
            "basalMetabolicRate": ["_id", "id", "end"],
            "bodyFat": ["_id", "id", "end"],
            "distance": ["_id", "id"],
            "exerciseSession": ["_id", "id"],
            "elevationGained": ["_id", "id"],
            "floorsClimbed": ["_id", "id"],
            "heartRate": ["_id", "id", "end", "start"],
            "height": ["_id", "id", "end"],
            "nutrition": ["_id", "id", "end"],
            "oxygenSaturation": ["_id", "id", "end"],
            "sleepSession": ["_id", "id"],
            "speed": ["_id", "id"],
            "steps": ["_id", "id", "end"],
            "totalCaloriesBurned": ["_id", "id"],
            "weight": ["_id", "id", "end"]
        }

    def read_csv(self, username, metric):
        """Read CSV file with error handling and validation."""
        try:
            file_path = Path(self.data_dir) / username / "Uncleaned" / f"{metric}_{username}.csv"
            if not file_path.exists():
                print(f"❌ {metric} CSV file does not exist")
                return None
            
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"❌ {metric} CSV file is empty")
                return None
                
            return df
        except Exception as e:
            print(f"❌ Error reading {metric} CSV: {str(e)}")
            return None

    def expand_data_column(self, df, metric):
        """Expands the data column into separate columns with proper error handling."""
        if df is None or df.empty:
            return pd.DataFrame()

        expanded_rows = []
        for _, row in df.iterrows():
            try:
                if "data" not in row:
                    continue
                    
                data_dict = ast.literal_eval(row["data"].replace("'", "\""))
                flattened_data = {}

                for key, value in data_dict.items():
                    if isinstance(value, dict):  
                        for sub_key, sub_value in value.items():
                            flattened_data[f"{metric}_{key}_{sub_key}"] = sub_value
                    else:
                        flattened_data[f"{metric}_{key}"] = value
                
                new_row = row.to_dict()
                new_row.pop("data")
                new_row.update(flattened_data)
                expanded_rows.append(new_row)
            except Exception as e:
                print(f"❌ Error processing row: {row['data']} - {e}")

        if not expanded_rows:
            return pd.DataFrame()

        df_expanded = pd.DataFrame(expanded_rows)
        return self._process_datetime_columns(df_expanded, metric)

    def _process_datetime_columns(self, df, metric):
        """Process datetime columns and calculate durations."""
        datetime_cols = ["start", "end"]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
                df[col] = df[col].dt.round("min")
        
        if "start" in df.columns and "end" in df.columns:
            df[f"{metric}_total_time"] = (df["end"] - df["start"]).dt.total_seconds() / 60
        
        return df

    def process_heart_rate(self, df):
        """Process heart rate data with improved error handling."""
        if df is None or df.empty:
            return pd.DataFrame()

        expanded_rows = []
        for _, row in df.iterrows():
            try:
                if "data" not in row:
                    continue

                data_parsed = ast.literal_eval(row["data"].replace("'", "\""))
                if not isinstance(data_parsed, dict) or "samples" not in data_parsed:
                    continue

                for sample in data_parsed["samples"]:
                    new_row = row.to_dict()
                    new_row.pop("data")
                    new_row.update(sample)
                    expanded_rows.append(new_row)
            except Exception as e:
                print(f"❌ Error processing heart rate row: {e}")

        if not expanded_rows:
            return pd.DataFrame()

        df_expanded = pd.DataFrame(expanded_rows)
        if "time" in df_expanded.columns:
            df_expanded["time"] = pd.to_datetime(df_expanded["time"], errors="coerce")
            df_expanded = df_expanded.sort_values(by="time")
            df_expanded["minute"] = df_expanded["time"].dt.round("min")

            df_grouped = df_expanded.groupby(["app", "minute"], as_index=False).agg(
                beatsPerMinute=("beatsPerMinute", lambda x: round(np.mean(x)))
            )
            df_grouped.rename(columns={"minute": "start"}, inplace=True)
            return df_grouped

        return df_expanded

    def process_sleep_data(self, df):
        """Process sleep data with improved validation."""
        if df is None or df.empty:
            return pd.DataFrame()

        new_rows = []
        for _, row in df.iterrows():
            try:
                if "data" not in row:
                    continue

                sleep_data = ast.literal_eval(row["data"].replace("'", "\""))
                sleep_stage_times = {f"sleep_stage_{i}": timedelta(0) for i in range(1, 9)}

                if isinstance(sleep_data, dict) and "stages" in sleep_data:
                    for stage_info in sleep_data["stages"]:
                        try:
                            start_time = datetime.fromisoformat(stage_info["startTime"].replace("Z", ""))
                            end_time = datetime.fromisoformat(stage_info["endTime"].replace("Z", ""))
                            duration = end_time - start_time
                            stage_key = f"sleep_stage_{stage_info['stage']}"
                            if stage_key in sleep_stage_times:
                                sleep_stage_times[stage_key] += duration
                        except (KeyError, ValueError) as e:
                            print(f"❌ Error processing sleep stage: {e}")
                            continue

                row_data = row.to_dict()
                row_data.pop("data", None)
                
                # Convert timedelta to minutes
                for stage, duration in sleep_stage_times.items():
                    row_data[stage] = duration.total_seconds() / 60

                # Calculate total sleep time
                if "start" in row and "end" in row:
                    try:
                        start_time = datetime.fromisoformat(row["start"].replace("Z", ""))
                        end_time = datetime.fromisoformat(row["end"].replace("Z", ""))
                        row_data["total_sleep_time"] = (end_time - start_time).total_seconds() / 60
                    except ValueError:
                        row_data["total_sleep_time"] = None

                new_rows.append(row_data)
            except Exception as e:
                print(f"❌ Error processing sleep row: {e}")

        if not new_rows:
            return pd.DataFrame()

        df_expanded = pd.DataFrame(new_rows)
        return self._process_datetime_columns(df_expanded, "sleep")

    def clean_and_save_data(self, username, metric):
        """Clean and save data for a specific metric."""
        try:
            df = self.read_csv(username, metric)
            if df is None:
                return

            # Drop unnecessary columns
            if metric in self.metrics_dict:
                df = df.drop(columns=self.metrics_dict[metric], errors='ignore')

            # Process data based on metric type
            if metric == "heartRate":
                df = self.process_heart_rate(df)
            elif metric == "sleepSession":
                df = self.process_sleep_data(df)
            else:
                df = self.expand_data_column(df, metric)

            if df.empty:
                print(f"No valid data to save for {metric}")
                return

            # Ensure output directory exists
            output_dir = Path(self.data_dir) / username / "Cleaned"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save processed data
            output_file = output_dir / f"{metric}_{username}_Cleaned.csv"
            df.to_csv(output_file, index=False)
            print(f"✅ Successfully saved cleaned {metric} data")

        except Exception as e:
            print(f"❌ Error processing {metric}: {str(e)}")

    def combine_cleaned_data(self, username):
        """Combine all cleaned data files into a single CSV file."""
        try:
            cleaned_dir = Path(self.data_dir) / username / "Cleaned"
            if not cleaned_dir.exists():
                print("❌ Cleaned directory does not exist")
                return

            # Get all cleaned CSV files
            cleaned_files = list(cleaned_dir.glob(f"*_{username}_Cleaned.csv"))
            if not cleaned_files:
                print("❌ No cleaned files found")
                return

            # Read and process each file
            dfs = []
            for file in cleaned_files:
                try:
                    df = pd.read_csv(file)
                    if not df.empty and 'start' in df.columns:
                        df['start'] = pd.to_datetime(df['start'])
                        dfs.append(df)
                    elif not df.empty:
                        print(f"⚠️ Skipping {file.name} - no 'start' column")
                    else:
                        print(f"⚠️ Empty file: {file.name}")
                except Exception as e:
                    print(f"❌ Error reading {file.name}: {str(e)}")

            if not dfs:
                print("❌ No valid data frames to combine")
                return

            # Combine all dataframes
            combined_df = pd.concat(dfs, axis=0)
            combined_df = combined_df.sort_values('start').reset_index(drop=True)

            # Save combined data
            output_file = cleaned_dir / f"combined_health_data_{username}.csv"
            combined_df.to_csv(output_file, index=False)
            print(f"✅ Successfully created combined health data file at {output_file}")

        except Exception as e:
            print(f"❌ Error combining data: {str(e)}")

def main():
    processor = HealthDataProcessor()
    username = "someshbgd3"  # or get from environment/config
    
    metrics = [
        "basalMetabolicRate", "bodyFat", "distance", "exerciseSession",
        "elevationGained", "floorsClimbed", "heartRate", "height",
        "nutrition", "oxygenSaturation", "sleepSession", "speed",
        "steps", "totalCaloriesBurned", "weight"
    ]
    
    for metric in metrics:
        print(f"\nProcessing {metric}...")
        processor.clean_and_save_data(username, metric)
    
    print("\nCombining all cleaned data files...")
    processor.combine_cleaned_data(username)

if __name__ == "__main__":
    main() 