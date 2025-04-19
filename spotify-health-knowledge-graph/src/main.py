import os
import sys
import logging
import pandas as pd
from datetime import datetime
from visualizations.health_visualizer import HealthVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():
    """Load and preprocess the health data."""
    try:
        # Define file paths
        heart_rate_path = '../combining-apple-spotify/final-dataset-apple-spotify/spotifyHeartRate.csv'
        step_count_path = '../combining-apple-spotify/final-dataset-apple-spotify/spotifyStepCount.csv'
        
        # Load data
        heart_rate_df = pd.read_csv(heart_rate_path)
        step_count_df = pd.read_csv(step_count_path)
        
        # Convert timestamps
        for df in [heart_rate_df, step_count_df]:
            df['startDate'] = pd.to_datetime(df['startDate'])
            df['hour'] = df['startDate'].dt.hour
            df['day_of_week'] = df['startDate'].dt.day_name()
        
        logging.info(f"Heart Rate DataFrame columns: {heart_rate_df.columns.tolist()}")
        logging.info(f"Step Count DataFrame columns: {step_count_df.columns.tolist()}")
        
        return heart_rate_df, step_count_df
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def main():
    """Main function to run the visualization pipeline."""
    try:
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Load data
        heart_rate_df, step_count_df = load_data()
        
        # Initialize visualizer
        visualizer = HealthVisualizer()
        
        # Combine DataFrames
        combined_df = pd.concat([
            heart_rate_df.assign(type='HeartRate'),
            step_count_df.assign(type='StepCount')
        ])
        
        # Generate visualizations
        visualizer.generate_all_visualizations(combined_df)
        
        logging.info("Successfully generated all visualizations!")
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 