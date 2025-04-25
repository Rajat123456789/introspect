import os
import sys
import logging
import pandas as pd
from datetime import datetime
from visualizations.health_visualizer import HealthVisualizer
from utils.logger import HealthLogger
from utils.health_metrics import HealthMetricsCalculator
from utils.save_metrics import save_metrics_to_log

def load_data(logger: HealthLogger):
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
        
        logger.log_info(f"Heart Rate DataFrame columns: {heart_rate_df.columns.tolist()}")
        logger.log_info(f"Step Count DataFrame columns: {step_count_df.columns.tolist()}")
        
        return heart_rate_df, step_count_df
        
    except Exception as e:
        logger.log_error("Error loading data", e)
        raise

def calculate_and_log_metrics(df: pd.DataFrame, logger: HealthLogger):
    """Calculate and log health metrics."""
    try:
        calculator = HealthMetricsCalculator(df)
        metrics = calculator.calculate_all_metrics()
        
        # Log basic metrics
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.log_metric(metric_name, value)
        
        # Log activity intensity
        intensity_df = metrics['activity_intensity']
        for _, row in intensity_df.iterrows():
            logger.log_metric(
                f"activity_intensity_{row['zone']}",
                row['percentage'],
                {'time_spent': row['time_spent']}
            )
        
        return metrics
        
    except Exception as e:
        logger.log_error("Error calculating metrics", e)
        raise

def main():
    """Main function to run the visualization pipeline."""
    try:
        # Initialize logger
        logger = HealthLogger()
        logger.log_info("Starting health metrics analysis")
        
        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
        # Load data
        heart_rate_df, step_count_df = load_data(logger)
        
        # Combine DataFrames
        combined_df = pd.concat([
            heart_rate_df.assign(type='HeartRate'),
            step_count_df.assign(type='StepCount')
        ])
        
        # Calculate and log metrics
        metrics = calculate_and_log_metrics(combined_df, logger)
        
        # Save metrics to log file
        log_file_path = 'logs/health_metrics_20250419_122156.log'
        save_metrics_to_log(metrics, log_file_path)
        logger.log_info(f"Saved detailed metrics to {log_file_path}")
        
        # Initialize visualizer and generate plots
        visualizer = HealthVisualizer()
        visualizer.generate_all_visualizations(combined_df)
        
        logger.log_info("Successfully completed health metrics analysis")
        
    except Exception as e:
        if 'logger' in locals():
            logger.log_error("Error in main pipeline", e)
        raise

if __name__ == "__main__":
    main() 