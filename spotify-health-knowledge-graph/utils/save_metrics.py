import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any

def save_metrics_to_log(metrics: Dict[str, Any], log_file_path: str):
    """
    Save health metrics to a log file.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing health metrics
        log_file_path (str): Path to the log file
    """
    with open(log_file_path, 'w') as f:
        # Write header
        f.write(f"Health Metrics Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic Metrics
        f.write("BASIC METRICS\n")
        f.write("-" * 40 + "\n")
        basic_metrics = {
            'heart_rate_mean': 'Average Heart Rate (bpm)',
            'heart_rate_std': 'Heart Rate Standard Deviation',
            'heart_rate_min': 'Minimum Heart Rate (bpm)',
            'heart_rate_max': 'Maximum Heart Rate (bpm)',
            'steps_mean': 'Average Steps',
            'steps_std': 'Steps Standard Deviation',
            'steps_min': 'Minimum Steps',
            'steps_max': 'Maximum Steps',
            'steps_total': 'Total Steps'
        }
        
        for key, description in basic_metrics.items():
            if key in metrics:
                f.write(f"{description}: {metrics[key]:.2f}\n")
        
        # Activity Intensity
        f.write("\nACTIVITY INTENSITY\n")
        f.write("-" * 40 + "\n")
        if 'activity_intensity' in metrics:
            intensity_df = metrics['activity_intensity']
            for _, row in intensity_df.iterrows():
                f.write(f"{row['zone']} Activity: {row['percentage']:.2f}% "
                       f"(Time spent: {row['time_spent']} minutes)\n")
        
        # Workout Metrics
        f.write("\nWORKOUT METRICS\n")
        f.write("-" * 40 + "\n")
        workout_metrics = {
            'workout_duration': 'Total Workout Duration (minutes)',
            'avg_workout_heart_rate': 'Average Workout Heart Rate (bpm)',
            'max_workout_heart_rate': 'Maximum Workout Heart Rate (bpm)'
        }
        
        for key, description in workout_metrics.items():
            if key in metrics:
                f.write(f"{description}: {metrics[key]:.2f}\n")
        
        # Correlation Metrics
        f.write("\nCORRELATION METRICS\n")
        f.write("-" * 40 + "\n")
        if 'heart_rate_steps_correlation' in metrics:
            f.write(f"Heart Rate vs Steps Correlation: "
                   f"{metrics['heart_rate_steps_correlation']:.3f}\n")
        
        # Time-based Metrics
        f.write("\nTIME-BASED METRICS\n")
        f.write("-" * 40 + "\n")
        
        if 'hourly_heart_rate' in metrics:
            f.write("\nHourly Heart Rate Averages:\n")
            hr_hourly = metrics['hourly_heart_rate']
            for _, row in hr_hourly.iterrows():
                hour = int(row['hour'])  # Convert to integer
                f.write(f"Hour {hour:02d}:00 - "
                       f"Mean: {row['mean']:.2f} bpm, "
                       f"Std: {row['std']:.2f}\n")
        
        if 'daily_heart_rate' in metrics:
            f.write("\nDaily Heart Rate Averages:\n")
            hr_daily = metrics['daily_heart_rate']
            for _, row in hr_daily.iterrows():
                f.write(f"{row['day_of_week']}: "
                       f"Mean: {row['mean']:.2f} bpm, "
                       f"Std: {row['std']:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Health Metrics Log\n") 