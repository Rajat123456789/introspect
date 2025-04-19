import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

class HealthMetricsCalculator:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the HealthMetricsCalculator.
        
        Args:
            df (pd.DataFrame): DataFrame containing health data
        """
        self.df = df
        self.heart_rate_df = df[df['type'] == 'HeartRate'].copy()
        self.step_count_df = df[df['type'] == 'StepCount'].copy()
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic health metrics."""
        metrics = {}
        
        # Heart Rate Metrics
        hr_metrics = self.heart_rate_df['value'].agg(['mean', 'std', 'min', 'max'])
        metrics.update({
            'heart_rate_mean': hr_metrics['mean'],
            'heart_rate_std': hr_metrics['std'],
            'heart_rate_min': hr_metrics['min'],
            'heart_rate_max': hr_metrics['max']
        })
        
        # Step Count Metrics
        step_metrics = self.step_count_df['value'].agg(['mean', 'std', 'min', 'max', 'sum'])
        metrics.update({
            'steps_mean': step_metrics['mean'],
            'steps_std': step_metrics['std'],
            'steps_min': step_metrics['min'],
            'steps_max': step_metrics['max'],
            'steps_total': step_metrics['sum']
        })
        
        return metrics
    
    def calculate_time_based_metrics(self) -> Dict[str, pd.DataFrame]:
        """Calculate time-based health metrics."""
        metrics = {}
        
        # Hourly metrics
        hr_hourly = self.heart_rate_df.groupby('hour')['value'].agg(['mean', 'std']).reset_index()
        steps_hourly = self.step_count_df.groupby('hour')['value'].agg(['mean', 'std']).reset_index()
        
        metrics['hourly_heart_rate'] = hr_hourly
        metrics['hourly_steps'] = steps_hourly
        
        # Daily metrics
        hr_daily = self.heart_rate_df.groupby('day_of_week')['value'].agg(['mean', 'std']).reset_index()
        steps_daily = self.step_count_df.groupby('day_of_week')['value'].agg(['mean', 'std']).reset_index()
        
        metrics['daily_heart_rate'] = hr_daily
        metrics['daily_steps'] = steps_daily
        
        return metrics
    
    def calculate_activity_intensity(self) -> pd.DataFrame:
        """Calculate activity intensity based on heart rate zones."""
        # Define heart rate zones
        zones = {
            'Rest': (0, 60),
            'Light': (60, 100),
            'Moderate': (100, 140),
            'Vigorous': (140, 180),
            'Maximum': (180, float('inf'))
        }
        
        # Calculate time spent in each zone
        zone_data = []
        for zone_name, (min_hr, max_hr) in zones.items():
            zone_time = self.heart_rate_df[
                (self.heart_rate_df['value'] >= min_hr) & 
                (self.heart_rate_df['value'] < max_hr)
            ].shape[0]
            
            zone_data.append({
                'zone': zone_name,
                'time_spent': zone_time,
                'percentage': (zone_time / len(self.heart_rate_df)) * 100
            })
        
        return pd.DataFrame(zone_data)
    
    def calculate_workout_metrics(self) -> Dict[str, float]:
        """Calculate workout-related metrics."""
        metrics = {}
        
        # Identify potential workout sessions (periods of elevated heart rate)
        hr_threshold = self.heart_rate_df['value'].mean() + self.heart_rate_df['value'].std()
        workout_periods = self.heart_rate_df[self.heart_rate_df['value'] > hr_threshold]
        
        # Calculate workout metrics
        metrics['workout_duration'] = len(workout_periods)  # in minutes
        metrics['avg_workout_heart_rate'] = workout_periods['value'].mean()
        metrics['max_workout_heart_rate'] = workout_periods['value'].max()
        
        return metrics
    
    def calculate_correlation_metrics(self) -> Dict[str, float]:
        """Calculate correlation metrics between health measurements."""
        # Merge heart rate and step count data on timestamp
        merged_data = pd.merge(
            self.heart_rate_df[['startDate', 'value']].rename(columns={'value': 'heart_rate'}),
            self.step_count_df[['startDate', 'value']].rename(columns={'value': 'steps'}),
            on='startDate',
            how='inner'
        )
        
        # Calculate correlations
        correlation = merged_data['heart_rate'].corr(merged_data['steps'])
        
        return {
            'heart_rate_steps_correlation': correlation
        }
    
    def calculate_all_metrics(self) -> Dict[str, any]:
        """Calculate all available health metrics."""
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_basic_metrics())
        
        # Time-based metrics
        metrics.update(self.calculate_time_based_metrics())
        
        # Activity intensity
        metrics['activity_intensity'] = self.calculate_activity_intensity()
        
        # Workout metrics
        metrics.update(self.calculate_workout_metrics())
        
        # Correlation metrics
        metrics.update(self.calculate_correlation_metrics())
        
        return metrics 