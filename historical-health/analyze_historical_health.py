import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List, Any
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze historical health data')
    parser.add_argument('data_dir', nargs='?', default="data/Fit", 
                        help='Directory containing the Google Fit data (default: data/Fit)')
    return parser.parse_args()

class HistoricalHealthAnalyzer:
    def __init__(self, data_dir: str = "data/Fit"):
        """Initialize the analyzer with the paths to historical health data."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Data paths
        self.daily_metrics_dir = self.data_dir / "Daily activity metrics"
        self.activities_dir = self.data_dir / "Activities"
        self.sessions_dir = self.data_dir / "All sessions"
        self.all_data_dir = self.data_dir / "All data"
        
        # Initialize dataframes
        self.daily_metrics_df = None
        self.activity_sessions_df = None
        self.step_data_df = None
        self.heart_rate_data_df = None
        
        # Load data
        self._load_daily_metrics()
        self._load_activity_sessions()
        self._load_step_data()
        
    def _load_daily_metrics(self):
        """Load daily activity metrics from CSV files."""
        logger.info("Loading daily activity metrics data...")
        
        # List all CSV files in the daily metrics directory
        csv_files = list(self.daily_metrics_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No daily metrics CSV files found!")
            return
        
        # Read and concatenate all CSV files
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Add date column from filename
                date_str = file.stem  # Get filename without extension
                df['date'] = pd.to_datetime(date_str)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}")
        
        if dfs:
            self.daily_metrics_df = pd.concat(dfs, ignore_index=True)
            # Convert time columns to datetime
            self.daily_metrics_df['start_datetime'] = pd.to_datetime(
                self.daily_metrics_df['date'].dt.strftime('%Y-%m-%d') + ' ' + 
                self.daily_metrics_df['Start time'].str.split('-').str[0]
            )
            self.daily_metrics_df['end_datetime'] = pd.to_datetime(
                self.daily_metrics_df['date'].dt.strftime('%Y-%m-%d') + ' ' + 
                self.daily_metrics_df['End time'].str.split('-').str[0]
            )
            
            logger.info(f"Loaded {len(csv_files)} daily metrics files with {len(self.daily_metrics_df)} records")
        else:
            logger.warning("No valid daily metrics data loaded")

    def _load_activity_sessions(self):
        """Load activity sessions from JSON files."""
        logger.info("Loading activity sessions data...")
        
        # List all JSON files in the sessions directory
        json_files = list(self.sessions_dir.glob("*.json"))
        if not json_files:
            logger.warning("No activity session JSON files found!")
            return
        
        # Read and process all JSON files
        sessions = []
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract base session info
                session = {
                    'activity_type': data.get('fitnessActivity', ''),
                    'start_time': data.get('startTime', ''),
                    'end_time': data.get('endTime', ''),
                    'duration_seconds': None,
                    'distance_meters': None,
                    'calories': None,
                    'steps': None,
                    'heart_minutes': None,
                    'active_minutes': None,
                    'avg_speed': None
                }
                
                # Calculate duration
                if 'duration' in data:
                    duration_str = data['duration']
                    if duration_str.endswith('s'):
                        try:
                            session['duration_seconds'] = float(duration_str[:-1])
                        except ValueError:
                            pass
                
                # Extract aggregated metrics
                if 'aggregate' in data:
                    for metric in data['aggregate']:
                        name = metric.get('metricName', '')
                        if 'com.google.calories.expended' in name and 'floatValue' in metric:
                            session['calories'] = metric['floatValue']
                        elif 'com.google.step_count.delta' in name and 'intValue' in metric:
                            session['steps'] = metric['intValue']
                        elif 'com.google.distance.delta' in name and 'floatValue' in metric:
                            session['distance_meters'] = metric['floatValue']
                        elif 'com.google.heart_minutes.summary' in name and 'floatValue' in metric:
                            session['heart_minutes'] = metric['floatValue']
                        elif 'com.google.active_minutes' in name and 'intValue' in metric:
                            session['active_minutes'] = metric['intValue']
                        elif 'com.google.speed.summary' in name and 'floatValue' in metric:
                            session['avg_speed'] = metric['floatValue']
                
                sessions.append(session)
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
        
        if sessions:
            self.activity_sessions_df = pd.DataFrame(sessions)
            # Convert time strings to datetime - use ISO8601 format to handle different ISO date formats
            try:
                self.activity_sessions_df['start_time'] = pd.to_datetime(self.activity_sessions_df['start_time'], format='ISO8601')
                self.activity_sessions_df['end_time'] = pd.to_datetime(self.activity_sessions_df['end_time'], format='ISO8601')
                # Add date column
                self.activity_sessions_df['date'] = self.activity_sessions_df['start_time'].dt.date
                logger.info(f"Loaded {len(json_files)} activity sessions")
            except Exception as e:
                logger.error(f"Error converting datetime: {e}")
                # Fallback to just extracting the date part manually
                logger.info("Trying alternative date extraction method...")
                try:
                    # Extract just the date part with regex or string methods
                    import re
                    def extract_date(date_str):
                        if not date_str:
                            return None
                        match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
                        if match:
                            return match.group(1)
                        return None
                    
                    self.activity_sessions_df['date'] = self.activity_sessions_df['start_time'].apply(extract_date)
                    self.activity_sessions_df['date'] = pd.to_datetime(self.activity_sessions_df['date'])
                    logger.info(f"Successfully extracted dates from {len(json_files)} activity sessions")
                except Exception as e2:
                    logger.error(f"Alternative date extraction failed: {e2}")
        else:
            logger.warning("No valid activity sessions data loaded")
    
    def _load_step_data(self):
        """Extract step count data from the derived data JSON files."""
        logger.info("Loading step data...")
        
        step_files = list(self.all_data_dir.glob("*step_count.delta*.json"))
        
        # Sample a few files to analyze
        step_data = []
        
        # Try to find a file with step data that's not too large
        for file in step_files:
            if file.stat().st_size < 5 * 1024 * 1024:  # Files smaller than 5MB
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if the file has point data
                    if 'Data point' in str(data):
                        for point in data.get('Data point', []):
                            if 'value' in point and 'intVal' in point['value']:
                                entry = {
                                    'start_time': point.get('startTimeNanos', 0),
                                    'end_time': point.get('endTimeNanos', 0),
                                    'steps': point['value'].get('intVal', 0)
                                }
                                step_data.append(entry)
                except Exception as e:
                    logger.error(f"Error processing step file {file}: {e}")
        
        if step_data:
            self.step_data_df = pd.DataFrame(step_data)
            # Convert nanoseconds to datetime
            self.step_data_df['start_time'] = pd.to_datetime(self.step_data_df['start_time'], unit='ns')
            self.step_data_df['end_time'] = pd.to_datetime(self.step_data_df['end_time'], unit='ns')
            self.step_data_df['date'] = self.step_data_df['start_time'].dt.date
            logger.info(f"Loaded {len(step_data)} step data points")
        else:
            logger.warning("No valid step data loaded")
            
            # If we couldn't extract step data from JSON, use the daily metrics
            if self.daily_metrics_df is not None:
                self.step_data_df = self.daily_metrics_df[['date', 'start_datetime', 'end_datetime', 'Step count']]
                self.step_data_df = self.step_data_df.rename(columns={
                    'start_datetime': 'start_time',
                    'end_datetime': 'end_time',
                    'Step count': 'steps'
                })
                self.step_data_df = self.step_data_df.dropna(subset=['steps'])
                logger.info(f"Using {len(self.step_data_df)} step data points from daily metrics")

    def analyze_activity_metrics(self):
        """Analyze and visualize activity metrics (steps, distance, calories)."""
        if self.daily_metrics_df is None or len(self.daily_metrics_df) == 0:
            logger.error("No daily metrics data available for analysis")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Daily step count
        ax1 = plt.subplot(2, 2, 1)
        daily_steps = self.daily_metrics_df.groupby('date')['Step count'].sum().reset_index()
        sns.lineplot(data=daily_steps, x='date', y='Step count', ax=ax1)
        ax1.set_title('Daily Step Count')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Steps')
        plt.xticks(rotation=45)
        
        # 2. Step count by hour of day
        ax2 = plt.subplot(2, 2, 2)
        self.daily_metrics_df['hour'] = self.daily_metrics_df['start_datetime'].dt.hour
        hourly_steps = self.daily_metrics_df.groupby('hour')['Step count'].mean().reset_index()
        sns.barplot(data=hourly_steps, x='hour', y='Step count', ax=ax2)
        ax2.set_title('Average Steps by Hour of Day')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Steps')
        
        # 3. Daily distance
        ax3 = plt.subplot(2, 2, 3)
        daily_distance = self.daily_metrics_df.groupby('date')['Distance (m)'].sum().reset_index()
        daily_distance['Distance (km)'] = daily_distance['Distance (m)'] / 1000
        sns.lineplot(data=daily_distance, x='date', y='Distance (km)', ax=ax3)
        ax3.set_title('Daily Distance')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Distance (km)')
        plt.xticks(rotation=45)
        
        # 4. Daily calories
        ax4 = plt.subplot(2, 2, 4)
        daily_calories = self.daily_metrics_df.groupby('date')['Calories (kcal)'].sum().reset_index()
        sns.lineplot(data=daily_calories, x='date', y='Calories (kcal)', ax=ax4)
        ax4.set_title('Daily Calories Burned')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Calories (kcal)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'activity_metrics_analysis.png')
        plt.close()
        
        logger.info("Activity metrics analysis complete")

    def analyze_exercise_sessions(self):
        """Analyze and visualize exercise sessions data."""
        if self.activity_sessions_df is None or len(self.activity_sessions_df) == 0:
            logger.error("No activity sessions data available for analysis")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Exercise duration by type
        ax1 = plt.subplot(2, 2, 1)
        exercise_duration = self.activity_sessions_df.groupby('activity_type')['duration_seconds'].sum().reset_index()
        exercise_duration['duration_minutes'] = exercise_duration['duration_seconds'] / 60
        sns.barplot(data=exercise_duration, x='activity_type', y='duration_minutes', ax=ax1)
        ax1.set_title('Total Exercise Duration by Type')
        ax1.set_xlabel('Exercise Type')
        ax1.set_ylabel('Duration (minutes)')
        
        # 2. Exercise sessions count by type
        ax2 = plt.subplot(2, 2, 2)
        exercise_count = self.activity_sessions_df['activity_type'].value_counts().reset_index()
        exercise_count.columns = ['activity_type', 'count']
        sns.barplot(data=exercise_count, x='activity_type', y='count', ax=ax2)
        ax2.set_title('Number of Exercise Sessions by Type')
        ax2.set_xlabel('Exercise Type')
        ax2.set_ylabel('Count')
        
        # 3. Exercise sessions trend over time
        ax3 = plt.subplot(2, 2, 3)
        sessions_by_date = self.activity_sessions_df.groupby('date').size().reset_index(name='count')
        sns.lineplot(data=sessions_by_date, x='date', y='count', ax=ax3)
        ax3.set_title('Exercise Sessions per Day')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Sessions')
        plt.xticks(rotation=45)
        
        # 4. Average exercise duration trend
        ax4 = plt.subplot(2, 2, 4)
        avg_duration = self.activity_sessions_df.groupby('date')['duration_seconds'].mean().reset_index()
        avg_duration['duration_minutes'] = avg_duration['duration_seconds'] / 60
        sns.lineplot(data=avg_duration, x='date', y='duration_minutes', ax=ax4)
        ax4.set_title('Average Exercise Duration per Day')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Average Duration (minutes)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise_sessions_analysis.png')
        plt.close()
        
        logger.info("Exercise sessions analysis complete")

    def analyze_walking_metrics(self):
        """Analyze and visualize walking-specific metrics."""
        if self.activity_sessions_df is None or len(self.activity_sessions_df) == 0:
            logger.error("No activity sessions data available for analysis")
            return
        
        # Filter for walking activities
        walking_sessions = self.activity_sessions_df[self.activity_sessions_df['activity_type'].str.lower() == 'walking']
        
        if len(walking_sessions) == 0:
            logger.warning("No walking sessions found")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Walking distance distribution
        ax1 = plt.subplot(2, 2, 1)
        sns.histplot(data=walking_sessions, x='distance_meters', bins=20, ax=ax1)
        ax1.set_title('Walking Distance Distribution')
        ax1.set_xlabel('Distance (meters)')
        ax1.set_ylabel('Count')
        
        # 2. Walking pace (speed) distribution
        ax2 = plt.subplot(2, 2, 2)
        walking_sessions['pace_min_per_km'] = 1000 / (walking_sessions['avg_speed'] * 60) if 'avg_speed' in walking_sessions.columns else np.nan
        if not walking_sessions['pace_min_per_km'].isna().all():
            sns.histplot(data=walking_sessions, x='pace_min_per_km', bins=20, ax=ax2)
            ax2.set_title('Walking Pace Distribution')
            ax2.set_xlabel('Pace (minutes per km)')
            ax2.set_ylabel('Count')
        else:
            ax2.text(0.5, 0.5, 'No pace data available', ha='center', va='center')
        
        # 3. Walking duration vs. distance scatter plot
        ax3 = plt.subplot(2, 2, 3)
        sns.scatterplot(
            data=walking_sessions, 
            x='duration_seconds', 
            y='distance_meters',
            ax=ax3
        )
        ax3.set_title('Walking Duration vs Distance')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Distance (meters)')
        
        # 4. Calories burned vs. distance scatter plot
        ax4 = plt.subplot(2, 2, 4)
        sns.scatterplot(
            data=walking_sessions, 
            x='distance_meters', 
            y='calories',
            ax=ax4
        )
        ax4.set_title('Calories Burned vs Distance')
        ax4.set_xlabel('Distance (meters)')
        ax4.set_ylabel('Calories')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'walking_metrics_analysis.png')
        plt.close()
        
        logger.info("Walking metrics analysis complete")

    def analyze_step_patterns(self):
        """Analyze and visualize step count patterns."""
        if self.daily_metrics_df is None or len(self.daily_metrics_df) == 0:
            logger.error("No daily metrics data available for analysis")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Daily step count with moving average
        ax1 = plt.subplot(2, 2, 1)
        daily_steps = self.daily_metrics_df.groupby('date')['Step count'].sum().reset_index()
        daily_steps = daily_steps.sort_values('date')
        # Calculate 7-day moving average
        daily_steps['7_day_avg'] = daily_steps['Step count'].rolling(window=7, min_periods=1).mean()
        # Plot both raw data and moving average
        sns.lineplot(data=daily_steps, x='date', y='Step count', label='Daily Steps', alpha=0.7, ax=ax1)
        sns.lineplot(data=daily_steps, x='date', y='7_day_avg', label='7-day Moving Average', color='red', ax=ax1)
        ax1.set_title('Daily Step Count with 7-day Moving Average')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Steps')
        ax1.legend()
        plt.xticks(rotation=45)
        
        # 2. Step count by day of week
        ax2 = plt.subplot(2, 2, 2)
        self.daily_metrics_df['day_of_week'] = self.daily_metrics_df['start_datetime'].dt.day_name()
        # Order days of week correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_steps = self.daily_metrics_df.groupby('day_of_week')['Step count'].sum().reindex(day_order).reset_index()
        sns.barplot(data=weekday_steps, x='day_of_week', y='Step count', ax=ax2)
        ax2.set_title('Total Steps by Day of Week')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Total Steps')
        plt.xticks(rotation=45)
        
        # 3. Step count distribution
        ax3 = plt.subplot(2, 2, 3)
        sns.histplot(data=daily_steps, x='Step count', bins=20, ax=ax3)
        ax3.set_title('Daily Step Count Distribution')
        ax3.set_xlabel('Step Count')
        ax3.set_ylabel('Frequency')
        
        # 4. Step count heatmap by hour and day of week
        ax4 = plt.subplot(2, 2, 4)
        self.daily_metrics_df['hour'] = self.daily_metrics_df['start_datetime'].dt.hour
        hourly_weekday = self.daily_metrics_df.groupby(['day_of_week', 'hour'])['Step count'].mean().reset_index()
        hourly_weekday_pivot = hourly_weekday.pivot(index='day_of_week', columns='hour', values='Step count')
        # Reindex to get days in correct order
        hourly_weekday_pivot = hourly_weekday_pivot.reindex(day_order)
        sns.heatmap(hourly_weekday_pivot, cmap='viridis', ax=ax4)
        ax4.set_title('Average Steps by Hour and Day of Week')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'step_patterns_analysis.png')
        plt.close()
        
        logger.info("Step patterns analysis complete")

    def analyze_heart_points(self):
        """Analyze and visualize heart points and active minutes."""
        if self.daily_metrics_df is None or len(self.daily_metrics_df) == 0:
            logger.error("No daily metrics data available for analysis")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Daily heart points
        ax1 = plt.subplot(2, 2, 1)
        daily_heart_points = self.daily_metrics_df.groupby('date')['Heart Points'].sum().reset_index()
        sns.lineplot(data=daily_heart_points, x='date', y='Heart Points', ax=ax1)
        ax1.set_title('Daily Heart Points')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Heart Points')
        plt.xticks(rotation=45)
        
        # 2. Daily active minutes
        ax2 = plt.subplot(2, 2, 2)
        daily_move_minutes = self.daily_metrics_df.groupby('date')['Move Minutes count'].sum().reset_index()
        sns.lineplot(data=daily_move_minutes, x='date', y='Move Minutes count', ax=ax2)
        ax2.set_title('Daily Active Minutes')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Active Minutes')
        plt.xticks(rotation=45)
        
        # 3. Heart points by hour of day
        ax3 = plt.subplot(2, 2, 3)
        hourly_heart_points = self.daily_metrics_df.groupby('hour')['Heart Points'].mean().reset_index()
        sns.barplot(data=hourly_heart_points, x='hour', y='Heart Points', ax=ax3)
        ax3.set_title('Average Heart Points by Hour of Day')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Average Heart Points')
        
        # 4. Heart points vs. steps correlation
        ax4 = plt.subplot(2, 2, 4)
        heart_vs_steps = self.daily_metrics_df[['Heart Points', 'Step count']].dropna()
        sns.scatterplot(data=heart_vs_steps, x='Step count', y='Heart Points', ax=ax4)
        ax4.set_title('Heart Points vs. Step Count')
        ax4.set_xlabel('Step Count')
        ax4.set_ylabel('Heart Points')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heart_points_analysis.png')
        plt.close()
        
        logger.info("Heart points analysis complete")

    def create_dashboard(self):
        """Create a comprehensive dashboard with key health metrics."""
        if self.daily_metrics_df is None or len(self.daily_metrics_df) == 0:
            logger.error("No daily metrics data available for dashboard")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Daily step count trend
        ax1 = plt.subplot(2, 2, 1)
        daily_steps = self.daily_metrics_df.groupby('date')['Step count'].sum().reset_index()
        daily_steps = daily_steps.sort_values('date')
        # Calculate 7-day moving average
        daily_steps['7_day_avg'] = daily_steps['Step count'].rolling(window=7, min_periods=1).mean()
        # Plot both raw data and moving average
        sns.lineplot(data=daily_steps, x='date', y='Step count', label='Daily Steps', alpha=0.7, ax=ax1)
        sns.lineplot(data=daily_steps, x='date', y='7_day_avg', label='7-day Moving Average', color='red', ax=ax1)
        ax1.set_title('Daily Step Count')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Steps')
        ax1.legend()
        plt.xticks(rotation=45)
        
        # 2. Daily activity minutes
        ax2 = plt.subplot(2, 2, 2)
        daily_active = self.daily_metrics_df.groupby('date')['Move Minutes count'].sum().reset_index()
        daily_active = daily_active.sort_values('date')
        sns.lineplot(data=daily_active, x='date', y='Move Minutes count', ax=ax2)
        ax2.set_title('Daily Active Minutes')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Minutes')
        plt.xticks(rotation=45)
        
        # 3. Weekly activity summary
        ax3 = plt.subplot(2, 2, 3)
        # Add week column
        self.daily_metrics_df['week'] = self.daily_metrics_df['date'].dt.isocalendar().week
        self.daily_metrics_df['year'] = self.daily_metrics_df['date'].dt.isocalendar().year
        self.daily_metrics_df['year_week'] = self.daily_metrics_df['year'].astype(str) + '-' + self.daily_metrics_df['week'].astype(str)
        
        weekly_data = self.daily_metrics_df.groupby('year_week').agg({
            'Step count': 'sum',
            'Move Minutes count': 'sum',
            'Heart Points': 'sum',
            'date': 'min'  # Get first date of the week for sorting
        }).reset_index()
        
        weekly_data = weekly_data.sort_values('date')
        
        # Plot weekly data
        width = 0.25
        x = np.arange(len(weekly_data))
        
        # Normalize data for better visualization
        weekly_data['Step count'] = weekly_data['Step count'] / 1000  # Convert to thousands
        weekly_data['Move Minutes count'] = weekly_data['Move Minutes count'] / 60  # Convert to hours
        
        ax3.bar(x - width, weekly_data['Step count'], width, label='Steps (thousands)')
        ax3.bar(x, weekly_data['Move Minutes count'], width, label='Active Hours')
        ax3.bar(x + width, weekly_data['Heart Points'], width, label='Heart Points')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(weekly_data['year_week'], rotation=45)
        ax3.set_title('Weekly Activity Summary')
        ax3.set_xlabel('Year-Week')
        ax3.set_ylabel('Activity Level')
        ax3.legend()
        
        # 4. Activity distribution by hour
        ax4 = plt.subplot(2, 2, 4)
        hourly_activity = self.daily_metrics_df.groupby('hour').agg({
            'Step count': 'mean',
            'Heart Points': 'mean'
        }).reset_index()
        
        # Create a secondary y-axis
        ax4_twin = ax4.twinx()
        
        # Plot steps on primary axis
        sns.barplot(data=hourly_activity, x='hour', y='Step count', alpha=0.7, ax=ax4, color='blue')
        ax4.set_ylabel('Average Steps', color='blue')
        
        # Plot heart points on secondary axis
        sns.lineplot(data=hourly_activity, x='hour', y='Heart Points', ax=ax4_twin, color='red', marker='o')
        ax4_twin.set_ylabel('Average Heart Points', color='red')
        
        ax4.set_title('Activity Distribution by Hour of Day')
        ax4.set_xlabel('Hour of Day')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'health_dashboard.png')
        plt.close()
        
        logger.info("Health dashboard created")

def main():
    # Parse command line arguments
    args = parse_args()
    data_dir = args.data_dir
    
    # Initialize the analyzer with the data directory
    logger.info(f"Initializing analyzer with data directory: {data_dir}")
    analyzer = HistoricalHealthAnalyzer(data_dir)
    
    # Generate all analyses
    analyzer.analyze_activity_metrics()
    analyzer.analyze_exercise_sessions()
    analyzer.analyze_walking_metrics()
    analyzer.analyze_step_patterns()
    analyzer.analyze_heart_points()
    analyzer.create_dashboard()
    
    logger.info("All analyses complete!")

if __name__ == "__main__":
    main() 