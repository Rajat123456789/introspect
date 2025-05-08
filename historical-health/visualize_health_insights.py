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
import calendar

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
    parser = argparse.ArgumentParser(description='Visualize historical health insights')
    parser.add_argument('data_dir', nargs='?', default="data/Fit", 
                        help='Directory containing the Google Fit data (default: data/Fit)')
    return parser.parse_args()

class HistoricalHealthVisualizer:
    def __init__(self, data_dir: str = "data/Fit"):
        """Initialize the visualizer with paths to data."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
        # Load processed data
        self.combined_data = self._load_combined_data()
        
    def _load_combined_data(self):
        """Load and combine all health data sources."""
        combined_data = {}
        
        # Load daily metrics
        daily_metrics = self._load_daily_metrics()
        if daily_metrics is not None:
            combined_data['daily_metrics'] = daily_metrics
            
        # Load activity sessions
        activity_sessions = self._load_activity_sessions()
        if activity_sessions is not None:
            combined_data['activity_sessions'] = activity_sessions
        
        return combined_data
    
    def _load_daily_metrics(self):
        """Load daily activity metrics from CSV files."""
        logger.info("Loading daily activity metrics data...")
        
        daily_metrics_dir = self.data_dir / "Daily activity metrics"
        # List all CSV files in the daily metrics directory
        csv_files = list(daily_metrics_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No daily metrics CSV files found!")
            return None
        
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
            daily_metrics_df = pd.concat(dfs, ignore_index=True)
            # Convert time columns to datetime
            daily_metrics_df['start_datetime'] = pd.to_datetime(
                daily_metrics_df['date'].dt.strftime('%Y-%m-%d') + ' ' + 
                daily_metrics_df['Start time'].str.split('-').str[0]
            )
            daily_metrics_df['end_datetime'] = pd.to_datetime(
                daily_metrics_df['date'].dt.strftime('%Y-%m-%d') + ' ' + 
                daily_metrics_df['End time'].str.split('-').str[0]
            )
            
            # Add derived columns for analysis
            daily_metrics_df['hour'] = daily_metrics_df['start_datetime'].dt.hour
            daily_metrics_df['day_of_week'] = daily_metrics_df['start_datetime'].dt.day_name()
            daily_metrics_df['month'] = daily_metrics_df['start_datetime'].dt.month_name()
            
            logger.info(f"Loaded {len(csv_files)} daily metrics files with {len(daily_metrics_df)} records")
            return daily_metrics_df
        else:
            logger.warning("No valid daily metrics data loaded")
            return None
    
    def _load_activity_sessions(self):
        """Load activity sessions from JSON files."""
        logger.info("Loading activity sessions data...")
        
        sessions_dir = self.data_dir / "All sessions"
        # List all JSON files in the sessions directory
        json_files = list(sessions_dir.glob("*.json"))
        if not json_files:
            logger.warning("No activity session JSON files found!")
            return None
        
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
            activities_df = pd.DataFrame(sessions)
            # Convert time strings to datetime - use ISO8601 format to handle different ISO date formats
            try:
                activities_df['start_time'] = pd.to_datetime(activities_df['start_time'], format='ISO8601')
                activities_df['end_time'] = pd.to_datetime(activities_df['end_time'], format='ISO8601')
                # Add date column
                activities_df['date'] = activities_df['start_time'].dt.date
                
                # Add derived columns for analysis
                activities_df['hour'] = activities_df['start_time'].dt.hour
                activities_df['day_of_week'] = activities_df['start_time'].dt.day_name()
                activities_df['month'] = activities_df['start_time'].dt.month_name()
                
                logger.info(f"Loaded {len(json_files)} activity sessions")
                return activities_df
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
                    
                    activities_df['date'] = activities_df['start_time'].apply(extract_date)
                    activities_df['date'] = pd.to_datetime(activities_df['date'])
                    
                    # Since we don't have proper datetime objects for start_time and end_time,
                    # let's create placeholder hour, day_of_week and month columns
                    activities_df['hour'] = 12  # Default to noon
                    activities_df['day_of_week'] = activities_df['date'].dt.day_name()
                    activities_df['month'] = activities_df['date'].dt.month_name()
                    
                    logger.info(f"Successfully extracted dates from {len(json_files)} activity sessions")
                    return activities_df
                except Exception as e2:
                    logger.error(f"Alternative date extraction failed: {e2}")
                    return None
        else:
            logger.warning("No valid activity sessions data loaded")
            return None

    def create_activity_insights(self):
        """Create comprehensive activity visualizations."""
        if 'daily_metrics' not in self.combined_data:
            logger.error("No daily metrics data available for activity insights")
            return
        
        df = self.combined_data['daily_metrics']
        
        plt.figure(figsize=(15, 10))
        
        # Daily steps
        plt.subplot(2, 2, 1)
        daily_steps = df.groupby('date')['Step count'].sum().reset_index()
        daily_steps = daily_steps.sort_values('date')
        plt.bar(daily_steps['date'], daily_steps['Step count'])
        plt.title('Daily Step Count')
        plt.xlabel('Date')
        plt.ylabel('Steps')
        plt.xticks(rotation=45)
        
        # Weekly activity patterns
        plt.subplot(2, 2, 2)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_steps = df.groupby('day_of_week')['Step count'].sum().reindex(day_order).reset_index()
        plt.bar(weekly_steps['day_of_week'], weekly_steps['Step count'])
        plt.title('Total Steps by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Total Steps')
        plt.xticks(rotation=45)
        
        # Activity by hour
        plt.subplot(2, 2, 3)
        hourly_steps = df.groupby('hour')['Step count'].mean().reset_index()
        plt.bar(hourly_steps['hour'], hourly_steps['Step count'])
        plt.title('Average Steps by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Steps')
        
        # Distance traveled trend
        plt.subplot(2, 2, 4)
        daily_distance = df.groupby('date')['Distance (m)'].sum().reset_index()
        daily_distance = daily_distance.sort_values('date')
        # Convert to kilometers for better readability
        daily_distance['Distance (km)'] = daily_distance['Distance (m)'] / 1000
        plt.plot(daily_distance['date'], daily_distance['Distance (km)'], marker='o')
        plt.title('Daily Distance Traveled')
        plt.xlabel('Date')
        plt.ylabel('Distance (km)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'activity_insights.png')
        plt.close()
        
        logger.info("Activity insights visualization created")

    def create_exercise_insights(self):
        """Create comprehensive exercise session visualizations."""
        if 'activity_sessions' not in self.combined_data:
            logger.error("No activity sessions data available for exercise insights")
            return
        
        df = self.combined_data['activity_sessions']
        
        plt.figure(figsize=(15, 10))
        
        # Exercise duration by type
        plt.subplot(2, 2, 1)
        exercise_duration = df.groupby('activity_type')['duration_seconds'].sum().reset_index()
        exercise_duration['duration_minutes'] = exercise_duration['duration_seconds'] / 60
        plt.bar(exercise_duration['activity_type'], exercise_duration['duration_minutes'])
        plt.title('Total Exercise Duration by Type')
        plt.xlabel('Exercise Type')
        plt.ylabel('Duration (minutes)')
        plt.xticks(rotation=45)
        
        # Exercise sessions by day of week
        plt.subplot(2, 2, 2)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sessions_by_day = df.groupby('day_of_week').size().reindex(day_order).reset_index(name='count')
        plt.bar(sessions_by_day['day_of_week'], sessions_by_day['count'])
        plt.title('Exercise Sessions by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Sessions')
        plt.xticks(rotation=45)
        
        # Exercise duration trend
        plt.subplot(2, 2, 3)
        daily_duration = df.groupby('date')['duration_seconds'].sum().reset_index()
        daily_duration = daily_duration.sort_values('date')
        daily_duration['duration_minutes'] = daily_duration['duration_seconds'] / 60
        plt.plot(daily_duration['date'], daily_duration['duration_minutes'], marker='o')
        plt.title('Daily Exercise Duration')
        plt.xlabel('Date')
        plt.ylabel('Duration (minutes)')
        plt.xticks(rotation=45)
        
        # Exercise by hour of day
        plt.subplot(2, 2, 4)
        hourly_exercise = df.groupby('hour').size().reset_index(name='count')
        plt.bar(hourly_exercise['hour'], hourly_exercise['count'])
        plt.title('Exercise Sessions by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Sessions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise_insights.png')
        plt.close()
        
        logger.info("Exercise insights visualization created")
    
    def create_steps_insights(self):
        """Create detailed step count visualizations."""
        if 'daily_metrics' not in self.combined_data:
            logger.error("No daily metrics data available for steps insights")
            return
        
        df = self.combined_data['daily_metrics']
        
        plt.figure(figsize=(15, 10))
        
        # Daily steps with moving average
        plt.subplot(2, 2, 1)
        daily_steps = df.groupby('date')['Step count'].sum().reset_index()
        daily_steps = daily_steps.sort_values('date')
        # Calculate 7-day moving average
        daily_steps['7_day_avg'] = daily_steps['Step count'].rolling(window=7, min_periods=1).mean()
        # Plot both raw data and moving average
        plt.plot(daily_steps['date'], daily_steps['Step count'], label='Daily Steps', alpha=0.7)
        plt.plot(daily_steps['date'], daily_steps['7_day_avg'], label='7-day Moving Average', color='red')
        plt.title('Daily Step Count with Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Steps')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Step count distribution
        plt.subplot(2, 2, 2)
        plt.hist(daily_steps['Step count'], bins=20)
        plt.title('Daily Step Count Distribution')
        plt.xlabel('Steps')
        plt.ylabel('Frequency')
        
        # Steps heatmap by hour and day
        plt.subplot(2, 2, 3)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_steps = df.groupby(['day_of_week', 'hour'])['Step count'].mean().reset_index()
        step_pivot = hourly_steps.pivot(index='day_of_week', columns='hour', values='Step count')
        step_pivot = step_pivot.reindex(day_order)
        sns.heatmap(step_pivot, cmap='viridis')
        plt.title('Average Steps by Hour and Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        
        # Step count vs distance correlation
        plt.subplot(2, 2, 4)
        step_distance = df.groupby('date').agg({'Step count': 'sum', 'Distance (m)': 'sum'}).reset_index()
        # Convert distance to kilometers for better visualization
        step_distance['Distance (km)'] = step_distance['Distance (m)'] / 1000
        plt.scatter(step_distance['Step count'], step_distance['Distance (km)'])
        # Calculate and plot trend line
        if len(step_distance) > 1:
            z = np.polyfit(step_distance['Step count'], step_distance['Distance (km)'], 1)
            p = np.poly1d(z)
            plt.plot(step_distance['Step count'], p(step_distance['Step count']), "r--")
        plt.title('Steps vs Distance Correlation')
        plt.xlabel('Step Count')
        plt.ylabel('Distance (km)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'steps_insights.png')
        plt.close()
        
        logger.info("Steps insights visualization created")

    def create_heart_points_insights(self):
        """Create heart points and active minutes visualizations."""
        if 'daily_metrics' not in self.combined_data:
            logger.error("No daily metrics data available for heart points insights")
            return
        
        df = self.combined_data['daily_metrics']
        
        plt.figure(figsize=(15, 10))
        
        # Daily heart points
        plt.subplot(2, 2, 1)
        daily_heart = df.groupby('date')['Heart Points'].sum().reset_index()
        daily_heart = daily_heart.sort_values('date')
        plt.plot(daily_heart['date'], daily_heart['Heart Points'], marker='o')
        plt.title('Daily Heart Points')
        plt.xlabel('Date')
        plt.ylabel('Heart Points')
        plt.xticks(rotation=45)
        
        # Daily active minutes
        plt.subplot(2, 2, 2)
        daily_active = df.groupby('date')['Move Minutes count'].sum().reset_index()
        daily_active = daily_active.sort_values('date')
        plt.plot(daily_active['date'], daily_active['Move Minutes count'], marker='o')
        plt.title('Daily Active Minutes')
        plt.xlabel('Date')
        plt.ylabel('Active Minutes')
        plt.xticks(rotation=45)
        
        # Heart points by day of week
        plt.subplot(2, 2, 3)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_heart = df.groupby('day_of_week')['Heart Points'].sum().reindex(day_order).reset_index()
        plt.bar(weekly_heart['day_of_week'], weekly_heart['Heart Points'])
        plt.title('Total Heart Points by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Heart Points')
        plt.xticks(rotation=45)
        
        # Heart points vs active minutes correlation
        plt.subplot(2, 2, 4)
        heart_active = df.groupby('date').agg({'Heart Points': 'sum', 'Move Minutes count': 'sum'}).reset_index()
        plt.scatter(heart_active['Move Minutes count'], heart_active['Heart Points'])
        # Calculate and plot trend line
        if len(heart_active) > 1:
            z = np.polyfit(heart_active['Move Minutes count'].fillna(0), heart_active['Heart Points'].fillna(0), 1)
            p = np.poly1d(z)
            plt.plot(heart_active['Move Minutes count'], p(heart_active['Move Minutes count']), "r--")
        plt.title('Heart Points vs Active Minutes Correlation')
        plt.xlabel('Active Minutes')
        plt.ylabel('Heart Points')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heart_points_insights.png')
        plt.close()
        
        logger.info("Heart points insights visualization created")

    def create_calories_insights(self):
        """Create calories burned visualizations."""
        if 'daily_metrics' not in self.combined_data:
            logger.error("No daily metrics data available for calories insights")
            return
        
        df = self.combined_data['daily_metrics']
        
        plt.figure(figsize=(15, 10))
        
        # Daily calories burned
        plt.subplot(2, 2, 1)
        daily_calories = df.groupby('date')['Calories (kcal)'].sum().reset_index()
        daily_calories = daily_calories.sort_values('date')
        plt.plot(daily_calories['date'], daily_calories['Calories (kcal)'], marker='o')
        plt.title('Daily Calories Burned')
        plt.xlabel('Date')
        plt.ylabel('Calories (kcal)')
        plt.xticks(rotation=45)
        
        # Calories by hour of day
        plt.subplot(2, 2, 2)
        hourly_calories = df.groupby('hour')['Calories (kcal)'].mean().reset_index()
        plt.bar(hourly_calories['hour'], hourly_calories['Calories (kcal)'])
        plt.title('Average Calories Burned by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Calories (kcal)')
        
        # Calories vs steps correlation
        plt.subplot(2, 2, 3)
        cal_steps = df.groupby('date').agg({'Calories (kcal)': 'sum', 'Step count': 'sum'}).reset_index()
        plt.scatter(cal_steps['Step count'], cal_steps['Calories (kcal)'])
        # Calculate and plot trend line
        if len(cal_steps) > 1:
            z = np.polyfit(cal_steps['Step count'].fillna(0), cal_steps['Calories (kcal)'].fillna(0), 1)
            p = np.poly1d(z)
            plt.plot(cal_steps['Step count'], p(cal_steps['Step count']), "r--")
        plt.title('Calories vs Steps Correlation')
        plt.xlabel('Step Count')
        plt.ylabel('Calories (kcal)')
        
        # Calories distribution
        plt.subplot(2, 2, 4)
        plt.hist(daily_calories['Calories (kcal)'], bins=20)
        plt.title('Daily Calories Distribution')
        plt.xlabel('Calories (kcal)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calories_insights.png')
        plt.close()
        
        logger.info("Calories insights visualization created")

    def create_comprehensive_dashboard(self):
        """Create a comprehensive health metrics dashboard."""
        if 'daily_metrics' not in self.combined_data:
            logger.error("No daily metrics data available for comprehensive dashboard")
            return
        
        df = self.combined_data['daily_metrics']
        
        plt.figure(figsize=(20, 15))
        
        # 1. Daily health metrics overview
        plt.subplot(2, 2, 1)
        
        # Calculate daily metrics
        daily_data = df.groupby('date').agg({
            'Step count': 'sum',
            'Heart Points': 'sum',
            'Move Minutes count': 'sum'
        }).reset_index()
        daily_data = daily_data.sort_values('date')
        
        # Normalize data for better visualization on the same scale
        daily_data['Step count (normalized)'] = daily_data['Step count'] / daily_data['Step count'].max() * 100
        daily_data['Heart Points (normalized)'] = daily_data['Heart Points'] / daily_data['Heart Points'].max() * 100 if daily_data['Heart Points'].max() > 0 else 0
        daily_data['Move Minutes (normalized)'] = daily_data['Move Minutes count'] / daily_data['Move Minutes count'].max() * 100 if daily_data['Move Minutes count'].max() > 0 else 0
        
        # Plot normalized metrics
        plt.plot(daily_data['date'], daily_data['Step count (normalized)'], label='Steps %', color='blue')
        plt.plot(daily_data['date'], daily_data['Heart Points (normalized)'], label='Heart Points %', color='red')
        plt.plot(daily_data['date'], daily_data['Move Minutes (normalized)'], label='Active Minutes %', color='green')
        
        plt.title('Daily Health Metrics (% of Maximum)')
        plt.xlabel('Date')
        plt.ylabel('Percentage of Maximum')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 2. Weekly activity summary
        plt.subplot(2, 2, 2)
        
        # Add week column
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.isocalendar().year
        df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str)
        
        weekly_data = df.groupby('year_week').agg({
            'Step count': 'sum',
            'Move Minutes count': 'sum',
            'Heart Points': 'sum',
            'date': 'min'  # Get first date of the week for sorting
        }).reset_index()
        
        weekly_data = weekly_data.sort_values('date')
        
        # Calculate weekly averages
        steps_weekly_avg = weekly_data['Step count'].mean()
        active_weekly_avg = weekly_data['Move Minutes count'].mean()
        heart_weekly_avg = weekly_data['Heart Points'].mean()
        
        # Plot weekly metrics with reference lines for averages
        plt.bar(weekly_data['year_week'], weekly_data['Step count'], alpha=0.3, label='Steps')
        plt.axhline(y=steps_weekly_avg, color='blue', linestyle='--', alpha=0.7, label=f'Avg Steps: {steps_weekly_avg:.0f}')
        
        # Create twin axes for the other metrics to have different scales
        ax2 = plt.twinx()
        ax2.plot(weekly_data['year_week'], weekly_data['Heart Points'], 'ro-', label='Heart Points')
        ax2.axhline(y=heart_weekly_avg, color='red', linestyle='--', alpha=0.7, label=f'Avg HP: {heart_weekly_avg:.0f}')
        ax2.set_ylabel('Heart Points', color='red')
        
        ax3 = plt.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(weekly_data['year_week'], weekly_data['Move Minutes count'], 'go-', label='Active Minutes')
        ax3.axhline(y=active_weekly_avg, color='green', linestyle='--', alpha=0.7, label=f'Avg Min: {active_weekly_avg:.0f}')
        ax3.set_ylabel('Active Minutes', color='green')
        
        plt.title('Weekly Activity Summary')
        plt.xlabel('Year-Week')
        plt.ylabel('Steps')
        plt.xticks(rotation=45)
        
        # Combine legends
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        # 3. Activity heatmap by hour and day
        plt.subplot(2, 2, 3)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_activity = df.groupby(['day_of_week', 'hour'])['Step count'].mean().reset_index()
        activity_pivot = hourly_activity.pivot(index='day_of_week', columns='hour', values='Step count')
        activity_pivot = activity_pivot.reindex(day_order)
        sns.heatmap(activity_pivot, cmap='viridis', annot=False)
        plt.title('Average Steps by Hour and Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        
        # 4. Correlation matrix of key metrics
        plt.subplot(2, 2, 4)
        
        # Correlations between key metrics
        key_metrics = df.groupby('date').agg({
            'Step count': 'sum',
            'Heart Points': 'sum',
            'Move Minutes count': 'sum',
            'Calories (kcal)': 'sum',
            'Distance (m)': 'sum'
        }).reset_index()
        
        key_metrics = key_metrics.rename(columns={
            'Step count': 'Steps',
            'Heart Points': 'Heart Points',
            'Move Minutes count': 'Active Minutes',
            'Calories (kcal)': 'Calories',
            'Distance (m)': 'Distance'
        })
        
        # Calculate correlation matrix
        corr_matrix = key_metrics.drop('date', axis=1).corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Health Metrics')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_dashboard.png')
        plt.close()
        
        logger.info("Comprehensive dashboard created")

def main():
    # Parse command line arguments
    args = parse_args()
    data_dir = args.data_dir
    
    # Initialize the visualizer with data directory
    logger.info(f"Initializing visualizer with data directory: {data_dir}")
    visualizer = HistoricalHealthVisualizer(data_dir)
    
    # Generate all visualizations
    visualizer.create_activity_insights()
    visualizer.create_exercise_insights()
    visualizer.create_steps_insights()
    visualizer.create_heart_points_insights()
    visualizer.create_calories_insights()
    visualizer.create_comprehensive_dashboard()
    
    logger.info("All visualizations complete!")

if __name__ == "__main__":
    main() 