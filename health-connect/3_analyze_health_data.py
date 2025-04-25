import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HealthDataAnalyzer:
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all data files
        self.sleep_data = pd.read_csv(self.data_dir / 'sleepSession_someshbgd3_Cleaned.csv')
        self.exercise_data = pd.read_csv(self.data_dir / 'exerciseSession_someshbgd3_Cleaned.csv')
        self.heart_rate_data = pd.read_csv(self.data_dir / 'heartRate_someshbgd3_Cleaned.csv')
        self.weight_data = pd.read_csv(self.data_dir / 'weight_someshbgd3_Cleaned.csv')
        self.body_fat_data = pd.read_csv(self.data_dir / 'bodyFat_someshbgd3_Cleaned.csv')
        self.nutrition_data = pd.read_csv(self.data_dir / 'nutrition_someshbgd3_Cleaned.csv')
        self.steps_data = pd.read_csv(self.data_dir / 'steps_someshbgd3_Cleaned.csv')
        self.calories_data = pd.read_csv(self.data_dir / 'totalCaloriesBurned_someshbgd3_Cleaned.csv')
        
        # Convert datetime columns
        self._convert_datetime_columns()
        
    def _convert_datetime_columns(self):
        """Convert string datetime columns to pandas datetime objects"""
        datetime_cols = ['start', 'end']
        
        for df in [self.sleep_data, self.exercise_data, self.heart_rate_data,
                  self.weight_data, self.body_fat_data, self.nutrition_data,
                  self.steps_data, self.calories_data]:
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    
    def analyze_sleep(self):
        """Analyze sleep patterns and create visualizations"""
        # Calculate sleep duration in hours
        self.sleep_data['sleep_duration'] = (self.sleep_data['end'] - self.sleep_data['start']).dt.total_seconds() / 3600
        
        # Convert datetime to numeric for regression
        self.sleep_data['days_since_start'] = (self.sleep_data['start'] - self.sleep_data['start'].min()).dt.total_seconds() / (24 * 3600)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Sleep duration trend
        ax1 = plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.sleep_data, x='days_since_start', y='sleep_duration', ax=ax1)
        sns.regplot(data=self.sleep_data, x='days_since_start', y='sleep_duration', 
                   scatter=False, ax=ax1, color='red')
        ax1.set_title('Sleep Duration Trend')
        ax1.set_xlabel('Days Since First Record')
        ax1.set_ylabel('Sleep Duration (hours)')
        
        # 2. Sleep stage distribution
        ax2 = plt.subplot(2, 2, 2)
        sleep_stages = self.sleep_data[[f'sleep_stage_{i}' for i in range(1, 9)]].mean()
        sleep_stages.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Time in Each Sleep Stage')
        ax2.set_xlabel('Sleep Stage')
        ax2.set_ylabel('Average Duration (minutes)')
        
        # 3. Sleep schedule consistency
        ax3 = plt.subplot(2, 2, 3)
        self.sleep_data['start_hour'] = self.sleep_data['start'].dt.hour
        self.sleep_data['end_hour'] = self.sleep_data['end'].dt.hour
        sns.boxplot(data=self.sleep_data[['start_hour', 'end_hour']], ax=ax3)
        ax3.set_title('Sleep Schedule Consistency')
        ax3.set_ylabel('Hour of Day')
        
        # 4. Sleep duration distribution
        ax4 = plt.subplot(2, 2, 4)
        sns.histplot(data=self.sleep_data, x='sleep_duration', bins=20, ax=ax4)
        ax4.set_title('Sleep Duration Distribution')
        ax4.set_xlabel('Sleep Duration (hours)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sleep_analysis.png')
        plt.close()

    def analyze_activity(self):
        """Analyze activity patterns and create visualizations"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Daily step count trends
        ax1 = plt.subplot(2, 2, 1)
        daily_steps = self.steps_data.groupby(self.steps_data['start'].dt.date)['steps_count'].sum()
        sns.lineplot(data=daily_steps, ax=ax1)
        ax1.set_title('Daily Step Count')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Steps')
        
        # 2. Exercise session types distribution
        ax2 = plt.subplot(2, 2, 2)
        exercise_types = self.exercise_data['exerciseSession_exerciseType'].value_counts()
        exercise_types.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Exercise Session Types Distribution')
        
        # 3. Exercise duration by type
        ax3 = plt.subplot(2, 2, 3)
        exercise_duration = self.exercise_data.groupby('exerciseSession_exerciseType')['exerciseSession_total_time'].mean()
        exercise_duration.plot(kind='bar', ax=ax3)
        ax3.set_title('Average Exercise Duration by Type')
        ax3.set_xlabel('Exercise Type')
        ax3.set_ylabel('Duration (minutes)')
        
        # 4. Activity patterns by time of day
        ax4 = plt.subplot(2, 2, 4)
        self.exercise_data['hour'] = self.exercise_data['start'].dt.hour
        hourly_activity = self.exercise_data.groupby('hour')['exerciseSession_total_time'].mean()
        sns.barplot(x=hourly_activity.index, y=hourly_activity.values, ax=ax4)
        ax4.set_title('Activity Patterns by Hour of Day')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Average Duration (minutes)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'activity_analysis.png')
        plt.close()

    def analyze_body_composition(self):
        """Analyze body composition metrics and create visualizations"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Weight trends with moving average
        ax1 = plt.subplot(2, 2, 1)
        self.weight_data.sort_values('start', inplace=True)
        plt.plot(self.weight_data['start'], self.weight_data['weight_weight_inKilograms'], 'o-', label='Weight')
        plt.title('Weight Trend')
        plt.xlabel('Date')
        plt.ylabel('Weight (kg)')
        plt.xticks(rotation=45)
        plt.legend()
        
        # 2. Body fat percentage vs weight correlation
        ax2 = plt.subplot(2, 2, 2)
        plt.plot(self.body_fat_data['start'], self.body_fat_data['bodyFat_percentage'], 'o-', label='Body Fat %')
        plt.title('Body Fat Percentage Trend')
        plt.xlabel('Date')
        plt.ylabel('Body Fat %')
        plt.xticks(rotation=45)
        plt.legend()
        
        # 3. BMI trend
        ax3 = plt.subplot(2, 2, 3)
        height_m = 1.75  # Assuming average height, adjust as needed
        self.weight_data['bmi'] = self.weight_data['weight_weight_inKilograms'] / (height_m ** 2)
        plt.plot(self.weight_data['start'], self.weight_data['bmi'], 'o-', label='BMI')
        plt.axhline(y=18.5, color='r', linestyle='--', alpha=0.5, label='Underweight threshold')
        plt.axhline(y=24.9, color='r', linestyle='--', alpha=0.5, label='Normal weight threshold')
        plt.title('BMI Trend')
        plt.xlabel('Date')
        plt.ylabel('BMI')
        plt.xticks(rotation=45)
        plt.legend()
        
        # 4. Weight distribution
        ax4 = plt.subplot(2, 2, 4)
        plt.hist(self.weight_data['weight_weight_inKilograms'], bins=20)
        plt.title('Weight Distribution')
        plt.xlabel('Weight (kg)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'body_composition_analysis.png')
        plt.close()

    def analyze_nutrition(self):
        """Analyze nutrition patterns and create visualizations"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Macronutrient distribution
        ax1 = plt.subplot(2, 2, 1)
        macro_cols = ['nutrition_totalCarbohydrate_inGrams', 'nutrition_totalFat_inGrams', 'nutrition_protein_inGrams']
        macro_avg = self.nutrition_data[macro_cols].mean()
        macro_avg.plot(kind='pie', ax=ax1, autopct='%1.1f%%')
        ax1.set_title('Average Macronutrient Distribution')
        
        # 2. Calorie intake vs burned
        ax2 = plt.subplot(2, 2, 2)
        calories_merged = pd.merge_asof(
            self.nutrition_data.sort_values('start')[['start', 'nutrition_energy_inKilocalories']],
            self.calories_data.sort_values('start')[['start', 'totalCaloriesBurned_energy_inKilocalories']],
            on='start',
            direction='nearest'
        )
        sns.scatterplot(data=calories_merged, x='nutrition_energy_inKilocalories', y='totalCaloriesBurned_energy_inKilocalories', ax=ax2)
        ax2.set_title('Calories Consumed vs Burned')
        ax2.set_xlabel('Calories Consumed')
        ax2.set_ylabel('Calories Burned')
        
        # 3. Nutrient intake trends
        ax3 = plt.subplot(2, 2, 3)
        nutrient_cols = ['nutrition_totalCarbohydrate_inGrams', 'nutrition_totalFat_inGrams', 'nutrition_protein_inGrams']
        for col in nutrient_cols:
            sns.lineplot(data=self.nutrition_data, x='start', y=col, 
                        label=col.replace('nutrition_', '').replace('_inGrams', ''), ax=ax3)
        ax3.set_title('Nutrient Intake Trends')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Grams')
        
        # 4. Meal timing patterns
        ax4 = plt.subplot(2, 2, 4)
        self.nutrition_data['hour'] = self.nutrition_data['start'].dt.hour
        meal_timing = self.nutrition_data.groupby('hour')['nutrition_energy_inKilocalories'].mean()
        sns.barplot(x=meal_timing.index, y=meal_timing.values, ax=ax4)
        ax4.set_title('Average Calorie Intake by Hour')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Average Calories')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'nutrition_analysis.png')
        plt.close()

    def analyze_cardiovascular(self):
        """Analyze cardiovascular health metrics and create visualizations"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Heart rate zones during exercise
        ax1 = plt.subplot(2, 2, 1)
        exercise_hr = pd.merge_asof(
            self.exercise_data.sort_values('start'),
            self.heart_rate_data.sort_values('start'),
            on='start',
            direction='nearest'
        )
        sns.boxplot(data=exercise_hr, x='exerciseSession_exerciseType', y='beatsPerMinute', ax=ax1)
        ax1.set_title('Heart Rate Zones by Exercise Type')
        ax1.set_xlabel('Exercise Type')
        ax1.set_ylabel('Heart Rate (bpm)')
        plt.xticks(rotation=45)
        
        # 2. Resting heart rate trend
        ax2 = plt.subplot(2, 2, 2)
        daily_min_hr = self.heart_rate_data.groupby(
            self.heart_rate_data['start'].dt.date)['beatsPerMinute'].min()
        sns.lineplot(data=daily_min_hr, ax=ax2)
        ax2.set_title('Daily Resting Heart Rate Trend')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Heart Rate (bpm)')
        
        # 3. Heart rate distribution
        ax3 = plt.subplot(2, 2, 3)
        sns.histplot(data=self.heart_rate_data, x='beatsPerMinute', bins=30, ax=ax3)
        ax3.set_title('Heart Rate Distribution')
        ax3.set_xlabel('Heart Rate (bpm)')
        ax3.set_ylabel('Count')
        
        # 4. Heart rate recovery after exercise
        ax4 = plt.subplot(2, 2, 4)
        # This would require more detailed time-series analysis
        # For now, showing average heart rate by hour
        self.heart_rate_data['hour'] = self.heart_rate_data['start'].dt.hour
        hourly_hr = self.heart_rate_data.groupby('hour')['beatsPerMinute'].mean()
        sns.lineplot(data=hourly_hr, ax=ax4)
        ax4.set_title('Average Heart Rate by Hour')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Average Heart Rate (bpm)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cardiovascular_analysis.png')
        plt.close()

    def create_combined_dashboard(self):
        """Create a combined health metrics dashboard"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Steps over time
        ax1 = plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.steps_data, x='start', y='steps_count', ax=ax1)
        ax1.set_title('Daily Steps')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Steps')
        plt.xticks(rotation=45)
        
        # 2. Weight over time
        ax2 = plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.weight_data, x='start', y='weight_weight_inKilograms', ax=ax2)
        ax2.set_title('Weight Trend')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Weight (kg)')
        plt.xticks(rotation=45)
        
        # 3. Exercise duration by type
        ax3 = plt.subplot(2, 2, 3)
        exercise_duration = self.exercise_data.groupby('exerciseSession_exerciseType')['exerciseSession_total_time'].mean()
        exercise_duration.plot(kind='bar', ax=ax3)
        ax3.set_title('Average Exercise Duration by Type')
        ax3.set_xlabel('Exercise Type')
        ax3.set_ylabel('Duration (minutes)')
        plt.xticks(rotation=45)
        
        # 4. Average heart rate by exercise type
        ax4 = plt.subplot(2, 2, 4)
        sns.boxplot(data=self.heart_rate_data, x='beatsPerMinute', ax=ax4)
        ax4.set_title('Heart Rate Distribution')
        ax4.set_xlabel('Beats Per Minute')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_dashboard.png')
        plt.close()

    def analyze_fitness_progress(self):
        """Analyze fitness progress metrics and create visualizations"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Exercise intensity over time
        ax1 = plt.subplot(2, 2, 1)
        self.exercise_data['intensity'] = self.exercise_data['exerciseSession_total_time']  # Using duration as proxy for intensity
        weekly_intensity = self.exercise_data.groupby(
            pd.Grouper(key='start', freq='W'))['intensity'].mean()
        sns.lineplot(data=weekly_intensity, ax=ax1)
        ax1.set_title('Weekly Exercise Intensity Trend')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Average Duration (minutes)')
        
        # 2. Workout duration distribution
        ax2 = plt.subplot(2, 2, 2)
        sns.histplot(data=self.exercise_data, x='exerciseSession_total_time', bins=20, ax=ax2)
        ax2.set_title('Workout Duration Distribution')
        ax2.set_xlabel('Duration (minutes)')
        ax2.set_ylabel('Count')
        
        # 3. Exercise type progression
        ax3 = plt.subplot(2, 2, 3)
        exercise_progression = self.exercise_data.pivot_table(
            index=pd.Grouper(key='start', freq='W'),
            columns='exerciseSession_exerciseType',
            values='exerciseSession_total_time',
            aggfunc='sum'
        ).fillna(0)
        exercise_progression.plot(kind='area', stacked=True, ax=ax3)
        ax3.set_title('Weekly Exercise Type Progression')
        ax3.set_xlabel('Week')
        ax3.set_ylabel('Total Duration (minutes)')
        
        # 4. Recovery patterns
        ax4 = plt.subplot(2, 2, 4)
        self.exercise_data['next_exercise'] = self.exercise_data['start'].shift(-1)
        self.exercise_data['recovery_time'] = (
            self.exercise_data['next_exercise'] - self.exercise_data['end']
        ).dt.total_seconds() / 3600  # Convert to hours
        sns.boxplot(
            data=self.exercise_data[self.exercise_data['recovery_time'] < 48],  # Filter out long gaps
            x='exerciseSession_exerciseType',
            y='recovery_time',
            ax=ax4
        )
        ax4.set_title('Recovery Time Between Exercises')
        ax4.set_xlabel('Exercise Type')
        ax4.set_ylabel('Recovery Time (hours)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fitness_progress.png')
        plt.close()

    def create_correlation_matrix(self, df):
        """Create correlation matrix for numerical health metrics."""
        # Select relevant numerical columns and give them readable names
        metric_columns = {
            'steps_count': 'Steps',
            'beatsPerMinute': 'Heart Rate',
            'sleep_duration': 'Sleep Duration',
            'weight_weight_inKilograms': 'Weight',
            'bodyFat_percentage': 'Body Fat %',
            'nutrition_energy_inKilocalories': 'Calories',
            'nutrition_protein_inGrams': 'Protein',
            'nutrition_totalCarbohydrate_inGrams': 'Carbs',
            'nutrition_totalFat_inGrams': 'Fat',
            'exerciseSession_total_time': 'Exercise Duration'
        }
        
        # Select only the most important metrics
        df_subset = df[metric_columns.keys()].rename(columns=metric_columns)
        
        # Create correlation matrix
        correlation_matrix = df_subset.corr()
        
        # Create figure with larger size
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with improved visibility
        sns.heatmap(correlation_matrix, 
                    annot=True,              # Show correlation values
                    fmt='.2f',               # Round to 2 decimal places
                    cmap='RdBu_r',          # Red-Blue diverging colormap
                    center=0,                # Center the colormap at 0
                    square=True,             # Make cells square
                    linewidths=0.5,          # Add cell borders
                    cbar_kws={"shrink": .8}, # Adjust colorbar size
                    vmin=-1, vmax=1)         # Fix scale from -1 to 1
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add title
        plt.title('Health Metrics Correlation Matrix', pad=20)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Initialize the analyzer with the data directory
    analyzer = HealthDataAnalyzer('Data/someshbgd3/Cleaned')
    
    # Generate all analyses
    analyzer.analyze_sleep()
    analyzer.analyze_activity()
    analyzer.analyze_body_composition()
    analyzer.analyze_nutrition()
    analyzer.analyze_cardiovascular()
    analyzer.create_combined_dashboard()
    analyzer.analyze_fitness_progress()
    analyzer.create_correlation_matrix(pd.concat([analyzer.sleep_data, analyzer.exercise_data, analyzer.heart_rate_data,
                                                analyzer.weight_data, analyzer.body_fat_data, analyzer.nutrition_data,
                                                analyzer.steps_data, analyzer.calories_data]))

if __name__ == "__main__":
    main() 