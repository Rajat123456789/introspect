import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import calendar
import logging

logger = logging.getLogger(__name__)

def load_data(user_id):
    """Load the combined health data."""
    file_path = Path(f"Data/{user_id}/Cleaned/combined_health_data_{user_id}.csv")
    df = pd.read_csv(file_path)
    df['start'] = pd.to_datetime(df['start'])
    df['date'] = df['start'].dt.date
    df['hour'] = df['start'].dt.hour
    df['weekday'] = df['start'].dt.day_name()
    df['month'] = df['start'].dt.month_name()
    return df

def create_heart_rate_insights(df):
    """Create comprehensive heart rate visualizations."""
    # 1. Daily Heart Rate Patterns
    plt.figure(figsize=(15, 10))
    
    # Daily average heart rate
    plt.subplot(2, 2, 1)
    daily_hr = df.groupby('date')['beatsPerMinute'].agg(['mean', 'min', 'max']).reset_index()
    plt.plot(daily_hr['date'], daily_hr['mean'], label='Average')
    plt.fill_between(daily_hr['date'], daily_hr['min'], daily_hr['max'], alpha=0.2)
    plt.title('Daily Heart Rate Patterns')
    plt.xlabel('Date')
    plt.ylabel('Heart Rate (BPM)')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Hourly heart rate distribution
    plt.subplot(2, 2, 2)
    hourly_hr = df.groupby('hour')['beatsPerMinute'].mean()
    plt.plot(hourly_hr.index, hourly_hr.values, marker='o')
    plt.title('Average Heart Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Heart Rate (BPM)')
    
    # Weekly heart rate patterns
    plt.subplot(2, 2, 3)
    weekly_hr = df.groupby('weekday')['beatsPerMinute'].mean()
    weekly_hr = weekly_hr.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    weekly_hr.plot(kind='bar')
    plt.title('Average Heart Rate by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Heart Rate (BPM)')
    
    # Heart rate distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=df, x='beatsPerMinute', bins=50)
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('analysis_output/heart_rate_insights.png')
    plt.close()

def create_activity_insights(df):
    """Create comprehensive activity visualizations."""
    plt.figure(figsize=(15, 10))
    
    # Daily steps
    plt.subplot(2, 2, 1)
    daily_steps = df.groupby('date')['steps_count'].sum().reset_index()
    plt.bar(daily_steps['date'], daily_steps['steps_count'])
    plt.title('Daily Step Count')
    plt.xlabel('Date')
    plt.ylabel('Steps')
    plt.xticks(rotation=45)
    
    # Weekly activity patterns
    plt.subplot(2, 2, 2)
    weekly_steps = df.groupby('weekday')['steps_count'].sum()
    weekly_steps = weekly_steps.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    weekly_steps.plot(kind='bar')
    plt.title('Total Steps by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Steps')
    
    # Exercise session types
    if 'exerciseSession_exerciseType' in df.columns:
        plt.subplot(2, 2, 3)
        exercise_types = df['exerciseSession_exerciseType'].value_counts()
        exercise_types.plot(kind='bar')
        plt.title('Exercise Session Types')
        plt.xlabel('Exercise Type')
        plt.ylabel('Number of Sessions')
    
    # Activity intensity over time
    plt.subplot(2, 2, 4)
    if 'speed_inMetersPerSecond' in df.columns:
        daily_speed = df.groupby('date')['speed_inMetersPerSecond'].mean().reset_index()
        plt.plot(daily_speed['date'], daily_speed['speed_inMetersPerSecond'])
        plt.title('Average Daily Speed')
        plt.xlabel('Date')
        plt.ylabel('Speed (m/s)')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_output/activity_insights.png')
    plt.close()

def create_weight_and_body_composition_insights(df):
    """Create comprehensive weight and body composition visualizations."""
    plt.figure(figsize=(15, 10))
    
    # Weight trends
    plt.subplot(2, 2, 1)
    if 'weight_inKilograms' in df.columns:
        weight_data = df[df['weight_inKilograms'].notna()]
        if len(weight_data) > 0:
            plt.plot(weight_data['date'], weight_data['weight_inKilograms'], marker='o')
            plt.title('Weight Trends')
            plt.xlabel('Date')
            plt.ylabel('Weight (kg)')
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No weight data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Weight Trends - No Data')
    
    # Body fat trends
    plt.subplot(2, 2, 2)
    if 'bodyFat_percentage' in df.columns:
        body_fat_data = df[df['bodyFat_percentage'].notna()]
        if len(body_fat_data) > 0:
            plt.plot(body_fat_data['date'], body_fat_data['bodyFat_percentage'], marker='o')
            plt.title('Body Fat Percentage Trends')
            plt.xlabel('Date')
            plt.ylabel('Body Fat (%)')
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No body fat data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Body Fat Trends - No Data')
    
    # Basal metabolic rate
    plt.subplot(2, 2, 3)
    if 'basalMetabolicRate_inKilocaloriesPerDay' in df.columns:
        bmr_data = df[df['basalMetabolicRate_inKilocaloriesPerDay'].notna()]
        if len(bmr_data) > 0:
            plt.plot(bmr_data['date'], bmr_data['basalMetabolicRate_inKilocaloriesPerDay'], marker='o')
            plt.title('Basal Metabolic Rate Trends')
            plt.xlabel('Date')
            plt.ylabel('BMR (kcal/day)')
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No BMR data available', horizontalalignment='center', verticalalignment='center')
        plt.title('BMR Trends - No Data')
    
    # Correlation between weight and body fat
    plt.subplot(2, 2, 4)
    if 'bodyFat_percentage' in df.columns and 'weight_inKilograms' in df.columns:
        combined_data = df[['weight_inKilograms', 'bodyFat_percentage']].dropna()
        if len(combined_data) > 0:
            plt.scatter(combined_data['weight_inKilograms'], combined_data['bodyFat_percentage'])
            plt.title('Weight vs Body Fat Percentage')
            plt.xlabel('Weight (kg)')
            plt.ylabel('Body Fat (%)')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for correlation', horizontalalignment='center', verticalalignment='center')
        plt.title('Weight vs Body Fat - No Data')
    
    plt.tight_layout()
    plt.savefig('analysis_output/weight_and_body_composition_insights.png')
    plt.close()

def create_nutrition_insights(df):
    """Create comprehensive nutrition visualizations."""
    plt.figure(figsize=(15, 10))
    
    # Daily protein intake
    plt.subplot(2, 2, 1)
    if 'protein_inGrams' in df.columns:
        daily_protein = df.groupby('date')['protein_inGrams'].sum().reset_index()
        if not daily_protein.empty:
            plt.plot(daily_protein['date'], daily_protein['protein_inGrams'], marker='o')
            plt.title('Daily Protein Intake')
            plt.xlabel('Date')
            plt.ylabel('Protein (g)')
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No protein data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Protein Intake - No Data')
    
    # Daily calories
    plt.subplot(2, 2, 2)
    if 'calories_inKilocalories' in df.columns:
        daily_calories = df.groupby('date')['calories_inKilocalories'].sum().reset_index()
        if not daily_calories.empty:
            plt.plot(daily_calories['date'], daily_calories['calories_inKilocalories'], marker='o')
            plt.title('Daily Calorie Intake')
            plt.xlabel('Date')
            plt.ylabel('Calories (kcal)')
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No calorie data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Calorie Intake - No Data')
    
    # Macronutrient distribution
    plt.subplot(2, 2, 3)
    required_macros = ['protein_inGrams', 'carbohydrate_inGrams', 'fat_inGrams']
    if all(col in df.columns for col in required_macros):
        macro_data = df[required_macros].mean().dropna()
        if not macro_data.empty and macro_data.sum() > 0:
            plt.pie(macro_data, labels=['Protein', 'Carbohydrates', 'Fat'], autopct='%1.1f%%')
            plt.title('Average Macronutrient Distribution')
        else:
            plt.text(0.5, 0.5, 'Insufficient macronutrient data', horizontalalignment='center', verticalalignment='center')
            plt.title('Macronutrients - Insufficient Data')
    else:
        plt.text(0.5, 0.5, 'No macronutrient data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Macronutrients - No Data')
    
    # Weekly nutrition patterns
    plt.subplot(2, 2, 4)
    if 'calories_inKilocalories' in df.columns:
        weekly_calories = df.groupby('weekday')['calories_inKilocalories'].mean()
        if not weekly_calories.empty:
            weekly_calories = weekly_calories.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            weekly_calories.plot(kind='bar')
            plt.title('Average Daily Calories by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Average Calories (kcal)')
    else:
        plt.text(0.5, 0.5, 'No calorie data available', horizontalalignment='center', verticalalignment='center')
        plt.title('Weekly Calories - No Data')
    
    plt.tight_layout()
    plt.savefig('analysis_output/nutrition_insights.png')
    plt.close()

def create_correlation_matrix(df):
    """Create correlation matrix for key health metrics."""
    # Select only the most important health metrics
    key_metrics = {
        'steps_count': 'Steps',
        'beatsPerMinute': 'Heart Rate',
        'weight_weight_inKilograms': 'Weight',
        'bodyFat_percentage': 'Body Fat %',
        'nutrition_energy_inKilocalories': 'Calories',
        'nutrition_protein_inGrams': 'Protein',
        'nutrition_totalCarbohydrate_inGrams': 'Carbs',
        'nutrition_totalFat_inGrams': 'Fat',
        'exerciseSession_total_time': 'Exercise Duration',
        'basalMetabolicRate_inKilocaloriesPerDay': 'BMR',
        'totalCaloriesBurned_energy_inKilocalories': 'Calories Burned',
        'distance_distance_inMeters': 'Distance'
    }
    
    # Create a dataframe with only the metrics we want
    metrics_df = pd.DataFrame()
    for col, name in key_metrics.items():
        if col in df.columns:
            metrics_df[name] = df[col]
    
    # Drop any columns that have all NaN values
    metrics_df = metrics_df.dropna(axis=1, how='all')
    
    # Calculate correlation matrix
    correlation_matrix = metrics_df.corr()
    
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
    
    # Add title with padding
    plt.title('Correlation Matrix of Key Health Metrics', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high DPI for better quality
    plt.savefig('analysis_output/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_advanced_insights(df):
    """Create advanced health insights visualizations."""
    plt.figure(figsize=(20, 25))
    
    # 1. Daily Energy Balance
    plt.subplot(4, 2, 1)
    if 'nutrition_energy_inKilocalories' in df.columns and 'totalCaloriesBurned_energy_inKilocalories' in df.columns:
        daily_energy = df.groupby('date').agg({
            'nutrition_energy_inKilocalories': 'sum',
            'totalCaloriesBurned_energy_inKilocalories': 'sum'
        }).reset_index()
        daily_energy['balance'] = daily_energy['nutrition_energy_inKilocalories'] - daily_energy['totalCaloriesBurned_energy_inKilocalories']
        colors = ['green' if x < 0 else 'red' for x in daily_energy['balance']]
        plt.bar(daily_energy['date'], daily_energy['balance'], color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Daily Energy Balance (Calories)')
        plt.xlabel('Date')
        plt.ylabel('Surplus/Deficit (kcal)')
        plt.xticks(rotation=45)
    
    # 2. Sleep Efficiency Score
    plt.subplot(4, 2, 2)
    sleep_stages = [col for col in df.columns if 'sleep_stage_' in col]
    if sleep_stages:
        sleep_data = df[sleep_stages].sum()
        total_sleep = sleep_data.sum()
        if total_sleep > 0:
            efficiency = (total_sleep - sleep_data['sleep_stage_1']) / total_sleep * 100
            plt.figure(plt.gcf().number)
            gauge = plt.pie([efficiency, 100-efficiency], colors=['green', 'lightgray'],
                          startangle=90, counterclock=False)
            plt.title(f'Sleep Efficiency: {efficiency:.1f}%')
    
    # 3. Resting Heart Rate Tracker
    plt.subplot(4, 2, 3)
    if 'beatsPerMinute' in df.columns:
        daily_min_hr = df.groupby('date')['beatsPerMinute'].min().reset_index()
        daily_min_hr['7day_avg'] = daily_min_hr['beatsPerMinute'].rolling(7).mean()
        baseline = daily_min_hr['7day_avg'].iloc[0] if not daily_min_hr.empty else 0
        alert_days = daily_min_hr[daily_min_hr['7day_avg'] > baseline + 5]['date']
        
        plt.plot(daily_min_hr['date'], daily_min_hr['beatsPerMinute'], 'b-', label='Daily Min HR')
        plt.plot(daily_min_hr['date'], daily_min_hr['7day_avg'], 'r-', label='7-day Avg')
        if not alert_days.empty:
            plt.scatter(alert_days, [baseline + 5] * len(alert_days), color='red', marker='^', label='Alert')
        plt.title('Resting Heart Rate Trend')
        plt.xlabel('Date')
        plt.ylabel('BPM')
        plt.legend()
        plt.xticks(rotation=45)
    
    # 4. Macro Breakdown
    plt.subplot(4, 2, 4)
    macro_cols = ['nutrition_totalCarbohydrate_inGrams', 'nutrition_totalFat_inGrams', 'nutrition_protein_inGrams']
    if all(col in df.columns for col in macro_cols):
        # Convert to calories (4 kcal/g for protein and carbs, 9 kcal/g for fat)
        macros = df[macro_cols].mean()
        calories = pd.Series({
            'Carbs': macros['nutrition_totalCarbohydrate_inGrams'] * 4,
            'Protein': macros['nutrition_protein_inGrams'] * 4,
            'Fat': macros['nutrition_totalFat_inGrams'] * 9
        })
        plt.pie(calories, labels=calories.index, autopct='%1.1f%%')
        plt.title('Macronutrient Distribution (% of Calories)')
    
    # 5. Training Load Heatmap
    plt.subplot(4, 2, 5)
    if 'exerciseSession_exerciseType' in df.columns and 'exerciseSession_total_time' in df.columns:
        df['weekday'] = df['start'].dt.day_name()
        df['week'] = df['start'].dt.isocalendar().week
        training_load = df.pivot_table(
            values='exerciseSession_total_time',
            index='week',
            columns='weekday',
            aggfunc='sum'
        ).fillna(0)
        sns.heatmap(training_load, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title('Weekly Training Load (Minutes)')
    
    # 6. Body Fat vs Weight Scatter
    plt.subplot(4, 2, 6)
    if 'bodyFat_percentage' in df.columns and 'weight_weight_inKilograms' in df.columns:
        # Get valid data points (non-null values)
        valid_data = df[['weight_weight_inKilograms', 'bodyFat_percentage']].dropna()
        
        if len(valid_data) >= 2:  # Need at least 2 points for a trend line
            plt.scatter(valid_data['weight_weight_inKilograms'], valid_data['bodyFat_percentage'])
            plt.xlabel('Weight (kg)')
            plt.ylabel('Body Fat %')
            plt.title('Body Composition Changes')
            
            try:
                # Add trend line with clean data
                z = np.polyfit(valid_data['weight_weight_inKilograms'], valid_data['bodyFat_percentage'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid_data['weight_weight_inKilograms'].min(), 
                                    valid_data['weight_weight_inKilograms'].max(), 
                                    100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8)
            except np.linalg.LinAlgError:
                logger.warning("Could not generate trend line for body composition plot")
        else:
            plt.text(0.5, 0.5, 'Insufficient data points\nfor body composition analysis',
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title('Body Composition Changes - Insufficient Data')
    else:
        plt.text(0.5, 0.5, 'No body composition data available',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Body Composition Changes - No Data')
    
    # 7. BMR vs Intake Dial
    plt.subplot(4, 2, 7)
    if 'basalMetabolicRate_inKilocaloriesPerDay' in df.columns and 'nutrition_energy_inKilocalories' in df.columns:
        bmr = df['basalMetabolicRate_inKilocaloriesPerDay'].mean()
        intake = df['nutrition_energy_inKilocalories'].mean()
        if bmr > 0:
            percentage = (intake / bmr) * 100
            colors = ['green' if percentage < 110 else 'yellow' if percentage < 120 else 'red']
            plt.pie([percentage, max(0, 200-percentage)], colors=colors + ['lightgray'],
                   startangle=90, counterclock=False)
            plt.title(f'Daily Intake vs BMR: {percentage:.1f}%')
    
    plt.tight_layout()
    plt.savefig('analysis_output/advanced_insights.png')
    plt.close()

def main():
    user_id = "someshbgd3"
    print(f"Generating comprehensive health insights for user: {user_id}")
    
    try:
        # Load data
        df = load_data(user_id)
        
        # Create output directory
        Path("analysis_output").mkdir(exist_ok=True)
        
        # Generate all visualizations including the new advanced insights
        if 'beatsPerMinute' in df.columns:
            create_heart_rate_insights(df)
            print("✅ Heart rate insights generated")
        else:
            print("⚠️ No heart rate data available, skipping heart rate insights")
            
        if 'steps_count' in df.columns:
            create_activity_insights(df)
            print("✅ Activity insights generated")
        else:
            print("⚠️ No activity data available, skipping activity insights")
            
        create_weight_and_body_composition_insights(df)
        print("✅ Weight and body composition insights generated")
        
        create_nutrition_insights(df)
        print("✅ Nutrition insights generated")
        
        create_advanced_insights(df)
        print("✅ Advanced insights generated")
        
        # Only generate correlation matrix if we have some numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            create_correlation_matrix(df)
            print("✅ Correlation matrix generated")
        else:
            print("⚠️ Not enough numeric data for correlation matrix")
        
        print("\nComprehensive health insights have been generated!")
        print("Visualizations have been saved to the analysis_output directory.")
        
        # Print some key insights
        print("\nKey Insights:")
        print(f"Data Collection Period: {df['date'].min()} to {df['date'].max()}")
        print(f"Total Days of Data: {(df['date'].max() - df['date'].min()).days}")
        
        if 'beatsPerMinute' in df.columns:
            print(f"\nHeart Rate Statistics:")
            print(f"Average: {df['beatsPerMinute'].mean():.1f} BPM")
            print(f"Maximum: {df['beatsPerMinute'].max():.1f} BPM")
            print(f"Minimum: {df['beatsPerMinute'].min():.1f} BPM")
            
        if 'steps_count' in df.columns:
            daily_steps = df.groupby('date')['steps_count'].sum()
            print(f"\nActivity Statistics:")
            print(f"Average Daily Steps: {daily_steps.mean():.0f}")
            print(f"Most Active Day: {daily_steps.max():.0f} steps")
            print(f"Least Active Day: {daily_steps.min():.0f} steps")
            
        if 'total_sleep_time' in df.columns:
            sleep_data = df[df['total_sleep_time'].notna()]
            if len(sleep_data) > 0:
                print(f"\nSleep Statistics:")
                print(f"Average Sleep Duration: {sleep_data['total_sleep_time'].mean() / 60:.1f} hours")
    except Exception as e:
        print(f"❌ Error generating health insights: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 