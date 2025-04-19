import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthVisualizer:
    def __init__(self, output_dir: str = 'visualizations'):
        """Initialize the HealthVisualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def save_plot(self, name: str):
        """Save the current plot to the output directory."""
        path = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Saved plot: {path}")
    
    def plot_metric_distribution(self, df: pd.DataFrame, metric_type: str):
        """Plot distribution of a health metric."""
        metric_df = df[df['type'] == metric_type]
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data=metric_df, x='value', kde=True)
        
        if metric_type == 'HeartRate':
            plt.xlabel('Heart Rate (bpm)')
            title = 'Heart Rate Distribution'
        else:
            plt.xlabel('Steps')
            title = 'Step Count Distribution'
            
        plt.title(title)
        self.save_plot(f"{metric_type.lower()}_distribution")
    
    def plot_metrics_by_hour(self, df: pd.DataFrame):
        """Plot average metrics by hour."""
        plt.figure(figsize=(15, 6))
        
        for metric_type in ['HeartRate', 'StepCount']:
            metric_df = df[df['type'] == metric_type]
            hourly_avg = metric_df.groupby('hour')['value'].agg(['mean', 'std']).reset_index()
            
            plt.errorbar(
                hourly_avg['hour'],
                hourly_avg['mean'],
                yerr=hourly_avg['std'],
                label=metric_type,
                capsize=5
            )
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Value')
        plt.title('Health Metrics by Hour')
        plt.legend()
        plt.xticks(range(0, 24))
        self.save_plot('metrics_by_hour')
    
    def plot_metrics_by_day(self, df: pd.DataFrame):
        """Plot average metrics by day of week."""
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        plt.figure(figsize=(15, 6))
        
        for metric_type in ['HeartRate', 'StepCount']:
            metric_df = df[df['type'] == metric_type]
            daily_avg = metric_df.groupby('day_of_week')['value'].agg(['mean', 'std']).reindex(days_order)
            
            plt.errorbar(
                range(len(days_order)),
                daily_avg['mean'],
                yerr=daily_avg['std'],
                label=metric_type,
                capsize=5
            )
        
        plt.xlabel('Day of Week')
        plt.ylabel('Value')
        plt.title('Health Metrics by Day')
        plt.legend()
        plt.xticks(range(len(days_order)), days_order, rotation=45)
        self.save_plot('metrics_by_day')
    
    def plot_metrics_by_artist(self, df: pd.DataFrame, top_n: int = 10):
        """Plot average metrics by top artists."""
        plt.figure(figsize=(15, 8))
        
        for metric_type in ['HeartRate', 'StepCount']:
            metric_df = df[df['type'] == metric_type]
            artist_avg = metric_df.groupby('artistName')['value'].agg(['mean', 'count'])
            top_artists = artist_avg.nlargest(top_n, 'count')
            
            plt.errorbar(
                range(len(top_artists)),
                top_artists['mean'],
                label=metric_type,
                capsize=5
            )
        
        plt.xlabel('Artist')
        plt.ylabel('Value')
        plt.title(f'Health Metrics by Top {top_n} Artists')
        plt.legend()
        plt.xticks(range(len(top_artists)), top_artists.index, rotation=45, ha='right')
        self.save_plot('metrics_by_artist')
    
    def generate_all_visualizations(self, df: pd.DataFrame):
        """Generate all visualizations from the combined DataFrame."""
        try:
            # Plot distributions
            self.plot_metric_distribution(df, 'HeartRate')
            self.plot_metric_distribution(df, 'StepCount')
            
            # Plot time-based metrics
            self.plot_metrics_by_hour(df)
            self.plot_metrics_by_day(df)
            
            # Plot artist-based metrics
            if 'artistName' in df.columns:
                self.plot_metrics_by_artist(df)
            
            logging.info("Successfully generated all visualizations")
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
            raise 