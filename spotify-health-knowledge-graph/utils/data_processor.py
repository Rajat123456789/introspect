import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    @staticmethod
    def load_heart_rate_data(file_path):
        """Load and process heart rate data from CSV."""
        try:
            logger.info(f"Loading heart rate data from {file_path}")
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['track_id'] = df['track_url'].apply(lambda x: x.split(':')[-1])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Extract time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['month'] = df['timestamp'].dt.month
            df['year'] = df['timestamp'].dt.year
            
            return df
        except Exception as e:
            logger.error(f"Error loading heart rate data: {str(e)}")
            raise

    @staticmethod
    def load_step_count_data(file_path):
        """Load and process step count data from CSV."""
        try:
            logger.info(f"Loading step count data from {file_path}")
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['track_id'] = df['track_url'].apply(lambda x: x.split(':')[-1])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Extract time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['month'] = df['timestamp'].dt.month
            df['year'] = df['timestamp'].dt.year
            
            return df
        except Exception as e:
            logger.error(f"Error loading step count data: {str(e)}")
            raise

    @staticmethod
    def calculate_health_metrics(df, metric_type):
        """Calculate aggregated health metrics."""
        try:
            metrics = df.groupby(['track_id', 'track_name', 'artist_name', 'album_name'])['value'].agg([
                'mean',
                'min',
                'max',
                'std',
                'count'
            ]).reset_index()
            
            # Rename columns based on metric type
            metrics.columns = [
                'track_id',
                'track_name',
                'artist_name',
                'album_name',
                f'avg_{metric_type.lower()}',
                f'min_{metric_type.lower()}',
                f'max_{metric_type.lower()}',
                f'std_{metric_type.lower()}',
                f'count_{metric_type.lower()}'
            ]
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating health metrics: {str(e)}")
            raise

    @staticmethod
    def merge_health_data(heart_rate_df, step_count_df):
        """Merge heart rate and step count data based on track and timestamp."""
        try:
            merged_df = pd.merge(
                heart_rate_df,
                step_count_df,
                on=['timestamp', 'track_id', 'track_name', 'artist', 'album'],
                suffixes=('_heart_rate', '_step_count')
            )
            return merged_df
        except Exception as e:
            logger.error(f"Error merging health data: {str(e)}")
            raise

    @staticmethod
    def get_time_based_metrics(df, metric_column):
        """Calculate time-based metrics for a given health metric."""
        try:
            hourly_metrics = df.groupby('hour')[metric_column].agg([
                'mean', 'median', 'std', 'count'
            ]).reset_index()
            
            daily_metrics = df.groupby('day_of_week')[metric_column].agg([
                'mean', 'median', 'std', 'count'
            ]).reset_index()
            
            return hourly_metrics, daily_metrics
        except Exception as e:
            logger.error(f"Error calculating time-based metrics: {str(e)}")
            raise

    @staticmethod
    def get_artist_metrics(df, metric_column):
        """Calculate artist-based metrics for a given health metric."""
        try:
            artist_metrics = df.groupby('artist')[metric_column].agg([
                'mean', 'median', 'std', 'count'
            ]).reset_index()
            
            # Sort by count and mean to get top artists
            artist_metrics = artist_metrics.sort_values(
                by=['count', 'mean'],
                ascending=[False, False]
            )
            
            return artist_metrics
        except Exception as e:
            logger.error(f"Error calculating artist metrics: {str(e)}")
            raise

def extract_track_id(url: str) -> str:
    """Extract track ID from Spotify URL."""
    try:
        return url.split('/')[-1]
    except:
        return None

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and perform basic preprocessing
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        # Convert timestamp columns to datetime
        if 'startDate' in df.columns:
            df['startDate'] = pd.to_datetime(df['startDate'])
        if 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Extract track_id from URL
        if 'url' in df.columns:
            df['track_id'] = df['url'].apply(extract_track_id)
            
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamp columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with additional time features
    """
    df = df.copy()
    
    if 'startDate' in df.columns:
        df['hour'] = df['startDate'].dt.hour
        df['day_of_week'] = df['startDate'].dt.day_name()
        df['month'] = df['startDate'].dt.month
        df['year'] = df['startDate'].dt.year
        
    return df

def calculate_health_metrics(df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    """
    Calculate aggregated health metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        metric_type (str): Type of health metric ('HeartRate' or 'StepCount')
        
    Returns:
        pd.DataFrame: DataFrame with calculated metrics
    """
    metrics = df.groupby(['track_id', 'track_name', 'artist_name', 'album_name'])['value'].agg([
        'mean',
        'min',
        'max',
        'std',
        'count'
    ]).reset_index()
    
    # Rename columns based on metric type
    metrics.columns = [
        'track_id',
        'track_name',
        'artist_name',
        'album_name',
        f'avg_{metric_type.lower()}',
        f'min_{metric_type.lower()}',
        f'max_{metric_type.lower()}',
        f'std_{metric_type.lower()}',
        f'count_{metric_type.lower()}'
    ]
    
    return metrics

def merge_health_data(heart_rate_df: pd.DataFrame, step_count_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge heart rate and step count data.
    
    Args:
        heart_rate_df (pd.DataFrame): Heart rate data
        step_count_df (pd.DataFrame): Step count data
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Create a copy of the DataFrames to avoid modifying the originals
    hr_df = heart_rate_df.copy()
    sc_df = step_count_df.copy()
    
    # Rename value columns to be more descriptive
    hr_df = hr_df.rename(columns={'value': 'heart_rate'})
    sc_df = sc_df.rename(columns={'value': 'step_count'})
    
    # Extract time features for both DataFrames
    hr_df = extract_time_features(hr_df)
    sc_df = extract_time_features(sc_df)
    
    # Merge the DataFrames on common columns
    merged_df = pd.merge(
        hr_df[['track_id', 'track_name', 'artist_name', 'album_name', 'startDate', 'heart_rate', 'hour', 'day_of_week']],
        sc_df[['track_id', 'startDate', 'step_count']],
        on=['track_id', 'startDate'],
        how='inner'
    )
    
    return merged_df 