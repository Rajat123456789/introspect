import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating and preprocessing health and music data."""
    
    REQUIRED_HEART_RATE_COLUMNS = ['timestamp', 'heart_rate', 'track_url']
    REQUIRED_STEP_COUNT_COLUMNS = ['timestamp', 'step_count', 'track_url']
    
    @staticmethod
    def validate_heart_rate_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate heart rate data format and content.
        
        Args:
            df (pd.DataFrame): Heart rate DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check required columns
        missing_cols = [col for col in DataValidator.REQUIRED_HEART_RATE_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
            
        # Check for null values
        null_counts = df[DataValidator.REQUIRED_HEART_RATE_COLUMNS].isnull().sum()
        if null_counts.any():
            return False, f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}"
            
        # Validate heart rate values
        invalid_rates = df[(df['heart_rate'] < 30) | (df['heart_rate'] > 220)]
        if not invalid_rates.empty:
            logger.warning(f"Found {len(invalid_rates)} heart rate values outside normal range (30-220)")
            
        # Validate track URLs
        invalid_urls = df[~df['track_url'].str.contains('spotify.com/track/', na=False)]
        if not invalid_urls.empty:
            return False, f"Found {len(invalid_urls)} invalid Spotify track URLs"
            
        return True, "Data validation successful"
        
    @staticmethod
    def validate_step_count_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate step count data format and content.
        
        Args:
            df (pd.DataFrame): Step count DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check required columns
        missing_cols = [col for col in DataValidator.REQUIRED_STEP_COUNT_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
            
        # Check for null values
        null_counts = df[DataValidator.REQUIRED_STEP_COUNT_COLUMNS].isnull().sum()
        if null_counts.any():
            return False, f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}"
            
        # Validate step count values
        invalid_steps = df[df['step_count'] < 0]
        if not invalid_steps.empty:
            return False, f"Found {len(invalid_steps)} negative step count values"
            
        # Validate track URLs
        invalid_urls = df[~df['track_url'].str.contains('spotify.com/track/', na=False)]
        if not invalid_urls.empty:
            return False, f"Found {len(invalid_urls)} invalid Spotify track URLs"
            
        return True, "Data validation successful"
        
    @staticmethod
    def preprocess_timestamps(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Preprocess timestamp data to ensure consistent format.
        
        Args:
            df (pd.DataFrame): DataFrame containing timestamp column
            timestamp_col (str): Name of the timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with processed timestamps
        """
        try:
            # Convert timestamps to datetime if they're not already
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                
            # Sort by timestamp
            df = df.sort_values(timestamp_col)
            
            # Add derived time features
            df['hour'] = df[timestamp_col].dt.hour
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            df['month'] = df[timestamp_col].dt.month
            df['year'] = df[timestamp_col].dt.year
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing timestamps: {str(e)}")
            raise
            
    @staticmethod
    def extract_track_id(url: str) -> str:
        """
        Extract Spotify track ID from track URL.
        
        Args:
            url (str): Spotify track URL
            
        Returns:
            str: Track ID
        """
        try:
            return url.split('track/')[-1].split('?')[0]
        except Exception:
            logger.error(f"Failed to extract track ID from URL: {url}")
            return ""
            
    @staticmethod
    def clean_heart_rate_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess heart rate data.
        
        Args:
            df (pd.DataFrame): Heart rate DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle outliers
        q1 = df['heart_rate'].quantile(0.25)
        q3 = df['heart_rate'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df['heart_rate'] < lower_bound) | (df['heart_rate'] > upper_bound)]
        if not outliers.empty:
            logger.warning(f"Found {len(outliers)} heart rate outliers")
            df = df[(df['heart_rate'] >= lower_bound) & (df['heart_rate'] <= upper_bound)]
            
        # Extract track IDs
        df['track_id'] = df['track_url'].apply(DataValidator.extract_track_id)
        
        # Process timestamps
        df = DataValidator.preprocess_timestamps(df)
        
        return df
        
    @staticmethod
    def clean_step_count_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess step count data.
        
        Args:
            df (pd.DataFrame): Step count DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle outliers
        q1 = df['step_count'].quantile(0.25)
        q3 = df['step_count'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df['step_count'] < lower_bound) | (df['step_count'] > upper_bound)]
        if not outliers.empty:
            logger.warning(f"Found {len(outliers)} step count outliers")
            df = df[(df['step_count'] >= lower_bound) & (df['step_count'] <= upper_bound)]
            
        # Extract track IDs
        df['track_id'] = df['track_url'].apply(DataValidator.extract_track_id)
        
        # Process timestamps
        df = DataValidator.preprocess_timestamps(df)
        
        return df
        
    @staticmethod
    def merge_health_data(heart_rate_df: pd.DataFrame, step_count_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge heart rate and step count data based on timestamp and track.
        
        Args:
            heart_rate_df (pd.DataFrame): Heart rate DataFrame
            step_count_df (pd.DataFrame): Step count DataFrame
            
        Returns:
            pd.DataFrame: Merged DataFrame
        """
        # Ensure both DataFrames have track_id and processed timestamps
        heart_rate_df = DataValidator.clean_heart_rate_data(heart_rate_df)
        step_count_df = DataValidator.clean_step_count_data(step_count_df)
        
        # Merge on timestamp and track_id
        merged_df = pd.merge(
            heart_rate_df,
            step_count_df,
            on=['timestamp', 'track_id'],
            how='outer',
            suffixes=('_hr', '_sc')
        )
        
        # Log merge statistics
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        logger.info(f"Number of matched records: {len(merged_df.dropna())}")
        
        return merged_df
        
    @staticmethod
    def validate_merged_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate merged health data.
        
        Args:
            df (pd.DataFrame): Merged DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check for required columns in merged data
        required_cols = ['timestamp', 'track_id', 'heart_rate', 'step_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns in merged data: {missing_cols}"
            
        # Check for reasonable date range
        date_range = df['timestamp'].max() - df['timestamp'].min()
        if date_range.days > 365:
            logger.warning(f"Data spans more than a year: {date_range.days} days")
            
        # Check for gaps in data
        time_diffs = df['timestamp'].diff()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(days=7)]
        if not large_gaps.empty:
            logger.warning(f"Found {len(large_gaps)} gaps > 7 days in data")
            
        return True, "Merged data validation successful" 