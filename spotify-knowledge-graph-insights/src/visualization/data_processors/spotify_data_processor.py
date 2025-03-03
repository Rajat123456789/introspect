"""
Spotify Data Processor Module

This module handles loading and preprocessing Spotify listening history data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyDataProcessor:
    """
    Class for processing Spotify listening history data.
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the data processor.
        
        Args:
            file_path (str, optional): Path to the Spotify history CSV file.
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self, file_path=None):
        """
        Load data from CSV file.
        
        Args:
            file_path (str, optional): Path to the Spotify history CSV file.
                If not provided, uses the path from initialization.
                
        Returns:
            pandas.DataFrame: The loaded data.
        """
        if file_path:
            self.file_path = file_path
            
        if not self.file_path:
            raise ValueError("No file path provided.")
            
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(self.data)} records")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self):
        """
        Clean and preprocess the data.
        
        Returns:
            pandas.DataFrame: The cleaned data.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Cleaning data...")
        
        # Make a copy to avoid modifying the original
        df = self.data.copy()
        
        # Convert timestamps to datetime
        try:
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['date'] = df['end_time'].dt.date
            df['hour'] = df['end_time'].dt.hour
            df['day_of_week'] = df['end_time'].dt.day_name()
        except Exception as e:
            logger.warning(f"Error processing timestamps: {e}")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                logger.info(f"Filling missing values in {col}")
                df[col] = df[col].fillna(df[col].median())
        
        # Drop duplicates based on track_id and end_time
        if 'track_id' in df.columns and 'end_time' in df.columns:
            before_count = len(df)
            df = df.drop_duplicates(subset=['track_id', 'end_time'])
            after_count = len(df)
            logger.info(f"Removed {before_count - after_count} duplicate entries")
        
        self.data = df
        logger.info("Data cleaning completed")
        return df
    
    def extract_features(self):
        """
        Extract additional features from the data.
        
        Returns:
            pandas.DataFrame: The data with additional features.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Extracting additional features...")
        
        df = self.data.copy()
        
        # Add listening session features
        if 'end_time' in df.columns and 'ms_played' in df.columns:
            # Sort by end_time
            df = df.sort_values('end_time')
            
            # Calculate time between consecutive songs
            df['time_diff'] = df['end_time'].diff().dt.total_seconds()
            
            # Mark new sessions (gap > 30 minutes)
            df['new_session'] = (df['time_diff'] > 1800) | (df['time_diff'].isna())
            
            # Assign session IDs
            df['session_id'] = df['new_session'].cumsum()
            
            # Calculate session duration
            session_durations = df.groupby('session_id')['ms_played'].sum() / 1000  # Convert to seconds
            df['session_duration'] = df['session_id'].map(session_durations)
        
        # Add genre simplification if track_genre exists
        if 'track_genre' in df.columns:
            df['main_genre'] = df['track_genre'].str.split('-').str[0]
        
        # Calculate normalized audio features
        audio_features = ['danceability', 'energy', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        for feature in audio_features:
            if feature in df.columns:
                # Min-max normalization
                df[f'{feature}_norm'] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
        
        self.data = df
        logger.info("Feature extraction completed")
        return df
    
    def get_listening_patterns(self):
        """
        Extract listening patterns from the data.
        
        Returns:
            dict: Dictionary containing various listening pattern metrics.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        df = self.data
        
        patterns = {
            'total_tracks': len(df),
            'unique_tracks': df['track_id'].nunique() if 'track_id' in df.columns else None,
            'unique_artists': df['artist_name'].nunique() if 'artist_name' in df.columns else None,
            'total_listening_time': df['ms_played'].sum() / (1000 * 60 * 60) if 'ms_played' in df.columns else None,  # in hours
        }
        
        # Add top artists if available
        if 'artist_name' in df.columns:
            top_artists = df['artist_name'].value_counts().head(10).to_dict()
            patterns['top_artists'] = top_artists
        
        # Add top genres if available
        if 'track_genre' in df.columns:
            top_genres = df['track_genre'].value_counts().head(10).to_dict()
            patterns['top_genres'] = top_genres
            
        # Add time patterns if available
        if 'hour' in df.columns:
            hour_counts = df['hour'].value_counts().sort_index().to_dict()
            patterns['listening_by_hour'] = hour_counts
            
        if 'day_of_week' in df.columns:
            # Ensure days are in correct order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['day_of_week'].value_counts().reindex(day_order).to_dict()
            patterns['listening_by_day'] = day_counts
        
        return patterns
    
    def get_audio_features_summary(self):
        """
        Get summary statistics for audio features.
        
        Returns:
            pandas.DataFrame: Summary statistics for audio features.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                         'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in audio_features if col in self.data.columns]
        
        if not available_features:
            logger.warning("No audio features found in the data")
            return None
        
        return self.data[available_features].describe()
    
    def process_pipeline(self, file_path=None):
        """
        Run the full data processing pipeline.
        
        Args:
            file_path (str, optional): Path to the Spotify history CSV file.
            
        Returns:
            tuple: (processed_data, listening_patterns, audio_features_summary)
        """
        self.load_data(file_path)
        self.clean_data()
        self.extract_features()
        
        listening_patterns = self.get_listening_patterns()
        audio_features_summary = self.get_audio_features_summary()
        
        return self.data, listening_patterns, audio_features_summary 