import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_spotify_data(file_path):
    """
    Load Spotify data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the Spotify data or None if an error occurs
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        logger.info(f"Loading Spotify data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Remove duplicate tracks (same track_id)
        if 'track_id' in df.columns:
            before_count = len(df)
            df = df.drop_duplicates(subset=['track_id', 'track_genre'], keep='first')
            after_count = len(df)
            logger.info(f"Removed {before_count - after_count} duplicate entries")
            
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return None 