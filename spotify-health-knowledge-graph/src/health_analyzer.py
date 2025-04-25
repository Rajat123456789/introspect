import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthAnalyzer:
    def __init__(self):
        """Initialize the HealthAnalyzer with Neo4j connection."""
        load_dotenv()
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', '')
        self.driver = None
        
    def connect_to_database(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise
            
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")
            
    def query_to_dataframe(self, query: str, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a Cypher query and return results as a pandas DataFrame.
        
        Args:
            query (str): Cypher query to execute
            parameters (Dict, optional): Query parameters
            
        Returns:
            pd.DataFrame: Query results as a DataFrame
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                records = list(result)
                if not records:
                    return pd.DataFrame()
                return pd.DataFrame([r.values() for r in records], columns=result.keys())
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
            
    def get_health_metrics_by_time(self, metric_type: str, time_unit: str = 'hour') -> pd.DataFrame:
        """
        Get average health metrics grouped by time unit.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            time_unit (str): Time unit for grouping ('hour', 'day', 'month')
            
        Returns:
            pd.DataFrame: Average metrics by time unit
        """
        time_function = {
            'hour': 'hour',
            'day': 'dayOfWeek',
            'month': 'month'
        }.get(time_unit)
        
        if not time_function:
            raise ValueError(f"Invalid time unit: {time_unit}")
            
        query = f"""
        MATCH (h:HealthMetric)
        WHERE h.type = $metric_type
        RETURN datetime.{time_function}(h.timestamp) as time_unit,
               avg(h.value) as avg_value,
               count(h) as count
        ORDER BY time_unit
        """
        
        return self.query_to_dataframe(query, {'metric_type': metric_type})
        
    def get_health_metrics_by_artist(self, metric_type: str, limit: int = 10) -> pd.DataFrame:
        """
        Get average health metrics grouped by artist.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            limit (int): Number of top artists to return
            
        Returns:
            pd.DataFrame: Average metrics by artist
        """
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = $metric_type
        WITH t.artist as artist,
             avg(h.value) as avg_value,
             count(h) as count
        ORDER BY count DESC
        LIMIT $limit
        RETURN artist, avg_value, count
        """
        
        return self.query_to_dataframe(query, {
            'metric_type': metric_type,
            'limit': limit
        })
        
    def get_health_metrics_by_track(self, metric_type: str, limit: int = 10) -> pd.DataFrame:
        """
        Get average health metrics grouped by track.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            limit (int): Number of top tracks to return
            
        Returns:
            pd.DataFrame: Average metrics by track
        """
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = $metric_type
        WITH t.name as track_name,
             t.artist as artist,
             avg(h.value) as avg_value,
             count(h) as count
        ORDER BY count DESC
        LIMIT $limit
        RETURN track_name, artist, avg_value, count
        """
        
        return self.query_to_dataframe(query, {
            'metric_type': metric_type,
            'limit': limit
        })
        
    def get_health_metrics_range(self, metric_type: str) -> Dict:
        """
        Get the range of values for a health metric.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            
        Returns:
            Dict: Min, max, avg, and std values for the metric
        """
        query = """
        MATCH (h:HealthMetric)
        WHERE h.type = $metric_type
        RETURN min(h.value) as min_value,
               max(h.value) as max_value,
               avg(h.value) as avg_value,
               stDev(h.value) as std_value
        """
        
        df = self.query_to_dataframe(query, {'metric_type': metric_type})
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
        
    def get_correlated_tracks(self, metric_type: str, value_range: Tuple[float, float]) -> pd.DataFrame:
        """
        Get tracks associated with specific health metric ranges.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            value_range (Tuple[float, float]): Range of values to filter by
            
        Returns:
            pd.DataFrame: Tracks and their average metric values within the range
        """
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = $metric_type
        AND h.value >= $min_value
        AND h.value <= $max_value
        WITH t.name as track_name,
             t.artist as artist,
             avg(h.value) as avg_value,
             count(h) as count
        WHERE count >= 3
        RETURN track_name, artist, avg_value, count
        ORDER BY count DESC
        """
        
        return self.query_to_dataframe(query, {
            'metric_type': metric_type,
            'min_value': value_range[0],
            'max_value': value_range[1]
        })
        
    def get_time_series_data(self, metric_type: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get time series data for a health metric within a date range.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            start_date (str): Start date in ISO format
            end_date (str): End date in ISO format
            
        Returns:
            pd.DataFrame: Time series data for the metric
        """
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = $metric_type
        AND h.timestamp >= datetime($start_date)
        AND h.timestamp <= datetime($end_date)
        RETURN h.timestamp as timestamp,
               h.value as value,
               t.name as track_name,
               t.artist as artist
        ORDER BY timestamp
        """
        
        return self.query_to_dataframe(query, {
            'metric_type': metric_type,
            'start_date': start_date,
            'end_date': end_date
        })
        
    def get_health_patterns(self, metric_type: str) -> pd.DataFrame:
        """
        Analyze patterns in health metrics across different time periods.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            
        Returns:
            pd.DataFrame: Health patterns data
        """
        query = """
        MATCH (h:HealthMetric)
        WHERE h.type = $metric_type
        WITH h,
             datetime.hour(h.timestamp) as hour,
             datetime.dayOfWeek(h.timestamp) as day
        RETURN day,
               hour,
               avg(h.value) as avg_value,
               count(h) as count
        ORDER BY day, hour
        """
        
        return self.query_to_dataframe(query, {'metric_type': metric_type})
        
    def analyze_health_trends(self, metric_type: str, window: str = 'day') -> pd.DataFrame:
        """
        Analyze trends in health metrics over time.
        
        Args:
            metric_type (str): Type of health metric ('heart_rate' or 'step_count')
            window (str): Time window for aggregation ('day', 'week', 'month')
            
        Returns:
            pd.DataFrame: Trend analysis data
        """
        time_function = {
            'day': 'date',
            'week': 'week',
            'month': 'month'
        }.get(window)
        
        if not time_function:
            raise ValueError(f"Invalid time window: {window}")
            
        query = f"""
        MATCH (h:HealthMetric)
        WHERE h.type = $metric_type
        WITH datetime.{time_function}(h.timestamp) as time_window,
             h.value as value
        RETURN time_window,
               avg(value) as avg_value,
               min(value) as min_value,
               max(value) as max_value,
               stDev(value) as std_value,
               count(*) as count
        ORDER BY time_window
        """
        
        return self.query_to_dataframe(query, {'metric_type': metric_type}) 