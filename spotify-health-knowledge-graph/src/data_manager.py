import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        """Initialize the DataManager with Neo4j connection."""
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
            
    def load_heart_rate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process heart rate data from CSV.
        
        Args:
            file_path (str): Path to the heart rate CSV file
            
        Returns:
            pd.DataFrame: Processed heart rate data
        """
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['month'] = df['timestamp'].dt.month
            df['year'] = df['timestamp'].dt.year
            
            # Extract track ID from URL
            df['track_id'] = df['track_url'].str.extract(r'track/([a-zA-Z0-9]+)')
            
            logger.info(f"Successfully loaded heart rate data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading heart rate data: {str(e)}")
            raise
            
    def load_step_count_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process step count data from CSV.
        
        Args:
            file_path (str): Path to the step count CSV file
            
        Returns:
            pd.DataFrame: Processed step count data
        """
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['month'] = df['timestamp'].dt.month
            df['year'] = df['timestamp'].dt.year
            
            # Extract track ID from URL
            df['track_id'] = df['track_url'].str.extract(r'track/([a-zA-Z0-9]+)')
            
            logger.info(f"Successfully loaded step count data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading step count data: {str(e)}")
            raise
            
    def create_track_node(self, tx, track_data: Dict) -> None:
        """Create a Track node in Neo4j."""
        query = """
        MERGE (t:Track {track_id: $track_id})
        SET t.name = $name,
            t.artist = $artist,
            t.album = $album,
            t.url = $url
        """
        tx.run(query, track_data)
        
    def create_health_metric_node(self, tx, metric_data: Dict) -> None:
        """Create a HealthMetric node in Neo4j."""
        query = """
        CREATE (h:HealthMetric {
            id: $id,
            type: $type,
            value: $value,
            unit: $unit,
            timestamp: datetime($timestamp),
            source: $source
        })
        WITH h
        MATCH (t:Track {track_id: $track_id})
        CREATE (t)-[:HAS_HEALTH_METRIC]->(h)
        """
        tx.run(query, metric_data)
        
    def load_data_to_neo4j(self, heart_rate_df: pd.DataFrame, step_count_df: pd.DataFrame) -> None:
        """
        Load processed data into Neo4j database.
        
        Args:
            heart_rate_df (pd.DataFrame): Processed heart rate data
            step_count_df (pd.DataFrame): Processed step count data
        """
        try:
            with self.driver.session() as session:
                # Create unique tracks first
                unique_tracks = pd.concat([
                    heart_rate_df[['track_id', 'track_name', 'artist', 'album', 'track_url']],
                    step_count_df[['track_id', 'track_name', 'artist', 'album', 'track_url']]
                ]).drop_duplicates(subset=['track_id'])
                
                for _, track in unique_tracks.iterrows():
                    track_data = {
                        'track_id': track['track_id'],
                        'name': track['track_name'],
                        'artist': track['artist'],
                        'album': track['album'],
                        'url': track['track_url']
                    }
                    session.write_transaction(self.create_track_node, track_data)
                
                # Create heart rate metrics
                for _, row in heart_rate_df.iterrows():
                    metric_data = {
                        'id': f"hr_{row.name}",
                        'type': 'heart_rate',
                        'value': float(row['value']),
                        'unit': 'bpm',
                        'timestamp': row['timestamp'].isoformat(),
                        'source': 'apple_health',
                        'track_id': row['track_id']
                    }
                    session.write_transaction(self.create_health_metric_node, metric_data)
                
                # Create step count metrics
                for _, row in step_count_df.iterrows():
                    metric_data = {
                        'id': f"sc_{row.name}",
                        'type': 'step_count',
                        'value': int(row['value']),
                        'unit': 'steps',
                        'timestamp': row['timestamp'].isoformat(),
                        'source': 'apple_health',
                        'track_id': row['track_id']
                    }
                    session.write_transaction(self.create_health_metric_node, metric_data)
                    
                logger.info("Successfully loaded all data into Neo4j database")
        except Exception as e:
            logger.error(f"Error loading data to Neo4j: {str(e)}")
            raise
            
    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            raise
            
    def create_constraints(self) -> None:
        """Create necessary constraints in Neo4j."""
        try:
            with self.driver.session() as session:
                # Create constraint on Track.track_id
                session.run("CREATE CONSTRAINT track_id IF NOT EXISTS FOR (t:Track) REQUIRE t.track_id IS UNIQUE")
                # Create constraint on HealthMetric.id
                session.run("CREATE CONSTRAINT health_metric_id IF NOT EXISTS FOR (h:HealthMetric) REQUIRE h.id IS UNIQUE")
                logger.info("Database constraints created successfully")
        except Exception as e:
            logger.error(f"Error creating constraints: {str(e)}")
            raise
            
    def get_statistics(self) -> Dict:
        """
        Get basic statistics about the graph.
        
        Returns:
            Dict: Statistics about nodes and relationships
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (t:Track)
                WITH count(t) as track_count
                MATCH (h:HealthMetric)
                WITH track_count, count(h) as metric_count
                MATCH ()-[r:HAS_HEALTH_METRIC]->()
                RETURN track_count, metric_count, count(r) as relationship_count
                """)
                stats = result.single()
                return {
                    'track_count': stats['track_count'],
                    'metric_count': stats['metric_count'],
                    'relationship_count': stats['relationship_count']
                }
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise 