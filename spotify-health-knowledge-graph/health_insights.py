import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neo4j import GraphDatabase
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyHealthInsights:
    def __init__(self, uri, username, password):
        """Initialize connection to Neo4j database"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info("Connected to Neo4j database")
        
    def close(self):
        """Close the driver connection"""
        self.driver.close()
        logger.info("Disconnected from Neo4j database")
        
    def run_query(self, query, parameters=None):
        """Run a Cypher query against the database"""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return list(result)
    
    def query_to_dataframe(self, query, parameters=None):
        """Convert query results to a pandas DataFrame"""
        results = self.run_query(query, parameters)
        if not results:
            return pd.DataFrame()
            
        # Extract keys from the first result
        keys = results[0].keys()
        
        # Convert results to a dataframe
        data = {key: [result[key] for result in results] for key in keys}
        return pd.DataFrame(data)
    
    def get_tracks_by_heart_rate_range(self, min_hr=0, max_hr=200):
        """Get tracks with heart rate measurements in specified range"""
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = 'HeartRate' AND h.value >= $min_hr AND h.value <= $max_hr
        RETURN t.name as track, t.artist as artist, h.value as heart_rate, h.timestamp as timestamp
        ORDER BY h.value DESC
        """
        
        return self.query_to_dataframe(query, {'min_hr': min_hr, 'max_hr': max_hr})
    
    def get_tracks_by_step_count_range(self, min_steps=0, max_steps=1000):
        """Get tracks with step counts in specified range"""
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = 'StepCount' AND h.value >= $min_steps AND h.value <= $max_steps
        RETURN t.name as track, t.artist as artist, h.value as steps, h.timestamp as timestamp
        ORDER BY h.value DESC
        """
        
        return self.query_to_dataframe(query, {'min_steps': min_steps, 'max_steps': max_steps})
    
    def get_health_metrics_by_time_of_day(self):
        """Get average heart rate and step count by hour of day"""
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WITH h, datetime(h.timestamp).hour as hour
        RETURN hour,
               avg(CASE WHEN h.type = 'HeartRate' THEN h.value ELSE null END) as avg_heart_rate,
               avg(CASE WHEN h.type = 'StepCount' THEN h.value ELSE null END) as avg_steps
        ORDER BY hour
        """
        
        return self.query_to_dataframe(query)
    
    def get_health_metrics_by_artist(self):
        """Get average heart rate and step count by artist"""
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        RETURN t.artist as artist,
               avg(CASE WHEN h.type = 'HeartRate' THEN h.value ELSE null END) as avg_heart_rate,
               avg(CASE WHEN h.type = 'StepCount' THEN h.value ELSE null END) as avg_steps,
               count(DISTINCT t) as track_count
        ORDER BY track_count DESC
        """
        
        return self.query_to_dataframe(query)
    
    def plot_heart_rate_distribution(self):
        """Plot distribution of heart rate measurements"""
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = 'HeartRate'
        RETURN h.value as heart_rate
        """
        
        df = self.query_to_dataframe(query)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='heart_rate', bins=30)
        plt.title('Distribution of Heart Rate Measurements')
        plt.xlabel('Heart Rate (bpm)')
        plt.ylabel('Count')
        plt.savefig('visualizations/heart_rate_distribution.png')
        plt.close()
        
    def plot_step_count_distribution(self):
        """Plot distribution of step counts"""
        query = """
        MATCH (t:Track)-[:HAS_HEALTH_METRIC]->(h:HealthMetric)
        WHERE h.type = 'StepCount'
        RETURN h.value as steps
        """
        
        df = self.query_to_dataframe(query)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='steps', bins=30)
        plt.title('Distribution of Step Counts')
        plt.xlabel('Steps')
        plt.ylabel('Count')
        plt.savefig('visualizations/step_count_distribution.png')
        plt.close()
        
    def plot_health_metrics_by_hour(self):
        """Plot average heart rate and step count by hour"""
        df = self.get_health_metrics_by_time_of_day()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['hour'], df['avg_heart_rate'], label='Average Heart Rate', marker='o')
        plt.plot(df['hour'], df['avg_steps'], label='Average Steps', marker='o')
        plt.title('Health Metrics by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/health_metrics_by_hour.png')
        plt.close()
        
    def plot_health_metrics_by_artist(self, top_n=10):
        """Plot average heart rate and step count by artist"""
        df = self.get_health_metrics_by_artist().head(top_n)
        
        plt.figure(figsize=(12, 6))
        x = range(len(df))
        width = 0.35
        
        plt.bar(x, df['avg_heart_rate'], width, label='Average Heart Rate')
        plt.bar([i + width for i in x], df['avg_steps'], width, label='Average Steps')
        
        plt.xlabel('Artist')
        plt.ylabel('Value')
        plt.title('Health Metrics by Artist')
        plt.xticks([i + width/2 for i in x], df['artist'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/health_metrics_by_artist.png')
        plt.close()

def main():
    # Connection parameters - modify these to match your Neo4j setup
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Initialize insights
    insights = SpotifyHealthInsights(uri, username, password)
    
    try:
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Generate plots
        insights.plot_heart_rate_distribution()
        insights.plot_step_count_distribution()
        insights.plot_health_metrics_by_hour()
        insights.plot_health_metrics_by_artist()
        
        # Print some insights
        print("\nTop tracks by heart rate:")
        print(insights.get_tracks_by_heart_rate_range(min_hr=80).head())
        
        print("\nTop tracks by step count:")
        print(insights.get_tracks_by_step_count_range(min_steps=100).head())
        
        print("\nHealth metrics by hour of day:")
        print(insights.get_health_metrics_by_time_of_day())
        
    finally:
        insights.close()

if __name__ == "__main__":
    main() 