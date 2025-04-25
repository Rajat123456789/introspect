import pandas as pd
from neo4j import GraphDatabase
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyHealthKnowledgeGraph:
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
            
    def clear_database(self):
        """Remove all nodes and relationships from the database"""
        logger.info("Clearing database...")
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
        
    def create_constraints(self):
        """Create constraints for uniqueness"""
        logger.info("Creating constraints...")
        
        try:
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Track) REQUIRE t.track_id IS UNIQUE")
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE")
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Album) REQUIRE (a.name, a.artist) IS UNIQUE")
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (h:HealthMetric) REQUIRE h.id IS UNIQUE")
        except Exception as e:
            logger.warning(f"Error creating constraints: {e}")
            
    def load_health_data(self, heart_rate_path, step_count_path):
        """Load health data from CSV files into Neo4j"""
        logger.info("Loading health data...")
        
        # Read CSV files
        heart_rate_df = pd.read_csv(heart_rate_path)
        step_count_df = pd.read_csv(step_count_path)
        
        # Process heart rate data
        for _, row in heart_rate_df.iterrows():
            self._create_heart_rate_record(row)
            
        # Process step count data
        for _, row in step_count_df.iterrows():
            self._create_step_count_record(row)
            
        logger.info("Health data loaded successfully")
        
    def _create_heart_rate_record(self, row):
        """Create heart rate record in Neo4j"""
        query = """
        MERGE (t:Track {track_id: $track_id})
        ON CREATE SET 
            t.name = $track_name,
            t.artist = $artist_name,
            t.album = $album_name,
            t.url = $url
            
        MERGE (h:HealthMetric {id: $metric_id})
        ON CREATE SET 
            h.type = 'HeartRate',
            h.value = $value,
            h.unit = $unit,
            h.timestamp = $timestamp,
            h.source = $source
            
        MERGE (t)-[:HAS_HEALTH_METRIC]->(h)
        """
        
        parameters = {
            'track_id': row['url'].split(':')[-1],
            'track_name': row['track_name'],
            'artist_name': row['artist_name'],
            'album_name': row['album_name'],
            'url': row['url'],
            'metric_id': f"hr_{row['startDate']}_{row['track_name']}",
            'value': float(row['value']),
            'unit': row['unit'],
            'timestamp': row['startDate'],
            'source': row['sourceName']
        }
        
        self.run_query(query, parameters)
        
    def _create_step_count_record(self, row):
        """Create step count record in Neo4j"""
        query = """
        MERGE (t:Track {track_id: $track_id})
        ON CREATE SET 
            t.name = $track_name,
            t.artist = $artist_name,
            t.album = $album_name,
            t.url = $url
            
        MERGE (h:HealthMetric {id: $metric_id})
        ON CREATE SET 
            h.type = 'StepCount',
            h.value = $value,
            h.unit = $unit,
            h.timestamp = $timestamp,
            h.source = $source
            
        MERGE (t)-[:HAS_HEALTH_METRIC]->(h)
        """
        
        parameters = {
            'track_id': row['url'].split(':')[-1],
            'track_name': row['track_name'],
            'artist_name': row['artist_name'],
            'album_name': row['album_name'],
            'url': row['url'],
            'metric_id': f"sc_{row['startDate']}_{row['track_name']}",
            'value': float(row['value']),
            'unit': row['unit'],
            'timestamp': row['startDate'],
            'source': row['sourceName']
        }
        
        self.run_query(query, parameters)
        
    def create_indexes(self):
        """Create indexes for better query performance"""
        logger.info("Creating indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (t:Track) ON (t.track_id)",
            "CREATE INDEX IF NOT EXISTS FOR (h:HealthMetric) ON (h.type)",
            "CREATE INDEX IF NOT EXISTS FOR (h:HealthMetric) ON (h.timestamp)"
        ]
        
        for index in indexes:
            self.run_query(index)
            
    def get_statistics(self):
        """Get basic statistics about the graph"""
        stats = {}
        
        # Get node counts
        node_counts = self.run_query("""
            MATCH (n)
            RETURN labels(n) as label, count(*) as count
        """)
        
        for record in node_counts:
            stats[f"{record['label'][0]}_count"] = record['count']
            
        # Get relationship counts
        rel_counts = self.run_query("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
        """)
        
        for record in rel_counts:
            stats[f"{record['type']}_count"] = record['count']
            
        return stats

def main():
    # Connection parameters - modify these to match your Neo4j setup
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Initialize graph
    graph = SpotifyHealthKnowledgeGraph(uri, username, password)
    
    try:
        # Clear existing data
        graph.clear_database()
        
        # Create constraints and indexes
        graph.create_constraints()
        graph.create_indexes()
        
        # Load data
        heart_rate_path = "../final-dataset-apple-spotify/spotifyHeartRate.csv"
        step_count_path = "../final-dataset-apple-spotify/spotifyStepCount.csv"
        graph.load_health_data(heart_rate_path, step_count_path)
        
        # Print statistics
        stats = graph.get_statistics()
        logger.info("Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
            
    finally:
        graph.close()

if __name__ == "__main__":
    main() 