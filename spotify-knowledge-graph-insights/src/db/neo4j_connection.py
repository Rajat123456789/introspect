from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Handles connections and operations with Neo4j database"""
    
    def __init__(self, uri, user, password):
        """Initialize connection to Neo4j database"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the database connection"""
        self.driver.close()
        
    def clear_database(self):
        """Clear all nodes and relationships in the database"""
        with self.driver.session() as session:
            logger.info("Clearing existing database...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully")

    def create_indexes(self):
        """Create indexes for faster querying"""
        with self.driver.session() as session:
            logger.info("Creating indexes...")
            # Create indexes for each node type
            session.run("CREATE INDEX track_id IF NOT EXISTS FOR (n:Track) ON (n.track_id)")
            session.run("CREATE INDEX artist_name IF NOT EXISTS FOR (n:Artist) ON (n.name)")
            session.run("CREATE INDEX album_name IF NOT EXISTS FOR (n:Album) ON (n.name)")
            session.run("CREATE INDEX genre_name IF NOT EXISTS FOR (n:Genre) ON (n.name)")
            logger.info("Indexes created successfully") 