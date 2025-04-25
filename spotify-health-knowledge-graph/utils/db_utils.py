import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_variables():
    """
    Load environment variables from .env file
    
    Returns:
        dict: Dictionary with environment variables
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Neo4j connection parameters
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    return {
        "uri": uri,
        "username": username,
        "password": password
    }

def get_db_connection():
    """
    Create a connection to the Neo4j database
    
    Returns:
        neo4j.Driver: Neo4j driver instance
    """
    env_vars = load_env_variables()
    
    try:
        driver = GraphDatabase.driver(
            env_vars["uri"], 
            auth=(env_vars["username"], env_vars["password"])
        )
        logger.info("Connected to Neo4j database")
        return driver
    except Exception as e:
        logger.error(f"Error connecting to Neo4j database: {e}")
        raise

def run_query(driver, query, parameters=None):
    """
    Run a Cypher query against the Neo4j database
    
    Args:
        driver (neo4j.Driver): Neo4j driver instance
        query (str): Cypher query
        parameters (dict, optional): Query parameters
        
    Returns:
        list: Query results
    """
    try:
        with driver.session() as session:
            result = session.run(query, parameters)
            return list(result)
    except Exception as e:
        logger.error(f"Error running query: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Parameters: {parameters}")
        raise

def clear_database(driver):
    """
    Remove all nodes and relationships from the database
    
    Args:
        driver (neo4j.Driver): Neo4j driver instance
    """
    logger.info("Clearing database...")
    run_query(driver, "MATCH (n) DETACH DELETE n")
    logger.info("Database cleared")

def create_constraints(driver):
    """
    Create constraints for uniqueness
    
    Args:
        driver (neo4j.Driver): Neo4j driver instance
    """
    logger.info("Creating constraints...")
    
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Track) REQUIRE t.track_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Album) REQUIRE (a.name, a.artist) IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (h:HealthMetric) REQUIRE h.id IS UNIQUE"
    ]
    
    for constraint in constraints:
        try:
            run_query(driver, constraint)
        except Exception as e:
            logger.warning(f"Error creating constraint: {e}")

def create_indexes(driver):
    """
    Create indexes for better query performance
    
    Args:
        driver (neo4j.Driver): Neo4j driver instance
    """
    logger.info("Creating indexes...")
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (t:Track) ON (t.track_id)",
        "CREATE INDEX IF NOT EXISTS FOR (h:HealthMetric) ON (h.type)",
        "CREATE INDEX IF NOT EXISTS FOR (h:HealthMetric) ON (h.timestamp)"
    ]
    
    for index in indexes:
        try:
            run_query(driver, index)
        except Exception as e:
            logger.warning(f"Error creating index: {e}")

def get_statistics(driver):
    """
    Get basic statistics about the graph
    
    Args:
        driver (neo4j.Driver): Neo4j driver instance
        
    Returns:
        dict: Dictionary with statistics
    """
    stats = {}
    
    # Get node counts
    node_counts = run_query(driver, """
        MATCH (n)
        RETURN labels(n) as label, count(*) as count
    """)
    
    for record in node_counts:
        stats[f"{record['label'][0]}_count"] = record['count']
        
    # Get relationship counts
    rel_counts = run_query(driver, """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
    """)
    
    for record in rel_counts:
        stats[f"{record['type']}_count"] = record['count']
        
    return stats 