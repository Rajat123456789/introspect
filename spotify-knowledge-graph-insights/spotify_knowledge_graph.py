#!/usr/bin/env python3
"""
Spotify Knowledge Graph Builder

This script creates a Neo4j knowledge graph from Spotify listening history data.
The graph represents tracks, artists, albums, genres, and listening patterns.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from db.neo4j_connection import Neo4jConnection
from models.spotify_data_loader import SpotifyDataLoader
from utils.file_loader import load_spotify_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_all_data(db_connection, data_file):
    """
    Load all Spotify data into the knowledge graph
    
    Args:
        db_connection: Neo4j database connection
        data_file: Path to the Spotify history CSV file
    """
    logger.info("Starting data loading process")
    
    # Load data from CSV
    spotify_df = load_spotify_data(data_file)
    if spotify_df is None:
        logger.error("Failed to load Spotify data")
        return False
    
    # Clear database and create indexes
    db_connection.clear_database()
    db_connection.create_indexes()
    
    # Initialize data loader
    data_loader = SpotifyDataLoader(db_connection)
    
    # Load all data
    logger.info("Loading track data...")
    data_loader.load_track_data(spotify_df)
    
    logger.info("Loading artist data and relationships...")
    data_loader.load_artists_and_relationships(spotify_df)
    
    logger.info("Loading album data and relationships...")
    data_loader.load_albums_and_relationships(spotify_df)
    
    logger.info("Loading genre data and relationships...")
    data_loader.load_genres_and_relationships(spotify_df)
    
    logger.info("Loading listening session data...")
    data_loader.load_listening_sessions(spotify_df)
    
    logger.info("Data loading completed successfully")
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Build a Spotify knowledge graph in Neo4j')
    parser.add_argument('--data', default='songs-parameters/spotify-history-annotated.csv',
                        help='Path to Spotify history CSV file')
    parser.add_argument('--uri', help='Neo4j URI')
    parser.add_argument('--user', help='Neo4j username')
    parser.add_argument('--password', help='Neo4j password')
    
    args = parser.parse_args()
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Get Neo4j connection details from arguments or environment variables
    neo4j_uri = args.uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = args.user or os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = args.password or os.getenv('NEO4J_PASSWORD', 'password')
    
    # Get full path to data file
    data_file = os.path.join(os.path.dirname(__file__), args.data)
    
    try:
        # Connect to Neo4j
        connection = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        
        # Load all data
        success = load_all_data(connection, data_file)
        
        # Close connection
        connection.close()
        
        if success:
            logger.info("Spotify knowledge graph created successfully")
        else:
            logger.error("Failed to create Spotify knowledge graph")
            return 1
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 