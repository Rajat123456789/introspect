import pandas as pd
import networkx as nx
from datetime import datetime
from neo4j import GraphDatabase
import logging
import os
import logging.handlers

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up file logging
log_file = 'youtube_kg_errors.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        """Clear all nodes and relationships in the database"""
        with self.driver.session() as session:
            logger.info("Clearing existing database...")
            result = session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully")

    def create_indexes(self):
        """Create indexes for faster querying"""
        with self.driver.session() as session:
            logger.info("Creating indexes...")
            # Create indexes for each node type
            session.run("DROP INDEX video_id IF EXISTS")
            session.run("DROP INDEX engagement IF EXISTS")
            session.run("DROP INDEX quality IF EXISTS")
            session.run("DROP INDEX format IF EXISTS")
            session.run("DROP INDEX category IF EXISTS")
            
            session.run("CREATE INDEX video_id IF NOT EXISTS FOR (n:Video) ON (n.video_id)")
            session.run("CREATE INDEX engagement IF NOT EXISTS FOR (n:Engagement) ON (n.name)")
            session.run("CREATE INDEX quality IF NOT EXISTS FOR (n:Quality) ON (n.name)")
            session.run("CREATE INDEX format IF NOT EXISTS FOR (n:Format) ON (n.name)")
            session.run("CREATE INDEX category IF NOT EXISTS FOR (n:MentalHealthCategory) ON (n.name)")
            logger.info("Indexes created successfully")

    def load_main_metadata(self, main_df):
        """Load the main video metadata into Neo4j"""
        invalid_timestamps = []  # Track rows with invalid timestamps
        
        with self.driver.session() as session:
            for idx, row in main_df.iterrows():
                try:
                    # Ensure video_id is an integer
                    video_id = int(float(row['video_id']))  # Handle both int and float formats
                    
                    # Try to parse timestamp if it looks like a timestamp
                    watched_at_str = None
                    if pd.notna(row['watched_at']) and isinstance(row['watched_at'], str):
                        if "PST" in row['watched_at']:
                            try:
                                watched_at = pd.to_datetime(row['watched_at'], format="%b %d, %Y, %I:%M:%S %p PST")
                                watched_at = watched_at.tz_localize('America/Los_Angeles')
                                watched_at = watched_at.tz_convert('UTC')
                                watched_at_str = watched_at.isoformat()
                            except Exception as e:
                                invalid_timestamps.append({
                                    'row_idx': idx,
                                    'video_id': video_id,
                                    'watched_at': row['watched_at'],
                                    'title': row['title']
                                })
                                logger.warning(f"Could not parse timestamp for row {idx}: {str(e)}")
                    else:
                        invalid_timestamps.append({
                            'row_idx': idx,
                            'video_id': video_id,
                            'watched_at': row['watched_at'],
                            'title': row['title']
                        })
                    
                    # Create video node with or without timestamp
                    if watched_at_str:
                        video_query = """
                        CREATE (v:Video {
                            video_id: $video_id,
                            title: $title,
                            watched_at: datetime($watched_at),
                            primary_category: $primary_category,
                            detailed_type: $detailed_type,
                            primary_format: $primary_format,
                            primary_purpose: $primary_purpose
                        })
                        """
                    else:
                        video_query = """
                        CREATE (v:Video {
                            video_id: $video_id,
                            title: $title,
                            primary_category: $primary_category,
                            detailed_type: $detailed_type,
                            primary_format: $primary_format,
                            primary_purpose: $primary_purpose
                        })
                        """
                    
                    params = {
                        "video_id": video_id,  # Store as integer
                        "title": str(row.get('title', '')),
                        "primary_category": str(row.get('primary_category', '')),
                        "detailed_type": str(row.get('detailed_type', '')),
                        "primary_format": str(row.get('primary_format', '')),
                        "primary_purpose": str(row.get('primary_purpose', ''))
                    }
                    
                    if watched_at_str:
                        params["watched_at"] = watched_at_str
                    
                    session.run(video_query, params)
                    
                except Exception as e:
                    logger.warning(f"Error creating video node for row {idx}: {str(e)}")
                    continue
            
            # After processing all rows, log summary of invalid timestamps
            if invalid_timestamps:
                logger.warning("\nRows with invalid timestamps:")
                for row in invalid_timestamps:
                    logger.warning(f"Row {row['row_idx']}: video_id={row['video_id']}, watched_at='{row['watched_at']}', title='{row['title']}'")
                logger.warning(f"\nTotal rows with invalid timestamps: {len(invalid_timestamps)}")

    def load_engagement_data(self, engagement_df):
        """Load the engagement data as relationships"""
        with self.driver.session() as session:
            for idx, row in engagement_df.iterrows():
                try:
                    # Convert video_id to integer like other methods
                    video_id = int(float(row['video_id']))
                    
                    # Add engagement relationships
                    if isinstance(row['audience_engagement'], list):
                        for engagement in row['audience_engagement']:
                            if pd.notna(engagement):
                                engagement_query = """
                                MERGE (e:Engagement {name: $engagement})
                                WITH e
                                MATCH (v:Video {video_id: $video_id})
                                MERGE (v)-[:HAS_ENGAGEMENT]->(e)
                                """
                                session.run(engagement_query,
                                          engagement=engagement.strip(),
                                          video_id=video_id)
                    
                    # Add quality relationships
                    if isinstance(row['production_quality'], list):
                        for quality in row['production_quality']:
                            if pd.notna(quality):
                                quality_query = """
                                MERGE (q:Quality {name: $quality})
                                WITH q
                                MATCH (v:Video {video_id: $video_id})
                                MERGE (v)-[:HAS_QUALITY]->(q)
                                """
                                session.run(quality_query,
                                          quality=quality.strip(),
                                          video_id=video_id)
                    
                    # Add format relationships
                    if isinstance(row['content_format'], list):
                        for format in row['content_format']:
                            if pd.notna(format):
                                format_query = """
                                MERGE (f:Format {name: $format})
                                WITH f
                                MATCH (v:Video {video_id: $video_id})
                                MERGE (v)-[:HAS_FORMAT]->(f)
                                """
                                session.run(format_query,
                                          format=format.strip(),
                                          video_id=video_id)
                except Exception as e:
                    logger.warning(f"Skipping malformed engagement row {idx}: {str(e)}")
                    continue

    def load_mental_health_data(self, mental_health_df):
        """Load mental health data as relationships with scores and sentiments"""
        with self.driver.session() as session:
            # Group by video_id to process all categories for each video
            for video_id, video_data in mental_health_df.groupby('video_id'):
                try:
                    # Convert video_id to integer
                    video_id = int(float(video_id))  # Handle both int and float formats
                    
                    # Process each category for the video
                    for _, row in video_data.iterrows():
                        try:
                            # Parse timestamp
                            timestamp = pd.to_datetime(row['timestamp'], format="%b %d, %Y, %I:%M:%S %p PST")
                            timestamp = timestamp.tz_localize('America/Los_Angeles')
                            timestamp = timestamp.tz_convert('UTC')
                            
                            # Create relationship with mental health category
                            query = """
                            MATCH (v:Video {video_id: $video_id})
                            MERGE (m:MentalHealthCategory {name: $category})
                            MERGE (v)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m)
                            SET r.score = $score,
                                r.sentiment = $sentiment,
                                r.timestamp = datetime($timestamp)
                            """
                            
                            session.run(query,
                                      video_id=video_id,  # Pass as integer
                                      category=row['category'],
                                      score=float(row['score']),
                                      sentiment=row['sentiment'],
                                      timestamp=timestamp.isoformat())
                            
                        except Exception as e:
                            logger.warning(f"Error processing mental health category for video {video_id}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing mental health data for video {video_id}: {str(e)}")
                    continue

def load_all_data():
    """Load all data files with error handling"""
    try:
        logger.info("Loading main metadata...")
        main_df = pd.read_csv('Youtube-Analysis-Files/youtube_analysis_20250223_010128_main.csv')
        logger.info(f"Main data shape: {main_df.shape}")
        logger.info("Main data columns:")
        logger.info(main_df.columns.tolist())
        
        logger.info("\nLoading mental health data...")
        mental_health_df = pd.read_csv('Youtube-Analysis-Files/youtube_analysis_20250223_010128_mental_health.csv')
        logger.info(f"Mental health data shape: {mental_health_df.shape}")
        
        # Merge main data with mental health data
        logger.info("\nMerging main and mental health data...")
        merged_df = pd.merge(
            main_df,
            mental_health_df,
            on='video_id',
            how='left',
            suffixes=('', '_mh')
        )
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        logger.info("\nLoading engagement data...")
        engagement_df = pd.read_csv('Youtube-Analysis-Files/youtube_analysis_20250223_010128_engagement.csv')
        engagement_df['audience_engagement'] = engagement_df['audience_engagement'].str.split(',')
        engagement_df['production_quality'] = engagement_df['production_quality'].str.split(',')
        engagement_df['content_format'] = engagement_df['content_format'].str.split(',')
        logger.info(f"Engagement data shape: {engagement_df.shape}")
        
        return merged_df, engagement_df
    
    except Exception as e:
        logger.error(f"Error loading data files: {str(e)}")
        raise

def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    
    try:
        # Load all data files
        main_df = pd.read_csv('Youtube-Analysis-Files/youtube_analysis_20250223_010128_main.csv')
        engagement_df = pd.read_csv('Youtube-Analysis-Files/youtube_analysis_20250223_010128_engagement.csv')
        mental_health_df = pd.read_csv('Youtube-Analysis-Files/youtube_analysis_20250223_010128_mental_health.csv')
        
        # Process engagement data
        engagement_df['audience_engagement'] = engagement_df['audience_engagement'].str.split(',')
        engagement_df['production_quality'] = engagement_df['production_quality'].str.split(',')
        engagement_df['content_format'] = engagement_df['content_format'].str.split(',')
        
        # Connect and setup database
        neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        neo4j_conn.clear_database()
        neo4j_conn.create_indexes()
        
        # Load data in correct order:
        # 1. First create all video nodes
        logger.info("Loading main metadata...")
        neo4j_conn.load_main_metadata(main_df)
        
        # 2. Then add engagement relationships
        logger.info("Loading engagement data...")
        neo4j_conn.load_engagement_data(engagement_df)
        
        # 3. Finally add mental health relationships
        logger.info("Loading mental health data...")
        neo4j_conn.load_mental_health_data(mental_health_df)
        
        neo4j_conn.close()
        logger.info("All data successfully loaded into Neo4j!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 