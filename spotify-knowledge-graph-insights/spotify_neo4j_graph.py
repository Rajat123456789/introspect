import pandas as pd
from neo4j import GraphDatabase
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyKnowledgeGraph:
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
        
        # Check if constraints already exist and create them if they don't
        try:
            # For Neo4j 4.x and later
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Track) REQUIRE t.track_id IS UNIQUE")
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE")
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Album) REQUIRE (a.name, a.artist) IS UNIQUE")
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
            self.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Listening) REQUIRE l.id IS UNIQUE")
        except Exception as e:
            # For Neo4j 3.x
            logger.warning(f"Error creating constraints: {e}. Trying alternative syntax...")
            try:
                self.run_query("CREATE CONSTRAINT ON (t:Track) ASSERT t.track_id IS UNIQUE")
                self.run_query("CREATE CONSTRAINT ON (a:Artist) ASSERT a.name IS UNIQUE")
                self.run_query("CREATE CONSTRAINT ON (a:Album) ASSERT (a.name, a.artist) IS UNIQUE")
                self.run_query("CREATE CONSTRAINT ON (g:Genre) ASSERT g.name IS UNIQUE")
                self.run_query("CREATE CONSTRAINT ON (l:Listening) ASSERT l.id IS UNIQUE")
            except Exception as e2:
                logger.error(f"Failed to create constraints: {e2}")
                
        logger.info("Constraints created")
        
    def load_spotify_data(self, csv_path):
        """Load Spotify history data from CSV and create the knowledge graph"""
        logger.info(f"Loading data from {csv_path}...")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Clean data and handle duplicates
        df = df.dropna(subset=['track_id'])
        
        # Create genres
        logger.info("Creating Genre nodes...")
        genres = df['track_genre'].dropna().unique()
        for genre in genres:
            query = """
            MERGE (g:Genre {name: $genre})
            """
            self.run_query(query, {"genre": genre})
            
        # Create artists
        logger.info("Creating Artist nodes...")
        artists = df['artists'].dropna().unique()
        for artist in artists:
            # Handle multiple artists separated by semicolons
            for individual_artist in artist.split(';'):
                individual_artist = individual_artist.strip()
                if individual_artist:
                    query = """
                    MERGE (a:Artist {name: $artist})
                    """
                    self.run_query(query, {"artist": individual_artist})
        
        # Create albums
        logger.info("Creating Album nodes and relationships...")
        album_data = df[['album_name_y', 'artists']].dropna().drop_duplicates()
        for _, row in album_data.iterrows():
            album_name = row['album_name_y']
            main_artist = row['artists'].split(';')[0].strip()  # Use first artist as main
            
            query = """
            MATCH (a:Artist {name: $artist})
            MERGE (album:Album {name: $album, artist: $artist})
            MERGE (album)-[:CREATED_BY]->(a)
            """
            self.run_query(query, {"album": album_name, "artist": main_artist})
        
        # Create tracks and relationships
        logger.info("Creating Track nodes and relationships...")
        track_count = len(df['track_id'].unique())
        processed = 0
        
        for _, row in df.drop_duplicates(subset=['track_id']).iterrows():
            track_id = row['track_id']
            track_name = row['track_name_y'] if not pd.isna(row['track_name_y']) else row['track_name_x']
            album_name = row['album_name_y'] if not pd.isna(row['album_name_y']) else row['album_name_x']
            
            # Get artists - could be multiple
            if not pd.isna(row['artists']):
                artists_list = [a.strip() for a in row['artists'].split(';')]
                main_artist = artists_list[0]  # First artist is considered main artist
            else:
                main_artist = row['artist_name']
                artists_list = [main_artist]
            
            # Get audio features
            audio_features = {}
            for feature in ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 
                            'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                            'liveness', 'valence', 'tempo', 'time_signature']:
                if feature in row and not pd.isna(row[feature]):
                    audio_features[feature] = float(row[feature]) if feature != 'explicit' else bool(row[feature])
            
            # Get genre
            genre = row['track_genre'] if not pd.isna(row['track_genre']) else None
            
            # Create track node with audio features
            query = """
            MERGE (t:Track {track_id: $track_id})
            ON CREATE SET t.name = $name, t += $features
            """
            self.run_query(query, {"track_id": track_id, "name": track_name, "features": audio_features})
            
            # Connect track to album
            if album_name and main_artist:
                query = """
                MATCH (t:Track {track_id: $track_id})
                MATCH (a:Album {name: $album, artist: $artist})
                MERGE (t)-[:PART_OF]->(a)
                """
                self.run_query(query, {"track_id": track_id, "album": album_name, "artist": main_artist})
            
            # Connect track to artists
            for artist in artists_list:
                query = """
                MATCH (t:Track {track_id: $track_id})
                MATCH (a:Artist {name: $artist})
                MERGE (t)-[:PERFORMED_BY]->(a)
                """
                self.run_query(query, {"track_id": track_id, "artist": artist})
            
            # Connect track to genre
            if genre:
                query = """
                MATCH (t:Track {track_id: $track_id})
                MATCH (g:Genre {name: $genre})
                MERGE (t)-[:BELONGS_TO]->(g)
                """
                self.run_query(query, {"track_id": track_id, "genre": genre})
            
            processed += 1
            if processed % 50 == 0:
                logger.info(f"Processed {processed}/{track_count} tracks")
        
        # Create listening events
        logger.info("Creating Listening events...")
        for idx, row in df.iterrows():
            if pd.isna(row['track_id']) or pd.isna(row['end_time']):
                continue
                
            track_id = row['track_id']
            end_time = row['end_time']
            ms_played = float(row['ms_played']) if not pd.isna(row['ms_played']) else 0
            
            # Create unique ID for listening event
            listening_id = f"{track_id}_{end_time}_{idx}"
            
            query = """
            MATCH (t:Track {track_id: $track_id})
            MERGE (l:Listening {id: $listening_id})
            ON CREATE SET l.end_time = $end_time, l.ms_played = $ms_played
            MERGE (l)-[:OF_TRACK]->(t)
            """
            self.run_query(query, {
                "track_id": track_id,
                "listening_id": listening_id,
                "end_time": end_time,
                "ms_played": ms_played
            })
            
        logger.info("Knowledge graph created successfully!")
        
    def create_indexes(self):
        """Create indexes for better performance"""
        logger.info("Creating indexes...")
        
        try:
            # For Neo4j 4.x and later
            self.run_query("CREATE INDEX track_name_idx IF NOT EXISTS FOR (t:Track) ON (t.name)")
            self.run_query("CREATE INDEX listening_time_idx IF NOT EXISTS FOR (l:Listening) ON (l.end_time)")
        except Exception as e:
            # For Neo4j 3.x
            logger.warning(f"Error creating indexes: {e}. Trying alternative syntax...")
            try:
                self.run_query("CREATE INDEX ON :Track(name)")
                self.run_query("CREATE INDEX ON :Listening(end_time)")
            except Exception as e2:
                logger.error(f"Failed to create indexes: {e2}")
                
        logger.info("Indexes created")
        
    def get_statistics(self):
        """Get statistics about the knowledge graph"""
        stats = {}
        
        # Count nodes by label
        for label in ["Track", "Artist", "Album", "Genre", "Listening"]:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = self.run_query(query)
            stats[f"{label}_count"] = result[0]["count"]
        
        # Count relationships
        query = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
        result = self.run_query(query)
        for record in result:
            stats[f"rel_{record['type']}"] = record["count"]
            
        return stats

def main():
    # Connection parameters - modify these to match your Neo4j setup
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"  # Change this to your actual password
    
    # CSV file path
    csv_path = "songs-parameters/spotify-history-annotated.csv"
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), csv_path)
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at {csv_path}")
            return
    
    # Initialize knowledge graph
    try:
        kg = SpotifyKnowledgeGraph(uri, username, password)
        
        # Clear database if needed (uncomment to clear on each run)
        # kg.clear_database()
        
        # Create constraints and indexes
        kg.create_constraints()
        
        # Load data
        kg.load_spotify_data(csv_path)
        
        # Create indexes for performance
        kg.create_indexes()
        
        # Get and display statistics
        stats = kg.get_statistics()
        logger.info("Knowledge Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Close connection
        kg.close()
        
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        raise

if __name__ == "__main__":
    main() 