import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neo4j import GraphDatabase
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyInsights:
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
    
    def get_top_artists(self, limit=10):
        """Get top artists by track count"""
        query = """
        MATCH (t:Track)-[:PERFORMED_BY]->(a:Artist)
        WITH a, count(DISTINCT t) as track_count
        RETURN a.name as artist, track_count
        ORDER BY track_count DESC
        LIMIT $limit
        """
        return self.query_to_dataframe(query, {"limit": limit})
    
    def get_top_tracks(self, limit=10):
        """Get top tracks by listen count"""
        query = """
        MATCH (l:Listening)-[:OF_TRACK]->(t:Track)
        RETURN t.name as track, count(l) as listen_count
        ORDER BY listen_count DESC
        LIMIT $limit
        """
        return self.query_to_dataframe(query, {"limit": limit})
    
    def get_top_genres(self, limit=10):
        """Get top genres by listen count"""
        query = """
        MATCH (l:Listening)-[:OF_TRACK]->(t:Track)-[:BELONGS_TO]->(g:Genre)
        RETURN g.name as genre, count(l) as listen_count
        ORDER BY listen_count DESC
        LIMIT $limit
        """
        return self.query_to_dataframe(query, {"limit": limit})
    
    def get_listening_by_hour(self):
        """Get listening patterns by hour"""
        query = """
        MATCH (l:Listening)
        RETURN substring(l.end_time, 11, 2) as hour, count(*) as listen_count
        ORDER BY hour
        """
        return self.query_to_dataframe(query)
    
    def get_audio_features_by_genre(self, features=None):
        """Get average audio features by genre"""
        if features is None:
            features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
            
        feature_clauses = [f"avg(t.{feature}) as {feature}" for feature in features]
        query = f"""
        MATCH (t:Track)-[:BELONGS_TO]->(g:Genre)
        RETURN g.name as genre, {', '.join(feature_clauses)}, count(t) as track_count
        ORDER BY track_count DESC
        """
        return self.query_to_dataframe(query)
    
    def get_similar_songs(self, track_name, limit=10):
        """Find similar songs based on audio features"""
        query = """
        MATCH (t1:Track)
        WHERE t1.name CONTAINS $track_name
        WITH t1 LIMIT 1
        MATCH (t2:Track)-[:PERFORMED_BY]->(a:Artist)
        WHERE t1 <> t2
        WITH t1, t2, a,
             abs(t1.danceability - t2.danceability) +
             abs(t1.energy - t2.energy) +
             abs(t1.valence - t2.valence) +
             abs(t1.acousticness - t2.acousticness) as difference
        ORDER BY difference ASC
        LIMIT $limit
        RETURN t2.name as track, a.name as artist, difference
        """
        return self.query_to_dataframe(query, {"track_name": track_name, "limit": limit})
    
    def get_artist_collaborations(self, limit=20):
        """Find artist collaborations (artists who performed on the same tracks)"""
        query = """
        MATCH (a1:Artist)<-[:PERFORMED_BY]-(t:Track)-[:PERFORMED_BY]->(a2:Artist)
        WHERE a1.name < a2.name
        WITH a1, a2, count(t) as collaboration_count
        WHERE collaboration_count > 1
        RETURN a1.name as artist1, a2.name as artist2, collaboration_count
        ORDER BY collaboration_count DESC
        LIMIT $limit
        """
        return self.query_to_dataframe(query, {"limit": limit})
    
    def plot_top_artists(self, limit=10):
        """Plot top artists by track count"""
        df = self.get_top_artists(limit)
        if df.empty:
            logger.warning("No data available for top artists")
            return
            
        plt.figure(figsize=(12, 6))
        sns.barplot(x='track_count', y='artist', data=df)
        plt.title('Top Artists by Number of Tracks')
        plt.tight_layout()
        plt.savefig('top_artists.png')
        logger.info("Saved top artists plot to 'top_artists.png'")
        
    def plot_top_genres(self, limit=10):
        """Plot top genres by listen count"""
        df = self.get_top_genres(limit)
        if df.empty:
            logger.warning("No data available for top genres")
            return
            
        plt.figure(figsize=(12, 6))
        sns.barplot(x='listen_count', y='genre', data=df)
        plt.title('Top Genres by Listening Count')
        plt.tight_layout()
        plt.savefig('top_genres.png')
        logger.info("Saved top genres plot to 'top_genres.png'")
        
    def plot_listening_by_hour(self):
        """Plot listening patterns by hour"""
        df = self.get_listening_by_hour()
        if df.empty:
            logger.warning("No data available for listening by hour")
            return
            
        # Convert hour to numeric
        df['hour'] = pd.to_numeric(df['hour'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='hour', y='listen_count', data=df)
        plt.title('Listening Patterns by Hour of Day')
        plt.xlabel('Hour of Day (24-hour format)')
        plt.ylabel('Number of Listens')
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig('listening_by_hour.png')
        logger.info("Saved listening by hour plot to 'listening_by_hour.png'")
        
    def plot_genre_audio_features(self, top_n=5):
        """Plot radar charts for audio features by genre"""
        df = self.get_audio_features_by_genre()
        if df.empty:
            logger.warning("No data available for genre audio features")
            return
            
        # Get top N genres by track count
        top_genres = df.nlargest(top_n, 'track_count')
        
        # Features to include in the radar chart
        features = ['danceability', 'energy', 'valence', 'acousticness']
        
        # Create radar chart
        plt.figure(figsize=(10, 8))
        
        # Number of variables
        N = len(features)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set up the plot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], features)
        
        # Draw the genre data
        for i, genre in enumerate(top_genres['genre']):
            values = top_genres.loc[top_genres['genre'] == genre, features].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=genre)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Audio Features by Genre')
        plt.tight_layout()
        plt.savefig('genre_audio_features.png')
        logger.info("Saved genre audio features plot to 'genre_audio_features.png'")

def main():
    # Connection parameters - modify these to match your Neo4j setup
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"  # Change this to your actual password
    
    try:
        insights = SpotifyInsights(uri, username, password)
        
        # Generate insights
        logger.info("Generating insights...")
        
        # Get and display top artists
        top_artists = insights.get_top_artists()
        logger.info("Top Artists:")
        logger.info(top_artists)
        
        # Get and display top tracks
        top_tracks = insights.get_top_tracks()
        logger.info("Top Tracks:")
        logger.info(top_tracks)
        
        # Get and display top genres
        top_genres = insights.get_top_genres()
        logger.info("Top Genres:")
        logger.info(top_genres)
        
        # Get and display listening by hour
        listening_by_hour = insights.get_listening_by_hour()
        logger.info("Listening by Hour:")
        logger.info(listening_by_hour)
        
        # Get and display similar songs for a sample track
        if not top_tracks.empty:
            sample_track = top_tracks.iloc[0]['track']
            similar_songs = insights.get_similar_songs(sample_track)
            logger.info(f"Songs similar to '{sample_track}':")
            logger.info(similar_songs)
        
        # Generate plots
        logger.info("Generating plots...")
        insights.plot_top_artists()
        insights.plot_top_genres()
        insights.plot_listening_by_hour()
        insights.plot_genre_audio_features()
        
        # Close connection
        insights.close()
        
        logger.info("All insights and visualizations have been generated!")
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise

if __name__ == "__main__":
    main() 