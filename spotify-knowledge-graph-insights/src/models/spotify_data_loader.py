import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SpotifyDataLoader:
    """Handles loading Spotify data into Neo4j"""
    
    def __init__(self, neo4j_connection):
        """Initialize with a Neo4j connection"""
        self.conn = neo4j_connection
    
    def load_track_data(self, df):
        """Load track nodes with their properties"""
        with self.conn.driver.session() as session:
            # Group by track_id to avoid duplicates
            for track_id, track_data in df.groupby('track_id'):
                if pd.isna(track_id):
                    continue
                    
                # Get first row for track properties
                row = track_data.iloc[0]
                
                try:
                    # Create track node
                    track_query = """
                    MERGE (t:Track {track_id: $track_id})
                    SET t.name = $name,
                        t.popularity = $popularity,
                        t.duration_ms = $duration_ms,
                        t.explicit = $explicit,
                        t.danceability = $danceability,
                        t.energy = $energy,
                        t.key = $key,
                        t.loudness = $loudness,
                        t.mode = $mode,
                        t.speechiness = $speechiness,
                        t.acousticness = $acousticness,
                        t.instrumentalness = $instrumentalness,
                        t.liveness = $liveness,
                        t.valence = $valence,
                        t.tempo = $tempo,
                        t.time_signature = $time_signature
                    """
                    
                    session.run(track_query, {
                        "track_id": track_id,
                        "name": str(row.get('track_name_y', row.get('track_name_x', ''))),
                        "popularity": float(row.get('popularity', 0)) if pd.notna(row.get('popularity')) else 0,
                        "duration_ms": float(row.get('duration_ms', 0)) if pd.notna(row.get('duration_ms')) else 0,
                        "explicit": bool(row.get('explicit', False)) if pd.notna(row.get('explicit')) else False,
                        "danceability": float(row.get('danceability', 0)) if pd.notna(row.get('danceability')) else 0,
                        "energy": float(row.get('energy', 0)) if pd.notna(row.get('energy')) else 0,
                        "key": float(row.get('key', 0)) if pd.notna(row.get('key')) else 0,
                        "loudness": float(row.get('loudness', 0)) if pd.notna(row.get('loudness')) else 0,
                        "mode": float(row.get('mode', 0)) if pd.notna(row.get('mode')) else 0,
                        "speechiness": float(row.get('speechiness', 0)) if pd.notna(row.get('speechiness')) else 0,
                        "acousticness": float(row.get('acousticness', 0)) if pd.notna(row.get('acousticness')) else 0,
                        "instrumentalness": float(row.get('instrumentalness', 0)) if pd.notna(row.get('instrumentalness')) else 0,
                        "liveness": float(row.get('liveness', 0)) if pd.notna(row.get('liveness')) else 0,
                        "valence": float(row.get('valence', 0)) if pd.notna(row.get('valence')) else 0,
                        "tempo": float(row.get('tempo', 0)) if pd.notna(row.get('tempo')) else 0,
                        "time_signature": float(row.get('time_signature', 4)) if pd.notna(row.get('time_signature')) else 4
                    })
                    
                except Exception as e:
                    logger.warning(f"Error creating track node for {track_id}: {str(e)}")
                    continue
    
    def load_artists_and_relationships(self, df):
        """Load artist nodes and their relationships to tracks"""
        with self.conn.driver.session() as session:
            for _, row in df.iterrows():
                if pd.isna(row.get('track_id')) or pd.isna(row.get('artists')):
                    continue
                    
                try:
                    # Handle multiple artists (semicolon separated)
                    artists = str(row.get('artists', '')).split(';')
                    
                    for artist in artists:
                        artist = artist.strip()
                        if artist:
                            artist_query = """
                            MERGE (a:Artist {name: $artist_name})
                            WITH a
                            MATCH (t:Track {track_id: $track_id})
                            MERGE (a)-[:PERFORMED]->(t)
                            """
                            
                            session.run(artist_query, {
                                "artist_name": artist,
                                "track_id": row['track_id']
                            })
                
                except Exception as e:
                    logger.warning(f"Error processing artist for track {row.get('track_id', '')}: {str(e)}")
                    continue
    
    def load_albums_and_relationships(self, df):
        """Load album nodes and their relationships to tracks and artists"""
        with self.conn.driver.session() as session:
            for _, row in df.iterrows():
                if pd.isna(row.get('track_id')) or pd.isna(row.get('album_name_y')) and pd.isna(row.get('album_name_x')):
                    continue
                    
                try:
                    album_name = str(row.get('album_name_y', row.get('album_name_x', '')))
                    
                    if album_name:
                        # Create album and relationship to track
                        album_query = """
                        MERGE (a:Album {name: $album_name})
                        WITH a
                        MATCH (t:Track {track_id: $track_id})
                        MERGE (a)-[:CONTAINS]->(t)
                        """
                        
                        session.run(album_query, {
                            "album_name": album_name,
                            "track_id": row['track_id']
                        })
                        
                        # Create relationships between artists and albums
                        if pd.notna(row.get('artists')):
                            artists = str(row.get('artists', '')).split(';')
                            
                            for artist in artists:
                                artist = artist.strip()
                                if artist:
                                    artist_album_query = """
                                    MATCH (a:Artist {name: $artist_name})
                                    MATCH (al:Album {name: $album_name})
                                    MERGE (a)-[:RELEASED]->(al)
                                    """
                                    
                                    session.run(artist_album_query, {
                                        "artist_name": artist,
                                        "album_name": album_name
                                    })
                
                except Exception as e:
                    logger.warning(f"Error processing album for track {row.get('track_id', '')}: {str(e)}")
                    continue
    
    def load_genres_and_relationships(self, df):
        """Load genre nodes and their relationships to tracks"""
        with self.conn.driver.session() as session:
            for _, row in df.iterrows():
                if pd.isna(row.get('track_id')) or pd.isna(row.get('track_genre')):
                    continue
                    
                try:
                    genre = str(row.get('track_genre', '')).strip()
                    
                    if genre:
                        genre_query = """
                        MERGE (g:Genre {name: $genre})
                        WITH g
                        MATCH (t:Track {track_id: $track_id})
                        MERGE (t)-[:HAS_GENRE]->(g)
                        """
                        
                        session.run(genre_query, {
                            "genre": genre,
                            "track_id": row['track_id']
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing genre for track {row.get('track_id', '')}: {str(e)}")
                    continue
    
    def load_listening_sessions(self, df):
        """Load user listening sessions as relationships"""
        with self.conn.driver.session() as session:
            # Create user node once
            session.run("MERGE (u:User {id: 'spotify_user'})")
            
            for _, row in df.iterrows():
                if pd.isna(row.get('track_id')) or pd.isna(row.get('end_time')):
                    continue
                    
                try:
                    # Parse timestamp 
                    end_time = pd.to_datetime(row['end_time'])
                    
                    # Create listening session relationship
                    listen_query = """
                    MATCH (u:User {id: 'spotify_user'})
                    MATCH (t:Track {track_id: $track_id})
                    CREATE (u)-[l:LISTENED_TO {
                        end_time: datetime($end_time),
                        ms_played: $ms_played
                    }]->(t)
                    """
                    
                    session.run(listen_query, {
                        "track_id": row['track_id'],
                        "end_time": end_time.isoformat(),
                        "ms_played": int(row.get('ms_played', 0)) if pd.notna(row.get('ms_played')) else 0
                    })
                
                except Exception as e:
                    logger.warning(f"Error processing listening session for track {row.get('track_id', '')}: {str(e)}")
                    continue 