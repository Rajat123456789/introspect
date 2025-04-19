import os
import logging
from typing import Dict, List, Optional
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyClient:
    """A client for interacting with the Spotify Web API."""
    
    def __init__(self):
        """Initialize the Spotify client using credentials from environment variables."""
        load_dotenv()
        
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            raise ValueError("Spotify credentials not found in environment variables")
        
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Spotify client initialized successfully")
    
    def get_track_info(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed information about a track.
        
        Args:
            track_id (str): Spotify track ID
            
        Returns:
            dict: Track information including name, artist, album, etc.
        """
        try:
            track = self.sp.track(track_id)
            return {
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'duration_ms': track['duration_ms'],
                'popularity': track['popularity'],
                'preview_url': track['preview_url'],
                'external_url': track['external_urls']['spotify']
            }
        except Exception as e:
            logger.error(f"Error getting track info for {track_id}: {str(e)}")
            return None
    
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a track.
        
        Args:
            track_id (str): Spotify track ID
            
        Returns:
            dict: Audio features including tempo, key, mode, etc.
        """
        try:
            features = self.sp.audio_features(track_id)[0]
            if features:
                return {
                    'tempo': features['tempo'],
                    'key': features['key'],
                    'mode': features['mode'],
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'loudness': features['loudness'],
                    'valence': features['valence'],
                    'instrumentalness': features['instrumentalness']
                }
            return None
        except Exception as e:
            logger.error(f"Error getting audio features for {track_id}: {str(e)}")
            return None
    
    def get_artist_info(self, artist_id: str) -> Optional[Dict]:
        """
        Get detailed information about an artist.
        
        Args:
            artist_id (str): Spotify artist ID
            
        Returns:
            dict: Artist information including name, genres, popularity, etc.
        """
        try:
            artist = self.sp.artist(artist_id)
            return {
                'id': artist['id'],
                'name': artist['name'],
                'genres': artist['genres'],
                'popularity': artist['popularity'],
                'followers': artist['followers']['total']
            }
        except Exception as e:
            logger.error(f"Error getting artist info for {artist_id}: {str(e)}")
            return None
    
    def get_recommendations(self, seed_tracks: List[str], limit: int = 10) -> List[Dict]:
        """
        Get track recommendations based on seed tracks.
        
        Args:
            seed_tracks (List[str]): List of Spotify track IDs to use as seeds
            limit (int): Maximum number of recommendations to return
            
        Returns:
            List[dict]: List of recommended tracks with their information
        """
        try:
            recommendations = self.sp.recommendations(
                seed_tracks=seed_tracks[:5],  # Spotify API allows max 5 seed tracks
                limit=limit
            )
            
            return [{
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name']
            } for track in recommendations['tracks']]
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for tracks using a query string.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            
        Returns:
            List[dict]: List of tracks matching the search query
        """
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks = results['tracks']['items']
            
            return [{
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name']
            } for track in tracks]
        except Exception as e:
            logger.error(f"Error searching tracks with query '{query}': {str(e)}")
            return []

    def get_track_analysis(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed audio analysis for a track.
        
        Args:
            track_id (str): Spotify track ID
            
        Returns:
            Optional[Dict]: Audio analysis or None if not found
        """
        try:
            analysis = self.sp.audio_analysis(track_id)
            return {
                'num_samples': analysis['track']['num_samples'],
                'duration': analysis['track']['duration'],
                'tempo': analysis['track']['tempo'],
                'time_signature': analysis['track']['time_signature'],
                'key': analysis['track']['key'],
                'mode': analysis['track']['mode'],
                'num_segments': len(analysis['segments']),
                'num_sections': len(analysis['sections']),
                'num_bars': len(analysis['bars']),
                'num_beats': len(analysis['beats'])
            }
        except Exception as e:
            logger.error(f"Error fetching audio analysis for {track_id}: {str(e)}")
            return None
            
    def get_related_artists(self, artist_id: str) -> List[Dict]:
        """
        Get list of artists related to the given artist.
        
        Args:
            artist_id (str): Spotify artist ID
            
        Returns:
            List[Dict]: List of related artists
        """
        try:
            related = self.sp.artist_related_artists(artist_id)
            return [{
                'id': artist['id'],
                'name': artist['name'],
                'genres': artist['genres'],
                'popularity': artist['popularity']
            } for artist in related['artists']]
        except Exception as e:
            logger.error(f"Error fetching related artists for {artist_id}: {str(e)}")
            return []
            
    def get_artist_top_tracks(self, artist_id: str, country: str = 'US') -> List[Dict]:
        """
        Get an artist's top tracks.
        
        Args:
            artist_id (str): Spotify artist ID
            country (str): Country code for market
            
        Returns:
            List[Dict]: List of top tracks
        """
        try:
            results = self.sp.artist_top_tracks(artist_id, country)
            return [{
                'id': track['id'],
                'name': track['name'],
                'popularity': track['popularity'],
                'preview_url': track['preview_url'],
                'external_url': track['external_urls']['spotify']
            } for track in results['tracks']]
        except Exception as e:
            logger.error(f"Error fetching top tracks for {artist_id}: {str(e)}")
            return [] 