from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyMusicAnalyzer:
    def __init__(self, client_id=None, client_secret=None):
        self.sp = Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
    
    def get_track_info(self, title):
        """Get track information from Spotify"""
        try:
            # Search for the track
            results = self.sp.search(q=title, type='track', limit=1)
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                
                # Get audio features
                audio_features = self.sp.audio_features([track['id']])[0]
                
                # Get artist genres
                artist = self.sp.artist(track['artists'][0]['id'])
                
                return {
                    'genres': artist['genres'],
                    'popularity': track['popularity'],
                    'audio_features': audio_features,
                    'source': 'spotify'
                }
        except Exception as e:
            print(f"[ERROR] Spotify analysis failed: {str(e)}")
        return None