 import requests
from typing import Optional, Dict, List

class LastFmAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
    
    def get_artist_info(self, artist_name: str) -> Optional[Dict]:
        """Get artist information from Last.fm"""
        params = {
            'method': 'artist.getinfo',
            'artist': artist_name,
            'api_key': self.api_key,
            'format': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'artist' in data:
                return {
                    'tags': [tag['name'] for tag in data['artist']['tags']['tag']],
                    'listeners': data['artist'].get('stats', {}).get('listeners'),
                    'playcount': data['artist'].get('stats', {}).get('playcount'),
                    'source': 'lastfm'
                }
        except Exception as e:
            print(f"[ERROR] Last.fm API request failed: {str(e)}")
        return None