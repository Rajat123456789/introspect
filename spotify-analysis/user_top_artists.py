import os
import json
from datetime import datetime
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load environment variables
load_dotenv()

def setup_spotify():
    """Set up Spotify client with proper authentication"""
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv('SPOTIFY_CLIENT_ID'),
        client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
        redirect_uri='http://localhost:8888/callback',  # Default Spotipy redirect URI
        scope='user-top-read'
    ))

def save_to_json(data, filename):
    """Save data to a JSON file with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename}_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nData saved to {filename}")

def get_top_artists(sp, time_range='medium_term', limit=20):
    """Fetch user's top artists for a given time range"""
    try:
        results = sp.current_user_top_artists(
            limit=limit,
            offset=0,
            time_range=time_range
        )
        return results['items']
    except Exception as e:
        print(f"Error fetching top artists: {str(e)}")
        return None

def display_artist_info(artist, rank):
    """Display formatted artist information"""
    print(f"\n{rank}. {artist['name']}")
    print("   " + "="*30)
    print(f"   Genres: {', '.join(artist['genres'])}")
    print(f"   Popularity: {artist['popularity']}/100")
    print(f"   Followers: {artist['followers']['total']:,}")
    print(f"   Spotify URL: {artist['external_urls']['spotify']}")

def main():
    try:
        # Initialize Spotify client
        print("Initializing Spotify client...")
        sp = setup_spotify()
        
        # Get user profile
        user_profile = sp.current_user()
        print(f"\nFetching top artists for user: {user_profile['display_name']}")
        
        # Time ranges to fetch
        time_ranges = {
            'short_term': 'Last 4 Weeks',
            'medium_term': 'Last 6 Months',
            'long_term': 'All Time'
        }
        
        # Store all data for JSON export
        all_data = {
            'user': user_profile,
            'timestamp': datetime.now().isoformat(),
            'top_artists': {}
        }
        
        # Fetch and display top artists for each time range
        for time_range, label in time_ranges.items():
            print(f"\n=== Top Artists - {label} ===")
            artists = get_top_artists(sp, time_range=time_range)
            
            if artists:
                all_data['top_artists'][time_range] = artists
                for idx, artist in enumerate(artists, 1):
                    display_artist_info(artist, idx)
            else:
                print(f"No data available for {label}")
        
        # Save all data to JSON file
        save_to_json(all_data, 'spotify_top_artists')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Ensure your .env file contains valid SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        print("2. Check your internet connection")
        print("3. Make sure you've set up your Spotify Developer application correctly")

if __name__ == '__main__':
    main()