import os
import json
import datetime
import webbrowser
import time
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

class YouTubeHistoryFetcher:
    def __init__(self, client_secrets_file="client_secret.json"):
        self.client_secrets_file = client_secrets_file
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
        self.credentials = None
        self.base_dir = Path("youtube_data")
        
    def authenticate(self):
        """Authenticate user with Google OAuth2"""
        # Create data directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)
        token_file = self.base_dir / "token.json"
        
        # Check if token already exists
        if token_file.exists():
            self.credentials = Credentials.from_authorized_user_file(str(token_file), self.scopes)
        
        # If credentials don't exist or are invalid, get new ones
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            else:
                print("Please authenticate with your Google account to access YouTube data.")
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file, self.scopes)
                self.credentials = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(token_file, "w") as token:
                token.write(self.credentials.to_json())
                
        return self.credentials
    
    def get_takeout_instructions(self):
        """Shows instructions on how to get YouTube history from Google Takeout"""
        print("\n=== GETTING YOUTUBE WATCH HISTORY ===")
        print("\nNOTE: YouTube API no longer provides direct access to watch history.")
        print("The best way to get your watch history is through Google Takeout.")
        print("\nInstructions:")
        print("1. Visit https://takeout.google.com/")
        print("2. Deselect all products")
        print("3. Scroll down and select only 'YouTube and YouTube Music'")
        print("4. Click 'All YouTube data included' and deselect everything except 'history'")
        print("5. Click 'OK' and then 'Next step'")
        print("6. Choose delivery method, frequency, and file type")
        print("7. Click 'Create export'")
        print("8. Google will email you when your export is ready to download")
        print("9. Download the export and extract watch-history.html or watch-history.json")
        print("10. Place the file in the user's directory in youtube_data folder")
        
        open_takeout = input("\nWould you like to open Google Takeout now? (y/n): ")
        if open_takeout.lower() == 'y':
            webbrowser.open("https://takeout.google.com/")
    
    def parse_history_from_takeout(self, username, file_path=None):
        """Parse watch history from Google Takeout file"""
        user_dir = self.base_dir / username
        user_dir.mkdir(exist_ok=True)
        
        if not file_path:
            # Look for watch-history files in user directory
            json_file = user_dir / "watch-history.json"
            html_file = user_dir / "watch-history.html"
            
            if json_file.exists():
                file_path = json_file
            elif html_file.exists():
                file_path = html_file
            else:
                print(f"No watch history file found in {user_dir}")
                print("Please download your history from Google Takeout and place it in this directory.")
                return None
        
        if str(file_path).endswith('.json'):
            return self._parse_json_history(file_path, username)
        elif str(file_path).endswith('.html'):
            print("HTML parsing not implemented yet. Please use the JSON file from Google Takeout.")
            return None
        else:
            print(f"Unsupported file format: {file_path}")
            return None
    
    def _parse_json_history(self, file_path, username):
        """Parse JSON watch history from Google Takeout"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter to get only videos from last week
            one_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
            
            # Extract video information
            recent_videos = []
            for item in data:
                if item.get('titleUrl') and item.get('time'):
                    try:
                        # Parse timestamp
                        timestamp = item['time']
                        # Google Takeout format: 2023-04-15T12:34:56.789Z
                        video_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                        
                        if video_time >= one_week_ago:
                            video_id = item['titleUrl'].split('v=')[1].split('&')[0]
                            recent_videos.append({
                                'video_id': video_id,
                                'title': item.get('title', ''),
                                'channel': item.get('subtitles', [{}])[0].get('name', '') if item.get('subtitles') else '',
                                'time': timestamp
                            })
                    except (KeyError, IndexError, ValueError) as e:
                        continue
            
            # Save filtered history
            output_file = self.base_dir / username / "recent_watch_history.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(recent_videos, f, indent=2)
                
            print(f"Extracted {len(recent_videos)} videos watched in the last week.")
            print(f"Results saved to {output_file}")
            
            return recent_videos
            
        except Exception as e:
            print(f"Error parsing watch history: {e}")
            return None
    
    def adjust_time_range(self, username, days=7):
        """Adjust the time range for history retrieval"""
        history_file = self.base_dir / username / "watch-history.json"
        
        if not history_file.exists():
            print(f"No watch history file found at {history_file}")
            return None
            
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter to get videos from specified days
            time_ago = datetime.datetime.now() - datetime.timedelta(days=days)
            
            # Extract video information
            filtered_videos = []
            for item in data:
                if item.get('titleUrl') and item.get('time'):
                    try:
                        # Parse timestamp
                        timestamp = item['time']
                        video_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                        
                        if video_time >= time_ago:
                            video_id = item['titleUrl'].split('v=')[1].split('&')[0]
                            filtered_videos.append({
                                'video_id': video_id,
                                'title': item.get('title', ''),
                                'channel': item.get('subtitles', [{}])[0].get('name', '') if item.get('subtitles') else '',
                                'time': timestamp
                            })
                    except (KeyError, IndexError, ValueError):
                        continue
            
            # Save filtered history
            output_file = self.base_dir / username / f"recent_watch_history_{days}days.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_videos, f, indent=2)
                
            print(f"Extracted {len(filtered_videos)} videos watched in the last {days} days.")
            print(f"Results saved to {output_file}")
            
            return filtered_videos
            
        except Exception as e:
            print(f"Error adjusting time range: {e}")
            return None

def main():
    print("YouTube Watch History Fetcher")
    print("-----------------------------")
    
    username = input("Enter username (used for organizing data): ")
    
    fetcher = YouTubeHistoryFetcher()
    
    # Show Google Takeout instructions
    fetcher.get_takeout_instructions()
    
    # Wait for user to download and place file
    proceed = input("\nHave you placed your watch-history.json file in the data directory? (y/n): ")
    
    if proceed.lower() == 'y':
        # Create user directory if it doesn't exist
        user_dir = fetcher.base_dir / username
        user_dir.mkdir(exist_ok=True)
        
        # Parse history
        file_path = input("Enter the path to your watch-history.json file (or press Enter to use default location): ")
        if file_path:
            fetcher.parse_history_from_takeout(username, file_path)
        else:
            fetcher.parse_history_from_takeout(username)
        
        # Adjust time range if needed
        adjust = input("\nWould you like to adjust the time range? (y/n): ")
        if adjust.lower() == 'y':
            days = int(input("Enter number of days to include: "))
            fetcher.adjust_time_range(username, days)
    
    print("\nProcess completed.")

if __name__ == "__main__":
    main() 