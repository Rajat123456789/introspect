import os
import json
import requests
from pathlib import Path
import time
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

class YouTubeTranscriptFetcher:
    def __init__(self):
        self.base_dir = Path("youtube_data")
        self.transcript_dir = Path("youtube_transcripts")
        self.transcript_api = YouTubeTranscriptApi()
        
        # Create directories if they don't exist
        self.base_dir.mkdir(exist_ok=True)
        self.transcript_dir.mkdir(exist_ok=True)
    
    def fetch_transcript_for_video(self, video_id, languages=None):
        """Fetch transcript for a specific video"""
        if languages is None:
            languages = ['en']
        
        try:
            # Try getting transcript in specified languages
            transcript = self.transcript_api.get_transcript(video_id, languages=languages)
            return {
                'video_id': video_id,
                'success': True,
                'transcript': transcript
            }
        except TranscriptsDisabled:
            return {
                'video_id': video_id,
                'success': False,
                'error': 'Transcripts are disabled for this video'
            }
        except NoTranscriptFound:
            return {
                'video_id': video_id,
                'success': False,
                'error': f'No transcript found in languages: {languages}'
            }
        except Exception as e:
            return {
                'video_id': video_id,
                'success': False,
                'error': str(e)
            }
    
    def fetch_transcripts_for_user(self, username, days=7):
        """Fetch transcripts for all videos in a user's recent history"""
        user_dir = self.base_dir / username
        history_file = user_dir / f"recent_watch_history_{days}days.json"
        
        if not history_file.exists():
            # Try the default file
            history_file = user_dir / "recent_watch_history.json"
            if not history_file.exists():
                print(f"No watch history file found for user {username}")
                return False
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                
            print(f"Found {len(history)} videos in history. Fetching transcripts...")
            
            # Create user transcript directory
            user_transcript_dir = self.transcript_dir / username
            user_transcript_dir.mkdir(exist_ok=True)
            
            results = []
            success_count = 0
            
            for i, video in enumerate(history):
                video_id = video['video_id']
                print(f"[{i+1}/{len(history)}] Fetching transcript for: {video['title']}")
                
                # Try to get transcript
                result = self.fetch_transcript_for_video(video_id)
                
                # Add video info to result
                result['title'] = video.get('title', '')
                result['channel'] = video.get('channel', '')
                result['time'] = video.get('time', '')
                
                # Save to results list
                results.append(result)
                
                if result['success']:
                    success_count += 1
                    # Save individual transcript
                    transcript_file = user_transcript_dir / f"{video_id}.json"
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                
                # Sleep to avoid hitting rate limits
                time.sleep(0.5)
            
            # Save all results to a single file
            result_file = user_transcript_dir / "all_transcripts.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nTranscript fetching completed.")
            print(f"Successfully fetched {success_count} out of {len(history)} transcripts.")
            print(f"Results saved to {result_file}")
            
            # Create a simple text file with all transcripts
            self.create_text_transcripts(username)
            
            return True
            
        except Exception as e:
            print(f"Error fetching transcripts: {e}")
            return False
    
    def create_text_transcripts(self, username):
        """Create a text version of all fetched transcripts"""
        user_transcript_dir = self.transcript_dir / username
        all_transcripts_file = user_transcript_dir / "all_transcripts.json"
        
        if not all_transcripts_file.exists():
            print(f"No transcript data found for user {username}")
            return False
        
        try:
            with open(all_transcripts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create text directory
            text_dir = user_transcript_dir / "text"
            text_dir.mkdir(exist_ok=True)
            
            # Process each video transcript
            for video_data in data:
                if not video_data['success']:
                    continue
                
                video_id = video_data['video_id']
                title = video_data['title']
                channel = video_data['channel']
                transcript = video_data['transcript']
                
                # Format transcript as text
                text_content = f"Title: {title}\n"
                text_content += f"Channel: {channel}\n"
                text_content += f"Video ID: {video_id}\n\n"
                text_content += "TRANSCRIPT:\n\n"
                
                for entry in transcript:
                    text_content += f"{entry['text']}\n"
                
                # Save to file
                text_file = text_dir / f"{video_id}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)
            
            # Create a combined file with all transcripts
            combined_text = ""
            for video_data in data:
                if not video_data['success']:
                    continue
                
                title = video_data['title']
                channel = video_data['channel']
                video_id = video_data['video_id']
                transcript = video_data['transcript']
                
                combined_text += f"\n\n{'='*50}\n"
                combined_text += f"Title: {title}\n"
                combined_text += f"Channel: {channel}\n"
                combined_text += f"Video ID: {video_id}\n\n"
                combined_text += "TRANSCRIPT:\n\n"
                
                for entry in transcript:
                    combined_text += f"{entry['text']}\n"
            
            # Save combined file
            combined_file = text_dir / "all_transcripts.txt"
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
                
            print(f"Text transcripts created in {text_dir}")
            return True
            
        except Exception as e:
            print(f"Error creating text transcripts: {e}")
            return False

    def fetch_transcript_from_alternative_api(self, video_id, language='en'):
        """Fallback method to fetch transcripts using a free API"""
        try:
            url = f"https://youtube-transcriber-api.vercel.app/v1/transcripts"
            params = {
                'id': video_id,
                'lang': language,
                'type': 'json'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return {
                    'video_id': video_id,
                    'success': True,
                    'transcript': response.json()
                }
            else:
                return {
                    'video_id': video_id,
                    'success': False,
                    'error': f'API error: {response.status_code}'
                }
        except Exception as e:
            return {
                'video_id': video_id,
                'success': False,
                'error': str(e)
            }

def main():
    print("YouTube Transcript Fetcher")
    print("-------------------------")
    
    username = input("Enter username (must match the folder with watch history): ")
    
    fetcher = YouTubeTranscriptFetcher()
    
    # Ask for time range
    days_input = input("Enter number of days to include (default is 7): ")
    days = 7
    
    if days_input.strip():
        try:
            days = int(days_input)
        except ValueError:
            print("Invalid input. Using default value of 7 days.")
    
    # Fetch transcripts
    fetcher.fetch_transcripts_for_user(username, days)
    
    print("\nProcess completed.")

if __name__ == "__main__":
    main() 