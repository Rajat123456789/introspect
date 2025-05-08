# YouTube Live Data Tools

This repository contains Python scripts to fetch YouTube watch history and video transcripts. These tools help you collect and organize your YouTube viewing history and the transcripts of videos you've watched.

## Requirements

- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`)

## Setup

1. Clone or download this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. For accessing YouTube API features, you'll need to create a Google API project:
   - Go to [Google Developer Console](https://console.developers.google.com/)
   - Create a new project
   - Enable the YouTube Data API v3
   - Create OAuth 2.0 credentials
   - Download the client secret JSON file and rename it to `client_secret.json`
   - Place the file in the root directory of this project

## Scripts

### 1. Fetch Watch History (`fetch_history.py`)

This script helps you retrieve your YouTube watch history from the last week (or a configurable time period).

**Note**: Due to YouTube API limitations, watch history is not directly accessible via the API. Instead, the script provides instructions for downloading your watch history through Google Takeout and then processes the downloaded file.

#### Usage:

```
python fetch_history.py
```

The script will:
1. Guide you through the Google Takeout process to download your YouTube watch history
2. Ask you to specify where you saved the watch history file
3. Process the history and extract video information
4. Save the processed history with video details in JSON format
5. Allow you to adjust the time range to include only videos from a specific number of days

### 2. Fetch Transcripts (`fetch_transcripts.py`)

This script fetches transcripts for all videos in your watch history.

#### Usage:

```
python fetch_transcripts.py
```

The script will:
1. Ask for your username (the same one used when fetching history)
2. Ask for the time range to include (default is 7 days)
3. Fetch transcripts for all videos in your history from that time period
4. Save transcripts in both JSON and text formats
5. Create individual transcript files for each video
6. Create a combined file with all transcripts

## Data Organization

The scripts create the following directory structure:

```
youtube_data/
  └── [username]/
      ├── watch-history.json                (original Google Takeout file)
      ├── recent_watch_history.json         (processed history - last 7 days)
      └── recent_watch_history_[days]days.json (processed history - custom timeframe)

youtube_transcripts/
  └── [username]/
      ├── [video_id].json                  (individual transcript files)
      ├── all_transcripts.json             (all transcripts in one JSON file)
      └── text/
          ├── [video_id].txt               (individual text transcripts)
          └── all_transcripts.txt          (all transcripts in one text file)
```

## Limitations

- YouTube API does not directly provide watch history
- Not all videos have transcripts available
- Some videos may have transcripts disabled
- The YouTube Transcript API might be rate-limited for frequent requests

## Troubleshooting

- If you encounter authentication issues, delete the `token.json` file and run the script again
- If transcripts aren't being fetched properly, ensure the video IDs are correctly extracted from your history
- For watch history issues, ensure you're downloading the correct file from Google Takeout

## License

This project is open source and available under the MIT License.

## Acknowledgements

This project utilizes:
- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) 