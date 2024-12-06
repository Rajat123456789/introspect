import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

# Set up your API key and base URLs
API_KEY = 'API KEY'
VIDEO_DETAILS_URL = 'https://www.googleapis.com/youtube/v3/videos'
CATEGORY_DETAILS_URL = 'https://www.googleapis.com/youtube/v3/videoCategories'


# Step 1: Parse the HTML file to extract video IDs and date-time details
def extract_video_data(html_file):
    video_data = []
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        # Find all video entries (assuming each video has a link with a date-time in nearby text)
        for entry in soup.find_all('a', href=True):
            # Extract video ID from the link
            match = re.search(r'v=([a-zA-Z0-9_-]+)', entry['href'])
            if match:
                video_id = match.group(1)
                video_url = entry['href']
                # Find date-time text adjacent to the link
                date_time = entry.find_next_sibling(text=True)
                if date_time:
                    date_time = date_time.strip()

                # Append video_id and date-time to the data list
                video_data.append({
                    'video_id': video_id,
                    'watched_at': date_time,
                    'video_url': video_url
                })
    return video_data


# Load video IDs and watch timestamps from the HTML file
video_watch_data = extract_video_data('/Users/rajatsharma/Documents/youtube/Gaurav-YouTube.html')

# Function to get video details
def get_video_details(video_id):
    try:
        params = {
            'part': 'snippet,contentDetails,statistics',
            'id': video_id,
            'key': API_KEY
        }
        response = requests.get(VIDEO_DETAILS_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to retrieve details for video ID {video_id}: {e}")
        return None


# Function to get category name by category ID
def get_category_name(category_id):
    try:
        params = {
            'part': 'snippet',
            'id': category_id,
            'key': API_KEY
        }
        response = requests.get(CATEGORY_DETAILS_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data['items'][0]['snippet']['title'] if data['items'] else None
    except requests.RequestException as e:
        print(f"Failed to retrieve category name for category ID {category_id}: {e}")
        return None


# List to store final video data for the CSV
final_video_data = []

# Step 2: Loop through each video ID and fetch details
for video in video_watch_data[:]:
    video_id = video['video_id']
    video_url = video['video_url']
    watched_at = video['watched_at']  # Extracted watch time from HTML file

    video_info = get_video_details(video_id)
    if video_info is None or 'items' not in video_info or not video_info['items']:
        continue  # Skip if video details are unavailable

    # Extract relevant information
    video_item = video_info['items'][0]
    snippet = video_item.get('snippet', {})
    statistics = video_item.get('statistics', {})
    content_details = video_item.get('contentDetails', {})

    # Get the category name using the category ID
    category_id = snippet.get('categoryId')
    category_name = get_category_name(category_id) if category_id else None

    # Collect the necessary fields, including watch date and time
    final_video_data.append({
        # 'Video ID': video_id,
        # 'Video Link': video_url,
        'Title': snippet.get('title'),
        'Description': snippet.get('description'),
        # 'Category Id': category_id,
        'Category': category_name,
        # 'Published At': snippet.get('publishedAt'),
        'Watched At': watched_at,  # Add watch date and time
        'Duration': content_details.get('duration'),
        # 'View Count': statistics.get('viewCount'),
        # 'Like Count': statistics.get('likeCount'),
        # 'Comment Count': statistics.get('commentCount')
    })

# Step 3: Create a DataFrame and save to CSV
df = pd.DataFrame(final_video_data)
df.to_csv('youtube_7654321.csv', index=False)
print("CSV file created successfully with watch timestamps.")
print(df)
