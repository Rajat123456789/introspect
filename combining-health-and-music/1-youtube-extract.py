import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
import re

# Set up API details
API_KEY = 'AIzaSyDpzbZa16fL5aCV25bdx_0NFXLxTiCUdbo'
VIDEO_DETAILS_URL = 'https://www.googleapis.com/youtube/v3/videos'
CATEGORY_DETAILS_URL = 'https://www.googleapis.com/youtube/v3/videoCategories'

MAX_VIDEOS = 50  # Maximum number of videos to process

# Function to extract video data from HTML
def extract_video_data(html_file, max_videos=MAX_VIDEOS):
    video_data = []
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
        links = soup.find_all('a', href=True, limit=max_videos)
        
        for entry in links:
            match = re.search(r'v=([a-zA-Z0-9_-]+)', entry['href'])
            if match:
                video_id = match.group(1)
                date_time = entry.find_next_sibling(string=True)  # Updated here
                date_time = date_time.strip() if date_time else "Unknown"
                
                video_data.append({'video_id': video_id, 'watched_at': date_time, 'video_url': entry['href']})
    return video_data[:max_videos]  # Ensure the list size is within limits

# Async function to fetch video details
async def fetch_video_details(session, video_id):
    params = {'part': 'snippet', 'id': video_id, 'key': API_KEY}
    async with session.get(VIDEO_DETAILS_URL, params=params) as response:
        if response.status == 200:
            data = await response.json()
            items = data.get('items', [])
            if not items:  # Check if items is empty
                return {}
            return items[0]  # Safe to access the first element
        return {}

# Async function to fetch category name
async def fetch_category_name(session, category_id):
    params = {'part': 'snippet', 'id': category_id, 'key': API_KEY}
    async with session.get(CATEGORY_DETAILS_URL, params=params) as response:
        if response.status == 200:
            data = await response.json()
            return data.get('items', [{}])[0].get('snippet', {}).get('title', 'Unknown')
        return 'Unknown'

# Async function to process videos in parallel
async def process_videos(video_watch_data):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_video_details(session, video['video_id']) for video in video_watch_data]
        video_info_list = await asyncio.gather(*tasks)
        
        final_video_data = []
        for video, video_info in zip(video_watch_data, video_info_list):
            if not video_info:
                continue
            snippet = video_info.get('snippet', {})
            category_id = snippet.get('categoryId', None)
            category_name = await fetch_category_name(session, category_id) if category_id else "Unknown"
            
            final_video_data.append({
                # 'Video ID': video['video_id'],
                # 'Video Link': video['video_url'],
                'Title': snippet.get('title', 'Unknown'),
                'Category': category_name,
                'Watched At': video['watched_at']
            })
    return final_video_data

# Main function to execute the script
def main():
    html_file = '/Users/rajatsharma/10-feb-introspect-proj/Gaurav-YouTube.html'
    video_watch_data = extract_video_data(html_file, 10000)
    final_video_data = asyncio.run(process_videos(video_watch_data))  # Fixed here

    df = pd.DataFrame(final_video_data)
    df.to_csv('10-feb-2.csv', index=False)
    print("CSV file created successfully!")

if __name__ == "__main__":
    main()
