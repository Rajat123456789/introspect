import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_extract.log'),
        logging.StreamHandler()  # This will print to console as well
    ]
)
logger = logging.getLogger(__name__)

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

def extract_youtube_history(html_file):
    """Extract YouTube watch history from HTML file into a structured format"""
    logger.info(f"Starting extraction from {html_file}")
    
    try:
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            logger.info("Successfully opened HTML file")
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Find all video entries
        entries = soup.find_all('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')
        logger.info(f"Found {len(entries)} potential video entries")
        
        data = []
        for i, entry in enumerate(entries, 1):
            if i % 100 == 0:  # Log progress every 100 entries
                logger.info(f"Processing entry {i}/{len(entries)}")
            
            # Extract video title
            title_elem = entry.find('a')
            if not title_elem:
                logger.debug(f"Skipping entry {i} - No title element found")
                continue
            title = title_elem.text.strip()
            
            # Extract category
            category = "Unknown"
            category_elem = entry.find('a', href=re.compile(r'/feed/history/.*_category.*'))
            if category_elem:
                category = category_elem.text.strip()
            
            # Extract timestamp
            timestamp_elem = entry.find(string=re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec).*'))
            if not timestamp_elem:
                logger.debug(f"Skipping entry {i} - No timestamp found for video: {title[:50]}...")
                continue
            timestamp = timestamp_elem.strip()
            
            data.append({
                'Title': title,
                'Category': category,
                'Watched At': timestamp
            })
            
        logger.info(f"Successfully extracted {len(data)} video entries")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}", exc_info=True)
        raise

def main():
    logger.info("Starting YouTube history extraction script")
    
    try:
        # Extract data from HTML file
        logger.info("Beginning data extraction...")
        df = extract_youtube_history('Gaurav-YouTube-v2.html')
        
        # Save to CSV
        output_file = 'youtube-gaurav.csv'
        logger.info(f"Saving data to {output_file}")
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(df)} entries to CSV")
        
        # Display first few entries
        logger.info("\nFirst few entries of the extracted data:")
        logger.info("\n" + str(df.head()))
        
    except FileNotFoundError:
        logger.error("HTML file not found. Please check the file path.", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
