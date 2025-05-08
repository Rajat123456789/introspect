"""
JSON Analysis Agent - Uses OpenAI's API to analyze JSON-formatted data.
Specifically designed for analyzing YouTube video data.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("json_analysis_agent")

# Configuration constants
ENABLE_SKIP_EXISTING = True  # Set to True to skip files that have already been analyzed

# Try to load OpenAI client, gracefully handle import error
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not found. Please install it using: pip install openai")
    OPENAI_AVAILABLE = False

class JSONAnalysisAgent:
    """
    Agent responsible for analyzing JSON data, particularly YouTube data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the JSON analysis agent.
        
        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client if API key is available
        self.client = None
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def analyze_youtube_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single YouTube video entry from the dataset.
        
        Args:
            video_data: Dictionary containing YouTube video data
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        if not OPENAI_AVAILABLE or not self.client:
            logger.error("OpenAI client not available. Cannot perform YouTube video analysis.")
            return {"error": "OpenAI client not available"}
        
        # Extract relevant information for the prompt
        video_id = video_data.get("id", "Unknown")
        video_title = video_data.get("title", "Unknown title")
        
        # Extract URL to get video ID if not explicitly provided
        url = video_data.get("url", "")
        if video_id == "Unknown" and "youtube.com/watch?v=" in url:
            video_id = url.split("youtube.com/watch?v=")[1].split("&")[0]
        
        # Create a prompt for YouTube video analysis focused on mental health impact
        analysis_prompt = f"""
        Analyze the following YouTube video information and provide a factual assessment of its content and potential mental health relevance:
        
        Video ID: {video_id}
        Title: {video_title}
        Channel: {video_data.get("channel", "Unknown")}
        Description: {video_data.get("description", "No description")}
        Tags: {", ".join(video_data.get("tags", [])) if "tags" in video_data else "No tags"}
        View Count: {video_data.get("view_count", "Unknown")}
        Duration: {video_data.get("duration", "Unknown")}
        URL: {url}
        
        Please provide a detailed factual analysis including:
        
        1. Content classification: Categorize the content type based on the title, description, and tags
        2. Mental health relevance: Identify topics related to mental health, emotions, or psychological states
        3. Audience demographic indicators: Based on content, specify likely target audiences
        4. Content attributes: Note elements such as educational content, entertainment value, emotional content
        5. Engagement metrics: Note any view counts, likes, or other engagement data
        
        Focus exclusively on the observable data - do not make recommendations or suggestions.
        Use specific categories and factual descriptions only.
        
        Structure your analysis with clear headings for each section.
        """
        
        logger.info(f"Analyzing YouTube video: {video_title}")
        start_time = time.time()
        
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a media content analyst who objectively categorizes and describes online content. You only report factual observations without making recommendations or suggestions."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1500
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            analysis_result = response.choices[0].message.content
            logger.info(f"Video analysis completed in {duration:.2f} seconds")
            
            return {
                "success": True,
                "analysis": analysis_result,
                "video_id": video_id,
                "video_title": video_title,
                "duration": duration,
                "timestamp": time.time(),
                "model": "gpt-4o",
            }
            
        except Exception as e:
            logger.error(f"Error during video analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "video_id": video_id,
                "video_title": video_title
            }
    
    def process_json_file(self, json_file_path: str, output_dir: str, max_videos: int = 30) -> Dict[str, Any]:
        """
        Process a JSON file containing YouTube data and analyze each video.
        
        Args:
            json_file_path: Path to the JSON file
            output_dir: Directory to save analysis results
            max_videos: Maximum number of videos to process
            
        Returns:
            Dictionary with summary of processing results
        """
        if not os.path.exists(json_file_path):
            logger.error(f"JSON file not found: {json_file_path}")
            return {"error": "JSON file not found"}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different possible JSON structures
            youtube_videos = []
            
            # Case 1: Direct list of videos (each object may have a "videos" array)
            if isinstance(json_data, list):
                for entry in json_data:
                    if isinstance(entry, dict) and "videos" in entry and isinstance(entry["videos"], list):
                        youtube_videos.extend(entry["videos"])
                    elif isinstance(entry, dict):
                        youtube_videos.append(entry)
                    else:
                        youtube_videos.append(entry)
            
            # Case 2: Dictionary with videos as a field
            elif isinstance(json_data, dict):
                if 'videos' in json_data and isinstance(json_data['videos'], list):
                    youtube_videos = json_data['videos']
                elif 'items' in json_data and isinstance(json_data['items'], list):
                    youtube_videos = json_data['items']
                # Add more potential structures as needed
                else:
                    # Extract first-level values that are lists or dicts
                    for key, value in json_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            youtube_videos = value
                            break
            
            # Remove duplicates based on title and URL
            unique_videos = []
            seen_urls = set()
            for video in youtube_videos:
                if isinstance(video, dict):
                    url = video.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_videos.append(video)
            
            youtube_videos = unique_videos
            
            if not youtube_videos:
                logger.error(f"No YouTube video data found in {json_file_path}")
                return {"error": "No YouTube video data found in the JSON file"}
            
            logger.info(f"Found {len(youtube_videos)} videos to analyze in {json_file_path}")
            
            # Process each video up to max_videos limit
            results = []
            successful = 0
            failed = 0
            videos_to_process = min(len(youtube_videos), max_videos)
            
            # Check for existing analyses to avoid reprocessing
            existing_analyses = {}
            for analysis_file in Path(output_dir).glob("*_analysis.txt"):
                existing_analyses[analysis_file.stem.rsplit('_', 1)[0]] = True
            
            for i, video_data in enumerate(youtube_videos[:videos_to_process]):
                if not isinstance(video_data, dict):
                    logger.warning(f"Skipping non-dictionary video data: {video_data}")
                    continue
                
                video_title = video_data.get("title", f"Untitled Video {i}")
                sanitized_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video_title)
                sanitized_title = sanitized_title[:50]  # Limit filename length
                
                # Skip if analysis already exists
                if sanitized_title in existing_analyses and ENABLE_SKIP_EXISTING:
                    logger.info(f"Skipping video {i+1}/{videos_to_process}: {video_title} (analysis already exists)")
                    successful += 1
                    continue
                
                logger.info(f"Processing video {i+1}/{videos_to_process}: {video_title}")
                
                result = self.analyze_youtube_video(video_data)
                
                # Save analysis to file
                if result.get("success", False):
                    output_file = os.path.join(output_dir, f"{sanitized_title}_analysis.txt")
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(f"# Analysis of YouTube Video: {video_title}\n\n")
                        f.write(f"Video ID: {result['video_id']}\n")
                        f.write(f"Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))}\n")
                        f.write(f"Processing time: {result['duration']:.2f} seconds\n\n")
                        f.write(result['analysis'])
                    
                    logger.info(f"Analysis saved to: {output_file}")
                    successful += 1
                else:
                    logger.error(f"Analysis failed for video: {video_title}")
                    failed += 1
                
                results.append(result)
                
                # Add a small delay to avoid rate limiting
                time.sleep(1.5)  # Increased from 0.5 to 1.5 seconds to avoid rate limiting
            
            # Create a summary report
            summary_file = os.path.join(output_dir, "processing_summary.txt")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"# YouTube Video Analysis Processing Summary\n\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"JSON File: {json_file_path}\n")
                f.write(f"Total videos in file: {len(youtube_videos)}\n")
                f.write(f"Max videos to process: {max_videos}\n")
                f.write(f"Videos processed: {videos_to_process}\n")
                f.write(f"Successful: {successful}\n")
                f.write(f"Failed: {failed}\n\n")
                
                f.write("## Processed Videos\n\n")
                for i, result in enumerate(results):
                    status = "✓ Success" if result.get("success", False) else "✗ Failed"
                    f.write(f"{i+1}. {result.get('video_title', 'Unknown')} - {status}\n")
            
            logger.info(f"Processing summary saved to: {summary_file}")
            
            return {
                "success": True,
                "total_videos": len(youtube_videos),
                "max_videos": max_videos,
                "processed": videos_to_process,
                "successful": successful,
                "failed": failed,
                "output_dir": output_dir,
                "summary_file": summary_file,
                "results": results
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {"error": f"JSON parsing error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error processing JSON file: {str(e)}")
            return {"error": f"Error processing JSON file: {str(e)}"}
    
    def generate_consolidated_report(self, analysis_dir: str, output_file: str) -> Dict[str, Any]:
        """
        Generate a consolidated report from individual YouTube video analyses.
        
        Args:
            analysis_dir: Directory containing individual video analyses
            output_file: Path to save the consolidated report
            
        Returns:
            Dictionary with report generation results
        """
        if not os.path.exists(analysis_dir):
            logger.error(f"Analysis directory not found: {analysis_dir}")
            return {"error": "Analysis directory not found"}
        
        # Get all analysis files
        analysis_files = list(Path(analysis_dir).glob("*_analysis.txt"))
        
        if not analysis_files:
            logger.warning(f"No analysis files found in {analysis_dir}")
            return {"error": "No analysis files found"}
        
        logger.info(f"Found {len(analysis_files)} analysis files to consolidate")
        
        # Read all analyses
        analyses = []
        for file_path in analysis_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                analyses.append({
                    "file_name": file_path.name,
                    "content": content
                })
        
        # Combine analyses for the consolidated report prompt
        combined_analyses = "\n\n===== NEXT ANALYSIS =====\n\n".join(
            [f"FILENAME: {a['file_name']}\n\n{a['content']}" for a in analyses]
        )
        
        # Create consolidated prompt for mental health impact of YouTube content
        system_message = "You are a digital media researcher who summarizes findings about online content. You only report factual observations without making recommendations or suggestions."
        consolidated_prompt = f"""
        I'll provide you with analyses of multiple YouTube videos. Your task is to create a comprehensive consolidated report that:
        
        1. Summarizes the findings across all analyzed videos
        2. Categorizes the content types and topics found across the videos 
        3. Presents statistical breakdowns (e.g., percentage of videos in each content category)
        4. Identifies patterns in content consumption or creation
        5. Notes mental health topics and themes that appear in the videos
        
        IMPORTANT: 
        - Only report factual observations from the data
        - Do NOT include any recommendations, advice, or suggestions
        - Focus on specific numbers, percentages, and categorizations
        - Present information in a structured, factual manner
        
        Organize the report in a clear, structured format with sections and subsections.
        
        Here are the individual analyses to consolidate:
        
        {combined_analyses}
        """
        
        if not OPENAI_AVAILABLE or not self.client:
            logger.error("OpenAI client not available. Cannot generate consolidated report.")
            return {"error": "OpenAI client not available"}
        
        logger.info(f"Generating consolidated report from {len(analyses)} analyses")
        start_time = time.time()
        
        try:
            # Call the OpenAI API for the consolidated report
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": consolidated_prompt}
                ],
                max_tokens=4000
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            consolidated_report = response.choices[0].message.content
            logger.info(f"Consolidated report generated in {duration:.2f} seconds")
            
            # Save the consolidated report
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Consolidated YouTube Content Analysis Report\n\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Based on analysis of {len(analyses)} YouTube videos\n\n")
                f.write(consolidated_report)
            
            logger.info(f"Consolidated report saved to: {output_file}")
            
            return {
                "success": True,
                "total_analyses": len(analyses),
                "output_file": output_file,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error generating consolidated report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_analyses": len(analyses)
            } 