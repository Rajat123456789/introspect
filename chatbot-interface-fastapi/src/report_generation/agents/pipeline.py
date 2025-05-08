"""
Agentic Pipeline - Main script to orchestrate the processing of various data sources.
"""

import os
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.report_generation.agents.image_analysis_agent import ImageAnalysisAgent
from src.report_generation.agents.json_analysis_agent import JSONAnalysisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agentic_pipeline")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "report_generation", "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "src", "report_generation", "reports")

# Data subdirectories
HEALTH_HISTORY_DIR = os.path.join(DATA_DIR, "Health_history")
HEALTH_LIVE_DIR = os.path.join(DATA_DIR, "Health_live")
YOUTUBE_HISTORY_DIR = os.path.join(DATA_DIR, "Youtube_history")
YOUTUBE_LIVE_DIR = os.path.join(DATA_DIR, "Youtube_live")

# Analysis output directories
ANALYSIS_DIR = os.path.join(REPORTS_DIR, "analysis")
HEALTH_HISTORY_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "health_history")
HEALTH_LIVE_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "health_live")
YOUTUBE_HISTORY_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "youtube_history")
YOUTUBE_LIVE_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "youtube_live")

# Final report paths
HEALTH_HISTORY_REPORT = os.path.join(REPORTS_DIR, "health_history_report.txt")
HEALTH_LIVE_REPORT = os.path.join(REPORTS_DIR, "health_live_report.txt")
YOUTUBE_HISTORY_REPORT = os.path.join(REPORTS_DIR, "youtube_history_report.txt")
YOUTUBE_LIVE_REPORT = os.path.join(REPORTS_DIR, "youtube_live_report.txt")

# Configuration for batch processing to avoid rate limits
MAX_RETRIES = 3
BATCH_SIZE = 15
MAX_YOUTUBE_VIDEOS = 30  # Default maximum number of YouTube videos to process
ENABLE_SKIP_EXISTING = True  # Set to True to skip files that have already been analyzed
DELAY_BETWEEN_PROCESSES = 5  # Seconds to wait between processing different data types

# Set to True to check for ENABLE_SKIP_EXISTING in other files
try:
    from src.report_generation.agents.json_analysis_agent import ENABLE_SKIP_EXISTING as JSON_SKIP_EXISTING
except ImportError:
    # Define ENABLE_SKIP_EXISTING as a global in json_analysis_agent.py
    pass

def setup():
    """Set up the environment for the pipeline."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not found.")
        logger.error("Please set the OPENAI_API_KEY environment variable.")
        return False
    
    # Create directories
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(HEALTH_HISTORY_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(HEALTH_LIVE_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(YOUTUBE_HISTORY_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(YOUTUBE_LIVE_ANALYSIS_DIR, exist_ok=True)
    
    # Check data directories
    for dir_path, dir_name in [
        (HEALTH_HISTORY_DIR, "Health_history"),
        (HEALTH_LIVE_DIR, "Health_live"),
        (YOUTUBE_HISTORY_DIR, "Youtube_history"),
        (YOUTUBE_LIVE_DIR, "Youtube_live")
    ]:
        if not os.path.exists(dir_path):
            logger.error(f"{dir_name} directory not found at: {dir_path}")
            return False
    
    return True

def process_health_history() -> Dict[str, Any]:
    """Process Health_history data (images)."""
    logger.info("=== Processing Health History Data ===")
    image_agent = ImageAnalysisAgent()
    
    # Check if final report already exists
    if os.path.exists(HEALTH_HISTORY_REPORT) and ENABLE_SKIP_EXISTING:
        logger.info(f"Final report already exists at: {HEALTH_HISTORY_REPORT}. Skipping processing.")
        return {
            "success": True,
            "message": "Report already exists, skipped processing",
            "output_file": HEALTH_HISTORY_REPORT
        }
    
    # Check if any individual analyses exist
    existing_analyses = []
    if ENABLE_SKIP_EXISTING:
        if os.path.exists(HEALTH_HISTORY_ANALYSIS_DIR):
            existing_analyses = list(Path(HEALTH_HISTORY_ANALYSIS_DIR).glob("*_analysis.txt"))
            if existing_analyses:
                logger.info(f"Found {len(existing_analyses)} existing analysis files in {HEALTH_HISTORY_ANALYSIS_DIR}")
    
    # Process all images
    result = image_agent.process_directory(
        directory_path=HEALTH_HISTORY_DIR,
        output_dir=HEALTH_HISTORY_ANALYSIS_DIR,
        analysis_type="health",
        skip_existing=ENABLE_SKIP_EXISTING
    )
    
    if result.get("success", False):
        logger.info(f"Successfully processed {result['successful']}/{result['total']} Health History images")
        
        # Generate consolidated report (with batching to avoid rate limits)
        if len(existing_analyses) > 0 and result.get("skipped", 0) == result.get("total", 0):
            # All files were skipped, just generate the report
            logger.info("All image analyses were already completed, generating consolidated report")
        
        report_result = image_agent.generate_consolidated_report(
            analysis_dir=HEALTH_HISTORY_ANALYSIS_DIR,
            output_file=HEALTH_HISTORY_REPORT,
            analysis_type="health",
            max_retries=MAX_RETRIES,
            batch_size=BATCH_SIZE
        )
        
        if report_result.get("success", False):
            logger.info(f"Health History consolidated report generated at: {HEALTH_HISTORY_REPORT}")
        else:
            logger.error(f"Failed to generate Health History consolidated report: {report_result.get('error', 'Unknown error')}")
    else:
        logger.error(f"Failed to process Health History images: {result.get('error', 'Unknown error')}")
    
    return result

def process_health_live() -> Dict[str, Any]:
    """Process Health_live data (images)."""
    logger.info("=== Processing Health Live Data ===")
    image_agent = ImageAnalysisAgent()
    
    # Check if final report already exists
    if os.path.exists(HEALTH_LIVE_REPORT) and ENABLE_SKIP_EXISTING:
        logger.info(f"Final report already exists at: {HEALTH_LIVE_REPORT}. Skipping processing.")
        return {
            "success": True,
            "message": "Report already exists, skipped processing",
            "output_file": HEALTH_LIVE_REPORT
        }
    
    # Process all images
    result = image_agent.process_directory(
        directory_path=HEALTH_LIVE_DIR,
        output_dir=HEALTH_LIVE_ANALYSIS_DIR,
        analysis_type="health",
        skip_existing=ENABLE_SKIP_EXISTING
    )
    
    if result.get("success", False):
        logger.info(f"Successfully processed {result['successful']}/{result['total']} Health Live images")
        
        # Generate consolidated report (with batching to avoid rate limits)
        report_result = image_agent.generate_consolidated_report(
            analysis_dir=HEALTH_LIVE_ANALYSIS_DIR,
            output_file=HEALTH_LIVE_REPORT,
            analysis_type="health",
            max_retries=MAX_RETRIES,
            batch_size=BATCH_SIZE
        )
        
        if report_result.get("success", False):
            logger.info(f"Health Live consolidated report generated at: {HEALTH_LIVE_REPORT}")
        else:
            logger.error(f"Failed to generate Health Live consolidated report: {report_result.get('error', 'Unknown error')}")
    else:
        logger.error(f"Failed to process Health Live images: {result.get('error', 'Unknown error')}")
    
    return result

def process_youtube_history() -> Dict[str, Any]:
    """Process Youtube_history data (images)."""
    logger.info("=== Processing YouTube History Data ===")
    image_agent = ImageAnalysisAgent()
    
    # Check if final report already exists
    if os.path.exists(YOUTUBE_HISTORY_REPORT) and ENABLE_SKIP_EXISTING:
        logger.info(f"Final report already exists at: {YOUTUBE_HISTORY_REPORT}. Skipping processing.")
        return {
            "success": True,
            "message": "Report already exists, skipped processing",
            "output_file": YOUTUBE_HISTORY_REPORT
        }
    
    # Process all images
    result = image_agent.process_directory(
        directory_path=YOUTUBE_HISTORY_DIR,
        output_dir=YOUTUBE_HISTORY_ANALYSIS_DIR,
        analysis_type="youtube",
        skip_existing=ENABLE_SKIP_EXISTING
    )
    
    if result.get("success", False):
        logger.info(f"Successfully processed {result['successful']}/{result['total']} YouTube History images")
        
        # Generate consolidated report (with batching to avoid rate limits)
        report_result = image_agent.generate_consolidated_report(
            analysis_dir=YOUTUBE_HISTORY_ANALYSIS_DIR,
            output_file=YOUTUBE_HISTORY_REPORT,
            analysis_type="youtube",
            max_retries=MAX_RETRIES,
            batch_size=BATCH_SIZE
        )
        
        if report_result.get("success", False):
            logger.info(f"YouTube History consolidated report generated at: {YOUTUBE_HISTORY_REPORT}")
        else:
            logger.error(f"Failed to generate YouTube History consolidated report: {report_result.get('error', 'Unknown error')}")
    else:
        logger.error(f"Failed to process YouTube History images: {result.get('error', 'Unknown error')}")
    
    return result

def process_youtube_live(max_videos: int = MAX_YOUTUBE_VIDEOS) -> Dict[str, Any]:
    """
    Process Youtube_live data (JSON).
    
    Args:
        max_videos: Maximum number of videos to process per JSON file
        
    Returns:
        Dictionary with processing results
    """
    logger.info("=== Processing YouTube Live Data ===")
    json_agent = JSONAnalysisAgent()
    
    # Check if final report already exists
    if os.path.exists(YOUTUBE_LIVE_REPORT) and ENABLE_SKIP_EXISTING:
        logger.info(f"Final report already exists at: {YOUTUBE_LIVE_REPORT}. Skipping processing.")
        return {
            "success": True,
            "message": "Report already exists, skipped processing",
            "output_file": YOUTUBE_LIVE_REPORT
        }
    
    # Find JSON files
    json_files = list(Path(YOUTUBE_LIVE_DIR).glob("*.json"))
    
    if not json_files:
        logger.error(f"No JSON files found in {YOUTUBE_LIVE_DIR}")
        return {"error": "No JSON files found"}
    
    results = []
    
    # Check for existing analyses
    existing_analyses = []
    if ENABLE_SKIP_EXISTING and os.path.exists(YOUTUBE_LIVE_ANALYSIS_DIR):
        existing_analyses = list(Path(YOUTUBE_LIVE_ANALYSIS_DIR).glob("*_analysis.txt"))
        if existing_analyses:
            logger.info(f"Found {len(existing_analyses)} existing analysis files in {YOUTUBE_LIVE_ANALYSIS_DIR}")
    
    # Process each JSON file
    for json_file in json_files:
        logger.info(f"Processing JSON file: {json_file}")
        
        result = json_agent.process_json_file(
            json_file_path=str(json_file),
            output_dir=YOUTUBE_LIVE_ANALYSIS_DIR,
            max_videos=max_videos
        )
        
        results.append(result)
        
        if result.get("success", False):
            logger.info(f"Successfully processed {result['successful']}/{result['processed']} videos from {json_file.name}")
        else:
            logger.error(f"Failed to process JSON file {json_file.name}: {result.get('error', 'Unknown error')}")
    
    # Generate consolidated report if any processing was successful
    if any(result.get("success", False) for result in results) or existing_analyses:
        report_result = json_agent.generate_consolidated_report(
            analysis_dir=YOUTUBE_LIVE_ANALYSIS_DIR,
            output_file=YOUTUBE_LIVE_REPORT
        )
        
        if report_result.get("success", False):
            logger.info(f"YouTube Live consolidated report generated at: {YOUTUBE_LIVE_REPORT}")
        else:
            logger.error(f"Failed to generate YouTube Live consolidated report: {report_result.get('error', 'Unknown error')}")
    
    return {
        "success": any(result.get("success", False) for result in results),
        "total_files": len(json_files),
        "results": results,
        "max_videos": max_videos
    }

def run_pipeline(data_types: Optional[List[str]] = None, youtube_live_max_videos: int = MAX_YOUTUBE_VIDEOS) -> Dict[str, Any]:
    """
    Run the complete pipeline for the specified data types.
    
    Args:
        data_types: List of data types to process. Options: "health_history", "health_live", 
                   "youtube_history", "youtube_live". If None, all data types are processed.
        youtube_live_max_videos: Maximum number of YouTube Live videos to process per file
    
    Returns:
        Dictionary with results for each data type.
    """
    logger.info("=== Starting Agentic Data Pipeline ===")
    start_time = time.time()
    
    # Set up the environment
    if not setup():
        logger.error("Pipeline setup failed. Exiting.")
        return {"error": "Pipeline setup failed"}
    
    # Determine which data types to process
    all_data_types = {
        "health_history": process_health_history,
        "health_live": process_health_live,
        "youtube_history": process_youtube_history,
        "youtube_live": lambda: process_youtube_live(max_videos=youtube_live_max_videos)
    }
    
    if data_types:
        # Filter to only the requested data types
        data_processors = {
            data_type: processor 
            for data_type, processor in all_data_types.items() 
            if data_type in data_types
        }
    else:
        # Process all data types
        data_processors = all_data_types
    
    logger.info(f"Will process the following data types: {', '.join(data_processors.keys())}")
    
    # Process each data type
    results = {}
    for data_type, processor in data_processors.items():
        data_start_time = time.time()
        logger.info(f"Starting processing of {data_type}")
        
        try:
            result = processor()
            results[data_type] = result
            
            data_end_time = time.time()
            data_duration = data_end_time - data_start_time
            
            logger.info(f"Completed processing {data_type} in {data_duration:.2f} seconds")
            
            # Add delay between processes to avoid rate limits
            if data_type != list(data_processors.keys())[-1]:  # Not the last item
                logger.info(f"Waiting {DELAY_BETWEEN_PROCESSES} seconds before next process to avoid rate limits")
                time.sleep(DELAY_BETWEEN_PROCESSES)
            
        except Exception as e:
            logger.error(f"Error processing {data_type}: {str(e)}", exc_info=True)
            results[data_type] = {"error": str(e)}
    
    # Calculate total duration
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"=== Pipeline completed in {total_duration:.2f} seconds ===")
    
    # Generate pipeline summary
    summary_file = os.path.join(REPORTS_DIR, "pipeline_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# Agentic Data Pipeline Summary\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total duration: {total_duration:.2f} seconds\n\n")
        
        for data_type, result in results.items():
            f.write(f"## {data_type.replace('_', ' ').title()}\n\n")
            
            if "error" in result:
                f.write(f"Status: Failed\n")
                f.write(f"Error: {result['error']}\n")
            elif "message" in result:
                f.write(f"Status: Skipped\n")
                f.write(f"Message: {result['message']}\n")
            else:
                f.write(f"Status: Success\n")
                
                if data_type in ["health_history", "health_live", "youtube_history"]:
                    f.write(f"Images total: {result.get('total', 'Unknown')}\n")
                    if "skipped" in result:
                        f.write(f"Images skipped: {result.get('skipped', 0)}\n")
                    f.write(f"Images processed: {result.get('processed', result.get('total', 'Unknown'))}\n")
                    f.write(f"Successful: {result.get('successful', 'Unknown')}\n")
                    f.write(f"Failed: {result.get('failed', 'Unknown')}\n")
                elif data_type == "youtube_live":
                    f.write(f"JSON files processed: {result.get('total_files', 'Unknown')}\n")
                    
                    if "results" in result:
                        total_videos = sum(r.get("total_videos", 0) for r in result["results"] if r.get("success", False))
                        processed_videos = sum(r.get("processed", 0) for r in result["results"] if r.get("success", False))
                        successful_videos = sum(r.get("successful", 0) for r in result["results"] if r.get("success", False))
                        
                        f.write(f"Total videos: {total_videos}\n")
                        f.write(f"Videos processed: {processed_videos}\n")
                        f.write(f"Successful: {successful_videos}\n")
            
            f.write("\n")
    
    logger.info(f"Pipeline summary saved to: {summary_file}")
    
    return {
        "success": all(result.get("success", False) for result in results.values() if "error" not in result),
        "duration": total_duration,
        "data_types_processed": list(data_processors.keys()),
        "results": results
    }

def main():
    """Main entry point for the agentic pipeline script."""
    # Access globals
    global ENABLE_SKIP_EXISTING, BATCH_SIZE, MAX_YOUTUBE_VIDEOS
    
    parser = argparse.ArgumentParser(description="Run agentic data analysis pipeline")
    parser.add_argument("--health-history", action="store_true", help="Process Health History data")
    parser.add_argument("--health-live", action="store_true", help="Process Health Live data")
    parser.add_argument("--youtube-history", action="store_true", help="Process YouTube History data")
    parser.add_argument("--youtube-live", action="store_true", help="Process YouTube Live data")
    parser.add_argument("--all", action="store_true", help="Process all data types (default)")
    parser.add_argument("--no-skip", action="store_true", help="Disable skipping of existing analyses")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for consolidated reports")
    parser.add_argument("--youtube-max-videos", type=int, default=None, 
                        help=f"Maximum number of YouTube Live videos to process (default: {MAX_YOUTUBE_VIDEOS})")
    
    args = parser.parse_args()
    
    # Update global settings based on arguments
    if args.no_skip:
        ENABLE_SKIP_EXISTING = False
        logger.info("Skipping existing analyses is disabled")
    
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
        logger.info(f"Batch size set to {BATCH_SIZE}")
        
    # Determine which data types to process
    data_types = []
    if args.health_history:
        data_types.append("health_history")
    if args.health_live:
        data_types.append("health_live")
    if args.youtube_history:
        data_types.append("youtube_history")
    if args.youtube_live:
        data_types.append("youtube_live")
    
    # Check if YouTube Live is selected and max_videos is required
    if args.youtube_live or args.all:
        if args.youtube_max_videos is None:
            if args.youtube_live:
                # Only show error if YouTube Live was explicitly selected
                logger.error("When processing YouTube Live data, you must specify --youtube-max-videos")
                print("ERROR: When processing YouTube Live data, you must specify --youtube-max-videos")
                print("Example: python -m src.report_generation.agents.pipeline --youtube-live --youtube-max-videos 30")
                return
            else:
                # Default to MAX_YOUTUBE_VIDEOS if using --all
                args.youtube_max_videos = MAX_YOUTUBE_VIDEOS
        else:
            # User provided a value
            MAX_YOUTUBE_VIDEOS = args.youtube_max_videos
            logger.info(f"Maximum YouTube Live videos to process set to {MAX_YOUTUBE_VIDEOS}")
    
    # If no specific data type is selected, process all
    if not data_types and not args.all:
        args.all = True
    
    # Run the pipeline
    result = run_pipeline(None if args.all else data_types, youtube_live_max_videos=MAX_YOUTUBE_VIDEOS)
    
    if result.get("success", False):
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline completed with errors. Check the logs for details.")

if __name__ == "__main__":
    main() 