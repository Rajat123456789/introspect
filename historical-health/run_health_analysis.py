#!/usr/bin/env python3
"""
Script to run both historical health data analysis and visualization.
"""

import logging
import subprocess
import os
import sys
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run historical health data analysis and visualization')
    parser.add_argument('data_dir', nargs='?', default="data/Fit", 
                        help='Directory containing the Google Fit data (default: data/Fit)')
    return parser.parse_args()

def main():
    """Run both analysis and visualization scripts."""
    # Parse command line arguments
    args = parse_args()
    data_dir = args.data_dir
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Define script paths
    analysis_script = script_dir / "analyze_historical_health.py"
    visualization_script = script_dir / "visualize_health_insights.py"
    
    # Check if scripts exist
    if not analysis_script.exists():
        logger.error(f"Analysis script not found: {analysis_script}")
        sys.exit(1)
        
    if not visualization_script.exists():
        logger.error(f"Visualization script not found: {visualization_script}")
        sys.exit(1)
    
    # Run analysis script
    logger.info(f"Running historical health data analysis with data from: {data_dir}...")
    try:
        result = subprocess.run(
            [sys.executable, str(analysis_script), data_dir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Analysis completed successfully.")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Analysis failed with exit code {e.returncode}:")
        logger.error(e.stderr)
        sys.exit(1)
    
    # Run visualization script
    logger.info(f"Running historical health data visualization with data from: {data_dir}...")
    try:
        result = subprocess.run(
            [sys.executable, str(visualization_script), data_dir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Visualization completed successfully.")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Visualization failed with exit code {e.returncode}:")
        logger.error(e.stderr)
        sys.exit(1)
    
    # Output directories
    analysis_output = script_dir / "analysis_output"
    visualization_output = script_dir / "visualizations"
    
    logger.info("All processing completed successfully!")
    logger.info(f"Analysis outputs available in: {analysis_output}")
    logger.info(f"Visualization outputs available in: {visualization_output}")

if __name__ == "__main__":
    main() 