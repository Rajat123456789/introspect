#!/usr/bin/env python3
"""
Spotify Data Visualization Script

This script provides a command-line interface to generate visualizations from Spotify listening history data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualization.visualization_pipeline import SpotifyVisualizationPipeline


def main():
    """
    Main function to parse arguments and run the visualization pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Generate visualizations from Spotify listening history data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the Spotify history CSV file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Directory to save visualizations to (default: spotify_visualizations_TIMESTAMP)'
    )
    
    parser.add_argument(
        '--no-basic',
        action='store_true',
        help='Skip basic visualizations'
    )
    
    parser.add_argument(
        '--no-advanced',
        action='store_true',
        help='Skip advanced visualizations'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive visualizations'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML report with all visualizations'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
        
    if not input_path.is_file():
        print(f"Error: '{args.input_file}' is not a file.")
        return 1
        
    # Create and run pipeline
    try:
        print(f"Initializing visualization pipeline for '{args.input_file}'...")
        pipeline = SpotifyVisualizationPipeline(args.input_file, args.output_dir)
        
        print("Running visualization pipeline...")
        pipeline.run_pipeline(
            generate_basic=not args.no_basic,
            generate_advanced=not args.no_advanced,
            generate_interactive=not args.no_interactive
        )
        
        # Generate report if requested
        if args.report:
            print("Generating HTML report...")
            report_path = pipeline.generate_report()
            print(f"Report generated at: {report_path}")
            
        print(f"Visualizations saved to: {pipeline.output_dir}")
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 