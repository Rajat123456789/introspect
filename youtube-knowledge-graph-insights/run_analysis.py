#!/usr/bin/env python3
"""
YouTube Mental Health Analysis Runner
Run this script to perform various analyses on YouTube viewing data
"""

import argparse
import sys
import logging
from youtube_mental_health_analysis import MentalHealthAnalyzer
import youtube_mental_health_main as analysis_runner

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("mental_health_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("YouTube_Analysis")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YouTube Mental Health Analysis Tool')
    parser.add_argument('--uri', default="bolt://localhost:7687", help='Neo4j URI')
    parser.add_argument('--user', default="neo4j", help='Neo4j username')
    parser.add_argument('--password', default="12345678", help='Neo4j password')
    parser.add_argument('--analysis', default="all", 
                        choices=["all", "temporal", "sentiment", "viewing", "addiction", "report"],
                        help='Type of analysis to run')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')
    parser.add_argument('--report-only', action='store_true', help='Generate report from existing analysis')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n========================================")
    print("YouTube Mental Health Analysis Tool")
    print("========================================\n")
    
    # Connect to Neo4j
    try:
        analyzer = MentalHealthAnalyzer(args.uri, args.user, args.password)
        analyzer.save_analysis_results = analysis_runner.save_analysis_results.__get__(analyzer, MentalHealthAnalyzer)
        
        save_results = not args.no_save
        
        # Run requested analysis
        if args.analysis == "all":
            print("Running comprehensive analysis...\n")
            result = analysis_runner.run_comprehensive_analysis(analyzer, save_results)
        elif args.analysis == "temporal":
            print("Running temporal trend analysis...\n")
            result = analysis_runner.run_temporal_analysis(analyzer, save_results)
        elif args.analysis == "sentiment":
            print("Running sentiment analysis...\n")
            result = analysis_runner.run_sentiment_analysis(analyzer, save_results)
        elif args.analysis == "viewing":
            print("Running viewing pattern analysis...\n")
            result = analysis_runner.run_viewing_pattern_analysis(analyzer, save_results)
        elif args.analysis == "addiction":
            print("Running content addiction analysis...\n")
            result = analysis_runner.run_addiction_analysis(analyzer, save_results)
        elif args.analysis == "report":
            print("Generating comprehensive report...\n")
            result = analysis_runner.run_comprehensive_analysis(analyzer, save_results)
        
        print("\n========================================")
        print("Analysis complete!")
        print("- Visualizations have been saved as PNG files")
        if save_results:
            print("- Data has been saved to the 'analysis_results' directory")
            print("- Reports have been saved to the 'analysis_reports' directory")
        print("========================================\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        print("Please check the log file for details.")
        return 1
    finally:
        if 'analyzer' in locals():
            analyzer.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 