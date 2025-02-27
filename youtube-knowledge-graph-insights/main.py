"""
Main entry point for YouTube Mental Health Analysis
"""
import logging
import argparse
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("mental_health_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Import analyzer modules
from analyzer.mental_health_analyzer import MentalHealthAnalyzer
from analyzer.temporal_analysis import get_temporal_trends, analyze_sentiment_trajectory, plot_temporal_trends
from analyzer.addiction_analysis import analyze_viewing_patterns, analyze_content_addiction, plot_addiction_risk
from analyzer.music_analysis import analyze_music_details
from analyzer.dashboard import plot_mental_health_dashboard

def main():
    parser = argparse.ArgumentParser(description='YouTube Mental Health Analysis Tool')
    parser.add_argument('--uri', default="bolt://localhost:7687", help='Neo4j URI')
    parser.add_argument('--user', default="neo4j", help='Neo4j username')
    parser.add_argument('--password', default="12345678", help='Neo4j password')
    parser.add_argument('--analysis', default="all", 
                        choices=["all", "temporal", "sentiment", "viewing", "addiction", "music", "dashboard"],
                        help='Type of analysis to run')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Create reports directory
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    analyzer = MentalHealthAnalyzer(args.uri, args.user, args.password)
    
    try:
        # Run the appropriate analysis based on command line arguments
        
        # Display welcome message
        print("\n========================================")
        print("YouTube Mental Health Analysis Tool")
        print("========================================\n")
        
        # Run temporal analysis
        if args.analysis in ["all", "temporal"]:
            print("Running temporal trend analysis...")
            temporal_analyses = get_temporal_trends(analyzer)
            if temporal_analyses:
                plot_temporal_trends(temporal_analyses)
                if not args.no_save:
                    analyzer.save_analysis_results(temporal_analyses, "temporal_trends")
                    
        # Run all other analyses similarly
        
        print("\n========================================")
        print("Analysis complete!")
        print(f"Reports saved to '{reports_dir}' directory")
        print("========================================\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        print("Please check the log file for details.")
        return 1
    finally:
        analyzer.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 