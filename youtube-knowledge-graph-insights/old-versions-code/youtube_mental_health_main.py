import argparse
import logging
import pandas as pd
from youtube_mental_health_analysis import MentalHealthAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("mental_health_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def run_temporal_analysis(analyzer, save_results=True):
    """Run temporal trend analysis"""
    logger.info("Running temporal trend analysis...")
    temporal_analyses = analyzer.get_temporal_trends()
    
    if temporal_analyses:
        analyzer.plot_temporal_trends(temporal_analyses)
        if save_results:
            analyzer.save_analysis_results(temporal_analyses, "temporal_trends")
    
    return temporal_analyses

def run_sentiment_analysis(analyzer, save_results=True):
    """Run sentiment trajectory analysis"""
    logger.info("Running sentiment trajectory analysis...")
    sentiment_data = analyzer.analyze_sentiment_trajectory()
    
    if sentiment_data is not None:
        analyzer.plot_sentiment_changes(sentiment_data)
        if save_results:
            analyzer.save_analysis_results(sentiment_data, "sentiment_trajectory")
    
    return sentiment_data

def run_viewing_pattern_analysis(analyzer, save_results=True):
    """Run viewing pattern analysis"""
    logger.info("Running viewing pattern analysis...")
    viewing_patterns = analyzer.analyze_viewing_patterns()
    
    if viewing_patterns is not None:
        analyzer.plot_viewing_habits(viewing_patterns)
        if save_results:
            analyzer.save_analysis_results(viewing_patterns, "viewing_patterns")
            
        # Generate report summary
        if viewing_patterns['binge_day'].sum() > 0:
            binge_pct = (viewing_patterns['binge_day'].sum() / len(viewing_patterns)) * 100
            logger.info(f"Binge watching occurred on {binge_pct:.1f}% of days")
    
    return viewing_patterns

def run_addiction_analysis(analyzer, save_results=True):
    """Run content addiction analysis"""
    logger.info("Running content addiction analysis...")
    content_addiction = analyzer.analyze_content_addiction()
    category_addiction = analyzer.analyze_category_addiction_risk()
    
    if content_addiction is not None and not content_addiction.empty:
        if save_results:
            analyzer.save_analysis_results(content_addiction, "content_addiction_risk")
    
    if category_addiction is not None and not category_addiction.empty:
        if save_results:
            analyzer.save_analysis_results(category_addiction, "category_addiction_risk")
    
    return content_addiction, category_addiction

def run_comprehensive_analysis(analyzer, save_results=True):
    """Run all analyses and generate a comprehensive report"""
    logger.info("Running comprehensive mental health analysis...")
    
    # Run all analyses
    temporal_analyses = run_temporal_analysis(analyzer, save_results)
    sentiment_data = run_sentiment_analysis(analyzer, save_results)
    viewing_patterns = run_viewing_pattern_analysis(analyzer, save_results)
    content_addiction, category_addiction = run_addiction_analysis(analyzer, save_results)
    
    # Category relationships
    logger.info("Analyzing category relationships...")
    category_relationships = analyzer.analyze_category_relationships()
    if not category_relationships.empty and save_results:
        analyzer.plot_category_correlations(category_relationships)
        analyzer.save_analysis_results(category_relationships, "category_relationships")
    
    # Mental health index
    logger.info("Creating mental health index...")
    mh_index = analyzer.create_personal_mental_health_index()
    if mh_index is not None:
        analyzer.plot_mental_health_index(mh_index)
        if save_results:
            analyzer.save_analysis_results(mh_index, "mental_health_index")
    
    # Music impact
    logger.info("Analyzing music content impact...")
    music_impact = analyzer.analyze_music_impact()
    if music_impact is not None and not music_impact.empty and save_results:
        analyzer.save_analysis_results(music_impact, "music_impact")
    
    # Content-category correlations
    logger.info("Analyzing content-category correlations...")
    content_correlations = analyzer.analyze_content_category_correlation()
    if content_correlations is not None and not content_correlations.empty and save_results:
        analyzer.save_analysis_results(content_correlations, "content_category_correlations")
    
    # Generate comprehensive report
    generate_report(temporal_analyses, sentiment_data, viewing_patterns, 
                   content_addiction, category_addiction, mh_index, 
                   music_impact, content_correlations, category_relationships)
    
    return "Analysis complete. Results saved to analysis_results directory."

def generate_report(temporal_analyses, sentiment_data, viewing_patterns, 
                   content_addiction, category_addiction, mh_index,
                   music_impact, content_correlations, category_relationships):
    """Generate a comprehensive HTML and text report"""
    import os
    from datetime import datetime
    
    # Create reports directory
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"{reports_dir}/mental_health_report_{timestamp}.html"
    text_report = f"{reports_dir}/mental_health_report_{timestamp}.txt"
    
    # Start HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Mental Health Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #1a73e8; }}
            h2 {{ color: #1a73e8; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #f2f8ff; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>YouTube Mental Health Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="summary">
            <h2>Executive Summary</h2>
    """
    
    # Text report content
    text_content = f"YouTube Mental Health Analysis Report\n"
    text_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    text_content += "EXECUTIVE SUMMARY\n==================\n"
    
    # Add mental health index trend if available
    if mh_index is not None and len(mh_index) > 1:
        first_half = mh_index.iloc[:len(mh_index)//2]['mental_health_index'].mean()
        second_half = mh_index.iloc[len(mh_index)//2:]['mental_health_index'].mean()
        change = second_half - first_half
        trend = "improving" if change > 0 else "declining"
        
        html_content += f"""
            <p><strong>Mental Health Trend:</strong> Overall mental health appears to be {trend} 
            (change: {change:.2f} points)</p>
        """
        
        text_content += f"Mental Health Trend: Overall mental health appears to be {trend} "
        text_content += f"(change: {change:.2f} points)\n"
    
    # Add viewing patterns summary if available
    if viewing_patterns is not None and not viewing_patterns.empty:
        binge_pct = (viewing_patterns['binge_day'].sum() / len(viewing_patterns)) * 100 if len(viewing_patterns) > 0 else 0
        late_night_pct = (viewing_patterns['late_night_count'] > 0).sum() / len(viewing_patterns) * 100 if len(viewing_patterns) > 0 else 0
        
        html_content += f"""
            <p><strong>Viewing Patterns:</strong> {viewing_patterns['videos_per_day'].mean():.1f} videos watched per day on average. 
            Binge watching occurred on {binge_pct:.1f}% of days. 
            Late night viewing (10PM-4AM) occurred on {late_night_pct:.1f}% of days.</p>
        """
        
        text_content += f"Viewing Patterns: {viewing_patterns['videos_per_day'].mean():.1f} videos watched per day on average.\n"
        text_content += f"Binge watching occurred on {binge_pct:.1f}% of days.\n"
        text_content += f"Late night viewing (10PM-4AM) occurred on {late_night_pct:.1f}% of days.\n"
    
    # Add content addiction risks if available
    if content_addiction is not None and not content_addiction.empty:
        top_category = content_addiction.iloc[0]['category'] if len(content_addiction) > 0 else "None"
        top_score = content_addiction.iloc[0]['addiction_risk_score'] if len(content_addiction) > 0 else 0
        
        html_content += f"""
            <p><strong>Content Addiction Risk:</strong> The category with highest addiction risk is 
            "{top_category}" with a risk score of {top_score:.2f}.</p>
        """
        
        text_content += f"Content Addiction Risk: The category with highest addiction risk is "
        text_content += f"\"{top_category}\" with a risk score of {top_score:.2f}.\n"
    
    # Close summary div
    html_content += """
        </div>
    """
    
    # Add detailed sections
    html_content += """
        <h2>Detailed Analysis</h2>
    """
    
    text_content += "\nDETAILED ANALYSIS\n=================\n"
    
    # Add each section with data tables if available
    # Mental Health Index
    if mh_index is not None and not mh_index.empty:
        html_content += """
            <h3>Mental Health Index</h3>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
        """
        
        stats = mh_index['mental_health_index'].describe()
        
        for stat, value in stats.items():
            html_content += f"""
                <tr>
                    <td>{stat}</td>
                    <td>{value:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        text_content += "\nMental Health Index:\n"
        text_content += f"{stats.to_string()}\n"
    
    # Viewing Patterns
    if viewing_patterns is not None and not viewing_patterns.empty:
        html_content += """
            <h3>Viewing Pattern Statistics</h3>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
        """
        
        stats = viewing_patterns['videos_per_day'].describe()
        
        for stat, value in stats.items():
            html_content += f"""
                <tr>
                    <td>{stat}</td>
                    <td>{value:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        text_content += "\nViewing Pattern Statistics:\n"
        text_content += f"{stats.to_string()}\n"
    
    # Content Addiction Risk
    if content_addiction is not None and not content_addiction.empty:
        html_content += """
            <h3>Content Categories with Addiction Risk</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>View Count</th>
                    <th>Negative Score</th>
                    <th>Risk Score</th>
                </tr>
        """
        
        for _, row in content_addiction.head().iterrows():
            html_content += f"""
                <tr>
                    <td>{row['category']}</td>
                    <td>{row['view_count']}</td>
                    <td>{row['avg_negative_score']:.2f}</td>
                    <td>{row['addiction_risk_score']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        text_content += "\nContent Categories with Addiction Risk:\n"
        text_content += f"{content_addiction[['category', 'view_count', 'avg_negative_score', 'addiction_risk_score']].head().to_string(index=False)}\n"
    
    # Finish HTML
    html_content += """
        <p>Analysis complete. Images of all visualizations are available in the main directory.</p>
    </body>
    </html>
    """
    
    text_content += "\nAnalysis complete. Images of all visualizations are available in the main directory.\n"
    
    # Write files
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    with open(text_report, 'w') as f:
        f.write(text_content)
    
    logger.info(f"Report generated: {report_file}")
    logger.info(f"Text report generated: {text_report}")
    
    return report_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YouTube Mental Health Analyzer')
    parser.add_argument('--uri', default="bolt://localhost:7687", help='Neo4j URI')
    parser.add_argument('--user', default="neo4j", help='Neo4j username')
    parser.add_argument('--password', default="12345678", help='Neo4j password')
    parser.add_argument('--analysis', default="all", 
                        choices=["all", "temporal", "sentiment", "viewing", "addiction"],
                        help='Type of analysis to run')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Connect to Neo4j
    analyzer = MentalHealthAnalyzer(args.uri, args.user, args.password)
    
    try:
        save_results = not args.no_save
        
        # Run requested analysis
        if args.analysis == "all":
            result = run_comprehensive_analysis(analyzer, save_results)
        elif args.analysis == "temporal":
            result = run_temporal_analysis(analyzer, save_results)
        elif args.analysis == "sentiment":
            result = run_sentiment_analysis(analyzer, save_results)
        elif args.analysis == "viewing":
            result = run_viewing_pattern_analysis(analyzer, save_results)
        elif args.analysis == "addiction":
            result = run_addiction_analysis(analyzer, save_results)
        
        print(f"Analysis complete: {args.analysis}")
        
    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}", exc_info=True)
        raise
    finally:
        analyzer.close()

if __name__ == "__main__":
    main() 