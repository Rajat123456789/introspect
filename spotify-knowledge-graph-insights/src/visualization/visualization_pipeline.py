"""
Spotify Visualization Pipeline

This module orchestrates the entire visualization process for Spotify listening history data.
It provides a simple interface to generate all visualizations from a single CSV file.
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.visualization.data_processors.spotify_data_processor import SpotifyDataProcessor
from src.visualization.plotters.basic_plotters import BasicPlotters
from src.visualization.plotters.advanced_plotters import AdvancedPlotters
from src.visualization.plotters.interactive_plotters import InteractivePlotters

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyVisualizationPipeline:
    """
    Pipeline for generating visualizations from Spotify listening history data.
    """
    
    def __init__(self, input_file=None, output_dir=None):
        """
        Initialize the visualization pipeline.
        
        Args:
            input_file (str, optional): Path to the Spotify history CSV file.
            output_dir (str, optional): Directory to save visualizations to.
        """
        self.input_file = input_file
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"spotify_visualizations_{timestamp}"
            
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_processor = SpotifyDataProcessor(input_file)
        self.basic_plotters = BasicPlotters(output_dir)
        self.advanced_plotters = AdvancedPlotters(output_dir)
        self.interactive_plotters = InteractivePlotters(output_dir)
        
        # Initialize data containers
        self.processed_data = None
        self.listening_patterns = None
        self.audio_features_summary = None
        
    def process_data(self, input_file=None):
        """
        Process the input data.
        
        Args:
            input_file (str, optional): Path to the Spotify history CSV file.
                If not provided, uses the path from initialization.
                
        Returns:
            tuple: (processed_data, listening_patterns, audio_features_summary)
        """
        if input_file:
            self.input_file = input_file
            
        logger.info(f"Processing data from {self.input_file}")
        
        # Run data processing pipeline
        self.processed_data, self.listening_patterns, self.audio_features_summary = \
            self.data_processor.process_pipeline(self.input_file)
            
        logger.info("Data processing completed")
        
        return self.processed_data, self.listening_patterns, self.audio_features_summary
    
    def generate_basic_visualizations(self):
        """
        Generate basic visualizations.
        
        Returns:
            dict: Dictionary of generated figures.
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Call process_data() first.")
            return {}
            
        logger.info("Generating basic visualizations")
        
        figures = {}
        
        # Listening patterns by time
        figures['listening_by_hour'] = self.basic_plotters.plot_listening_by_time(
            self.processed_data, time_col='hour', save_as='listening_by_hour.png')
            
        figures['listening_by_day'] = self.basic_plotters.plot_listening_by_time(
            self.processed_data, time_col='day_of_week', save_as='listening_by_day.png')
            
        # Top items
        if 'artist_name' in self.processed_data.columns:
            figures['top_artists'] = self.basic_plotters.plot_top_items(
                self.processed_data, item_col='artist_name', save_as='top_artists.png')
                
        if 'track_genre' in self.processed_data.columns:
            figures['top_genres'] = self.basic_plotters.plot_top_items(
                self.processed_data, item_col='track_genre', save_as='top_genres.png')
                
        # Audio features distribution
        figures['audio_features_dist'] = self.basic_plotters.plot_audio_features_distribution(
            self.processed_data, save_as='audio_features_distribution.png')
            
        # Feature correlations
        figures['feature_correlations'] = self.basic_plotters.plot_feature_correlations(
            self.processed_data, save_as='feature_correlations.png')
            
        # Genre word cloud
        if 'track_genre' in self.processed_data.columns:
            figures['genre_wordcloud'] = self.basic_plotters.plot_genre_wordcloud(
                self.processed_data, save_as='genre_wordcloud.png')
                
        # Listening timeline
        figures['listening_timeline'] = self.basic_plotters.plot_listening_timeline(
            self.processed_data, time_unit='W', save_as='listening_timeline_weekly.png')
            
        logger.info(f"Generated {len(figures)} basic visualizations")
        
        return figures
    
    def generate_advanced_visualizations(self):
        """
        Generate advanced visualizations.
        
        Returns:
            dict: Dictionary of generated figures.
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Call process_data() first.")
            return {}
            
        logger.info("Generating advanced visualizations")
        
        figures = {}
        
        # Feature radar chart
        figures['feature_radar'] = self.advanced_plotters.plot_feature_radar_chart(
            self.processed_data, save_as='feature_radar_chart.png')
            
        # PCA analysis
        if 'track_genre' in self.processed_data.columns:
            figures['pca_analysis'] = self.advanced_plotters.plot_pca_analysis(
                self.processed_data, color_by='track_genre', save_as='pca_analysis.png')
        else:
            figures['pca_analysis'] = self.advanced_plotters.plot_pca_analysis(
                self.processed_data, save_as='pca_analysis.png')
                
        # Clustering analysis
        figures['clustering'], _ = self.advanced_plotters.plot_clustering_analysis(
            self.processed_data, n_clusters=5, save_as='clustering_analysis.png')
            
        # Feature by time
        for feature in ['valence', 'energy', 'danceability']:
            if feature in self.processed_data.columns:
                figures[f'{feature}_by_time'] = self.advanced_plotters.plot_feature_by_time(
                    self.processed_data, feature=feature, time_unit='M', 
                    save_as=f'{feature}_by_time.png')
                    
        # Feature comparison by genre
        if 'track_genre' in self.processed_data.columns:
            figures['feature_by_genre'] = self.advanced_plotters.plot_feature_comparison_by_genre(
                self.processed_data, save_as='feature_comparison_by_genre.png')
                
        logger.info(f"Generated {len(figures)} advanced visualizations")
        
        return figures
    
    def generate_interactive_visualizations(self):
        """
        Generate interactive visualizations.
        
        Returns:
            dict: Dictionary of generated figures.
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Call process_data() first.")
            return {}
            
        logger.info("Generating interactive visualizations")
        
        figures = {}
        
        # Listening patterns heatmap
        figures['listening_heatmap'] = self.interactive_plotters.plot_listening_patterns_heatmap(
            self.processed_data, save_as='listening_patterns_heatmap.html')
            
        # Audio features radar
        figures['audio_radar'] = self.interactive_plotters.plot_audio_features_radar(
            self.processed_data, save_as='audio_features_radar.html')
            
        # Genre distribution sunburst
        if 'track_genre' in self.processed_data.columns:
            figures['genre_sunburst'] = self.interactive_plotters.plot_genre_distribution_sunburst(
                self.processed_data, save_as='genre_distribution_sunburst.html')
                
        # Listening timeline
        figures['interactive_timeline'] = self.interactive_plotters.plot_listening_timeline_interactive(
            self.processed_data, save_as='listening_timeline_interactive.html')
            
        # Top artists bar
        if 'artist_name' in self.processed_data.columns:
            figures['top_artists_bar'] = self.interactive_plotters.plot_top_artists_bar(
                self.processed_data, save_as='top_artists_interactive.html')
                
        # Audio features scatter
        figures['features_scatter'] = self.interactive_plotters.plot_audio_features_scatter(
            self.processed_data, save_as='audio_features_scatter.html')
            
        logger.info(f"Generated {len(figures)} interactive visualizations")
        
        return figures
    
    def run_pipeline(self, input_file=None, generate_basic=True, 
                    generate_advanced=True, generate_interactive=True):
        """
        Run the full visualization pipeline.
        
        Args:
            input_file (str, optional): Path to the Spotify history CSV file.
            generate_basic (bool): Whether to generate basic visualizations.
            generate_advanced (bool): Whether to generate advanced visualizations.
            generate_interactive (bool): Whether to generate interactive visualizations.
            
        Returns:
            dict: Dictionary of all generated figures.
        """
        # Process data
        self.process_data(input_file)
        
        all_figures = {}
        
        # Generate visualizations
        if generate_basic:
            basic_figures = self.generate_basic_visualizations()
            all_figures.update(basic_figures)
            
        if generate_advanced:
            advanced_figures = self.generate_advanced_visualizations()
            all_figures.update(advanced_figures)
            
        if generate_interactive:
            interactive_figures = self.generate_interactive_visualizations()
            all_figures.update(interactive_figures)
            
        logger.info(f"Visualization pipeline completed. Generated {len(all_figures)} visualizations.")
        logger.info(f"Visualizations saved to {self.output_dir}")
        
        return all_figures
    
    def generate_report(self):
        """
        Generate a simple HTML report with all visualizations.
        
        Returns:
            str: Path to the generated report.
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Call process_data() first.")
            return None
            
        logger.info("Generating HTML report")
        
        # Create report directory
        report_dir = Path(self.output_dir) / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Spotify Listening History Analysis</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1, h2, h3 { color: #1DB954; }",
            "        .section { margin-bottom: 30px; }",
            "        .viz-container { display: flex; flex-wrap: wrap; }",
            "        .viz-item { margin: 10px; max-width: 600px; }",
            "        .viz-item img { max-width: 100%; border: 1px solid #ddd; }",
            "        .viz-item iframe { border: 1px solid #ddd; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Spotify Listening History Analysis</h1>",
            f"    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "    <div class='section'>",
            "        <h2>Listening Patterns</h2>"
        ]
        
        # Add listening patterns summary
        if self.listening_patterns:
            html_content.extend([
                "        <table>",
                "            <tr><th>Metric</th><th>Value</th></tr>"
            ])
            
            for key, value in self.listening_patterns.items():
                if not isinstance(value, dict):
                    html_content.append(f"            <tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>")
                    
            html_content.append("        </table>")
        
        # Add basic visualizations section
        html_content.extend([
            "    </div>",
            "    <div class='section'>",
            "        <h2>Basic Visualizations</h2>",
            "        <div class='viz-container'>"
        ])
        
        # Add links to basic visualization images
        basic_images = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        for img in basic_images:
            img_path = f"../{img}"  # Relative path from report directory
            html_content.extend([
                "            <div class='viz-item'>",
                f"                <h3>{img.replace('.png', '').replace('_', ' ').title()}</h3>",
                f"                <img src='{img_path}' alt='{img}'>",
                "            </div>"
            ])
            
        # Add interactive visualizations section
        html_content.extend([
            "        </div>",
            "    </div>",
            "    <div class='section'>",
            "        <h2>Interactive Visualizations</h2>",
            "        <div class='viz-container'>"
        ])
        
        # Add links to interactive visualizations
        interactive_htmls = [f for f in os.listdir(self.output_dir) if f.endswith('.html')]
        for html_file in interactive_htmls:
            html_path = f"../{html_file}"  # Relative path from report directory
            html_content.extend([
                "            <div class='viz-item'>",
                f"                <h3>{html_file.replace('.html', '').replace('_', ' ').title()}</h3>",
                f"                <iframe src='{html_path}' width='600' height='400'></iframe>",
                f"                <p><a href='{html_path}' target='_blank'>Open in new tab</a></p>",
                "            </div>"
            ])
            
        # Close HTML
        html_content.extend([
            "        </div>",
            "    </div>",
            "</body>",
            "</html>"
        ])
        
        # Write HTML file
        report_path = report_dir / "index.html"
        with open(report_path, 'w') as f:
            f.write('\n'.join(html_content))
            
        logger.info(f"Report generated at {report_path}")
        
        return str(report_path)


def main():
    """
    Main function to run the visualization pipeline from command line.
    """
    parser = argparse.ArgumentParser(description='Generate visualizations from Spotify listening history data.')
    parser.add_argument('input_file', help='Path to the Spotify history CSV file')
    parser.add_argument('--output-dir', '-o', help='Directory to save visualizations to')
    parser.add_argument('--no-basic', action='store_true', help='Skip basic visualizations')
    parser.add_argument('--no-advanced', action='store_true', help='Skip advanced visualizations')
    parser.add_argument('--no-interactive', action='store_true', help='Skip interactive visualizations')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SpotifyVisualizationPipeline(args.input_file, args.output_dir)
    pipeline.run_pipeline(
        generate_basic=not args.no_basic,
        generate_advanced=not args.no_advanced,
        generate_interactive=not args.no_interactive
    )
    
    # Generate report if requested
    if args.report:
        report_path = pipeline.generate_report()
        print(f"Report generated at: {report_path}")


if __name__ == "__main__":
    main() 

