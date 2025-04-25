import os
import json
import glob
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
import shutil
import PIL.Image
import base64
import io
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_report(report_path):
    """Load a JSON report file."""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON report {report_path}: {str(e)}")
        return None

def find_report_files(base_dir):
    """Find all pattern analysis report directories."""
    # Look for directories that match the pattern_analysis pattern
    pattern_dirs = glob.glob(os.path.join(base_dir, "pattern_analysis*"))
    return pattern_dirs

def extract_date_range_from_dir(dir_path):
    """Extract date range from directory name if available."""
    # Try to extract date from directory name
    dir_name = os.path.basename(dir_path)
    if "pattern_analysis_" in dir_name:
        date_str = dir_name.replace("pattern_analysis_", "")
        try:
            return datetime.strptime(date_str, "%Y%m%d").strftime("%B %d, %Y")
        except ValueError:
            pass
    return "Unknown Date"

def find_visualization_files(dir_path):
    """Find all visualization files in a directory."""
    viz_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        viz_files.extend(glob.glob(os.path.join(dir_path, ext)))
    return viz_files

def find_specific_visualizations(base_dir, specific_files):
    """Find specific visualization files in the directory structure."""
    found_files = []
    
    for file_path in specific_files:
        # Check if the file exists directly
        if os.path.exists(file_path):
            found_files.append(file_path)
            logger.info(f"Found file: {file_path}")
            continue
        
        # If not, try to find it in the directory structure
        file_name = os.path.basename(file_path)
        for root, _, files in os.walk(base_dir):
            if file_name in files:
                full_path = os.path.join(root, file_name)
                found_files.append(full_path)
                logger.info(f"Found file: {full_path}")
                break
    
    return found_files

def analyze_visualization_with_openai(image_path):
    """Analyze a visualization using OpenAI API and return a summary."""
    try:
        # Check if OpenAI API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found in environment variables")
            # Ask for the API key
            print("\nOpenAI API key not found in environment variables.")
            api_key = input("Please enter your OpenAI API key: ").strip()
            if not api_key:
                logger.error("No OpenAI API key provided")
                return None
            # Set the API key in the environment for future use
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info("OpenAI API key set from user input")
        
        # Read the image file
        with open(image_path, "rb") as image_file:
            # Encode the image to base64
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create a more specific prompt based on the filename
        filename = os.path.basename(image_path).lower()
        
        # Base context about the individual's mental health data
        context = "This visualization shows mental health data for a specific individual who has been tracking their mental wellbeing over time. The data represents personal mental health scores and indicators that reflect their emotional state, stress levels, and overall psychological wellbeing. The individual has been monitoring these metrics to understand patterns in their mental health and identify factors that may influence their emotional state."
        
        prompt = f"{context}\n\nAnalyze this data visualization chart and provide a detailed summary of what it shows. Include the type of chart, the main metrics or variables being displayed, any trends or patterns you can identify, and the general purpose of the visualization."
        
        if "mental_health" in filename:
            prompt = f"{context}\n\nAnalyze this mental health visualization in detail. This is a matplotlib chart showing the individual's mental health scores over time. Describe the trends in their mental health indicators, noting any significant changes, peaks, or valleys. Identify if there are any patterns that might indicate improving or declining mental health for this person. Pay attention to the y-axis which represents mental health scores and the x-axis which shows time periods. How do these scores change over time and what might this indicate about the individual's mental wellbeing?"
        elif "trend" in filename:
            prompt = f"{context}\n\nAnalyze this trend visualization in detail. This is a matplotlib chart showing trends in the individual's mental health metrics over time. Describe the direction and magnitude of the trends shown, noting any significant changes over time. Identify if there are any seasonal patterns or cyclical behavior in their mental health. Pay attention to the y-axis values and how they change across the x-axis time periods. What do these trends suggest about the individual's mental health patterns and potential triggers or alleviators?"
        elif "sentiment" in filename:
            prompt = f"{context}\n\nAnalyze this sentiment analysis visualization in detail. This is a matplotlib chart showing sentiment scores from the individual's content consumption over time. Describe how their sentiment scores change over time, noting any correlations with other metrics. Identify if there are any patterns in sentiment that might relate to their mental health. Pay attention to positive vs negative sentiment trends and how they might indicate changes in the individual's emotional wellbeing."
        elif "forecast" in filename:
            prompt = f"{context}\n\nAnalyze this forecasting visualization in detail. This is a matplotlib chart showing predicted mental health trends for the individual. Describe the predicted trends and compare them with historical data. Identify any potential future patterns or changes in their mental health indicators. Pay attention to the confidence intervals and how they might indicate uncertainty in the predictions. What might these forecasts suggest about the individual's future mental wellbeing?"
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        # Make the API request
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error analyzing image with OpenAI {image_path}: {str(e)}")
        return None

def analyze_visualization(image_path):
    """Analyze a visualization using AI and return a summary."""
    # Use OpenAI for image analysis
    description = analyze_visualization_with_openai(image_path)
    
    # If OpenAI fails, return a default message
    if description is None:
        return "Image analysis not available. Please try again later."
    
    return description

def create_html_report(report_dirs, output_file, specific_files=None, additional_dirs=None):
    """Create an HTML report combining all pattern analysis reports."""
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))
    
    # Add custom filters
    env.filters['basename'] = os.path.basename
    
    # Dictionary to store image analyses
    image_analyses = {}
    
    def relative_path_filter(file_path):
        """Convert absolute path to relative path for HTML."""
        # Get the directory of the output file
        output_dir = os.path.dirname(output_file)
        # Create a directory for visualizations in the output directory
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Copy the visualization file to the output directory
        file_name = os.path.basename(file_path)
        target_path = os.path.join(viz_dir, file_name)
        
        # Copy the file if it doesn't exist in the target directory
        if not os.path.exists(target_path):
            shutil.copy2(file_path, target_path)
        
        # Generate AI description for the visualization if not already done
        if file_path not in image_analyses:
            logger.info(f"Analyzing image: {file_path}")
            image_analyses[file_path] = analyze_visualization(file_path)
        
        return os.path.join("visualizations", file_name)
    
    env.filters['relative_path'] = relative_path_filter
    
    # Add the analyses to the environment so they can be accessed in the template
    env.globals['get_analysis'] = lambda file_path: image_analyses.get(file_path, "Analysis not available")
    
    template = env.get_template('report_template.html')
    
    # Collect data for each report
    reports_data = []
    
    # Add specific visualizations if provided
    if specific_files:
        specific_viz_files = find_specific_visualizations(os.path.dirname(os.path.abspath(__file__)), specific_files)
        if specific_viz_files:
            specific_report = {
                'date': 'Selected Visualizations',
                'json_data': None,
                'visualizations': specific_viz_files,
                'analyses': {}
            }
            # Pre-analyze all visualizations
            for viz in specific_viz_files:
                specific_report['analyses'][viz] = analyze_visualization(viz)
            
            reports_data.append(specific_report)
            logger.info(f"Added {len(specific_viz_files)} specific visualizations")
    
    # Add visualizations from additional directories if provided
    if additional_dirs:
        for dir_name, dir_path in additional_dirs.items():
            if os.path.exists(dir_path):
                viz_files = find_visualization_files(dir_path)
                if viz_files:
                    additional_report = {
                        'date': dir_name,
                        'json_data': None,
                        'visualizations': viz_files,
                        'analyses': {}
                    }
                    # Pre-analyze all visualizations
                    for viz in viz_files:
                        additional_report['analyses'][viz] = analyze_visualization(viz)
                    
                    reports_data.append(additional_report)
                    logger.info(f"Added {len(viz_files)} visualizations from {dir_name}")
    
    for dir_path in report_dirs:
        report_data = {}
        
        # Extract date range
        report_data['date'] = extract_date_range_from_dir(dir_path)
        
        # Load JSON report
        json_path = os.path.join(dir_path, 'analysis_results.json')
        if os.path.exists(json_path):
            report_data['json_data'] = load_json_report(json_path)
        else:
            report_data['json_data'] = None
            logger.warning(f"No JSON report found in {dir_path}")
        
        # Find visualization files
        viz_files = find_visualization_files(dir_path)
        report_data['visualizations'] = viz_files
        
        # Pre-analyze all visualizations
        report_data['analyses'] = {}
        for viz in viz_files:
            report_data['analyses'][viz] = analyze_visualization(viz)
        
        # Add to reports list
        reports_data.append(report_data)
    
    # Sort reports by date (newest first)
    reports_data.sort(key=lambda x: x['date'], reverse=True)
    
    # Render template
    html_content = template.render(reports=reports_data)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {output_file}")

def main():
    """Main function to combine reports."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all report directories
    report_dirs = find_report_files(script_dir)
    
    if not report_dirs:
        logger.warning("No pattern analysis reports found.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, "combined_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create HTML report
    output_file = os.path.join(output_dir, "combined_pattern_analysis.html")
    
    # Specific visualizations to include
    specific_files = [
        os.path.join(script_dir, "analysis_reports", "20250302_170343", "category_sentiment_viz.png"),
        os.path.join(script_dir, "analysis_reports", "20250302_170343", "mental_health_monthly_trends.png"),
        os.path.join(script_dir, "analysis_reports", "20250302_170343", "mental_health_weekly_trends.png"),
        os.path.join(script_dir, "analysis_reports", "content_category_impact.png"),
        os.path.join(script_dir, "analysis_reports", "mental_health_forecast_simple.png")
    ]
    
    # Include visualizations from additional directories
    additional_dirs = {
        "Analysis Reports": os.path.join(script_dir, "analysis_reports"),
        "Insights": os.path.join(script_dir, "insights")
    }
    
    create_html_report(report_dirs, output_file, specific_files=specific_files, additional_dirs=additional_dirs)

if __name__ == "__main__":
    main() 