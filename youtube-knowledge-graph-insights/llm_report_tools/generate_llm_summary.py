import os
import json
import argparse
from bs4 import BeautifulSoup
import re
from datetime import datetime

def extract_visualization_data(html_content):
    """Extract detailed information about visualizations from the HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    visualizations = []
    
    # Find all visualization elements
    viz_elements = soup.select('.all-visualization')
    
    for viz in viz_elements:
        title_elem = viz.select_one('.all-visualization-title')
        img_elem = viz.select_one('img')
        
        if title_elem and img_elem:
            title = title_elem.text.strip()
            img_src = img_elem.get('src', '')
            
            # Determine if it's a full-width visualization
            is_full_width = 'full-width' in viz.get('class', [])
            
            # Extract visualization type based on filename
            viz_type = "general"
            if "mental_health" in img_src.lower():
                viz_type = "mental_health"
            elif "sentiment" in img_src.lower():
                viz_type = "sentiment"
            elif "category" in img_src.lower():
                viz_type = "category"
            elif "trend" in img_src.lower():
                viz_type = "trend"
            
            # Extract time period if available
            time_period = "unknown"
            if "daily" in img_src.lower():
                time_period = "daily"
            elif "weekly" in img_src.lower():
                time_period = "weekly"
            elif "monthly" in img_src.lower():
                time_period = "monthly"
            
            visualizations.append({
                'title': title,
                'image_path': img_src,
                'is_full_width': is_full_width,
                'type': viz_type,
                'time_period': time_period
            })
    
    return visualizations

def extract_pattern_summary(html_content):
    """Extract pattern summary from the HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    pattern_summary = {}
    
    # Find pattern summary section
    pattern_items = soup.select('.pattern-item')
    
    for item in pattern_items:
        name_elem = item.select_one('.pattern-name')
        value_elem = item.select_one('.pattern-value')
        
        if name_elem and value_elem:
            name = name_elem.text.strip().rstrip(':')
            value = value_elem.text.strip()
            pattern_summary[name] = value
    
    return pattern_summary

def extract_recommendations(html_content):
    """Extract recommendations from the HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    recommendations = []
    
    # Find recommendations section
    rec_list = soup.select('.pattern-summary ul li')
    
    for rec in rec_list:
        recommendations.append(rec.text.strip())
    
    return recommendations

def categorize_visualizations(visualizations):
    """Categorize visualizations by type and time period."""
    categorized = {
        'mental_health': {
            'daily': [],
            'weekly': [],
            'monthly': [],
            'other': []
        },
        'sentiment': [],
        'category': [],
        'trend': [],
        'other': []
    }
    
    for viz in visualizations:
        if viz['type'] == 'mental_health':
            if viz['time_period'] in ['daily', 'weekly', 'monthly']:
                categorized['mental_health'][viz['time_period']].append(viz)
            else:
                categorized['mental_health']['other'].append(viz)
        elif viz['type'] in ['sentiment', 'category', 'trend']:
            categorized[viz['type']].append(viz)
        else:
            categorized['other'].append(viz)
    
    return categorized

def generate_llm_summary(html_file_path, output_file_path=None):
    """Generate a comprehensive summary report for LLM analysis."""
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract information
    visualizations = extract_visualization_data(html_content)
    pattern_summary = extract_pattern_summary(html_content)
    recommendations = extract_recommendations(html_content)
    
    # Categorize visualizations
    categorized_viz = categorize_visualizations(visualizations)
    
    # Create a structured report
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source_html': html_file_path
        },
        'visualizations': {
            'all': visualizations,
            'categorized': categorized_viz
        },
        'pattern_summary': pattern_summary,
        'recommendations': recommendations
    }
    
    # Create a text version of the report optimized for LLM consumption
    text_report = "# YouTube Pattern Analysis Report for LLM\n\n"
    
    # Add metadata
    text_report += "## Report Metadata\n\n"
    text_report += f"- Generated: {report['metadata']['generated_at']}\n"
    text_report += f"- Source: {report['metadata']['source_html']}\n\n"
    
    # Add visualizations section
    text_report += "## Visualizations Overview\n\n"
    text_report += f"Total visualizations: {len(visualizations)}\n\n"
    
    # Add categorized visualizations
    text_report += "### Mental Health Visualizations\n\n"
    
    # Daily mental health trends
    if categorized_viz['mental_health']['daily']:
        text_report += "#### Daily Mental Health Trends\n\n"
        for viz in categorized_viz['mental_health']['daily']:
            text_report += f"- **{viz['title']}**\n"
    
    # Weekly mental health trends
    if categorized_viz['mental_health']['weekly']:
        text_report += "\n#### Weekly Mental Health Trends\n\n"
        for viz in categorized_viz['mental_health']['weekly']:
            text_report += f"- **{viz['title']}**\n"
    
    # Monthly mental health trends
    if categorized_viz['mental_health']['monthly']:
        text_report += "\n#### Monthly Mental Health Trends\n\n"
        for viz in categorized_viz['mental_health']['monthly']:
            text_report += f"- **{viz['title']}**\n"
    
    # Other mental health visualizations
    if categorized_viz['mental_health']['other']:
        text_report += "\n#### Other Mental Health Visualizations\n\n"
        for viz in categorized_viz['mental_health']['other']:
            text_report += f"- **{viz['title']}**\n"
    
    # Sentiment visualizations
    if categorized_viz['sentiment']:
        text_report += "\n### Sentiment Visualizations\n\n"
        for viz in categorized_viz['sentiment']:
            text_report += f"- **{viz['title']}**\n"
    
    # Category visualizations
    if categorized_viz['category']:
        text_report += "\n### Category Visualizations\n\n"
        for viz in categorized_viz['category']:
            text_report += f"- **{viz['title']}**\n"
    
    # Trend visualizations
    if categorized_viz['trend']:
        text_report += "\n### Trend Visualizations\n\n"
        for viz in categorized_viz['trend']:
            text_report += f"- **{viz['title']}**\n"
    
    # Other visualizations
    if categorized_viz['other']:
        text_report += "\n### Other Visualizations\n\n"
        for viz in categorized_viz['other']:
            text_report += f"- **{viz['title']}**\n"
    
    # Add pattern summary section
    text_report += "\n## Pattern Summary\n\n"
    for pattern, value in pattern_summary.items():
        text_report += f"- **{pattern}**: {value}\n"
    
    # Add recommendations section
    text_report += "\n## Recommendations\n\n"
    for rec in recommendations:
        text_report += f"- {rec}\n"
    
    # Add LLM analysis instructions
    text_report += "\n## Instructions for LLM Analysis\n\n"
    text_report += "Please analyze this YouTube viewing pattern data and provide insights on:\n\n"
    text_report += "1. Mental health trends over time (daily, weekly, monthly)\n"
    text_report += "2. Sentiment analysis of content consumed\n"
    text_report += "3. Content category preferences and their impact\n"
    text_report += "4. Overall viewing patterns and potential concerns\n"
    text_report += "5. Specific recommendations based on the data\n\n"
    text_report += "Focus on actionable insights that could help improve the user's YouTube viewing habits and mental well-being.\n"
    
    # Save the report
    if output_file_path:
        # Save JSON version
        json_output = output_file_path.replace('.txt', '.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Save text version
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"LLM summary report saved to {output_file_path} and {json_output}")
        return output_file_path, json_output
    
    return text_report, report

def main():
    parser = argparse.ArgumentParser(description='Generate LLM-friendly summary from HTML report')
    parser.add_argument('html_file', help='Path to the HTML report file')
    parser.add_argument('--output', '-o', help='Path to save the output report (optional)')
    
    args = parser.parse_args()
    
    # If no output path is provided, use the HTML file name with _llm_summary.txt extension
    if not args.output:
        base_name = os.path.splitext(args.html_file)[0]
        args.output = f"{base_name}_llm_summary.txt"
    
    generate_llm_summary(args.html_file, args.output)

if __name__ == "__main__":
    main() 