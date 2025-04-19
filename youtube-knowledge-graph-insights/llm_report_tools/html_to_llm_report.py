import os
import re
from bs4 import BeautifulSoup
import json
import argparse

def extract_visualizations(html_content):
    """Extract all visualizations and their descriptions from the HTML."""
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
            
            visualizations.append({
                'title': title,
                'image_path': img_src,
                'is_full_width': is_full_width
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

def create_llm_report(html_file_path, output_file_path=None):
    """Create a report optimized for LLM consumption from the HTML file."""
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract information
    visualizations = extract_visualizations(html_content)
    pattern_summary = extract_pattern_summary(html_content)
    recommendations = extract_recommendations(html_content)
    
    # Create a structured report
    report = {
        'visualizations': visualizations,
        'pattern_summary': pattern_summary,
        'recommendations': recommendations
    }
    
    # Create a text version of the report
    text_report = "# YouTube Pattern Analysis Report\n\n"
    
    # Add visualizations section
    text_report += "## Visualizations\n\n"
    for viz in visualizations:
        text_report += f"- **{viz['title']}**\n"
        if viz['is_full_width']:
            text_report += "  - *Full-width visualization*\n"
    
    # Add pattern summary section
    text_report += "\n## Pattern Summary\n\n"
    for pattern, value in pattern_summary.items():
        text_report += f"- **{pattern}**: {value}\n"
    
    # Add recommendations section
    text_report += "\n## Recommendations\n\n"
    for rec in recommendations:
        text_report += f"- {rec}\n"
    
    # Save the report
    if output_file_path:
        # Save JSON version
        json_output = output_file_path.replace('.txt', '.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Save text version
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"Report saved to {output_file_path} and {json_output}")
        return output_file_path, json_output
    
    return text_report, report

def main():
    parser = argparse.ArgumentParser(description='Convert HTML report to LLM-friendly format')
    parser.add_argument('html_file', help='Path to the HTML report file')
    parser.add_argument('--output', '-o', help='Path to save the output report (optional)')
    
    args = parser.parse_args()
    
    # If no output path is provided, use the HTML file name with .txt extension
    if not args.output:
        base_name = os.path.splitext(args.html_file)[0]
        args.output = f"{base_name}_llm_report.txt"
    
    create_llm_report(args.html_file, args.output)

if __name__ == "__main__":
    main() 