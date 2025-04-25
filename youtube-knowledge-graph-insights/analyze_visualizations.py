import os
import json
import argparse
from PIL import Image
import base64
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import re

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_gpt4(image_path, api_key):
    """Analyze an image using GPT-4 Vision API."""
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this visualization from a YouTube pattern analysis report. Provide a detailed summary of what the visualization shows, its key insights, and what it might indicate about the user's YouTube viewing patterns and mental health. Focus on actionable insights."
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
        "max_tokens": 500
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error analyzing image: {response.text}"

def extract_visualizations_from_html(html_file):
    """Extract visualization paths from HTML file."""
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    visualizations = []
    for img in soup.select('.all-visualization img'):
        src = img.get('src', '')
        alt = img.get('alt', '')
        if src and not src.startswith('http'):
            visualizations.append({
                'path': src,
                'name': alt
            })
    
    return visualizations

def update_html_with_analyses(html_file, analyses):
    """Update HTML file with AI-generated analyses."""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update visualization summaries
    for analysis in analyses:
        name = analysis['name']
        summary = analysis['summary']
        
        # Create a pattern to match the placeholder in the HTML
        pattern = f'<div class="all-visualization-summary">\s*<!-- AI will analyze this image and provide insights -->\s*<p><em>AI analysis will be generated for this visualization.</em></p>\s*</div>'
        
        # Replace the first occurrence of the pattern with the analysis
        replacement = f'<div class="all-visualization-summary">\n                            {summary}\n                        </div>'
        content = re.sub(pattern, replacement, content, count=1)
    
    # Update the AI Analysis section
    ai_analysis_section = '<div class="ai-analysis">\n            <h3>Visualization Insights</h3>\n            <div id="ai-summaries">\n'
    
    for analysis in analyses:
        name = analysis['name']
        summary = analysis['summary']
        ai_analysis_section += f'                <div class="pattern-item">\n                    <span class="pattern-name">{name}:</span>\n                    <span class="pattern-value">\n                        {summary}\n                    </span>\n                </div>\n'
    
    ai_analysis_section += '            </div>\n        </div>'
    
    # Replace the placeholder in the AI Analysis section
    pattern = '<div class="ai-analysis">\s*<h3>Visualization Insights</h3>\s*<div id="ai-summaries">.*?</div>\s*</div>'
    content = re.sub(pattern, ai_analysis_section, content, flags=re.DOTALL)
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description='Analyze visualizations in HTML report')
    parser.add_argument('html_file', help='Path to the HTML report file')
    parser.add_argument('--api-key', help='OpenAI API key', required=True)
    args = parser.parse_args()
    
    # Extract visualizations from HTML
    visualizations = extract_visualizations_from_html(args.html_file)
    print(f"Found {len(visualizations)} visualizations to analyze")
    
    # Analyze each visualization
    analyses = []
    for viz in visualizations:
        print(f"Analyzing {viz['name']}...")
        summary = analyze_image_with_gpt4(viz['path'], args.api_key)
        analyses.append({
            'name': viz['name'],
            'path': viz['path'],
            'summary': summary
        })
    
    # Update HTML with analyses
    update_html_with_analyses(args.html_file, analyses)
    print(f"Updated {args.html_file} with AI-generated analyses")

if __name__ == "__main__":
    main() 