import os
import argparse
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def analyze_with_openai(report_path, api_key=None):
    """Analyze the report using OpenAI's API."""
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
    
    # Read the report
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    # Prepare the prompt
    prompt = f"""Please analyze the following YouTube viewing pattern data and provide insights:

{report_content}

Based on this data, please provide:
1. A summary of the user's YouTube viewing patterns
2. Analysis of mental health trends over time
3. Insights about content preferences and their potential impact
4. Specific recommendations for improving viewing habits

Format your response as a comprehensive report with clear sections and actionable insights."""

    # Call OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are an expert in analyzing YouTube viewing patterns and their impact on mental health."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    analysis = result["choices"][0]["message"]["content"]
    
    return analysis

def analyze_with_local_llm(report_path, model_path):
    """Analyze the report using a local LLM."""
    # This is a placeholder for using a local LLM
    # You would need to implement the specific API calls for your local LLM
    print(f"Analyzing report with local LLM at {model_path}")
    print("This is a placeholder function. Implement the specific API calls for your local LLM.")
    return "Local LLM analysis placeholder"

def main():
    parser = argparse.ArgumentParser(description='Analyze YouTube pattern report with an LLM')
    parser.add_argument('report_file', help='Path to the LLM report file')
    parser.add_argument('--output', '-o', help='Path to save the analysis (optional)')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (optional, can use environment variable)')
    parser.add_argument('--local', '-l', action='store_true', help='Use local LLM instead of OpenAI')
    parser.add_argument('--model-path', '-m', help='Path to local LLM model (required if --local is used)')
    
    args = parser.parse_args()
    
    # Determine output path
    if not args.output:
        base_name = os.path.splitext(args.report_file)[0]
        args.output = f"{base_name}_analysis.txt"
    
    # Analyze the report
    if args.local:
        if not args.model_path:
            print("Error: --model-path is required when using --local")
            return
        analysis = analyze_with_local_llm(args.report_file, args.model_path)
    else:
        analysis = analyze_with_openai(args.report_file, args.api_key)
    
    if analysis:
        # Save the analysis
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(analysis)
        
        print(f"Analysis saved to {args.output}")
        
        # Print a preview
        print("\nAnalysis Preview:")
        print("-" * 50)
        preview_length = min(500, len(analysis))
        print(analysis[:preview_length] + "..." if len(analysis) > preview_length else analysis)
        print("-" * 50)
        print(f"Full analysis saved to {args.output}")

if __name__ == "__main__":
    main() 