import openai
import os
import argparse

def extract_key_insights(markdown_file, api_key, output_file="insights_summary.md"):
    """
    Extract key insights from the complete analysis using OpenAI's o4-mini model.
    
    Args:
        markdown_file (str): Path to the markdown file containing the analysis
        api_key (str): OpenAI API key
        output_file (str): Path to save the extracted insights
        
    Returns:
        str: Path to the file containing the extracted insights
    """   
    
    # Set up OpenAI client
    openai.api_key = api_key
    client = openai.OpenAI()
    
    # Read the markdown file
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {markdown_file} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Create the prompt for the API
    prompt = f"""
    Extract the key insights from the following analysis of the user's online activity. 
    Focus on patterns and trends related to mental and emotional health.
    The insights should be useful as context for a mental and emotional health introspection assistant.
    Identify viewing patterns, content preferences, and potential emotional triggers.
    Format the response as concise, actionable bullet points.
    
    Here's the complete analysis:
    {content}
    """
    
    # Call the OpenAI API with o4-mini model
    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are an expert in behavioral analysis and mental health."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_completion_tokens=1000
        )
        
        # Extract the insights from the response
        insights = response.choices[0].message.content
        
        # Save the insights to a file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(insights)
        
        print(f"Insights successfully extracted and saved to {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error during API call or saving results: {e}")
        return None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract key insights from markdown analysis")
    parser.add_argument("--markdown_file", help="Path to the markdown file containing the analysis")
    parser.add_argument("--api_key", help="OpenAI API key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--output", default="insights_summary.md", help="Path to save the extracted insights")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: OpenAI API key is required. Provide it with --api_key or set OPENAI_API_KEY environment variable.")
        exit(1)
    
    extract_key_insights(args.markdown_file, args.api_key, args.output)
