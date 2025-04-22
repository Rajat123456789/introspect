#!/usr/bin/env python3
"""
Generate a final prompt for the Introspect assistant by combining different components based on command line arguments.
"""

import json
import argparse
import os
from datetime import datetime

def load_json_file(file_path):
    """Load and return JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def generate_prompt(args):
    """Generate the final prompt based on provided arguments."""
    # Define paths
    content_dir = os.path.join(os.path.dirname(__file__), "content_for_prompt")
    system_prompts_path = os.path.join(content_dir, "system_prompts.json")
    user_prompts_path = os.path.join(content_dir, "user_prompts.json")
    
    if args.context_type == "insights":
        context_path = os.path.join(content_dir, "context_insights.json")
    else:  # raw
        context_path = os.path.join(content_dir, "context_raw.json")
    
    # Load JSON files
    system_prompts = load_json_file(system_prompts_path)
    user_prompts = load_json_file(user_prompts_path)
    context_data = load_json_file(context_path)
    
    if not all([system_prompts, user_prompts, context_data]):
        print("Error: Failed to load required JSON files")
        return None
    
    # Extract the appropriate system prompt
    system_prompt = system_prompts.get(args.system_prompt_type, "")
    if not system_prompt:
        print(f"Warning: System prompt type '{args.system_prompt_type}' not found")
    
    # Extract the appropriate user prompt
    user_prompt = user_prompts.get(args.user_prompt_type, "")
    if not user_prompt:
        print(f"Warning: User prompt type '{args.user_prompt_type}' not found")
    
    # Build context based on flags
    context_sections = []
    
    # Check if any context flags were specified
    any_context_specified = args.context_youtube or args.context_spotify or args.context_health
    
    # Function to add a context section if it exists and should be included
    def add_context_section(context_key, context_title, should_include):
        if (should_include or not any_context_specified) and context_key in context_data:
            if args.context_type == "insights":
                context_sections.append(f"{context_title} Activity:\n" + str(context_data[context_key]))
            else:
                context_sections.append(f"{context_title} Activity:\n" + json.dumps(context_data[context_key], indent=2))
    
    # Add context sections based on flags (or all if none specified)
    add_context_section("youtube", "YouTube", args.context_youtube)
    add_context_section("spotify", "Spotify", args.context_spotify)
    add_context_section("health", "Health", args.context_health)
    
    context_str = "\n\n".join(context_sections)
    
    # Build the final prompt
    final_prompt = f"""
<system_prompt>
{system_prompt}
</system_prompt>

<context>
{context_str}
</context>

<user_prompt>
{user_prompt}
</user_prompt>
"""
    
    return final_prompt

def save_prompt_to_file(prompt, output_path=None):
    """Save the generated prompt to a file."""
    if not output_path:
        # Use a fixed filename instead of timestamp-based naming
        output_dir = os.path.join(os.path.dirname(__file__), "generated_prompt")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "latest_prompt.txt")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Prompt successfully saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving prompt to file: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a prompt for the Introspect assistant")
    
    # Define command-line arguments
    parser.add_argument("--system_prompt_type", choices=["openai_gpt_4o", "llama4_maverick", "introspect_llm"], 
                        default="introspect_llm", help="Model to use")
    parser.add_argument("--user_prompt_type", choices=["mental", "emotional"], 
                        default="mental", help="Type of introspection")
    parser.add_argument("--context_type", choices=["insights", "raw"], 
                        default="insights", help="Type of context data to use")
    parser.add_argument("--context_youtube", action="store_true", help="Include YouTube context")
    parser.add_argument("--context_spotify", action="store_true", help="Include Spotify context")
    parser.add_argument("--context_health", action="store_true", help="Include health context")
    parser.add_argument("--output", help="Path to save the generated prompt")
    
    args = parser.parse_args()
    
    # Generate the prompt
    final_prompt = generate_prompt(args)
    
    if final_prompt:
        # Save the prompt to a file
        save_prompt_to_file(final_prompt, args.output)
