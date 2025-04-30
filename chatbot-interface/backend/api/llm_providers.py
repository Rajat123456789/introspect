import os
import json
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
import google.generativeai as genai
from openai import OpenAI


# Load environment variables again to ensure they're available in this module
load_dotenv()

# Get API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

system_prompts = json.load(open('content_for_prompt/system_prompts.json'))
# Debug information
print(f"In llm_providers.py - OpenAI API Key available: {bool(openai_api_key)} (length: {len(openai_api_key) if openai_api_key else 0})")
print(f"In llm_providers.py - Gemini API Key available: {bool(gemini_api_key)} (length: {len(gemini_api_key) if gemini_api_key else 0})")


# Configure OpenAI - use modern client approach
openai_client = None
if openai_api_key:
    try:
        # Initialize with the api_key parameter
        openai_client = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")

# Configure Gemini
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        print("Gemini API configured successfully")
    except Exception as e:
        print(f"Error configuring Gemini API: {str(e)}")

# Default model configurations (can be overridden by environment variables)
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

class LLMProviders:
    @staticmethod
    def get_openai_response(user_message, system_prompt=None, model=None):
        """Get a response from OpenAI's API."""
        try:
            # Use environment variable for model if not specified
            model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
            
            # Prepare the messages array for the OpenAI API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})
            
            # Use the modern client approach
            if openai_client:
                try:
                    # Using the newest OpenAI API format with chat.completions
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800
                    )
                    
                    # Access response content based on new client format
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Error with OpenAI client approach: {str(e)}")
                    raise e
                
        except Exception as e:
            print(f"Error getting OpenAI response: {str(e)}")
            return f"Sorry, I encountered an error with the OpenAI API: {str(e)}"
    
    @staticmethod
    def get_gemini_response(user_message, system_prompt=None, model=None):
        """Get a response from Google's Gemini API."""
        try:
            # Use environment variable for model if not specified
            model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            
            # Initialize the model
            gemini_model = GenerativeModel(model)
            
            prompt = ""
            if system_prompt:
                prompt = f"{system_prompt}\n\nUser: {user_message}"
            else:
                prompt = user_message
            
            # Add debug print    
            print(f"Sending to Gemini: model={model}, prompt length={len(prompt)}")
            
            response = gemini_model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"Error getting Gemini response: {str(e)}")
            return f"Sorry, I encountered an error with the Gemini API: {str(e)}"
        

SYSTEM_PROMPTS = {
    "base": {
        "openai": system_prompts["openai_gpt_4o"],
        "gemini": system_prompts["openai_gpt_4o"]
    },
    "health": {
        "openai": system_prompts["llama4_maverick"],
        "gemini": system_prompts["llama4_maverick"]
    },
    "introspect": {
        "openai": system_prompts["introspect_llm"],
        "gemini": system_prompts["introspect_llm"]
    }
} 