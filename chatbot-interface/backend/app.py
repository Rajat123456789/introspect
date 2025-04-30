from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from api.introspect import introspect_bp, PROMPT_ENG_DIR
from api.llm_providers import LLMProviders, SYSTEM_PROMPTS

# Load environment variables
load_dotenv()

# Debug: Print API keys status
openai_key = os.getenv("OPENAI_API_KEY", "")
gemini_key = os.getenv("GEMINI_API_KEY", "")
print(f"OpenAI API Key loaded: {bool(openai_key)} (length: {len(openai_key) if openai_key else 0})")
print(f"Gemini API Key loaded: {bool(gemini_key)} (length: {len(gemini_key) if gemini_key else 0})")

app = Flask(__name__)

# Configure CORS globally
CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     expose_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False,
     max_age=3600)

# Register blueprints
app.register_blueprint(introspect_bp, url_prefix='/api/introspect')

# Generic OPTIONS handler for CORS preflight requests
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check the health status of the API and configuration"""
    # Check API keys
    api_status = {
        "openai": bool(openai_key),
        "gemini": bool(gemini_key)
    }
    
    # Check content directory and files
    content_dir_status = {
        "directory_exists": os.path.exists(PROMPT_ENG_DIR),
        "files": {}
    }
    
    if os.path.exists(PROMPT_ENG_DIR):
        content_dir_status["files"]["raw_data"] = os.path.exists(os.path.join(PROMPT_ENG_DIR, 'context_raw.json'))
        content_dir_status["files"]["insights"] = os.path.exists(os.path.join(PROMPT_ENG_DIR, 'context_insights.json'))
    
    return jsonify({
        "status": "healthy", 
        "message": "Backend server is running",
        "api_status": api_status,
        "content_status": content_dir_status
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat requests from the frontend"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "status": "error",
                "message": "Invalid request. Message is required."
            }), 400
        
        # Extract parameters
        user_message = data.get('message', '')
        model_type = data.get('model_type', 'base')
        use_raw_data = data.get('use_raw_data', False)
        api_provider = data.get('api_provider', 'openai')  # Default to OpenAI if not specified
        
        # Validate API provider
        if api_provider not in ['openai', 'gemini']:
            return jsonify({
                "status": "error",
                "message": f"Invalid API provider: {api_provider}. Supported providers are 'openai' and 'gemini'."
            }), 400
        
        # Check if API key is available
        if api_provider == 'openai' and not openai_key:
            return jsonify({
                "status": "error",
                "message": "OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable."
            }), 500
        
        if api_provider == 'gemini' and not gemini_key:
            return jsonify({
                "status": "error",
                "message": "Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable."
            }), 500
        
        # Get system prompt based on model type and API provider
        system_prompt = SYSTEM_PROMPTS.get(model_type, {}).get(api_provider, "")
        
        # Add introspection data to the prompt if needed
        if model_type == 'introspect':
            introspect_data = data.get('introspect_data', {})
            introspect_insights = data.get('introspect_insights', {})
            
            if introspect_data or introspect_insights:
                introspect_context = "Additional context based on user data:\n"
                if introspect_data:
                    introspect_context += json.dumps(introspect_data, indent=2) + "\n\n"
                if introspect_insights:
                    introspect_context += json.dumps(introspect_insights, indent=2) + "\n\n"
                
                system_prompt += f"\n\n{introspect_context}"
        
        # Get response from the appropriate provider
        if api_provider == 'openai':
            response_message = LLMProviders.get_openai_response(user_message, system_prompt)
        else:  # gemini
            response_message = LLMProviders.get_gemini_response(user_message, system_prompt)
        
        # Return a fallback response if necessary
        if not response_message:
            response_message = f"No response generated for message: {user_message}. Please try again."
        
        return jsonify({
            "message": response_message
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing chat request: {str(e)}"
        }), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    return jsonify({
        "status": "success",
        "message": "Conversation history cleared"
    })

@app.route('/api/routes', methods=['GET'])
def list_routes():
    """List all available routes for debugging"""
    routes = [
        {
            'endpoint': rule.endpoint,
            'methods': [method for method in rule.methods if method != 'OPTIONS' and method != 'HEAD'],
            'path': str(rule)
        } for rule in app.url_map.iter_rules()
    ]
    
    return jsonify({
        'routes': routes,
        'content_directory': str(PROMPT_ENG_DIR),
        'content_files': os.listdir(PROMPT_ENG_DIR) if os.path.exists(PROMPT_ENG_DIR) else []
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # Use 127.0.0.1 to avoid IPv6 binding issues
    app.run(host='127.0.0.1', port=port, debug=True) 