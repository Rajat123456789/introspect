import os
import json
from flask import Blueprint, jsonify, request, make_response
from pathlib import Path

introspect_bp = Blueprint('introspect', __name__)

# Simple CORS headers for all responses
@introspect_bp.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Authorization,X-Requested-With')
    return response

# Handle OPTIONS requests for all routes
@introspect_bp.route('/<path:path>', methods=['OPTIONS'])
@introspect_bp.route('/', methods=['OPTIONS'], defaults={'path': ''})
def options_handler(path):
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# Use the specified directory for content files
PROMPT_ENG_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../content_for_prompt'))
PROMPT_ENG_DIR = Path(PROMPT_ENG_DIR_PATH)

# Log and create the directory if needed
print(f"Using content directory at: {PROMPT_ENG_DIR_PATH}")
if not os.path.exists(PROMPT_ENG_DIR_PATH):
    print(f"Directory not found, creating: {PROMPT_ENG_DIR_PATH}")
    os.makedirs(PROMPT_ENG_DIR_PATH, exist_ok=True)
else:
    print(f"Found existing directory at {PROMPT_ENG_DIR_PATH}")

# File path helper function to reduce duplication
def get_file_path(filename):
    return os.path.join(PROMPT_ENG_DIR_PATH, filename)

# Standard error response helper
def error_response(message, status_code=500):
    return jsonify({"status": "error", "message": message}), status_code

# Debug endpoint for checking configuration
@introspect_bp.route('/debug', methods=['GET'])
def debug_routes():
    """Debug endpoint to check if routes are properly registered"""
    raw_exists = os.path.exists(get_file_path('context_raw.json'))
    insights_exists = os.path.exists(get_file_path('context_insights.json'))
    
    return jsonify({
        'status': 'ok',
        'directory_info': {
            'path': str(PROMPT_ENG_DIR),
            'exists': PROMPT_ENG_DIR.exists(),
        },
        'files': {
            'context_raw.json': raw_exists,
            'context_insights.json': insights_exists
        },
        'routes': [
            '/api/introspect/debug',
            '/api/introspect/data',
            '/api/introspect/insights'
        ]
    })

@introspect_bp.route('/data', methods=['GET'])
def get_introspection_data():
    """API endpoint to get raw introspection data"""
    try:
        file_path = get_file_path('context_raw.json')
        print(f"Attempting to read data from: {file_path}")
        
        if not os.path.exists(file_path):
            return error_response(f"No context data file found at {file_path}", 404)
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return jsonify(data)
    except Exception as e:
        print(f"Error reading context data: {str(e)}")
        return error_response(f"Error reading context data: {str(e)}")

@introspect_bp.route('/insights', methods=['GET'])
def get_introspection_insights():
    """API endpoint to get processed introspection insights"""
    try:
        file_path = get_file_path('context_insights.json')
        print(f"Attempting to read insights from: {file_path}")
        
        if not os.path.exists(file_path):
            return error_response(f"No insights file found at {file_path}", 404)
            
        with open(file_path, 'r') as f:
            insights = json.load(f)
            
        return jsonify(insights)
    except Exception as e:
        print(f"Error reading insights data: {str(e)}")
        return error_response(f"Error reading insights data: {str(e)}")

# Routes for writing data
@introspect_bp.route('/data', methods=['POST'])
def update_introspection_data():
    """API endpoint to update raw introspection data"""
    try:
        data = request.get_json()
        if not data:
            return error_response("No data provided in request", 400)
            
        file_path = get_file_path('context_raw.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return jsonify({"status": "success", "message": "Raw data updated successfully"})
    except Exception as e:
        print(f"Error updating raw data: {str(e)}")
        return error_response(f"Error updating raw data: {str(e)}")

@introspect_bp.route('/insights', methods=['POST'])
def update_introspection_insights():
    """API endpoint to update processed introspection insights"""
    try:
        data = request.get_json()
        if not data:
            return error_response("No data provided in request", 400)
            
        file_path = get_file_path('context_insights.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return jsonify({"status": "success", "message": "Insights updated successfully"})
    except Exception as e:
        print(f"Error updating insights: {str(e)}")
        return error_response(f"Error updating insights: {str(e)}") 