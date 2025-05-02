import os
import json
from flask import Blueprint, jsonify, request, make_response, current_app
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

def ensure_dirs():
    """Ensure all necessary directories exist"""
    os.makedirs(PROMPT_ENG_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROMPT_ENG_DIR, 'youtube_data'), exist_ok=True)

@introspect_bp.route('/data', methods=['GET'])
def get_introspection_data():
    """Get the raw introspection data for the user"""
    ensure_dirs()
    
    try:
        # Path to the context raw file
        context_raw_file = os.path.join(PROMPT_ENG_DIR, 'context_raw.json')
        
        # If the file doesn't exist, check if we have YouTube data
        if not os.path.exists(context_raw_file):
            youtube_data_file = os.path.join(PROMPT_ENG_DIR, 'youtube_data', 'youtube_history.json')
            
            if os.path.exists(youtube_data_file):
                try:
                    with open(youtube_data_file, 'r') as f:
                        youtube_data = json.load(f)
                    
                    # Create a basic context_raw.json with YouTube data
                    context_data = {
                        'youtube': {
                            'last_updated': youtube_data[0]['timestamp'] if youtube_data else "",
                            'recent_videos': youtube_data[0]['videos'] if youtube_data else []
                        }
                    }
                    
                    # Save this data for future use
                    with open(context_raw_file, 'w') as f:
                        json.dump(context_data, f, indent=2)
                    
                    return jsonify(context_data)
                except Exception as e:
                    print(f"Error creating context from YouTube data: {str(e)}")
            
            # If no data found, return an empty object
            return jsonify({})
        
        # If the file exists, read it
        with open(context_raw_file, 'r') as f:
            context_data = json.load(f)
        
        # Check if there's YouTube data to include
        youtube_data_file = os.path.join(PROMPT_ENG_DIR, 'youtube_data', 'youtube_history.json')
        if os.path.exists(youtube_data_file) and (not context_data.get('youtube') or not context_data['youtube'].get('recent_videos')):
            try:
                with open(youtube_data_file, 'r') as f:
                    youtube_data = json.load(f)
                
                if youtube_data:
                    context_data['youtube'] = {
                        'last_updated': youtube_data[0]['timestamp'] if youtube_data else "",
                        'recent_videos': youtube_data[0]['videos'] if youtube_data else []
                    }
                    
                    # Save the updated data
                    with open(context_raw_file, 'w') as f:
                        json.dump(context_data, f, indent=2)
            except Exception as e:
                print(f"Error updating context with YouTube data: {str(e)}")
        
        return jsonify(context_data)
    except Exception as e:
        print(f"Error retrieving introspection data: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving introspection data: {str(e)}"
        }), 500

@introspect_bp.route('/insights', methods=['GET'])
def get_introspection_insights():
    """Get the analyzed introspection insights for the user"""
    ensure_dirs()
    
    try:
        # Path to the context insights file
        context_insights_file = os.path.join(PROMPT_ENG_DIR, 'context_insights.json')
        
        # If the file doesn't exist, check if we can generate basic insights from YouTube data
        if not os.path.exists(context_insights_file):
            youtube_data_file = os.path.join(PROMPT_ENG_DIR, 'youtube_data', 'youtube_history.json')
            
            if os.path.exists(youtube_data_file):
                try:
                    with open(youtube_data_file, 'r') as f:
                        youtube_data = json.load(f)
                    
                    if youtube_data and youtube_data[0].get('videos'):
                        videos = youtube_data[0]['videos']
                        
                        # Generate basic YouTube insights
                        youtube_insights = {
                            'summary': f"You have watched {len(videos)} YouTube videos recently.",
                            'patterns': generate_youtube_insights(videos)
                        }
                        
                        insights_data = {
                            'youtube': youtube_insights
                        }
                        
                        # Save these insights for future use
                        with open(context_insights_file, 'w') as f:
                            json.dump(insights_data, f, indent=2)
                        
                        return jsonify(insights_data)
                except Exception as e:
                    print(f"Error generating insights from YouTube data: {str(e)}")
            
            # If no data found, return an empty object
            return jsonify({})
        
        # If the file exists, read it
        with open(context_insights_file, 'r') as f:
            insights_data = json.load(f)
        
        # Check if there's YouTube data to update insights from
        youtube_data_file = os.path.join(PROMPT_ENG_DIR, 'youtube_data', 'youtube_history.json')
        if os.path.exists(youtube_data_file) and not insights_data.get('youtube'):
            try:
                with open(youtube_data_file, 'r') as f:
                    youtube_data = json.load(f)
                
                if youtube_data and youtube_data[0].get('videos'):
                    videos = youtube_data[0]['videos']
                    
                    # Generate basic YouTube insights
                    youtube_insights = {
                        'summary': f"You have watched {len(videos)} YouTube videos recently.",
                        'patterns': generate_youtube_insights(videos)
                    }
                    
                    insights_data['youtube'] = youtube_insights
                    
                    # Save the updated insights
                    with open(context_insights_file, 'w') as f:
                        json.dump(insights_data, f, indent=2)
            except Exception as e:
                print(f"Error updating insights with YouTube data: {str(e)}")
        
        return jsonify(insights_data)
    except Exception as e:
        print(f"Error retrieving introspection insights: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving introspection insights: {str(e)}"
        }), 500

def generate_youtube_insights(videos):
    """Generate basic insights from YouTube video data"""
    if not videos:
        return []
    
    # Extract channel and title information
    channels = {}
    categories = set()
    watch_times = []
    
    for video in videos:
        channel = video.get('channel', 'Unknown')
        title = video.get('title', 'Unknown video')
        
        # Count channel appearances
        if channel in channels:
            channels[channel] += 1
        else:
            channels[channel] = 1
        
        # Extract possible categories from title
        words = title.split()
        for word in words:
            if len(word) > 3:  # Only consider somewhat meaningful words
                categories.add(word.lower())
    
    # Get top channels
    top_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)[:3]
    
    insights = []
    
    # Add channel preference insight
    if top_channels:
        channels_text = ", ".join([f"{channel} ({count} videos)" for channel, count in top_channels])
        insights.append(f"Your most watched channels are {channels_text}.")
    
    # Add variety insight
    if len(channels) > 5:
        insights.append(f"You watch a diverse range of content across {len(channels)} different channels.")
    elif len(channels) > 1:
        insights.append(f"You tend to focus on a few specific channels ({len(channels)} in total).")
    else:
        insights.append("You've been watching videos from a single channel recently.")
    
    return insights

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