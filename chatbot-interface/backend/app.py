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
        
        # Load YouTube data if available - for both base and introspect models
        youtube_data = None
        try:
            context_raw_file = os.path.join(PROMPT_ENG_DIR, 'context_raw.json')
            if os.path.exists(context_raw_file):
                with open(context_raw_file, 'r') as f:
                    context_data = json.load(f)
                    if isinstance(context_data, dict) and 'youtube' in context_data:
                        youtube_data = context_data['youtube']
        except Exception as e:
            print(f"Error loading YouTube data: {str(e)}")
        
        # Add introspection data to the prompt if needed
        if model_type == 'introspect':
            introspect_data = data.get('introspect_data', {})
            introspect_insights = data.get('introspect_insights', {})
            
            # Add explicit YouTube data if available
            if youtube_data and (use_raw_data or not introspect_data):
                if not introspect_data:
                    introspect_data = {}
                introspect_data['youtube'] = youtube_data
            
            if introspect_data or introspect_insights:
                introspect_context = "Additional context based on user data:\n"
                if introspect_data:
                    introspect_context += json.dumps(introspect_data, indent=2) + "\n\n"
                if introspect_insights:
                    introspect_context += json.dumps(introspect_insights, indent=2) + "\n\n"
                
                system_prompt += f"\n\n{introspect_context}"
        
        # For base model, add YouTube data if available and raw data is requested
        elif model_type == 'base' and youtube_data and use_raw_data:
            youtube_context = "Recent YouTube viewing history:\n"
            youtube_context += json.dumps(youtube_data, indent=2) + "\n\n"
            system_prompt += f"\n\n{youtube_context}"
            
            # Add guidance for the base model on how to use YouTube data
            youtube_guidance = (
                "The user has shared their YouTube viewing history. You can use this to personalize your responses. "
                "Feel free to reference patterns you observe in their viewing habits if it's relevant to their question, "
                "but always maintain a respectful tone and focus primarily on answering their query."
            )
            system_prompt += f"\n\n{youtube_guidance}"
        
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

@app.route('/api/youtube/history', methods=['POST'])
def youtube_history():
    """Receive and store YouTube watch history data from the browser extension"""
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'videos' not in data:
            return jsonify({
                "status": "error",
                "message": "Invalid request. Videos data is required."
            }), 400
        
        videos = data.get('videos', [])
        timestamp = data.get('timestamp', '')
        source = data.get('source', 'unknown')
        
        print(f"Received {len(videos)} videos from {source}")
        
        # Create a directory to store the YouTube data if it doesn't exist
        youtube_data_dir = os.path.join(PROMPT_ENG_DIR, 'youtube_data')
        os.makedirs(youtube_data_dir, exist_ok=True)
        
        # Save the raw data
        youtube_data_file = os.path.join(youtube_data_dir, 'youtube_history.json')
        
        # Load existing data if available
        existing_data = []
        try:
            if os.path.exists(youtube_data_file):
                with open(youtube_data_file, 'r') as f:
                    existing_data = json.load(f)
        except Exception as e:
            print(f"Error loading existing YouTube data: {str(e)}")
        
        # Prepare new data entry
        new_data = {
            'timestamp': timestamp,
            'source': source,
            'videos': videos
        }
        
        # Add new data to existing data (limiting to most recent 100 entries)
        if isinstance(existing_data, list):
            existing_data.insert(0, new_data)
            existing_data = existing_data[:100]  # Keep only the 100 most recent entries
        else:
            existing_data = [new_data]
        
        # Save the updated data
        with open(youtube_data_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        # Also save to context_raw.json for introspection
        context_raw_file = os.path.join(PROMPT_ENG_DIR, 'context_raw.json')
        
        # Load existing context data if available
        context_data = {}
        try:
            if os.path.exists(context_raw_file):
                with open(context_raw_file, 'r') as f:
                    context_data = json.load(f)
        except Exception as e:
            print(f"Error loading existing context data: {str(e)}")
        
        # Update YouTube section in context data
        if not isinstance(context_data, dict):
            context_data = {}
        
        context_data['youtube'] = {
            'last_updated': timestamp,
            'recent_videos': videos
        }
        
        # Save the updated context data
        with open(context_raw_file, 'w') as f:
            json.dump(context_data, f, indent=2)
        
        # Generate insights about the videos using our AI agent
        # First, generate the basic insights for backward compatibility
        basic_video_insights = generate_youtube_insights(videos)
        
        # Then, use our new AI-powered analysis for more advanced insights
        ai_video_insights = analyze_youtube_patterns(videos)
        
        # Combine all insights
        all_insights = basic_video_insights + ai_video_insights
        
        # Update or create context_insights.json file
        context_insights_file = os.path.join(PROMPT_ENG_DIR, 'context_insights.json')
        
        # Load existing insights if available
        insights_data = {}
        try:
            if os.path.exists(context_insights_file):
                with open(context_insights_file, 'r') as f:
                    insights_data = json.load(f)
        except Exception as e:
            print(f"Error loading existing insights data: {str(e)}")
        
        # Update YouTube section in insights data
        if not isinstance(insights_data, dict):
            insights_data = {}
        
        insights_data['youtube'] = {
            'summary': f"You have watched {len(videos)} YouTube videos recently.",
            'patterns': all_insights
        }
        
        # Save the updated insights data
        with open(context_insights_file, 'w') as f:
            json.dump(insights_data, f, indent=2)
        
        # Generate model-specific insights
        if videos and len(videos) > 0:
            # Get most recent video for immediate analysis
            recent_video = videos[0]
            video_title = recent_video.get('title', 'Unknown video')
            video_channel = recent_video.get('channel', 'Unknown channel')
            video_description = recent_video.get('description', '')
            
            # Generate model-specific insights using our new AI-powered approach
            model_insights = {
                'base': generate_base_model_insight(video_title, video_channel, video_description),
                'health': generate_health_model_insight(video_title, video_channel, video_description),
                'introspect': generate_introspect_model_insight(video_title, video_channel, video_description, ai_video_insights)
            }
            
            # Save model-specific insights to a file
            model_insights_file = os.path.join(PROMPT_ENG_DIR, 'model_insights.json')
            with open(model_insights_file, 'w') as f:
                json.dump(model_insights, f, indent=2)
            
            # Return the insights along with the success response
            return jsonify({
                "status": "success",
                "message": f"Successfully received and stored {len(videos)} videos",
                "videos_count": len(videos),
                "insights": all_insights,
                "model_insights": model_insights
            })
        
        return jsonify({
            "status": "success",
            "message": f"Successfully received and stored {len(videos)} videos",
            "videos_count": len(videos),
            "insights": all_insights
        })
    
    except Exception as e:
        print(f"Error processing YouTube history: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing YouTube history: {str(e)}"
        }), 500

@app.route('/api/youtube/model_insights', methods=['GET'])
def get_youtube_model_insights():
    """Get model-specific insights for YouTube data"""
    try:
        # Path to the model insights file
        model_insights_file = os.path.join(PROMPT_ENG_DIR, 'model_insights.json')
        
        # Check if the file exists
        if not os.path.exists(model_insights_file):
            return jsonify({
                "status": "error",
                "message": "No model insights available"
            }), 404
        
        # Read the model insights
        with open(model_insights_file, 'r') as f:
            model_insights = json.load(f)
        
        # Get the specified model type from query parameter
        model_type = request.args.get('model_type', 'all')
        
        if model_type == 'all':
            return jsonify({
                "status": "success",
                "model_insights": model_insights
            })
        elif model_type in model_insights:
            return jsonify({
                "status": "success",
                "insight": model_insights[model_type]
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"No insights available for model type: {model_type}"
            }), 404
    
    except Exception as e:
        print(f"Error retrieving model insights: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error retrieving model insights: {str(e)}"
        }), 500

@app.route('/api/youtube/analyze', methods=['POST'])
def analyze_youtube_data():
    """Manually trigger analysis of existing YouTube data using AI agent"""
    try:
        # Get existing YouTube data
        context_raw_file = os.path.join(PROMPT_ENG_DIR, 'context_raw.json')
        if not os.path.exists(context_raw_file):
            return jsonify({
                "status": "error",
                "message": "No YouTube data found. Please upload data first."
            }), 404
            
        # Load the data
        with open(context_raw_file, 'r') as f:
            context_data = json.load(f)
            
        if not isinstance(context_data, dict) or 'youtube' not in context_data or 'recent_videos' not in context_data['youtube']:
            return jsonify({
                "status": "error",
                "message": "Invalid YouTube data format in context file."
            }), 400
            
        videos = context_data['youtube']['recent_videos']
        
        if not videos or len(videos) == 0:
            return jsonify({
                "status": "error",
                "message": "No videos found in YouTube data."
            }), 404
            
        # Generate AI-powered insights
        ai_video_insights = analyze_youtube_patterns(videos)
        print(f"AI video insights: {ai_video_insights}")
        
        # Get the most recent video for individual analysis
        recent_video = videos[0]
        video_title = recent_video.get('title', 'Unknown video')
        video_channel = recent_video.get('channel', 'Unknown channel')
        video_description = recent_video.get('description', '')
        
        # Generate model-specific insights
        model_insights = {
            'base': generate_base_model_insight(video_title, video_channel, video_description),
            'health': generate_health_model_insight(video_title, video_channel, video_description),
            'introspect': generate_introspect_model_insight(video_title, video_channel, video_description, ai_video_insights)
        }
        
        # Save the generated insights
        model_insights_file = os.path.join(PROMPT_ENG_DIR, 'model_insights.json')
        with open(model_insights_file, 'w') as f:
            json.dump(model_insights, f, indent=2)
            
        # Return the generated insights
        return jsonify({
            "status": "success",
            "message": f"Successfully analyzed {len(videos)} videos",
            "videos_count": len(videos),
            "ai_insights": ai_video_insights,
            "model_insights": model_insights
        })
        
    except Exception as e:
        print(f"Error analyzing YouTube data: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error analyzing YouTube data: {str(e)}"
        }), 500

def generate_youtube_insights(videos):
    """Generate basic insights from YouTube video data"""
    if not videos:
        return []
    
    # Extract channel and title information
    channels = {}
    categories = set()
    watched_recently = []
    
    for video in videos:
        channel = video.get('channel', 'Unknown')
        title = video.get('title', 'Unknown video')
        
        # Store recent videos for summary
        if len(watched_recently) < 3:
            watched_recently.append(title)
        
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
    
    # Add recent videos insight
    if watched_recently:
        recent_text = ", ".join([f'"{title}"' for title in watched_recently])
        insights.append(f"You recently watched: {recent_text}.")
    
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

def generate_base_model_insight(title, channel, description):
    """Generate a casual insight from the base model perspective"""
    
    # Analyze the video content from title and description
    topics = []
    if "music" in title.lower() or "song" in title.lower() or "audio" in title.lower():
        topics.append("music")
    if "tutorial" in title.lower() or "how to" in title.lower() or "guide" in title.lower():
        topics.append("tutorial")
    if "news" in title.lower() or "update" in title.lower() or "latest" in title.lower():
        topics.append("news")
    if "game" in title.lower() or "gaming" in title.lower() or "gameplay" in title.lower():
        topics.append("gaming")
    if "review" in title.lower() or "analysis" in title.lower():
        topics.append("review")
    
    # Default topic if none detected
    if not topics:
        topics.append("content")
    
    # Generate a casual comment based on the topic
    topic = topics[0]
    
    comments = {
        "music": f"I see you've been listening to '{title}' from {channel}. Interesting music choice! Would you like me to find similar songs?",
        "tutorial": f"You checked out '{title}' - looks like you're learning something new! Let me know if you need any help understanding this topic.",
        "news": f"I noticed you watched '{title}' from {channel}. Staying updated on current events is always good. Any particular news topic you're following?",
        "gaming": f"You watched '{title}' - looks like an interesting game! Are you a fan of this type of gaming content?",
        "review": f"I see you watched a review of '{title}'. Looking for opinions on something you're interested in?",
        "content": f"Interesting choice with '{title}' from {channel}. What did you find most engaging about this content?"
    }
    
    return comments.get(topic, f"I notice you watched '{title}' from {channel}. Interesting stuff! What made you choose this video?")

def generate_health_model_insight(title, channel, description):
    """Generate a health-focused insight based on the video content"""
    
    # Look for health-related keywords in the title and description
    health_keywords = ["health", "fitness", "workout", "exercise", "diet", "nutrition", 
                      "wellness", "mental health", "meditation", "yoga", "mindfulness",
                      "sleep", "stress", "anxiety", "depression"]
    
    # Check if this is explicitly health content
    is_health_content = any(keyword in (title + " " + description).lower() for keyword in health_keywords)
    
    if is_health_content:
        return f"I notice you're watching '{title}', which relates to health and wellness. Remember that consistent small steps are key to maintaining good health habits. Would you like to discuss any specific health topics related to this content?"
    
    # For non-health content, provide a gentle wellness reminder
    if "music" in title.lower():
        return f"Music like '{title}' can be a great way to manage stress and improve mood. Did you know that listening to music you enjoy can reduce anxiety and even improve cognitive performance?"
    
    if any(term in title.lower() for term in ["game", "gaming", "gameplay", "stream"]):
        return f"While enjoying content like '{title}', remember to take regular breaks (about 5 minutes every hour) to reduce eye strain and prevent sedentary behavior. How long do you typically watch or play in one session?"
    
    # Default health insight for general content
    return f"Taking time to enjoy content like '{title}' can be good for mental wellness, especially when balanced with physical activity. Have you considered pairing your viewing time with some light stretching or standing breaks?"

def generate_introspect_model_insight(title, channel, description, patterns):
    """Generate a deeper introspective insight based on the video and viewing patterns using AI"""
    
    # Instead of predefined reflections, we'll use an AI to generate a personalized reflection
    try:
        # Create a rich context about the video and patterns
        video_context = {
            "title": title,
            "channel": channel,
            "description": description,
            "viewing_patterns": patterns if patterns else []
        }
        
        # Create a prompt for the introspect model
        prompt = f"""
As an introspection guide, analyze this user's YouTube viewing data and generate a thoughtful reflection that encourages self-awareness.

Video details:
- Title: {title}
- Channel: {channel}
- Description: {description[:200]}... (truncated)

Viewing patterns detected:
{json.dumps(patterns, indent=2) if patterns else "No specific patterns detected yet."}

Your task:
1. Identify potential themes, interests, or emotional needs reflected in this content choice
2. Create a thoughtful, non-judgmental reflection that helps the user better understand their viewing choices
3. End with an open-ended question that encourages deeper self-reflection
4. Keep your response conversational and friendly, like a trusted friend noticing patterns
5. You need to keep track of patterns in Addiction, Rabbit Holes, Escaping if the user is watching a lot of these videos, offer help to get out of it
6. You need to keep track of patterns in Learning, Teaching, Exploring if the user is watching a lot of these videos, offer help to get more out of it

Example insight format: "
<Theme Identification>: `[Theme] like Addiction, Rabbit Holes, Escaping, Learning, Teaching, Exploring`
<IntroSpect Model thoughts>: `I notice you've been watching several videos about [topic]. This might suggest [potential meaning]. How does [reflective question]?"
<Reflection>: This is a reflection of the user's viewing choices, 2-3 sentences maximum, followed by a thoughtful question"

Important: Focus on genuine insight rather than generic observations. Avoid being prescriptive or judgmental.
"""
        # print(f"<generate_introspect_model_insight>Prompt: {prompt}")
        
        # Use the appropriate LLM provider to generate the reflection
        # We'll use OpenAI by default for consistency, but this could be configured
        reflection = LLMProviders.get_openai_response(prompt, SYSTEM_PROMPTS.get("introspect", {}).get("openai", ""))
        print(f"<generate_introspect_model_insight>Reflection: {reflection}")
        # If we couldn't get a reflection from the AI, fall back to a simple template
        if not reflection:
            import random
            fallback_reflections = [
                f"Your choice to watch '{title}' might reflect your current interests or mood. What drew you to this particular content today?",
                f"The content we consume often mirrors aspects of ourselves we're exploring. Did '{title}' connect with you on a personal level?",
                f"Sometimes our viewing choices reveal patterns about our values or current focus. Does '{title}' relate to something you're working on or thinking about in your life?"
            ]
            reflection = random.choice(fallback_reflections)
        
        return reflection
    
    except Exception as e:
        print(f"Error generating AI reflection: {str(e)}")
        # Fall back to a simple reflection if the AI approach fails
        return f"I notice you watched '{title}'. What drew you to this content, and what does it reveal about your current interests?"

def analyze_youtube_patterns(videos, max_videos=25):
    """Use AI to analyze patterns across multiple YouTube videos for deeper insights"""
    if not videos or len(videos) == 0:
        return []
    
    # Limit the number of videos to analyze to avoid token limits
    videos_to_analyze = videos[:max_videos]
    
    try:
        # Create a structured representation of the videos
        video_data = []
        for i, video in enumerate(videos_to_analyze):
            video_data.append({
                "index": i + 1,
                "title": video.get('title', 'Unknown'),
                "channel": video.get('channel', 'Unknown'),
                "description": video.get('description', '')[:100] + '...' if video.get('description', '') else ''
            })
        
        # Create a prompt for the AI to analyze patterns
        prompt = f"""
        As an introspection guide, analyze this collection of {len(videos_to_analyze)} recently watched YouTube videos. 
        Identify meaningful patterns, themes, or trends that might reveal the user's current interests, emotional state, or learning goals.

        Videos (most recent first):
        {json.dumps(video_data, indent=2)}

        Your task:
        1. Identify 2-3 meaningful patterns or themes across these videos and the theme of the video
        2. For each pattern, suggest what it might reveal about the user's current interests, needs, or state of mind
        3. Format your response as a JSON array of insights, where each insight is a string that describes a pattern and its potential meaning
        4. Each insight should be conversational and end with a thoughtful question that encourages self-reflection
        5. Keep each insight to 2-3 sentences, followed by a question
        6. You need to keep track of patterns in Addiction, Rabbit Holes, Escaping if the user is watching a lot of these videos, offer help to get out of it
        7. You need to keep track of patterns in Learning, Teaching, Exploring if the user is watching a lot of these videos, offer help to get more out of it

        Example insight format: "I notice you've been watching several videos about [topic]. This might suggest [potential meaning]. How does [reflective question]?
        Theme Identification: [Theme] like Addiction, Rabbit Holes, Escaping, Learning, Teaching, Exploring"

        Important: Focus on genuine insights rather than superficial observations. Be thoughtful, non-judgmental, and curious.
"""
        
        # Use the LLM to generate the insights
        # We'll use the introspect model which is designed for this type of reflection
        response = LLMProviders.get_openai_response(prompt, SYSTEM_PROMPTS.get("introspect", {}).get("openai", ""))
        # print(f"<analyze_youtube_patterns>LLM response: {response}")
        # Parse the JSON response
        try:
            # The response might have markdown formatting or extra text, so try to extract just the JSON part
            import re
            json_pattern = r'\[[\s\S]*\]'  # Pattern to match JSON array
            match = re.search(json_pattern, response)
            
            if match:
                json_str = match.group(0)
                insights = json.loads(json_str)
                return insights
            else:
                # If we couldn't find a JSON array, treat the whole response as a single insight
                return [response]
                
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response as a single insight
            return [response]
    
    except Exception as e:
        print(f"Error analyzing YouTube patterns: {str(e)}")
        # Return a fallback insight
        return ["I've noticed you've been watching a variety of YouTube content recently. Our viewing choices often reflect our current interests and curiosities. What themes have you noticed in your own viewing patterns lately?"]

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    # Use 127.0.0.1 to avoid IPv6 binding issues
    app.run(host='127.0.0.1', port=port, debug=True) 