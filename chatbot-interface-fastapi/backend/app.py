from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Debug: Print API keys status
openai_key = os.getenv("OPENAI_API_KEY", "")
gemini_key = os.getenv("GEMINI_API_KEY", "")
print(f"OpenAI API Key loaded: {bool(openai_key)} (length: {len(openai_key) if openai_key else 0})")
print(f"Gemini API Key loaded: {bool(gemini_key)} (length: {len(gemini_key) if gemini_key else 0})")

# Load all report files into memory
def load_report_files():
    reports = {}
    report_files = [
        "health_history_report.txt",
        "health_live_report.txt",
        "youtube_history_report.txt",
        "youtube_live_report.txt"
    ]
    
    for file_name in report_files:
        file_path = Path(f"reports/{file_name}")
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reports[file_name] = f.read()
                print(f"Loaded report: {file_name} ({len(reports[file_name])} characters)")
            except Exception as e:
                print(f"Error loading report file {file_name}: {str(e)}")
        else:
            print(f"Report file not found: {file_path}")
    
    return reports

# Load reports at application startup
USER_REPORTS = load_report_files()

app = FastAPI(title="Introspect API", description="Backend API for Introspect application")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    max_age=3600,
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static/"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates/")

# System prompts for different model types
SYSTEM_PROMPTS = {
    "base": {
        "openai": "You are a helpful assistant. Respond to the user's questions directly and concisely.",
        "gemini": "You are a helpful assistant. Respond to the user's questions directly and concisely."
    },
    "health": {
        "openai": "You are a health and wellness expert assistant. Provide accurate, evidence-based information about health, fitness, nutrition, and medical topics.",
        "gemini": "You are a health and wellness expert assistant. Provide accurate, evidence-based information about health, fitness, nutrition, and medical topics."
    },
    "introspect": {
        "openai": "You are an introspective assistant that helps users reflect on their digital and health data. Guide them to insights about their behaviors and patterns.",
        "gemini": "You are an introspective assistant that helps users reflect on their digital and health data. Guide them to insights about their behaviors and patterns."
    }
}

# Augment system prompts with report data for context
def get_enhanced_system_prompt(model_type, api_provider):
    base_prompt = SYSTEM_PROMPTS.get(model_type, {}).get(api_provider, "")
    
    # Only augment the introspect model type with the report data
    if model_type == "introspect" and USER_REPORTS:
        report_context = "\n\nHere is important contextual information about the user's health and digital behavior:\n\n"
        
        # Add each report to the context
        for report_name, report_content in USER_REPORTS.items():
            if report_content:
                # Add a section for each report with a reasonable content limit to avoid token issues
                report_summary = report_content[:3000]  # Taking first 3000 chars as a summary
                report_context += f"--- {report_name} ---\n{report_summary}\n\n"
        
        return base_prompt + report_context
    
    return base_prompt

# Define model for chat request
class ChatRequest(BaseModel):
    message: str
    model_type: str = 'base'
    use_raw_data: bool = False
    api_provider: str = 'openai'
    introspect_data: Optional[Dict[str, Any]] = None
    introspect_insights: Optional[Dict[str, Any]] = None

# Define model for chat response
class ChatResponse(BaseModel):
    message: str
    status: Optional[str] = None

# Define model for error response
class ErrorResponse(BaseModel):
    status: str
    message: str

# Helper class for LLM providers
class LLMProviders:
    @staticmethod
    def get_openai_response(user_message, system_prompt):
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return f"Sorry, there was an error with the OpenAI service: {str(e)}"
    
    @staticmethod
    def get_gemini_response(user_message, system_prompt):
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-pro')
            
            chat = model.start_chat(history=[
                {"role": "user", "parts": [system_prompt]},
                {"role": "model", "parts": ["I understand and will act accordingly."]}
            ])
            
            response = chat.send_message(user_message)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return f"Sorry, there was an error with the Gemini service: {str(e)}"

# Frontend route - main UI
@app.get("/", tags=["frontend"])
async def index(request: Request):
    """Serve the main frontend UI"""
    return templates.TemplateResponse("index.html", {"request": request})

# Visualization template routes
@app.get("/health_history", name="Health_history", tags=["visualization"])
async def health_history(request: Request):
    """Serve the Health History visualization page"""
    return templates.TemplateResponse("Health_history.html", {"request": request})

@app.get("/health_live", name="Health_live", tags=["visualization"])
async def health_live(request: Request):
    """Serve the Health Live visualization page"""
    return templates.TemplateResponse("Health_live.html", {"request": request})

@app.get("/youtube_history", name="Youtube_history", tags=["visualization"])
async def youtube_history(request: Request):
    """Serve the YouTube History visualization page"""
    return templates.TemplateResponse("Youtube_history.html", {"request": request})

# Health check endpoint
@app.get("/api/health", response_model=Dict[str, Any], tags=["health"])
async def health_check():
    """Check the health status of the API and configuration"""
    # Check API keys
    api_status = {
        "openai": bool(openai_key),
        "gemini": bool(gemini_key)
    }
    
    # Also check if report files are loaded
    reports_loaded = {name: bool(content) for name, content in USER_REPORTS.items()}
    
    return {
        "status": "healthy", 
        "message": "Backend server is running",
        "api_status": api_status,
        "reports_loaded": reports_loaded
    }

@app.post("/api/chat", response_model=Union[ChatResponse, ErrorResponse], tags=["chat"])
async def chat(request: ChatRequest):
    """Process chat requests from the frontend"""
    try:
        # Extract parameters
        user_message = request.message
        model_type = request.model_type
        use_raw_data = request.use_raw_data
        api_provider = request.api_provider
        
        # Validate API provider
        if api_provider not in ['openai', 'gemini']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid API provider: {api_provider}. Supported providers are 'openai' and 'gemini'."
            )
        
        # Check if API key is available
        if api_provider == 'openai' and not openai_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable."
            )
        
        if api_provider == 'gemini' and not gemini_key:
            raise HTTPException(
                status_code=500,
                detail="Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable."
            )
        
        # Get enhanced system prompt with report context
        system_prompt = get_enhanced_system_prompt(model_type, api_provider)
        
        # Get response from the appropriate provider
        if api_provider == 'openai':
            response_message = LLMProviders.get_openai_response(user_message, system_prompt)
        else:  # gemini
            response_message = LLMProviders.get_gemini_response(user_message, system_prompt)
        
        # Return a fallback response if necessary
        if not response_message:
            response_message = f"No response generated for message: {user_message}. Please try again."
        
        return {"message": response_message}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.post("/api/clear_history", response_model=Dict[str, str], tags=["chat"])
async def clear_history():
    """Clear conversation history"""
    return {
        "status": "success",
        "message": "Conversation history cleared"
    }

@app.get("/api/routes", response_model=Dict[str, Any], tags=["debug"])
async def list_routes():
    """List all available routes for debugging"""
    routes = []
    
    for route in app.routes:
        routes.append({
            'endpoint': route.name,
            'methods': list(route.methods),
            'path': route.path
        })
    
    return {
        'routes': routes
    }

@app.get("/api/reports", response_model=Dict[str, Any], tags=["debug"])
async def list_loaded_reports():
    """List all loaded report files for debugging"""
    return {
        "reports": {name: len(content) for name, content in USER_REPORTS.items()}
    }

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 5000))
    # Use 127.0.0.1 to avoid IPv6 binding issues
    uvicorn.run("app:app", host='127.0.0.1', port=port, reload=True) 