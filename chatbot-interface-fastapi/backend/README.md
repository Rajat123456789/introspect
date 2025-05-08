# Introspect Chatbot Backend

This directory contains the backend server for the Introspect chatbot interface. It provides API endpoints for chat functionality and serves the web frontend.

## Features

- FastAPI backend with integrated frontend
- Multiple model types: base, health, and introspect
- Support for OpenAI and Gemini API providers
- Health check endpoints for monitoring
- Introspection data integration

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in this directory with the following variables:
     ```
     OPENAI_API_KEY=your_openai_api_key
     GEMINI_API_KEY=your_gemini_api_key
     PORT=5000  # Optional, default is 5000
     ```

## Running the application

Use the run script:

```
python run.py
```

Or on Windows, you can use the batch file:

```
run.bat
```

The application will be available at http://localhost:5000

## API Endpoints

- `GET /`: Main frontend UI
- `GET /api/health`: Health check endpoint
- `POST /api/chat`: Process chat messages
- `POST /api/clear_history`: Clear conversation history
- `GET /api/introspect/data`: Get introspection data
- `GET /api/introspect/insights`: Get introspection insights
- `GET /api/routes`: List all available routes (debug endpoint)

## Project Structure

- `app.py`: Main FastAPI application
- `run.py`: Script to run the application
- `requirements.txt`: Python dependencies
- `api/`: API modules
  - `introspect.py`: Introspection data endpoints
  - `llm_providers.py`: LLM integration (OpenAI, Gemini)
- `static/`: Static files for the frontend
  - `css/`: CSS stylesheets
  - `js/`: JavaScript files
- `templates/`: Jinja2 templates for the frontend
- `content_for_prompt/`: Content files for LLM prompts

## Development

For development mode with auto-reload:

```
uvicorn app:app --reload --port 5000
```

## API Documentation

FastAPI automatically generates interactive API documentation:
- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc 