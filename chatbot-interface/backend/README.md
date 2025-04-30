# Backend Server

This is the backend server for the chatbot interface, built with Flask.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

1. Make sure your virtual environment is activated
2. Run the Flask application:
```bash
python app.py
```

The server will start on http://localhost:5000 by default.

## API Endpoints

- `GET /api/health`: Health check endpoint to verify the server is running

## Environment Variables

Create a `.env` file in the root directory with the following variables:
- `PORT`: Port number for the server (default: 5000)
- `FLASK_ENV`: Environment mode (development/production) 