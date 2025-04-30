# Health LLM Chat Interface

A modern chat interface for interacting with different AI models, including OpenAI (GPT) and Google's Gemini, with a focus on health-related information.

## Features

- Multi-model chat interface with three different modes:
  - Base Model: Standard AI responses
  - Health LLM: Healthcare-focused responses
  - Introspective Assistant: Personalized health data analysis
- OpenAI API integration (GPT-3.5/4)
- Google Gemini API integration
- Toggle between different AI providers
- Markdown support for rich responses
- Session management
- Modern, responsive UI

## Setup

### Prerequisites

- Node.js 16+ for frontend
- Python 3.8+ for backend
- API keys for OpenAI and/or Google Gemini

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd chatbot-interface/backend
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   ```

5. Edit the `.env` file with your actual API keys:
   ```
   OPENAI_API_KEY=your_actual_openai_key
   GEMINI_API_KEY=your_actual_gemini_key
   ```

6. Start the backend server:
   ```
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd chatbot-interface/frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. Open the application in your browser at `http://localhost:5173`

## Usage

1. Select the API provider (OpenAI or Gemini) from the dropdown at the bottom
2. Type your message in the input box at the bottom of the screen
3. Press Enter or click the Send button to send your message
4. View responses from all three model types side by side

### Mode Descriptions

- **Base Model**: General purpose AI assistant
- **Health LLM**: Medical assistant with healthcare domain knowledge
- **Introspective Assistant**: Analyzes your health and digital data to provide personalized insights

## Environment Variables

### Backend (.env)

- `OPENAI_API_KEY`: Your OpenAI API key
- `GEMINI_API_KEY`: Your Google Gemini API key
- `OPENAI_MODEL`: (Optional) OpenAI model to use (defaults to "gpt-3.5-turbo")
- `GEMINI_MODEL`: (Optional) Gemini model to use (defaults to "gemini-1.0-pro")
- `PORT`: (Optional) Port for the backend server (defaults to 5000)

## License

MIT 