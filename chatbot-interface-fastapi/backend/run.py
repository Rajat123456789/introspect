import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 5000))
    
    print(f"Starting Introspect server on http://localhost:{port}")
    
    # Run the FastAPI application
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True) 