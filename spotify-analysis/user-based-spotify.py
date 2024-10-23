import os
import base64
from requests import post, get
import json
from dotenv import load_dotenv
from urllib.parse import urlencode
import sys

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = {
    "SPOTIFY_CLIENT_ID": os.getenv("SPOTIFY_CLIENT_ID"),
    "SPOTIFY_CLIENT_SECRET": os.getenv("SPOTIFY_CLIENT_SECRET"),
    "SPOTIFY_REDIRECT_URI": os.getenv("SPOTIFY_REDIRECT_URI")
}

# Validate environment variables
missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    print("Error: Missing required environment variables:", ", ".join(missing_vars))
    print("Please ensure these are set in your .env file")
    sys.exit(1)

client_id = required_env_vars["SPOTIFY_CLIENT_ID"]
client_secret = required_env_vars["SPOTIFY_CLIENT_SECRET"]
redirect_uri = required_env_vars["SPOTIFY_REDIRECT_URI"]
scope = "user-read-private"

def get_authorization_url():
    """Generate the Spotify authorization URL."""
    auth_url = "https://accounts.spotify.com/authorize"
    query_params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "show_dialog": True  # Forces prompt for user consent
    }
    return f"{auth_url}?{urlencode(query_params)}"

def get_token(auth_code):
    """Exchange authorization code for access token."""
    try:
        auth_string = f"{client_id}:{client_secret}"
        auth_base64 = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": redirect_uri
        }

        response = post(url, headers=headers, data=data)
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        return response.json()

    except Exception as e:
        print(f"\nError getting token: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        return None

def get_user_profile(access_token):
    """Get user profile information using access token."""
    try:
        url = "https://api.spotify.com/v1/me"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        response = get(url, headers=headers)
        response.raise_for_status()
        
        return response.json()

    except Exception as e:
        print(f"\nError getting user profile: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        return None

def main():
    # Step 1: Get authorization URL
    auth_url = get_authorization_url()
    print("\n1. Visit this URL in your browser to authorize the application:")
    print(auth_url)
    print("\nAfter authorizing, you'll be redirected to your redirect URI.")
    print("Copy the 'code' parameter from the redirected URL.")

    # Step 2: Get authorization code from user
    auth_code = input("\n2. Enter the authorization code: ").strip()
    
    # Step 3: Exchange authorization code for token
    print("\n3. Exchanging authorization code for access token...")
    token_info = get_token(auth_code)
    
    if not token_info:
        print("Failed to get access token. Please check your authorization code and try again.")
        return

    access_token = token_info.get('access_token')
    
    # Step 4: Get user profile
    print("\n4. Getting user profile...")
    user_profile = get_user_profile(access_token)
    
    if not user_profile:
        print("Failed to get user profile.")
        return

    print("\nAuthentication successful!")
    print(f"User ID: {user_profile['id']}")
    print(f"Display Name: {user_profile['display_name']}")
    print(f"Email: {user_profile.get('email', 'Not available')}")
    
    # Save token info for future use if needed
    print(f"\nAccess Token (save this for future use):\n{access_token}")

if __name__ == "__main__":
    main()