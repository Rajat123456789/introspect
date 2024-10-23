from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json

load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes),"utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)

    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def search_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"
    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("No artist found")
        return None

    return json_result[0]

def get_songs_by_artist(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks"
    headers = get_auth_header(token)
    query = "?market=US"
    query_url = url + query
    result = get(query_url, headers=headers)
    songs = json.loads(result.content)["tracks"]

    top_songs = songs[:10]

    song_list = []
    for song in top_songs:
        song_list.append({
            "id": song["id"],
            "name": song["name"],
            "popularity": song["popularity"]
        })

    return song_list

def get_albums(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    headers = get_auth_header(token)
    query = "?include_groups=album&limit=50" 
    query_url = url + query
    result = get(query_url, headers=headers)
    albums = json.loads(result.content)["items"]
    
    album_list = []
    for album in albums:
        album_list.append({
            "name": album["name"],
            "release_date": album["release_date"],
            "total_tracks": album["total_tracks"]
        })

    return album_list

def get_song_features(token, song_id):
    url = f"https://api.spotify.com/v1/audio-features/{song_id}"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    features = json.loads(result.content)
    return {
        "danceability": features["danceability"],
        "energy": features["energy"],
        "tempo": features["tempo"],
        "valence": features["valence"],
        "acousticness": features["acousticness"],
        "instrumentalness": features["instrumentalness"],
        "liveness": features["liveness"],
        "speechiness": features["speechiness"],
        "loudness": features["loudness"]

    }

def calculate_artist_feature_score(top_songs, token):
    total_danceability = 0
    total_energy = 0
    total_tempo = 0
    num_songs = len(top_songs)

    for song in top_songs:
        features = get_song_features(token, song["id"])
        total_danceability += features["danceability"]
        total_energy += features["energy"]
        total_tempo += features["tempo"]
    

    mean_danceability = total_danceability / num_songs
    mean_energy = total_energy / num_songs
    mean_tempo = total_tempo / num_songs

    return {
        "mean_danceability": mean_danceability,
        "mean_energy": mean_energy,
        "mean_tempo": mean_tempo
    }



# Example usage:
token = get_token()

# Get artist details
artist = search_artist(token, "Arijit Singh")
if artist:
    print(f"Artist found: {artist['name']} (ID: {artist['id']})")
    
    albums = get_albums(token, artist["id"])
    print(f"Found {len(albums)} albums:")
    for album in albums:
        print(f"Album: {album['name']}, Release Date: {album['release_date']}, Total Tracks: {album['total_tracks']}")
    
    top_songs = get_songs_by_artist(token, artist["id"])
    print("\nTop 10 Songs:")
    for idx, song in enumerate(top_songs):
        print(f"{idx+1}. {song['name']} (Popularity: {song['popularity']})")

        features = get_song_features(token, song["id"])
        print(f"  Danceability: {features['danceability']}, Energy: {features['energy']}, Tempo: {features['tempo']}, Valence: {features['valence']}, Acousticness: {features['acousticness']}, Instrumentalness: {features['instrumentalness']}, Liveness: {features['liveness']}, Speechiness: {features['speechiness']}, Loudness: {features['loudness']}")


artist = search_artist(token, "Arijit Singh")
if artist:
    print(f"Artist found: {artist['name']} (ID: {artist['id']})")
    
    albums = get_albums(token, artist["id"])
    
    # Get top 10 songs
    top_songs = get_songs_by_artist(token, artist["id"])
    
    # Calculate the aggregate score based on features
    feature_score = calculate_artist_feature_score(top_songs, token)
    
    print("\nAggregate Artist Feature Score:")
    print(f"Danceability: {feature_score['mean_danceability']:.2f}")
    print(f"Energy: {feature_score['mean_energy']:.2f}")
    print(f"Tempo: {feature_score['mean_tempo']:.2f} BPM")
