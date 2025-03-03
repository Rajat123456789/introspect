# Spotify Knowledge Graph Insights

A tool to create a Neo4j knowledge graph from Spotify listening history data. This graph represents the relationships between tracks, artists, albums, genres, and your listening patterns.

## Structure

The codebase is organized into the following structure:

```
spotify-knowledge-graph-insights/
├── src/
│   ├── db/             - Database connection code
│   ├── models/         - Data models and loading logic
│   ├── utils/          - Utility functions
├── spotify_knowledge_graph.py - Main script
├── songs-parameters/   - Spotify listening history data
└── .env.example        - Example environment variables
```

## Requirements

- Python 3.6+
- Neo4j database
- Required Python packages:
  - pandas
  - neo4j
  - python-dotenv

## Setup

1. Clone the repository
2. Create a virtual environment and install dependencies:
   ```
   pip install pandas neo4j python-dotenv
   ```
3. Copy `.env.example` to `.env` and configure your Neo4j connection details:
   ```
   cp .env.example .env
   ```
4. Edit the `.env` file with your Neo4j connection information

## Usage

Run the script with:

```
python spotify_knowledge_graph.py
```

Options:
- `--data`: Path to Spotify history CSV file (default: songs-parameters/spotify-history-annotated.csv)
- `--uri`: Neo4j URI (overrides .env setting)
- `--user`: Neo4j username (overrides .env setting)
- `--password`: Neo4j password (overrides .env setting)

## Knowledge Graph Structure

The knowledge graph consists of the following node types:
- **Track**: Songs with audio features (danceability, energy, etc.)
- **Artist**: Musicians who performed the tracks
- **Album**: Collections of tracks
- **Genre**: Music genres
- **User**: Represents the listening user

And the following relationships:
- **User LISTENED_TO Track**: Listening sessions with timestamps
- **Track HAS_GENRE Genre**: Genres of each track
- **Artist PERFORMED Track**: Artists who performed each track
- **Album CONTAINS Track**: Albums containing each track
- **Artist RELEASED Album**: Artists who released each album

## Example Queries

Once the graph is loaded, you can run Cypher queries in Neo4j:

```cypher
// Find most listened to genres
MATCH (u:User)-[l:LISTENED_TO]->(t:Track)-[:HAS_GENRE]->(g:Genre)
RETURN g.name AS genre, COUNT(l) AS listen_count
ORDER BY listen_count DESC
LIMIT 10

// Find your most danceable tracks
MATCH (u:User)-[:LISTENED_TO]->(t:Track)
RETURN t.name AS track, t.danceability AS danceability
ORDER BY danceability DESC
LIMIT 10
``` 