# Spotify Knowledge Graph with Neo4j

This project creates a knowledge graph of your Spotify listening history using Neo4j, a graph database. The knowledge graph represents relationships between tracks, artists, albums, genres, and your listening events.

## Project Structure

- `spotify_neo4j_graph.py`: Main script for creating the Neo4j knowledge graph
- `songs-parameters/spotify-history-annotated.csv`: Your Spotify listening history data with audio features

## Knowledge Graph Schema

The knowledge graph consists of the following nodes and relationships:

### Nodes
- **Track**: Represents a song with its audio features
- **Artist**: Represents a musician or band
- **Album**: Represents a music album
- **Genre**: Represents a music genre
- **Listening**: Represents a listening event (when you played a track)

### Relationships
- **PERFORMED_BY**: Track → Artist (who performed the track)
- **PART_OF**: Track → Album (which album the track belongs to)
- **CREATED_BY**: Album → Artist (who created the album)
- **BELONGS_TO**: Track → Genre (which genre the track belongs to)
- **OF_TRACK**: Listening → Track (which track was listened to)

## Prerequisites

- Python 3.6+
- Neo4j Database (installed locally or accessible remotely)
- Python packages: `pandas`, `neo4j`

## Installation

1. Install required Python packages:
   ```bash
   pip install pandas neo4j
   ```

2. Start your Neo4j database (if running locally)

3. Update the connection parameters in `spotify_neo4j_graph.py`:
   ```python
   uri = "bolt://localhost:7687"  # Change if your Neo4j is on a different host/port
   username = "neo4j"             # Change to your Neo4j username
   password = "password"          # Change to your Neo4j password
   ```

## Usage

Run the script to create the knowledge graph:

```bash
python spotify_neo4j_graph.py
```

## Example Cypher Queries

Once the knowledge graph is created, you can run these example queries in the Neo4j Browser:

1. **Find your most listened tracks**:
   ```cypher
   MATCH (l:Listening)-[:OF_TRACK]->(t:Track)
   RETURN t.name, count(l) as listen_count
   ORDER BY listen_count DESC
   LIMIT 10
   ```

2. **Find artists with most songs in your history**:
   ```cypher
   MATCH (t:Track)-[:PERFORMED_BY]->(a:Artist)
   WITH a, count(DISTINCT t) as track_count
   RETURN a.name, track_count
   ORDER BY track_count DESC
   LIMIT 10
   ```

3. **Find your favorite genres**:
   ```cypher
   MATCH (l:Listening)-[:OF_TRACK]->(t:Track)-[:BELONGS_TO]->(g:Genre)
   RETURN g.name, count(l) as listen_count
   ORDER BY listen_count DESC
   LIMIT 10
   ```

4. **Find similar songs based on audio features**:
   ```cypher
   MATCH (t1:Track {name: "Your Favorite Song"})
   MATCH (t2:Track)
   WHERE t1 <> t2
   WITH t1, t2,
        abs(t1.danceability - t2.danceability) +
        abs(t1.energy - t2.energy) +
        abs(t1.valence - t2.valence) as difference
   ORDER BY difference ASC
   LIMIT 10
   RETURN t2.name, difference
   ```

5. **Find your listening patterns by time**:
   ```cypher
   MATCH (l:Listening)
   RETURN substring(l.end_time, 11, 2) as hour, count(*) as listen_count
   ORDER BY hour
   ```

## Visualizing the Graph

To visualize your knowledge graph in Neo4j Browser:

1. Open Neo4j Browser (typically at http://localhost:7474)
2. Run a query like:
   ```cypher
   MATCH (n)-[r]-(m)
   RETURN n, r, m
   LIMIT 100
   ```

## Extending the Graph

You can extend this knowledge graph by:

1. Adding user nodes to connect multiple users' listening history
2. Incorporating Spotify's recommendation data
3. Adding temporal relationships to track listening patterns over time
4. Connecting to lyrics databases or sentiment analysis results 