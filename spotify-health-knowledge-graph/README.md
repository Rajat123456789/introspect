# Spotify Health Knowledge Graph

This project creates a knowledge graph from Spotify listening data combined with health metrics (heart rate and step count). It uses Neo4j as the graph database and provides various insights and visualizations about the relationship between music listening and health metrics.

## Features

- Loads heart rate and step count data from CSV files
- Creates a knowledge graph with tracks, artists, and health metrics
- Provides various insights and visualizations:
  - Heart rate distribution
  - Step count distribution
  - Health metrics by hour of day
  - Health metrics by artist
  - Tracks by heart rate range
  - Tracks by step count range

## Prerequisites

- Python 3.8+
- Neo4j Database (local or remote)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your Neo4j credentials:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

## Usage

1. Load data into Neo4j:
   ```bash
   python spotify_health_neo4j.py
   ```

2. Generate insights and visualizations:
   ```bash
   python health_insights.py
   ```

## Data Structure

The knowledge graph consists of the following nodes and relationships:

### Nodes
- Track
  - Properties: track_id, name, artist, album, url
- HealthMetric
  - Properties: id, type, value, unit, timestamp, source

### Relationships
- (Track)-[:HAS_HEALTH_METRIC]->(HealthMetric)

## Visualizations

The following visualizations are generated in the `visualizations` directory:
- heart_rate_distribution.png
- step_count_distribution.png
- health_metrics_by_hour.png
- health_metrics_by_artist.png

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 