# Spotify Knowledge Graph Insights

A comprehensive tool to analyze and visualize your Spotify listening history data. This project consists of two main components:

1. **Knowledge Graph Creation**: Build a Neo4j knowledge graph representing the relationships between tracks, artists, albums, genres, and your listening patterns.
2. **Data Visualization**: Generate static and interactive visualizations of your Spotify listening habits and music preferences.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating the Knowledge Graph](#creating-the-knowledge-graph)
  - [Generating Visualizations](#generating-visualizations)
- [Visualization Types](#visualization-types)
- [Knowledge Graph Structure](#knowledge-graph-structure)
- [Example Queries](#example-queries)
- [Troubleshooting](#troubleshooting)

## Project Structure

The codebase is organized into the following structure:

```
spotify-knowledge-graph-insights/
├── src/
│   ├── db/                  - Database connection code
│   ├── models/              - Data models and loading logic
│   ├── utils/               - Utility functions
│   ├── visualization/       - Visualization components
│       ├── data_processors/ - Data processing for visualizations
│       ├── plotters/        - Visualization generation code
│       ├── utils/           - Visualization utility functions
├── songs-parameters/        - Spotify listening history data
├── spotify_knowledge_graph.py - Main script for knowledge graph creation
├── visualize_spotify_data.py  - Main script for data visualization
├── requirements.txt         - Python dependencies
└── .env.example             - Example environment variables
```

## Requirements

- Python 3.6+
- Neo4j database (for knowledge graph component)
- Required Python packages (see requirements.txt):
  - pandas
  - neo4j
  - python-dotenv
  - matplotlib
  - seaborn
  - plotly
  - scikit-learn
  - wordcloud

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd spotify-knowledge-graph-insights
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n spotify-insights python=3.8
   conda activate spotify-insights
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Neo4j connection (for knowledge graph component):
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your Neo4j connection information:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

## Usage

### Creating the Knowledge Graph

The knowledge graph component creates a Neo4j graph database from your Spotify listening history.

1. Ensure your Neo4j database is running and configured in the `.env` file.

2. Run the knowledge graph script:
   ```bash
   python spotify_knowledge_graph.py
   ```

   Options:
   - `--data`: Path to Spotify history CSV file (default: songs-parameters/spotify-history-annotated.csv)
   - `--uri`: Neo4j URI (overrides .env setting)
   - `--user`: Neo4j username (overrides .env setting)
   - `--password`: Neo4j password (overrides .env setting)

   Example with custom data file:
   ```bash
   python spotify_knowledge_graph.py --data path/to/your/spotify-data.csv
   ```

### Generating Visualizations

The visualization component generates static and interactive visualizations from your Spotify listening history.

1. Run the visualization script:
   ```bash
   python visualize_spotify_data.py songs-parameters/spotify-history-annotated.csv
   ```

   Options:
   - `input_file` (required): Path to the Spotify history CSV file
   - `--output-dir` or `-o`: Directory to save visualizations (default: spotify_visualizations_TIMESTAMP)
   - `--no-basic`: Skip basic visualizations
   - `--no-advanced`: Skip advanced visualizations
   - `--no-interactive`: Skip interactive visualizations
   - `--report`: Generate an HTML report with all visualizations

   Examples:
   ```bash
   # Generate all visualizations with default settings
   python visualize_spotify_data.py songs-parameters/spotify-history-annotated.csv
   
   # Generate only basic visualizations
   python visualize_spotify_data.py songs-parameters/spotify-history-annotated.csv --no-advanced --no-interactive
   
   # Generate all visualizations and an HTML report
   python visualize_spotify_data.py songs-parameters/spotify-history-annotated.csv --report
   
   # Specify a custom output directory
   python visualize_spotify_data.py songs-parameters/spotify-history-annotated.csv -o my_visualizations
   ```

2. View the generated visualizations:
   - Static visualizations (PNG files): Open with any image viewer
   - Interactive visualizations (HTML files): Open in a web browser

## Visualization Types

The visualization pipeline generates three types of visualizations:

### Basic Visualizations
- Listening by hour
- Listening by day
- Top artists
- Top genres
- Audio features distribution
- Feature correlations
- Genre wordcloud
- Listening timeline weekly

### Advanced Visualizations
- Feature radar chart
- PCA analysis
- Clustering analysis
- Valence by time
- Energy by time
- Danceability by time
- Feature comparison by genre

### Interactive Visualizations
- Listening patterns heatmap
- Audio features radar
- Genre distribution sunburst
- Listening timeline interactive
- Top artists interactive
- Audio features scatter

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

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - If using conda, make sure to install packages in the correct environment: `conda install -y wordcloud`

2. **Neo4j connection issues**:
   - Verify Neo4j is running
   - Check connection details in the `.env` file
   - Ensure Neo4j authentication is correctly configured

3. **Visualization errors**:
   - Check that your input CSV file has the expected format
   - Ensure you have sufficient disk space for generated visualizations

### Getting Help

If you encounter issues not covered in this README, please:
1. Check the logs for detailed error messages
2. Consult the documentation for the specific libraries (Neo4j, Pandas, Matplotlib, etc.)
3. Open an issue in the project repository with a detailed description of the problem 