# introspect.ai

Enhance your self-awareness through guided introspection based on daily activity data analysis.

## Project Overview

introspect.ai is a comprehensive data analysis toolkit that helps users gain insights into their digital behavior patterns and how they might relate to mental well-being. By analyzing data from various sources like Spotify, YouTube, and Apple Health, the platform provides personalized insights and visualizations to foster self-awareness.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Collection](#data-collection)
  - [YouTube Data](#youtube-data)
  - [Spotify Data](#spotify-data)
  - [Apple Health Data](#apple-health-data)
- [Components](#components)
  - [YouTube Analysis](#youtube-analysis)
  - [YouTube Knowledge Graph](#youtube-knowledge-graph)
  - [Spotify Analysis](#spotify-analysis)
  - [Spotify Knowledge Graph](#spotify-knowledge-graph)
  - [Apple Health Analysis](#apple-health-analysis)
  - [Combined Analysis](#combined-analysis)
- [Usage Examples](#usage-examples)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- **Data Analysis**: Process and analyze user data from multiple platforms
- **Knowledge Graph Creation**: Build semantic networks representing relationships in user activity
- **Visualization**: Generate compelling visualizations of data patterns
- **Correlation Analysis**: Identify potential relationships between digital behavior and well-being
- **Personalized Insights**: Receive tailored recommendations based on data analysis

## Project Structure

```
introspect.ai/
├── youtube-analysis/                # YouTube data processing & analysis
├── youtube-knowledge-graph-insights/# YouTube knowledge graph & visualization
├── spotify-analysis/                # Spotify data processing & analysis  
├── spotify-knowledge-graph-insights/# Spotify knowledge graph & visualization
├── apple-health/                    # Apple Health data processing & analysis
├── combining-apple-spotify/         # Combined Apple Health & Spotify analysis
├── combining-health-and-music/      # Health & music correlation analysis
├── analysis_reports/                # Generated reports from analysis runs
├── requirements.txt                 # Core dependencies for the project
├── LICENSE                          # Project license information
└── README.md                        # This documentation file
```

## Installation

### Prerequisites

- Python 3.7+
- Neo4j Database (4.0+) for knowledge graph components
- Pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/introspect.ai.git
   cd introspect.ai
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Neo4j (for knowledge graph components):
   - Download and install [Neo4j Desktop](https://neo4j.com/download/)
   - Create a new database with appropriate credentials

## Data Collection

Before using introspect.ai, you need to collect your personal data from the platforms you want to analyze. This section explains how to export your data from each supported platform.

### YouTube Data

To collect your YouTube viewing history:

1. Go to [Google Takeout](https://takeout.google.com/)
2. Sign in with your Google account
3. Deselect all products, then select only "YouTube and YouTube Music"
4. Under YouTube, select only "history" and ensure "YouTube Watch History" is checked
5. Click "Next step" and choose your delivery method (email link is recommended)
6. Choose file type (ZIP), size and frequency (one-time export)
7. Click "Create export"
8. Download the ZIP file when ready (usually within a few minutes to hours)
9. Extract the ZIP file and locate the `watch-history.json` file
10. Place this file in the `youtube-analysis/data/` or `youtube-knowledge-graph-insights/Youtube-Analysis-Files/` directory

The JSON file contains your entire YouTube viewing history with timestamps, video titles, and channel information.

### Spotify Data

To collect your Spotify listening history:

1. Go to your [Spotify Account](https://www.spotify.com/account/privacy/)
2. Sign in if required
3. Scroll down to "Download your data"
4. Click "Request" to request a copy of your data
5. You'll receive an email when your data is ready (can take up to 30 days, but typically a few days)
6. Download the data package and extract it
7. Locate the `StreamingHistory*.json` files (there may be multiple)
8. Process these files into a single CSV file with appropriate columns
9. Place the processed CSV file in the `spotify-knowledge-graph-insights/songs-parameters/` directory as `spotify-history-annotated.csv`

For the best results, the CSV file should include these columns:
- `artist` - Artist name
- `track` - Track name
- `album` - Album name
- `ts` - Timestamp (ISO format)
- Audio features such as `danceability`, `energy`, `valence`, etc. (can be added using the Spotify API)

### Apple Health Data

To export your Apple Health data:

1. On your iPhone, open the Health app
2. Tap on your profile picture in the top-right corner
3. Scroll down and tap "Export All Health Data"
4. Confirm the export and wait for the archive to be created
5. Choose how to share the export (AirDrop, email, etc.)
6. Transfer the exported ZIP file to your computer
7. Extract the ZIP file to get the `export.xml` file
8. Place this file in the `apple-health/data/` directory

The XML file contains your health metrics including activity, heart rate, sleep, and more.

## Components

### YouTube Analysis

Basic analysis of YouTube viewing history data.

**Key Features:**
- Parse YouTube viewing history exports
- Analyze viewing patterns and trends
- Categorize content consumption

### YouTube Knowledge Graph

Advanced analysis of YouTube data using knowledge graphs.

#### Running the YouTube Knowledge Graph

1. Navigate to the YouTube knowledge graph directory:
   ```bash
   cd youtube-knowledge-graph-insights
   ```

2. Ensure Neo4j is running with appropriate credentials (default: neo4j/12345678).

3. Run the knowledge graph builder:
   ```bash
   python youtube_knowledge_graph.py
   ```

4. Run the main analysis script:
   ```bash
   python youtube_mental_health_analysis.py
   ```

5. Generate visualizations:
   ```bash
   python visualize_results.py
   ```

**Visualizations Generated:**
- Mental health trend lines
- Viewing pattern calendars
- Content category impact analysis
- Sentiment trajectory charts
- Addiction risk assessments
- Music impact analysis

### Spotify Analysis

Basic analysis of Spotify listening history data.

**Key Features:**
- Parse Spotify streaming history exports
- Analyze listening patterns
- Examine music preferences over time

### Spotify Knowledge Graph

Advanced analysis of Spotify data using knowledge graphs and visualization tools.

#### Running the Spotify Knowledge Graph

1. Navigate to the Spotify knowledge graph directory:
   ```bash
   cd spotify-knowledge-graph-insights
   ```

2. Set up environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```
   
3. Edit the `.env` file with your Neo4j connection details:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

4. Build the knowledge graph:
   ```bash
   python spotify_knowledge_graph.py --data songs-parameters/spotify-history-annotated.csv
   ```

5. Generate visualizations:
   ```bash
   python visualize_spotify_data.py songs-parameters/spotify-history-annotated.csv
   ```

**Visualization Options:**
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

**Visualizations Generated:**
- Basic: listening patterns, top artists/genres, audio features
- Advanced: feature radar charts, PCA analysis, clustering analysis
- Interactive: heatmaps, sunburst charts, timeline visualizations

### Apple Health Analysis

Analysis of Apple Health export data.

**Key Features:**
- Parse Apple Health data exports
- Track physical activity metrics
- Monitor sleep and heart rate data
- Identify patterns in health metrics over time

### Combined Analysis

Correlate data across platforms to identify relationships between:
- Music consumption and physical activity
- Video consumption and sleep patterns
- Entertainment habits and overall well-being metrics

## Usage Examples

### Example 1: Analyze YouTube Viewing Patterns

```bash
cd youtube-knowledge-graph-insights
python youtube_mental_health_analysis.py
```

### Example 2: Create Spotify Knowledge Graph and Visualizations

```bash
cd spotify-knowledge-graph-insights
python spotify_knowledge_graph.py
python visualize_spotify_data.py songs-parameters/spotify-history-annotated.csv --report
```

### Example 3: Run Combined Health and Music Analysis

```bash
cd combining-health-and-music
python combined_analysis.py
```

## Dependencies

Core dependencies include:
- pandas (≥2.0.0)
- numpy (≥1.24.0)
- transformers (≥4.45.2)
- torch (≥2.0.0)
- spacy (≥3.7.0)
- neo4j (for knowledge graph components)
- matplotlib, seaborn, plotly (for visualizations)

Each component may have additional specific dependencies listed in their respective requirements.txt files.

## License

This project is licensed under the terms of the license included in the LICENSE file.
