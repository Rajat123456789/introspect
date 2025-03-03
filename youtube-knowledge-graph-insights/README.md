# YouTube Mental Health Analysis

A data analytics tool for examining the relationship between YouTube consumption patterns and mental health indicators using Neo4j graph database and Python.

## Overview

This project analyzes YouTube viewing history data to identify potential correlations between content consumption and mental health. By processing viewing patterns, content categories, and other metadata, the tool provides insights into:

- Temporal mental health trends
- Sentiment trajectories over time
- Addiction risk patterns
- Category-specific mental health impacts
- Music content influence on mood
- Problematic viewing behaviors (binge watching, late-night viewing)

## Installation

### Prerequisites

- Python 3.7+
- Neo4j Database (4.0+)
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/youtube-mental-health.git
   cd youtube-mental-health
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Neo4j is running with appropriate credentials (default: neo4j/12345678).

## Neo4j Database Setup

This tool requires a Neo4j database with YouTube viewing data. The database should contain:

- Video nodes with properties like title, category, watched_at timestamp
- Mental health category nodes
- Relationships between videos and mental health aspects

You can use `youtube_knowledge_graph.py` to build the initial graph from YouTube data.

## Usage

### Running the Analysis

Since you have the original monolithic script, you should run:

```bash
python youtube_mental_health_analysis.py
```

This will execute the full analysis with default Neo4j connection parameters.

### Alternative Entry Points

The repository includes multiple ways to run the analysis:

1. Using the main analysis script:
   ```bash
   python youtube_mental_health_analysis.py
   ```

2. Using the alternative main script:
   ```bash
   python youtube_mental_health_main.py
   ```

3. With the runner script:
   ```bash
   python run_analysis.py
   ```

## Analysis Capabilities

### Temporal Analysis
Examines how mental health indicators change over time, tracking daily, weekly, and monthly trends.

### Viewing Pattern Analysis
Identifies potentially problematic viewing patterns such as:
- Binge watching (15+ videos per day)
- Late-night viewing (10 PM - 4 AM)
- Content-specific consumption trends

### Addiction Risk Analysis
Evaluates content categories for addiction potential based on:
- Viewing frequency
- Negative mental health impact scores
- Clickbait title patterns
- Series-based content

### Music Impact Analysis
Analyzes music content for mental health effects by:
- Extracting genre from video titles
- Categorizing emotional energy (calming, energizing, etc.)
- Correlating with mental health indicators

## Output Files

Analysis results are saved to the `analysis_reports` directory:

- CSV files: Tabular data for each analysis type
- JSON files: Structured data for programmatic use
- PNG images: Visualizations and dashboards

Example outputs:
- `viewing_patterns_[timestamp].csv`: Daily viewing statistics
- `sentiment_trajectory_[timestamp].json`: Sentiment changes over time
- `title_addiction_patterns_[timestamp].csv`: Clickbait and addiction metrics
- `binge_triggers_[timestamp].csv`: Content that triggers binge sessions

## Utility Scripts

### organize_reports.py

This utility script helps organize analysis output files into timestamp-based folders for better organization.

#### Purpose

When running analyses before the automatic timestamp folder creation was implemented, output files were saved directly in the `analysis_reports` directory with timestamps in their filenames (e.g., `viewing_patterns_20230515_120145.csv`). This script organizes these files into dedicated timestamp folders.

#### How It Works

1. Scans all files in the `analysis_reports` directory
2. Identifies files with timestamp patterns in their names (format: `basename_YYYYMMDD_HHMMSS.extension`)
3. Creates folders named with the timestamps
4. Moves files into their corresponding timestamp folders, removing the timestamp from the filename

For example:
- `viewing_patterns_20230515_120145.csv` → `analysis_reports/20230515_120145/viewing_patterns.csv`
- `mental_health_index_20230515_120145.json` → `analysis_reports/20230515_120145/mental_health_index.json`

#### Usage

Run the script from the project root directory:

```bash
python organize_reports.py
```

This is particularly useful when:
- You've run multiple analyses before implementing the timestamp folder structure
- You want to organize legacy analysis outputs
- You need to clean up the main reports directory

The script is safe to run and only moves files - it doesn't delete any content.

## Project Structure

```
youtube-mental-health/
├── youtube_mental_health_analysis.py  # Main analysis script
├── youtube_mental_health_main.py      # Alternative entry point
├── run_analysis.py                    # Runner script
├── youtube_knowledge_graph.py         # Graph database builder
└── analysis_reports/                  # Output directory
    ├── viewing_patterns_*.csv
    ├── sentiment_trajectory_*.json/csv
    ├── title_addiction_patterns_*.csv
    ├── binge_triggers_*.json/csv
    └── [various visualization PNGs]

## Future Development

A modular approach as outlined in the project structure would make the codebase more maintainable. Consider refactoring the code into separate modules:

```
analyzer/
├── __init__.py
├── mental_health_analyzer.py  # Core analyzer class
├── utils.py                   # Utility functions
├── temporal_analysis.py       # Time-based analysis functions
├── addiction_analysis.py      # Addiction pattern analysis
├── music_analysis.py          # Music content analysis
└── dashboard.py               # Dashboard and visualization
```

This would improve code organization, readability, and make it easier to add new features.

## Visualization Examples

The tool generates various visualizations:

- Mental health trend lines
- Addiction risk by category
- Viewing pattern calendars
- Music impact charts
- Sentiment distribution
- Comprehensive dashboards

## Contributing

Contributions are welcome! Ways to extend the project:

1. Modularize the existing code for better maintainability
2. Add new analysis types
3. Enhance visualizations
4. Improve Neo4j queries for better performance
5. Add machine learning models to predict mental health impacts

## License

[MIT License]

## Acknowledgments

This project leverages Neo4j graph database, pandas, matplotlib, and other open-source libraries to enable mental health analysis from YouTube data. 