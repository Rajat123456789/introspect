# Agentic Data Analysis Pipeline

This module implements a simple agentic pipeline for analyzing multimodal data using OpenAI's API.

## Overview

The pipeline processes four types of data:
1. **Health History** (images) - Health data visualizations from historical data
2. **Health Live** (images) - Real-time health data visualizations
3. **YouTube History** (images) - Historical YouTube consumption data visualizations
4. **YouTube Live** (JSON) - Real-time YouTube video data

For each data type, the pipeline:
1. Analyzes each individual file (image or JSON entry)
2. Stores individual analyses as text files
3. Generates a consolidated report combining insights from all individual analyses

## Requirements

- Python 3.8+
- OpenAI API key (must be set as environment variable `OPENAI_API_KEY`)
- Required packages: `openai`, `dotenv`

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required packages:
```
pip install openai python-dotenv
```
3. Set up your OpenAI API key as an environment variable:
```
export OPENAI_API_KEY=your_api_key_here  # Linux/Mac
set OPENAI_API_KEY=your_api_key_here     # Windows
```
Or add it to your `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Directory Structure

The pipeline expects data to be organized as follows:

```
src/
  report_generation/
    data/
      Health_history/         # Health history visualization images
      Health_live/            # Health live visualization images
      Youtube_history/        # YouTube history visualization images
      Youtube_live/           # YouTube live data JSON files
    reports/                  # Output reports will be stored here
      analysis/               # Individual file analyses
        health_history/
        health_live/
        youtube_history/
        youtube_live/
```

## Usage

### Running the Pipeline

To run the complete pipeline for all data types:

```
python -m src.report_generation.agents.pipeline --all --youtube-max-videos 30
```

To run the pipeline for specific data types:

```
# Process health history data
python -m src.report_generation.agents.pipeline --health-history

# Process YouTube Live data (must specify max videos)
python -m src.report_generation.agents.pipeline --youtube-live --youtube-max-videos 30

# Process multiple data types
python -m src.report_generation.agents.pipeline --health-history --youtube-live --youtube-max-videos 20
```

### Command-line Options

- `--health-history`: Process Health History data
- `--health-live`: Process Health Live data
- `--youtube-history`: Process YouTube History data
- `--youtube-live`: Process YouTube Live data (requires `--youtube-max-videos`)
- `--all`: Process all data types (default)
- `--youtube-max-videos`: **Required** when processing YouTube Live data, specifies maximum number of videos to process
- `--batch-size`: Number of analyses to batch when generating consolidated reports (default: 15)
- `--no-skip`: Force processing of all files, even if analyses already exist

### Examples

```
# Process only YouTube Live data, limit to 10 videos
python -m src.report_generation.agents.pipeline --youtube-live --youtube-max-videos 10

# Process all data types with a custom batch size and YouTube video limit
python -m src.report_generation.agents.pipeline --all --batch-size 20 --youtube-max-videos 50

# Process health data and force reprocessing of existing analyses
python -m src.report_generation.agents.pipeline --health-history --health-live --no-skip
```

### Output

The pipeline generates:

1. Individual analysis files for each input file in `reports/analysis/{data_type}/`
2. Consolidated reports for each data type in `reports/`
3. A pipeline summary in `reports/pipeline_summary.txt`

## Using Individual Agents

The pipeline consists of two main agent classes that can be used independently:

### ImageAnalysisAgent

For analyzing image visualizations:

```python
from src.report_generation.agents.image_analysis_agent import ImageAnalysisAgent

# Initialize the agent
agent = ImageAnalysisAgent()

# Analyze a single image
result = agent.analyze_image("path/to/image.png", analysis_type="health")

# Process all images in a directory
results = agent.process_directory("path/to/images", "path/to/output", analysis_type="health")

# Generate consolidated report from individual analyses
report = agent.generate_consolidated_report("path/to/analyses", "output_report.txt", analysis_type="health")
```

### JSONAnalysisAgent

For analyzing JSON data (YouTube videos):

```python
from src.report_generation.agents.json_analysis_agent import JSONAnalysisAgent

# Initialize the agent
agent = JSONAnalysisAgent()

# Process a JSON file with a limit of 30 videos
results = agent.process_json_file("path/to/data.json", "path/to/output", max_videos=30)

# Generate consolidated report from individual analyses
report = agent.generate_consolidated_report("path/to/analyses", "output_report.txt")
```

## Rate Limit Handling

The pipeline includes several features to handle OpenAI API rate limits:

1. Batching of analyses when generating consolidated reports (configurable via `--batch-size`)
2. Delays between processing different data types (5 seconds by default)
3. Skip processing for files that already have analysis results (disable with `--no-skip`)
4. Increased delay between API calls when processing YouTube videos (1.5 seconds)

## Extending the Pipeline

To add support for new data types:

1. Create a new processing function in `pipeline.py`
2. Add the new data type to the `all_data_types` dictionary in the `run_pipeline` function
3. Implement appropriate analysis logic for the new data type 