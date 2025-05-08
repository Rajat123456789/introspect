# Report Generation Module

This module is responsible for analyzing multimodal data and generating insightful reports.

## Components

The module consists of the following components:

### Agentic Pipeline

The agentic pipeline is a simple implementation that:
1. Analyzes images using OpenAI's Vision API (GPT-4o)
2. Analyzes JSON data using OpenAI's text completion API
3. Generates consolidated reports from the analyses

To use the agentic pipeline, see the [Agents README](agents/README.md).

### Data Sources

The pipeline is designed to process four types of data:
- `Health_history`: Health data visualization images (historical)
- `Health_live`: Health data visualization images (real-time)
- `Youtube_history`: YouTube consumption data visualization images
- `Youtube_live`: YouTube video metadata in JSON format

All data sources are located in the `data/` directory.

### Reports

The pipeline generates individual analysis files and consolidated reports in the `reports/` directory.

## Using the Pipeline

To run the analysis pipeline for all data types:

```bash
python -m src.report_generation.agents.pipeline --all --youtube-max-videos 30
```

To run the pipeline for specific data types:

```bash
# Process health history data
python -m src.report_generation.agents.pipeline --health-history

# Process YouTube Live data (must specify max videos)
python -m src.report_generation.agents.pipeline --youtube-live --youtube-max-videos 30

# Process multiple data types
python -m src.report_generation.agents.pipeline --health-history --youtube-live --youtube-max-videos 20
```

### Important Parameters

- `--youtube-max-videos`: **Required** when processing YouTube Live data. Specifies the maximum number of videos to process per JSON file (default: 30 if using `--all`)
- `--batch-size`: Number of analyses to batch when generating consolidated reports (default: 15)
- `--no-skip`: Force processing of all files, even if analyses already exist

### Examples

```bash
# Process YouTube Live data with a limit of 10 videos
python -m src.report_generation.agents.pipeline --youtube-live --youtube-max-videos 10

# Process all data types but limit YouTube Live to 50 videos
python -m src.report_generation.agents.pipeline --all --youtube-max-videos 50

# Process health data and force reprocessing of existing analyses
python -m src.report_generation.agents.pipeline --health-history --health-live --no-skip
```

## Requirements

- Python 3.8+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Required packages: `openai`, `python-dotenv`

## Installation

Install the required packages:

```bash
pip install openai python-dotenv
```

Set up your OpenAI API key:

```bash
# Linux/Mac
export OPENAI_API_KEY=your_api_key_here

# Windows
set OPENAI_API_KEY=your_api_key_here
```

Or add it to your `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## Handling Rate Limits

The pipeline includes several features to handle OpenAI API rate limits:

1. Automatic batching of analyses when generating consolidated reports
2. Delays between processing different data types
3. Skip processing for files that already have analysis results
4. Increased delay between API calls when processing YouTube videos

## Testing

To test the agents:

```bash
# Test image analysis agent with a specific image
python -m src.report_generation.agents.test_agents --test-image-agent --image path/to/image.png

# Test JSON analysis agent with a YouTube video data file
python -m src.report_generation.agents.test_agents --test-json-agent --json path/to/youtube_data.json
``` 