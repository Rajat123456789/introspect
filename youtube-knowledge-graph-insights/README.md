# YouTube Viewing Pattern Analysis

This project analyzes YouTube viewing patterns to identify behavioral patterns that may impact mental health and digital wellbeing.

## Features

- **Pattern Detection**: Identifies multiple viewing patterns including:
  - Doom scrolling (rapid video consumption in short periods)
  - Rabbit holes (sequences of related videos)
  - Addiction patterns (consistent high consumption)
  - Escapism (entertainment during work hours)
  - Negative mood indicators (content with negative themes)
  - Unhealthy comparison (beauty/lifestyle idealization content)

- **Comprehensive Analysis**:
  - Daily and monthly trend analysis
  - Day of week patterns
  - Hour of day patterns
  - Correlation between different patterns
  - Time-based pattern detection

- **Visualizations**: Creates multiple visualizations:
  - Pattern time series
  - Daily and monthly trends
  - Pattern summaries and distributions
  - Day of week and hour of day heatmaps
  - Pattern correlations

- **Reporting**: Generates detailed reports with:
  - Key findings and statistics
  - Insights on viewing patterns
  - Personalized recommendations

## Requirements

- Python 3.7+
- Neo4j database with YouTube viewing data
- Required Python libraries:
  - pandas
  - matplotlib
  - seaborn
  - numpy
  - neo4j

Install required libraries:

```
pip install pandas matplotlib seaborn numpy neo4j
```

## Running the Analysis

### Option 1: Using the runner script

The simplest way to run the analysis is to use the provided runner script, which will analyze the last 30 days of data:

```
python run_analysis.py
```

Add the `-o` flag to automatically open the output directory after analysis:

```
python run_analysis.py -o
```

### Option 2: Run with custom parameters

For more control, run the main script directly with custom parameters:

```
python date_range_analysis.py --start_date="2023-06-01T00:00:00+00:00" --end_date="2023-06-30T23:59:59+00:00" --output_dir="june_analysis"
```

### Option 3: Import and use in your own code

```python
from date_range_analysis import run_date_range_analysis

# Run analysis for a specific date range
results = run_date_range_analysis(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="12345678",
    start_date="2023-06-01T00:00:00+00:00",
    end_date="2023-06-30T23:59:59+00:00",
    output_dir="june_analysis"
)
```

## Neo4j Configuration

The analysis requires a Neo4j database with YouTube viewing data. Set your Neo4j connection details as environment variables:

```
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

On Windows:

```
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=your_password
```

## Output

The analysis generates several output files in the specified directory:

- `report.txt`: Plain text report with key findings and recommendations
- `daily_pattern_trends.csv`: CSV file with daily pattern data
- `monthly_pattern_trends.csv`: CSV file with monthly pattern data
- Multiple PNG files with visualizations:
  - `daily_pattern_counts.png`
  - `daily_pattern_percentages.png`
  - `monthly_pattern_counts.png`
  - `pattern_time_series.png`
  - `day_of_week_patterns.png`
  - `hour_of_day_patterns.png`
  - `pattern_correlation_heatmap.png`
  - And many more...

## Customizing the Analysis

The pattern detection algorithms have default parameters that you can adjust for your specific use case:

- **Doom Scrolling**: 
  - `threshold`: Number of videos in time window (default: 25)
  - `time_window_hours`: Hours to check for high consumption (default: 1)

- **Rabbit Holes**:
  - `min_sequence`: Minimum videos in sequence (default: 6)
  - `max_time_gap`: Maximum time between videos (default: 30 minutes)
  - `min_keyword_overlap`: Minimum keyword matches (default: 3)

- **Addiction Pattern**:
  - `daily_threshold`: Minimum videos per day (default: 15)
  - `daily_consecutive_days`: Required consecutive days (default: 5)

Modify these parameters in the functions or create wrapper functions for custom analysis. 