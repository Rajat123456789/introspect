# Historical Health Data Analysis

This directory contains scripts for analyzing and visualizing historical health data exported from Google Fit.

## Data Structure

The scripts expect data to be organized in the following structure:

```
data/Fit/
  ├── Daily activity metrics/     - CSV files with daily activity metrics by 15-min intervals
  ├── Activities/                - TCX files with detailed activity data
  ├── All sessions/              - JSON files with activity session summaries
  └── All data/                  - JSON files with raw health data
```

## Scripts

This directory contains the following scripts:

1. **analyze_historical_health.py** - Analyzes historical health data and generates visualizations for activity metrics, exercise sessions, walking patterns, step patterns, and heart points.

2. **visualize_health_insights.py** - Creates comprehensive visualizations including activity insights, exercise insights, step patterns, heart points, calories burned, and a comprehensive dashboard.

3. **run_health_analysis.py** - Runs both analysis and visualization scripts in sequence.

## How to Run

To run the complete analysis:

```bash
python run_health_analysis.py
```

Or run the individual scripts separately:

```bash
python analyze_historical_health.py
python visualize_health_insights.py
```

## Output

The scripts generate the following output:

- **analysis_output/** - Contains analysis charts from the `analyze_historical_health.py` script.
- **visualizations/** - Contains comprehensive visualizations from the `visualize_health_insights.py` script.

## Analyses Included

The scripts provide the following analyses:

### Basic Analysis
- Daily step count trends
- Exercise session types and durations
- Walking metrics (distance, pace, calories)
- Step patterns by hour and day of week
- Heart points and active minutes

### Advanced Visualizations
- Activity insights by hour and day
- Exercise patterns across time
- Step count distributions and correlations
- Heart points and active minutes patterns
- Calories burned metrics
- Comprehensive health dashboard with correlation matrix

## Requirements

- Python 3.6+
- pandas
- matplotlib
- seaborn
- numpy

## Note

These scripts are designed to work with Google Fit export data, but can be adapted for other health data sources by modifying the data loading methods. 