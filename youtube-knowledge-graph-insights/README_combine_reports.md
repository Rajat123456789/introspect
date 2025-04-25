# YouTube Pattern Analysis Report Combiner

This script combines multiple YouTube pattern analysis reports into a single HTML report for easier viewing and comparison.

## Features

- Combines multiple pattern analysis reports into a single HTML file
- Organizes reports by date (newest first)
- Displays summaries, visualizations, and detailed data for each report
- Provides a tabbed interface for easy navigation
- Copies visualization files to the output directory for proper display

## Requirements

- Python 3.6+
- Required Python packages:
  - pandas
  - matplotlib
  - seaborn
  - jinja2

## Installation

1. Make sure you have the required Python packages installed:

```bash
pip install pandas matplotlib seaborn jinja2
```

2. Place the `combine_reports.py` and `report_template.html` files in the same directory as your pattern analysis reports.

## Usage

1. Run the script from the command line:

```bash
python combine_reports.py
```

2. The script will:
   - Find all directories matching the pattern `pattern_analysis*`
   - Extract data from each report's JSON file
   - Find all visualization files (PNG, JPG, JPEG)
   - Generate a combined HTML report in the `combined_reports` directory

3. Open the generated HTML file (`combined_reports/combined_pattern_analysis.html`) in a web browser to view the combined report.

## Output

The script creates a `combined_reports` directory containing:

- `combined_pattern_analysis.html`: The main HTML report file
- `visualizations/`: A directory containing copies of all visualization files

## Report Structure

Each report in the combined HTML file includes:

1. **Summary Tab**: Shows pattern summaries and recommendations
2. **Visualizations Tab**: Displays all visualization images
3. **Details Tab**: Shows the complete JSON data

## Troubleshooting

- If no reports are found, make sure your pattern analysis directories follow the naming convention `pattern_analysis*`
- If visualizations don't appear, check that the image files are in the correct format (PNG, JPG, or JPEG)
- If JSON data is missing, ensure that `analysis_results.json` exists in each pattern analysis directory 