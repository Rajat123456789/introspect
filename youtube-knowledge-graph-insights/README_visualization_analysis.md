# Visualization Analysis Tool

This tool uses GPT-4 Vision to analyze visualizations in your YouTube pattern analysis reports and generate detailed summaries for each visualization.

## Features

- Automatically extracts visualizations from HTML reports
- Uses GPT-4 Vision to analyze each visualization
- Generates detailed summaries of what each visualization shows
- Updates the HTML report with AI-generated analyses
- Provides insights about YouTube viewing patterns and mental health

## Requirements

- Python 3.7+
- OpenAI API key with access to GPT-4 Vision
- Required Python packages (see requirements_visualization_analysis.txt)

## Installation

1. Install the required packages:

```bash
pip install -r requirements_visualization_analysis.txt
```

2. Make sure you have an OpenAI API key with access to GPT-4 Vision.

## Usage

Run the script with the path to your HTML report and your OpenAI API key:

```bash
python analyze_visualizations.py path/to/your/report.html --api-key YOUR_OPENAI_API_KEY
```

The script will:
1. Extract all visualizations from the HTML report
2. Analyze each visualization using GPT-4 Vision
3. Update the HTML report with AI-generated summaries
4. Add a new section with all visualization analyses

## Example

```bash
python analyze_visualizations.py combined_reports/combined_report.html --api-key sk-...
```

## Output

The script will update the HTML report with:
1. Summaries for each visualization in the visualization cards
2. A new "AI Analysis of Visualizations" section with detailed analyses of all visualizations

## Troubleshooting

- If you encounter errors with the OpenAI API, make sure your API key is valid and has access to GPT-4 Vision.
- If the script can't find visualizations, check that the HTML report has the expected structure with `.all-visualization img` elements.
- If the script can't update the HTML, check that the HTML report has the expected placeholders for AI-generated summaries. 