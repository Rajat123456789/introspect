#!/bin/bash

# Generate and Analyze YouTube Pattern Report
# This script automates the process of generating an LLM-friendly report from an HTML report
# and then analyzing it with an LLM.

# Check if the HTML report file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <html_report_file> [openai_api_key]"
    echo "  html_report_file: Path to the HTML report file"
    echo "  openai_api_key: (Optional) OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable"
    exit 1
fi

HTML_REPORT=$1
API_KEY=$2

# Check if the HTML report file exists
if [ ! -f "$HTML_REPORT" ]; then
    echo "Error: HTML report file '$HTML_REPORT' not found."
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -r requirements_llm_report.txt

# Generate the LLM summary
echo "Generating LLM summary from HTML report..."
SUMMARY_FILE="${HTML_REPORT%.*}_llm_summary.txt"
python generate_llm_summary.py "$HTML_REPORT" --output "$SUMMARY_FILE"

# Check if the summary was generated successfully
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "Error: Failed to generate LLM summary."
    exit 1
fi

echo "LLM summary generated: $SUMMARY_FILE"

# Ask if the user wants to analyze the report with an LLM
read -p "Do you want to analyze this report with an LLM? (y/n): " ANALYZE

if [ "$ANALYZE" = "y" ] || [ "$ANALYZE" = "Y" ]; then
    echo "Analyzing report with LLM..."
    
    # If API key is provided, use it
    if [ -n "$API_KEY" ]; then
        python analyze_with_llm.py "$SUMMARY_FILE" --api-key "$API_KEY"
    else
        python analyze_with_llm.py "$SUMMARY_FILE"
    fi
    
    echo "Analysis complete!"
else
    echo "Skipping LLM analysis."
    echo "You can analyze the report later by running:"
    echo "  python analyze_with_llm.py $SUMMARY_FILE"
fi

echo "Done!" 