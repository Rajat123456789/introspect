@echo off
REM Generate and Analyze YouTube Pattern Report
REM This script automates the process of generating an LLM-friendly report from an HTML report
REM and then analyzing it with an LLM.

REM Check if the HTML report file is provided
if "%~1"=="" (
    echo Usage: %0 ^<html_report_file^> [openai_api_key]
    echo   html_report_file: Path to the HTML report file
    echo   openai_api_key: (Optional) OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable
    exit /b 1
)

set HTML_REPORT=%~1
set API_KEY=%~2

REM Check if the HTML report file exists
if not exist "%HTML_REPORT%" (
    echo Error: HTML report file '%HTML_REPORT%' not found.
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip install -r requirements_llm_report.txt

REM Generate the LLM summary
echo Generating LLM summary from HTML report...
set SUMMARY_FILE=%HTML_REPORT:~0,-4%_llm_summary.txt
python generate_llm_summary.py "%HTML_REPORT%" --output "%SUMMARY_FILE%"

REM Check if the summary was generated successfully
if not exist "%SUMMARY_FILE%" (
    echo Error: Failed to generate LLM summary.
    exit /b 1
)

echo LLM summary generated: %SUMMARY_FILE%

REM Ask if the user wants to analyze the report with an LLM
set /p ANALYZE=Do you want to analyze this report with an LLM? (y/n): 

if /i "%ANALYZE%"=="y" (
    echo Analyzing report with LLM...
    
    REM If API key is provided, use it
    if not "%API_KEY%"=="" (
        python analyze_with_llm.py "%SUMMARY_FILE%" --api-key "%API_KEY%"
    ) else (
        python analyze_with_llm.py "%SUMMARY_FILE%"
    )
    
    echo Analysis complete!
) else (
    echo Skipping LLM analysis.
    echo You can analyze the report later by running:
    echo   python analyze_with_llm.py %SUMMARY_FILE%
)

echo Done! 