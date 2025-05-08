@echo off
echo Running Historical Health Data Analysis...
echo.

REM Set the path to your Google Fit data directory
set DATA_PATH=data\Fit

echo Step 1: Running data analysis script...
python analyze_historical_health.py %DATA_PATH%
if %errorlevel% neq 0 (
    echo Error: Data analysis failed!
    exit /b %errorlevel%
)
echo.

echo Step 2: Running visualization script...
python visualize_health_insights.py %DATA_PATH%
if %errorlevel% neq 0 (
    echo Error: Visualization failed!
    exit /b %errorlevel%
)
echo.

echo Analysis complete!
echo Results can be found in:
echo - analysis_output/
echo - visualizations/
echo.

pause 