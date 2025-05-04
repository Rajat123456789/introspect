# Personal Data Introspection Engine

## Project Goal

This project aims to merge personal data from various sources, including YouTube history, Spotify listening habits, Google Fit/Fitbit data, and live Android health data via Health Connect. By analyzing this combined data, the goal is to provide users with insights into their mental and physical health patterns, facilitating self-introspection.

## Key Features

*   **Multi-Source Data Integration:** Combines data from Google Takeout (YouTube, Health), Spotify API, and Health Connect API.
*   **Automated Data Processing:** Cleans and structures data from diverse formats (e.g., YouTube HTML Takeout).
*   **Data Enhancement:** Enriches raw data using APIs and models (e.g., adding audio features to Spotify tracks, calculating mental health scores for YouTube videos).
*   **Time-Series Analysis:** Structures health and media consumption data into time series for pattern identification.
*   **Knowledge Graph Storage:** Organizes enhanced data into dedicated knowledge graphs.
*   **Pattern Identification:** Analyzes data for patterns related to mental well-being (e.g., media addiction, escapism, comparison) and physical activity.
*   **AI-Powered Reporting:** Generates HTML reports with visualizations and AI-generated textual analysis for each insight.
*   **Multi-Model Introspection Chatbot:** Provides a chatbot interface powered by three different language models, allowing users to interact with their insights and compare the models' introspection capabilities.

## Project Deliverables

1.  **Historical Data Analysis Report:** An HTML report summarizing patterns and insights derived from the user's historical data.
2.  **Introspection Model Comparison:** An evaluation or interface allowing comparison of the performance of the three different base models used in the chatbot for introspection tasks.
3.  **Introspective Agent:** The functional chatbot interface designed to interact with users and help them explore their personal data patterns.

## Project Workflow

1.  **Data Acquisition:** Collect historical data via Google Takeout and live data via Health Connect and Spotify APIs.
2.  **Data Merging & Enhancement:**
    *   Parse and structure YouTube Takeout data using the YouTube API.
    *   Clean and consolidate health data (steps, heart rate, SpO2) into a unified time-series format.
    *   Enhance media data with calculated scores (e.g., mental health for videos) and features (e.g., Spotify audio features).
3.  **Knowledge Graph Generation:** Store the processed and enhanced time-series data in separate knowledge graphs for YouTube and Health data.
4.  **Analysis & Reporting:**
    *   Query knowledge graphs to identify significant patterns in media consumption and health metrics.
    *   Generate HTML reports visualizing these patterns.
    *   Employ AI models to add descriptive insights to report visualizations.
5.  **Introspection:** Feed the AI-generated report messages into a specialized model (or models within the chatbot interface) to help users reflect on the identified patterns.

## Setup (High-Level)

*   **Data Sources:** You will need to provide data from:
    *   Google Takeout (specifically YouTube history and Google Fit/Fitbit data).
    *   Spotify Account (via Spotify API).
    *   Android Device (via Health Connect API).
*   **Dependencies:** (Details TBC - likely Python packages in `introspect_env`).
*   **Configuration:** (Details TBC - likely involves setting up API keys/credentials for Spotify, Google Cloud/YouTube, Health Connect).

## Running the Project (High-Level)

*   The main data processing pipeline (Data Acquisition -> Processing -> Knowledge Graph -> Reporting) execution method is **TBC**.
*   The `run-dev.bat` script is used **only** to start the frontend and backend services for the `chatbot-interface` after the data pipeline has been run and reports generated.

## Directory Structure (Tentative)

*   `apple-health/`, `health-connect/`: Health data acquisition/processing.
*   `spotify-analysis/`, `youtube-analysis/`: Media data acquisition/processing.
*   `combining-apple-spotify/`, `combining-health-and-music/`: Data integration modules.
*   `spotify-health-knowledge-graph/`, `spotify-knowledge-graph-insights/`, `youtube-knowledge-graph-insights/`: Knowledge graph management and querying.
*   `analysis_reports/`: Storage for generated reports.
*   `finetune_introspect-llm/`, `prompt-eng/`: AI model fine-tuning and prompt engineering.
*   `chatbot-interface/`: Multi-model chatbot interface for user introspection and model comparison.
*   `introspect_env/`: Python virtual environment.
*   `.git/`, `.vscode/`: Development environment configuration.
*   `run-dev.bat`: Script to run the frontend and backend of the `chatbot-interface`.
*   `.gitignore`, `LICENSE`: Standard project files.

*(This README is preliminary. More specific setup and usage instructions are needed.)* 