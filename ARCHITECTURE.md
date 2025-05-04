# Project Architecture

This document outlines the architecture of the Personal Data Introspection Engine, based on the provided project details.

## Guiding Principles

*   **Modularity:** Components are designed to handle specific tasks (data acquisition, processing, analysis, presentation) independently.
*   **Automation:** The pipeline aims for minimal manual intervention after initial setup.
*   **Data-Centric:** The core focus is on transforming raw personal data into actionable insights for introspection.
*   **Extensibility:** The structure allows for potential future integration of new data sources or analysis modules.

## System Components

The system comprises the following logical components:

1.  **Data Acquisition Layer:**
    *   **Responsibilities:** Interfaces with external data sources to fetch raw user data.
    *   **Modules:**
        *   `Google Takeout Processor`: Handles parsing of `.html` YouTube history and historical health data (Google Fit/Fitbit) from user-provided Takeout archives.
        *   `Spotify API Client`: Interacts with the Spotify Web API to retrieve listening history and analyzed track audio features (e.g., tempo).
        *   `Health Connect API Client`: Connects to the Android Health Connect API to get live health metrics.
        *   `Apple Health Module`: (Handles interaction with Apple Health data - cross-platform capability mentioned).
    *   **Outputs:** Raw data streams/files (HTML, API responses, health data exports).

2.  **Data Processing & Enhancement Layer:**
    *   **Responsibilities:** Cleans, transforms, structures, and enriches the raw data into enhanced time-series formats.
    *   **Modules:**
        *   `YouTube Data Enhancer` (`youtube-analysis`): Parses unstructured YouTube `.html` Takeout data, utilizes the YouTube API to convert it into a structured format, applies models to calculate mental health scores for each video, and organizes this into time-series data.
        *   `Health Data Consolidator` (`apple-health`, `health-connect`, processing Takeout data): Cleans health data from Google Takeout (Fit/Fitbit), Health Connect, and Apple Health. Joins steps, heart BPM, and SpO2 data into a single time-series dataframe where each row represents a 10-minute time window.
        *   `Spotify Data Enhancer` (`spotify-analysis`): Processes listening history and integrates associated audio feature analysis values from the Spotify API.
        *   `Data Merging Modules` (`combining-health-and-music`, `combining-apple-spotify`): Facilitates analysis across different enhanced data domains.
    *   **Outputs:** Enhanced time-series datasets for YouTube, Health, and Spotify data.

3.  **Data Storage Layer (Knowledge Graphs):**
    *   **Responsibilities:** Stores the enhanced time-series data from different domains in separate, queryable knowledge graphs.
    *   **Modules:**
        *   `YouTube Knowledge Graph Module` (`youtube-knowledge-graph-insights`): Manages the storage and querying of enhanced YouTube time-series data.
        *   `Health Knowledge Graph Module` (`spotify-health-knowledge-graph` - *Note: Directory name might be misleading, seems health-focused*): Manages the storage and querying of consolidated health time-series data.
        *   `Spotify Knowledge Graph Module` (`spotify-knowledge-graph-insights`): Manages the storage and querying of enhanced Spotify time-series data.
    *   **Technology:** Utilizes a graph database technology.
    *   **Outputs:** Persistent storage of enhanced data within distinct knowledge graphs.

4.  **Analysis & Reporting Layer:**
    *   **Responsibilities:** Queries the knowledge graphs to identify specific patterns, generates visualizations in HTML reports, and creates AI-generated textual analysis for *each image* within the reports.
    *   **Modules:**
        *   `Pattern Analysis Engine`: Executes queries against the KGs, specifically looking for patterns related to YouTube consumption (addiction, rabbit holes, escapism, negative comparison) and health data (exercise patterns).
        *   `Report Generator`: Creates HTML reports (`analysis_reports/`) incorporating visualizations (graphs, charts) of these identified patterns.
        *   `AI Insight Generator` (`prompt-eng`): Uses language models to generate a specific message (analysis/insight) for each individual image/visualization presented in the HTML report.
    *   **Outputs:** HTML reports containing visualizations, each accompanied by its own AI-generated message.

5.  **Introspection Interface Layer:**
    *   **Responsibilities:** Provides an interactive chatbot interface using the AI-generated messages from reports as context, allowing users to explore insights and compare three different base AI models.
    *   **Modules:**
        *   `Chatbot Backend` (`chatbot-interface`): Manages user interaction, utilizes the AI-generated messages from reports as context, and interfaces with the three language models.
        *   `Language Model Interfaces` (`finetune_introspect-llm`): Handles communication with the three distinct base models trained/configured for introspection tasks.
        *   `Frontend Interface` (`chatbot-interface`): The user-facing web interface for the chatbot, enabling interaction and model comparison.
    *   **Outputs:** Interactive user experience for data introspection, powered by report-derived context and multiple AI models.

## Data Flow

```
+-----------------------+      +---------------------------+      +--------------------------+      +---------------------------+      +---------------------------+
| Data Acquisition      | ---> | Data Processing/Enhancing | ---> | Knowledge Graph Storage  | ---> | Analysis & Reporting      | ---> | Introspection Interface   |
| (Takeout [.html/etc], |      | (Clean, Structure, Score, |      | (Separate KGs for        |      | (Pattern Analysis, HTML   |      | (Chatbot using Report     |
| Spotify API,          |      | Time-Series Conversion)   |      | YouTube, Health, Spotify)|      | Report, AI msg per image)|      | Messages, 3 Models)       |
| Health Connect API)   |      |                           |      |                          |      |                           |      |                           |
+-----------------------+      +---------------------------+      +--------------------------+      +---------------------------+      +---------------------------+
                                                                                                      |
                                                                                                      +---> User
```

## Key Technologies

*   **Programming Language:** Python (`introspect_env`)
*   **APIs:** Google Takeout processing, Spotify Web API, Google YouTube API, Android Health Connect API
*   **Data Storage:** Knowledge Graph Database
*   **AI/ML:** Language Models (3 base models, potentially fine-tuned), models for scoring YouTube videos.
*   **Frontend:** HTML, CSS, JavaScript (for reports and chatbot interface).
*   **Orchestration:**
    *   Chatbot Interface: Batch script (`run-dev.bat`) starts frontend/backend.
    *   Main Data Pipeline (Acquisition -> Reporting): Execution method TBD.

*(This architecture document reflects the specific details provided about the project's implementation.)* 