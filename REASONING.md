
Okay, here is a complete and extensive mind map of the Personal Data Introspection Engine project, synthesizing all the information provided, including the reasoning behind each component:

*   **Core Goal: Personal Introspection via Data Analysis**
    *   *Reasoning:* To empower users with deeper self-understanding by automatically collecting, merging, analyzing, and presenting patterns from their digital footprint (media consumption) and physical activity, facilitating reflection on mental and physical well-being.

*   **I. Data Acquisition Layer** (Gathering Raw Information)
    *   *Reasoning:* To collect the necessary base data from various user sources, both historical and live.
    *   **Component: Google Takeout Processor**
        *   *Input:* User-provided Google Takeout archive.
        *   *Sub-tasks:*
            *   Parse YouTube History (`.html` format) - *Reasoning:* Extracts raw viewing data (unstructured).
            *   Parse Health Data (Google Fit/Fitbit) - *Reasoning:* Extracts historical physical activity records.
    *   **Component: Spotify API Client** (`spotify-analysis` related)
        *   *Input:* User's Spotify account authorization.
        *   *Sub-tasks:*
            *   Fetch Listening History - *Reasoning:* Captures music choices reflecting mood/preferences over time.
            *   Fetch Track Audio Features (e.g., tempo) - *Reasoning:* Obtains objective metrics about the music listened to.
    *   **Component: Health Connect API Client** (`health-connect` related)
        *   *Input:* Authorized Android device connection.
        *   *Sub-tasks:* Fetch live/recent health metrics - *Reasoning:* Captures up-to-date physical state information.
    *   **Component: Apple Health Module** (`apple-health` related)
        *   *Input:* Authorized Apple Health connection.
        *   *Sub-tasks:* Fetch health metrics - *Reasoning:* Enables cross-platform health data integration (iOS users).

*   **II. Data Processing & Enhancement Layer** (Creating Meaningful Data)
    *   *Reasoning:* To transform the raw, disparate data into clean, structured, enriched, and time-aligned formats suitable for analysis.
    *   **Component: YouTube Data Enhancer** (`youtube-analysis`)
        *   *Input:* Raw YouTube HTML data.
        *   *Sub-tasks:*
            *   HTML Parsing - *Reasoning:* Extracts usable information from the unstructured Takeout file.
            *   YouTube API Integration - *Reasoning:* Structures the data and fetches additional video metadata.
            *   Mental Health Scoring Model - *Reasoning:* Applies a model to assign a calculated score related to mental well-being impact for each video watched.
            *   Time-Series Conversion - *Reasoning:* Organizes viewing data chronologically for pattern analysis.
    *   **Component: Health Data Consolidator** (`apple-health`, `health-connect`, Takeout processing)
        *   *Input:* Raw health data from Takeout, Health Connect, Apple Health.
        *   *Sub-tasks:*
            *   Data Cleaning - *Reasoning:* Ensures data quality and consistency.
            *   Metric Standardization - *Reasoning:* Unifies key metrics (Steps, Heart Rate BPM, SpO2) across different sources.
            *   Time-Series Conversion (10-minute intervals) - *Reasoning:* Creates a consistent temporal structure for analyzing health trends.
    *   **Component: Spotify Data Enhancer** (`spotify-analysis`)
        *   *Input:* Raw Spotify history and audio features.
        *   *Sub-tasks:*
            *   History Processing & Feature Integration - *Reasoning:* Combines listening events with objective track characteristics.
            *   Time-Series Conversion - *Reasoning:* Organizes listening data chronologically.
    *   **Component: Data Merging Modules** (`combining-health-and-music`, `combining-apple-spotify`)
        *   *Reasoning:* To prepare data for potential cross-domain analysis by aligning timelines or creating combined views (specific joining logic TBD).

*   **III. Data Storage Layer** (Persisting Enhanced Data)
    *   *Reasoning:* To store the processed, enhanced time-series data in a way that allows efficient querying for complex patterns and relationships.
    *   **Component: Knowledge Graphs**
        *   *Technology:* Graph Database - *Reasoning:* Suitable for representing interconnected data points and performing complex relationship-based queries over time.
        *   *Instances:*
            *   YouTube Knowledge Graph (`youtube-knowledge-graph-insights`) - *Reasoning:* Stores enhanced YouTube time-series data.
            *   Health Knowledge Graph (`spotify-health-knowledge-graph`) - *Reasoning:* Stores consolidated health time-series data.
            *   Spotify Knowledge Graph (`spotify-knowledge-graph-insights`) - *Reasoning:* Stores enhanced Spotify time-series data.

*   **IV. Analysis & Reporting Layer** (Extracting and Presenting Insights)
    *   *Reasoning:* To identify meaningful patterns within the stored data and present them to the user in an understandable format, augmented with AI interpretation.
    *   **Component: Pattern Analysis Engine**
        *   *Input:* Queries to Knowledge Graphs.
        *   *Sub-tasks:* Identify specific patterns:
            *   YouTube Analysis: Addiction, Rabbit Holes, Escapism, Negative Comparison - *Reasoning:* Focuses on predefined themes relevant to media consumption and mental well-being.
            *   Health Analysis: Exercise Patterns - *Reasoning:* Focuses on trends in physical activity.
    *   **Component: Report Generator**
        *   *Input:* Identified patterns.
        *   *Sub-tasks:*
            *   Create HTML Reports (`analysis_reports/`) - *Reasoning:* Provides a standard, accessible format for presenting findings.
            *   Generate Visualizations (Charts, Graphs) - *Reasoning:* Offers intuitive visual representation of data trends.
    *   **Component: AI Insight Generator** (`prompt-eng`)
        *   *Input:* Visualizations within the HTML report.
        *   *Technology:* Language Models (LLMs).
        *   *Sub-tasks:* Generate textual analysis/message for *each individual image* - *Reasoning:* Provides specific, AI-driven interpretation context for every piece of visual data presented in the report.

*   **V. Introspection Interface Layer** (User Interaction & Reflection)
    *   *Reasoning:* To provide an interactive tool for the user to explore the generated insights and facilitate the introspection process, leveraging multiple AI models.
    *   **Component: Chatbot Interface** (`chatbot-interface`)
        *   *Modules:* Frontend (UI), Backend (Logic).
        *   *Core Functionality:*
            *   Uses AI-generated messages from reports as *context* - *Reasoning:* Grounds the conversation in the specific findings from the user's data analysis.
            *   Utilizes 3 distinct base Language Models (`finetune_introspect-llm` related) - *Reasoning:* Allows the user (and developers) to compare the effectiveness and nuances of different AI approaches for the subjective task of introspection.
            *   Facilitates user exploration of patterns - *Reasoning:* Enables interactive dialogue about the insights surfaced in the reports.

*   **VI. Orchestration & Execution** (Running the System)
    *   *Reasoning:* Defines how the different parts of the system are initiated and managed.
    *   **Component: Main Data Pipeline Execution (Acquisition -> Reporting)**
        *   *Method:* **To Be Determined (TBD)** - *Reasoning:* The specific script or process to run the core data processing workflow is not yet specified.
    *   **Component: Chatbot Interface Execution**
        *   *Method:* `run-dev.bat` script - *Reasoning:* Starts the frontend and backend services *only* for the chatbot, intended to be run *after* the main pipeline has generated reports.
    *   **Environment:** Python (`introspect_env`) - *Reasoning:* The primary development and execution environment for the codebase.

*   **VII. Project Deliverables** (Expected Outputs)
    *   *Reasoning:* Defines the key tangible outcomes of the project.
    *   **Deliverable 1:** Historical Data Analysis Report (HTML format) - *Reasoning:* The summarized findings from the user's data.
    *   **Deliverable 2:** Introspection Model Comparison - *Reasoning:* The ability to evaluate the 3 different AI models within the chatbot.
    *   **Deliverable 3:** Operational Introspective Agent (The Chatbot) - *Reasoning:* The final interactive tool for the user.
