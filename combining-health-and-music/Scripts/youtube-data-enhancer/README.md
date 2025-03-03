# YouTube Content Analysis System

## Overview
This system analyzes YouTube watch history to identify mental health themes, content patterns, and engagement metrics. It's designed to process large datasets efficiently using parallel processing.

## Dependencies

### Core Libraries
```requirements
pandas>=1.3.0
numpy>=1.19.0
transformers>=4.15.0
textblob>=0.15.3
spacy>=3.0.0
tqdm>=4.62.0
```

### Language Models
1. **Zero-Shot Classification**
   - Model: `facebook/bart-large-mnli`
   - Framework: PyTorch
   - Usage: Content classification, mental health theme detection
   - Memory Requirements: ~1.6GB

2. **Sentiment Analysis**
   - Model: `distilbert-base-uncased-finetuned-sst-2-english`
   - Framework: PyTorch
   - Usage: Emotional tone detection
   - Memory Requirements: ~260MB

3. **SpaCy NLP**
   - Model: `en_core_web_sm`
   - Usage: Topic extraction, text processing
   - Memory Requirements: ~13MB

4. **CLIP Model**
   - Model: `openai/clip-vit-base-patch32`
   - Usage: Advanced content tagging
   - Memory Requirements: ~600MB

## Analysis Components

### 1. Content Classification
- **Model**: `facebook/bart-large-mnli`
- **Categories**:
  ```python
  youtube_categories = [
      "Music", "Gaming", "Education", "Entertainment",
      "Sports", "News", "Technology", "Comedy",
      "Vlogs", "Tutorial", "Reviews", "Podcast", "Shorts"
  ]
  ```
- **Confidence Threshold**: 0.3

### 2. Mental Health Analysis
- **Model**: `facebook/bart-large-mnli`
- **Categories**:
  ```python
  mental_health_categories = [
      "anxiety and stress management",
      "depression and emotional support",
      "self-improvement and personal growth",
      "mindfulness and mental wellness",
      "emotional health discussion",
      "mental health education",
      "therapy and mental health support",
      "addiction awareness and recovery",
      "sleep and mental wellness",
      "social anxiety and relationships",
      "work-life balance and burnout",
      "trauma awareness and healing",
      "self-care practices",
      "motivation and mental strength",
      "mental health experiences"
  ]
  ```
- **Context Bonus**: 0.05 per additional context
- **Base Threshold**: 0.3

### 3. Music Content Analysis
- **Categories**:
  ```python
  music_categories = {
      'mood': ['happy', 'sad', 'energetic', 'calm', 'aggressive', 
               'relaxed', 'dark', 'atmospheric'],
      'genre': ['electronic', 'rock', 'pop', 'hip hop', 'classical',
                'jazz', 'metal', 'folk', 'ambient', 'indie'],
      'characteristics': ['instrumental', 'vocal', 'fast', 'slow', 
                         'melodic', 'rhythmic', 'acoustic', 'synthetic'],
      'context': ['party', 'study', 'workout', 'sleep', 'meditation',
                 'dance', 'focus', 'background']
  }
  ```
- **Threshold**: 0.2

### 4. Sentiment Analysis
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Output**: POSITIVE/NEGATIVE with confidence score
- **Use**: Emotional context detection

## Input Requirements

### Required CSV Format
- Minimum columns: 'Title', 'Watched At'
- Optional columns: 'Description', 'Category', 'Tags'
- Example:
  ```csv
  Title,Watched At
  "Example Video","2024-02-23 10:30:00"
  ```

## Output Files

All output is saved to the `../output` directory:

1. `youtube_analysis_[TIMESTAMP]_main.csv` - Main analysis with video types and sentiment
2. `youtube_analysis_[TIMESTAMP]_mental_health.csv` - Detailed mental health indicators
3. `youtube_analysis_[TIMESTAMP]_patterns.csv` - Content patterns identified in videos
4. `youtube_analysis_[TIMESTAMP]_engagement.csv` - Audience engagement metrics

A log file `youtube_analysis_[TIMESTAMP].log` contains the complete analysis output, and `youtube_analysis_[TIMESTAMP]_summary.log` provides statistics about the processing.

## CSV Files Explained

### Main CSV
Contains core analysis for each video:
- video_id: Original row index from YouTube history
- title: Video title
- watched_at: When the video was watched
- primary_category: Main content type (e.g., Entertainment, Educational)
- detailed_type: More specific content categorization
- sentiment: POSITIVE, NEGATIVE, or NEUTRAL
- sentiment_score: Confidence score for sentiment
- primary_format: Format of content (e.g., tutorial, reaction)
- primary_purpose: Purpose of content (e.g., entertainment, education)
- style: Content style attributes
- confidence: Confidence score for primary category

### Mental Health CSV
Contains mental health themes detected:
- video_id: Links to the main CSV
- category: Mental health theme (e.g., "anxiety awareness")
- score: Relevance score (0-1)
- timestamp: When the video was watched
- sentiment: Associated sentiment
- sentiment_score: Confidence of sentiment

### Patterns CSV
Contains recurring patterns in content:
- video_id: Links to the main CSV
- pattern_type: Category of pattern
- pattern: Specific pattern detected
- timestamp: When the video was watched
- category: Content category

### Engagement CSV
Contains engagement metrics:
- video_id: Links to the main CSV
- timestamp: When the video was watched
- content_type: Primary content category
- audience_engagement: Engagement metrics
- production_quality: Production quality indicators
- content_format: Format indicators
- content_purpose: Purpose indicators

## Performance Considerations

### System Requirements
- Minimum: 4GB RAM
- Recommended: 8GB RAM
- Storage: ~2.5GB for models
- CPU: Multi-core recommended
- GPU: Optional, improves processing speed

### Processing Speed
- Average: 1-2 videos per second
- Factors affecting speed:
  - Available metadata
  - Content complexity
  - System resources
  - Batch size

## Error Handling
- Graceful degradation for missing metadata
- Default values for failed analyses
- Continuous processing despite individual errors
- Detailed error logging

## Usage Example
```python
from youtube_content_analyzer import YouTubeContentAnalyzer

# Initialize analyzer
analyzer = YouTubeContentAnalyzer()

# Process YouTube history
analyses = analyzer.analyze_youtube_history('youtube-history.csv')
```

## Knowledge Graph Preparation
The output files are structured for easy import into a knowledge graph system:

### Entity Types
1. Videos
2. Mental Health Themes
3. Content Patterns
4. Temporal Relationships

### Relationship Types
1. Video-Theme Relations
2. Theme-Theme Relations
3. Pattern-Content Relations
4. Temporal Connections

## Limitations
1. Title-only analysis when metadata unavailable
2. English-language focus
3. Pattern recognition dependent on video title clarity
4. Mental health detection is inference-based, not diagnostic

## Future Improvements
1. Multi-language support
2. GPU acceleration
3. Additional metadata extraction
4. Enhanced pattern recognition
5. Custom model training options

## Contributing
[Your contribution guidelines]

## License
[Your license information]

## Running the Analyzer

### Basic Usage
```bash
python youtube_content_analyzer.py --file your_youtube_history.csv
```

### Command Line Arguments
- `--file`: Path to the YouTube history CSV file (required)
- `--resume`: Resume analysis from a specific row number
- `--processes`: Number of parallel processes to use (default: 4)
- `--max-runs`: Limit the number of videos to process (useful for testing)

### Examples

1. Basic analysis:
```bash
python youtube_content_analyzer.py --file youtube-history.csv
```

2. Resume from a specific point:
```bash
python youtube_content_analyzer.py --file youtube-history.csv --resume 5590
```

3. Use more parallel processes (faster on multi-core systems):
```bash
python youtube_content_analyzer.py --file youtube-history.csv --processes 8
```

4. Run a quick test with limited videos:
```bash
python youtube_content_analyzer.py --file youtube-history.csv --max-runs 100
```

5. Resume and limit processing (for testing resumption):
```bash
python youtube_content_analyzer.py --file youtube-history.csv --resume 5000 --max-runs 50
```

### Handling Interruptions
If the script is interrupted, it saves progress to a checkpoint file. You can resume by running:
```bash
python youtube_content_analyzer.py --file youtube-history.csv
```
The script will automatically find the checkpoint and continue from where it left off.

A sample output tail would look like this:
```bash

D:\IntrospectAI\combining-health-and-music\Scripts>python youtube_content_analyzer.py --file youtube-gaurav.csv --processes 8 --max-runs 8000
....
Processing chunk 8/8

[PROGRESS] Processed 8000/8000 videos
Processing rate: 0.09 videos/second
Estimated time remaining: 0.0 minutes
Content types so far: {'Shorts': 166, 'Reviews': 40, 'Sports': 92, 'Entertainment': 203, 'News': 37, 'Comedy': 96, 'Technology': 96, 'Music': 117, 'Podcast': 38, 'Tutorial': 31, 'Gaming': 80, 'Vlogs': 1, 'Education': 3}
Mental health mentions: {'self-improvement and personal growth': 864, 'motivation and mental strength': 761, 'addiction awareness and recovery': 435, 'mental health experiences': 632, 'mindfulness and mental wellness': 478, 'trauma awareness and healing': 283, 'self-care practices': 559, 'therapy and mental health support': 219, 'emotional health discussion': 394, 'mental health education': 340, 'depression and emotional support': 187, 'social anxiety and relationships': 291, 'work-life balance and burnout': 338, 'anxiety and stress management': 245, 'sleep and mental wellness': 252}
--------------------------------------------------------------------------------

All results saved to:
- Main analysis: D:\IntrospectAI\combining-health-and-music\Scripts\..\output\youtube_analysis_20250228_042945_main.csv
- Mental health data: D:\IntrospectAI\combining-health-and-music\Scripts\..\output\youtube_analysis_20250228_042945_mental_health.csv
- Pattern analysis: D:\IntrospectAI\combining-health-and-music\Scripts\..\output\youtube_analysis_20250228_042945_patterns.csv
- Engagement metrics: D:\IntrospectAI\combining-health-and-music\Scripts\..\output\youtube_analysis_20250228_042945_engagement.csv

Summary log written to: D:\IntrospectAI\combining-health-and-music\Scripts\..\output\youtube_analysis_20250228_042945_summary.log

Analysis Complete!
Total videos processed: 8000/8000
Total time: 84694.33 seconds
Average processing rate: 0.09 videos/second
```
