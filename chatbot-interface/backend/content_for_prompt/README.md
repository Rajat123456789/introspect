# Content Directory for Introspect Chat

This directory contains the context data files used by the introspection chatbot.

## Required Files

The following files should be placed in this directory:

1. `context_raw.json` - Contains raw data about user activities and metrics
2. `context_insights.json` - Contains processed insights derived from the raw data

## File Format

Both files should be valid JSON files. The structure should match what the chat interface expects.

## Example Structure

### context_raw.json
```json
{
  "youtube": {
    "watch_time": {
      "weekday_avg": 2.3,
      "weekend_avg": 3.7
    },
    "topics": ["technology", "health"]
  },
  "spotify": {
    "listening_time": {
      "weekday_avg": 1.8,
      "weekend_avg": 2.5
    },
    "genres": ["pop", "rock", "ambient"]
  }
}
```

### context_insights.json
```json
{
  "youtube": {
    "escapism_index": 72,
    "educational_value": 58
  },
  "spotify": {
    "mood_regulation": 75,
    "genre_exploration": 62
  },
  "health_correlations": {
    "digital_to_sleep": {
      "correlation": "Moderate negative",
      "confidence": 68
    }
  }
}
```

## Usage

The chatbot will load these files when using the 'introspect' model type. If the files are missing, the chat will inform the user and continue without personalized insights. 