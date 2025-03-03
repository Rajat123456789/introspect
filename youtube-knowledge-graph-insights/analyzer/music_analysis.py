"""
Functions for analyzing music content and its mental health effects
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_music_details(analyzer):
    """Detailed analysis of music content and its impact"""
    query = """
    MATCH (v:Video)
    WHERE v.primary_category = 'Music' OR 
          toLower(v.title) CONTAINS "music" OR
          toLower(v.title) CONTAINS "song" OR
          toLower(v.title) CONTAINS "album" OR
          toLower(v.title) CONTAINS "concert"
    
    WITH v, 
         // Extract music genre/type from title
         CASE 
             WHEN toLower(v.title) CONTAINS "rock" THEN "Rock"
             WHEN toLower(v.title) CONTAINS "pop" THEN "Pop"
             WHEN toLower(v.title) CONTAINS "rap" OR toLower(v.title) CONTAINS "hip hop" THEN "Rap/Hip-Hop"
             WHEN toLower(v.title) CONTAINS "jazz" THEN "Jazz"
             WHEN toLower(v.title) CONTAINS "classical" THEN "Classical"
             WHEN toLower(v.title) CONTAINS "metal" THEN "Metal"
             WHEN toLower(v.title) CONTAINS "country" THEN "Country"
             WHEN toLower(v.title) CONTAINS "edm" OR toLower(v.title) CONTAINS "electronic" THEN "Electronic"
             ELSE "Unknown"
         END as music_genre,
         
         // Extract emotional energy
         CASE 
             WHEN toLower(v.title) CONTAINS "relax" OR 
                  toLower(v.title) CONTAINS "calm" OR 
                  toLower(v.title) CONTAINS "sleep" OR
                  toLower(v.title) CONTAINS "meditation" THEN "Calming"
             WHEN toLower(v.title) CONTAINS "sad" OR 
                  toLower(v.title) CONTAINS "melancholy" OR 
                  toLower(v.title) CONTAINS "emotional" THEN "Melancholic"
             WHEN toLower(v.title) CONTAINS "hype" OR 
                  toLower(v.title) CONTAINS "energetic" OR 
                  toLower(v.title) CONTAINS "workout" OR
                  toLower(v.title) CONTAINS "party" THEN "Energizing"
             WHEN toLower(v.title) CONTAINS "angry" OR 
                  toLower(v.title) CONTAINS "rage" OR 
                  toLower(v.title) CONTAINS "aggressive" THEN "Aggressive"
             ELSE "Neutral"
         END as energy_type

    // Match with mental health aspects
    OPTIONAL MATCH (v)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
    
    WITH v, music_genre, energy_type, 
         COLLECT(DISTINCT {category: m.name, score: r.score, sentiment: r.sentiment}) as mental_health_impacts
    
    // Get watch time patterns using string functions 
    RETURN v.video_id as video_id,
           v.title as title,
           music_genre,
           energy_type,
           mental_health_impacts,
           CASE WHEN v.watched_at IS NOT NULL 
                THEN substring(toString(v.watched_at), 11, 2)  
                ELSE null 
           END as watch_hour
    ORDER BY size(mental_health_impacts) DESC
    """
    return analyzer.execute_query(query) 