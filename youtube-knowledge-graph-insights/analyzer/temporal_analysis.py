"""
Functions for temporal trend analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_temporal_trends(analyzer):
    """Analyze mental health scores over time at different frequencies"""
    query = """
    MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
    WHERE r.timestamp IS NOT NULL
    RETURN v.video_id, m.name as category, r.score as score, 
           r.sentiment as sentiment, 
           toString(r.timestamp) as timestamp_str
    ORDER BY r.timestamp
    """
    df = analyzer.execute_query(query)
    
    if df.empty:
        return None
    
    # Convert timestamp strings to pandas datetime
    df['timestamp'] = pd.to_datetime(df['timestamp_str'])
    
    # Create different time-based aggregations
    analyses = {
        'daily': df.set_index('timestamp').groupby([pd.Grouper(freq='D'), 'category'])['score'].mean(),
        'weekly': df.set_index('timestamp').groupby([pd.Grouper(freq='W'), 'category'])['score'].mean(),
        'monthly': df.set_index('timestamp').groupby([pd.Grouper(freq='ME'), 'category'])['score'].mean()
    }
    
    return analyses

def analyze_sentiment_trajectory(analyzer):
    """Analyze how sentiment changes over time for different mental health categories"""
    query = """
    MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
    WHERE r.timestamp IS NOT NULL
    RETURN m.name as category, 
           r.sentiment as sentiment,
           toString(r.timestamp) as timestamp_str,
           r.score as score
    ORDER BY r.timestamp
    """
    df = analyzer.execute_query(query)
    
    if df.empty:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp_str'])
    return df

def plot_temporal_trends(analyses):
    """Plot mental health trends at different time frequencies"""
    # Create reports directory if it doesn't exist
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    frequencies = {'daily': 'Daily', 'weekly': 'Weekly', 'monthly': 'Monthly'}
    
    for freq, data in analyses.items():
        plt.figure(figsize=(15, 8))
        
        # Unstack the multi-index to plot multiple lines
        data_unstacked = data.unstack()
        
        # Plot each category
        for category in data_unstacked.columns:
            plt.plot(data_unstacked.index, data_unstacked[category], 
                    label=category, marker='o', markersize=4)
        
        plt.title(f'{frequencies[freq]} Mental Health Category Scores')
        plt.xlabel('Time')
        plt.ylabel('Average Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{reports_dir}/mental_health_{freq}_trends.png')
        plt.close() 