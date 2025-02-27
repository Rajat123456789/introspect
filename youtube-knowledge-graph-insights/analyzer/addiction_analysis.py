"""
Functions for analyzing addiction patterns in YouTube consumption
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_viewing_patterns(analyzer):
    """Analyze potentially problematic viewing patterns (binge watching, late night, etc.)"""
    query = """
    MATCH (v:Video)
    WHERE v.watched_at IS NOT NULL
    
    // Using string functions for timestamps
    WITH toString(v.watched_at) as watched_time_str, v.video_id as video_id, 
         v.title as title, v.primary_category as category
    
    // Group by date portion of the timestamp
    WITH substring(watched_time_str, 0, 10) as view_date,
         collect({time: watched_time_str, id: video_id, category: category, title: title}) as daily_views
    
    // Count videos per day and late night videos
    WITH 
        view_date,
        size(daily_views) as videos_per_day,
        [v in daily_views WHERE substring(v.time, 11, 2) >= "22" OR substring(v.time, 11, 2) <= "04" | v.id] as late_night_videos
    
    RETURN 
        view_date,
        videos_per_day,
        late_night_videos,
        size(late_night_videos) as late_night_count,
        CASE 
            WHEN videos_per_day > 15 THEN true 
            ELSE false 
        END as binge_day
    ORDER BY view_date
    """
    df = analyzer.execute_query(query)
    
    if df.empty:
        return None
        
    df['date'] = pd.to_datetime(df['view_date'])
    return df

def analyze_content_addiction(analyzer):
    """Analyze which content categories might present addiction risk"""
    query = """
    MATCH (v:Video)
    OPTIONAL MATCH (v)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
    WHERE m.name IN ['Anxiety', 'Depression', 'Addiction', 'Stress']
    
    WITH v, 
         count(r) as negative_aspects,
         avg(CASE WHEN r.score < 0 THEN r.score ELSE null END) as negative_score
    
    WHERE v.primary_category IS NOT NULL
    
    WITH v.primary_category as category,
         count(v) as video_count,
         sum(negative_aspects) as total_negative_aspects,
         avg(negative_score) as avg_negative_score
    
    RETURN 
        category,
        video_count,
        total_negative_aspects,
        avg_negative_score,
        (video_count * 0.3) + (total_negative_aspects * 0.3) + (coalesce(abs(avg_negative_score), 0) * 0.4) as addiction_risk_score
    ORDER BY addiction_risk_score DESC
    """
    df = analyzer.execute_query(query)
    
    if df.empty:
        return None
    
    # Normalize addiction risk score to 0-1 range
    if len(df) > 0:
        max_score = df['addiction_risk_score'].max()
        if max_score > 0:
            df['addiction_risk_score'] = df['addiction_risk_score'] / max_score
    
    return df

def plot_addiction_risk(addiction_data):
    """Create visualizations for addiction risk factors"""
    if addiction_data is None or addiction_data.empty:
        return

    # Create reports directory if it doesn't exist
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Plot 1: Addiction risk by video category
    plt.figure(figsize=(14, 8))
    category_risks = addiction_data.groupby('category')['addiction_risk_score'].mean().sort_values(ascending=False)
    
    # Create bar chart with custom colors based on risk level
    bars = plt.bar(category_risks.index, category_risks.values)
    
    # Color bars by risk level
    for i, bar in enumerate(bars):
        risk = category_risks.values[i]
        if risk > 0.7:
            bar.set_color('darkred')
        elif risk > 0.5:
            bar.set_color('red')
        elif risk > 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.title('Addiction Risk by Content Category')
    plt.xlabel('Content Category')
    plt.ylabel('Addiction Risk Score (0-1)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/addiction_risk_by_category.png')
    plt.close() 