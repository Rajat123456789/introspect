import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def get_temporal_trends(self):
        """Analyze mental health scores over time at different frequencies"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE r.timestamp IS NOT NULL
            RETURN v.video_id, m.name as category, r.score as score, 
                   r.sentiment as sentiment, 
                   toString(r.timestamp) as timestamp_str
            ORDER BY r.timestamp
            """
            result = session.run(query)
            
            # Convert to DataFrame and handle timestamps
            df = pd.DataFrame([dict(record) for record in result])
            if df.empty:
                logger.warning("No temporal data found")
                return None
            
            # Convert timestamp strings to pandas datetime - use timestamp_str which is guaranteed to be a string
            df['timestamp'] = pd.to_datetime(df['timestamp_str'])
            
            # Create different time-based aggregations
            analyses = {
                'daily': df.set_index('timestamp').groupby([pd.Grouper(freq='D'), 'category'])['score'].mean(),
                'weekly': df.set_index('timestamp').groupby([pd.Grouper(freq='W'), 'category'])['score'].mean(),
                'monthly': df.set_index('timestamp').groupby([pd.Grouper(freq='ME'), 'category'])['score'].mean()
            }
            
            return analyses

    def analyze_category_relationships(self):
        """Analyze relationships between mental health categories"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r1:HAS_MENTAL_HEALTH_ASPECT]->(m1:MentalHealthCategory)
            MATCH (v)-[r2:HAS_MENTAL_HEALTH_ASPECT]->(m2:MentalHealthCategory)
            WHERE m1.name < m2.name  // Avoid duplicates
            RETURN m1.name as category1, m2.name as category2,
                   avg(r1.score) as score1, avg(r2.score) as score2,
                   count(*) as frequency
            ORDER BY frequency DESC
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def analyze_engagement_impact(self):
        """Analyze relationship between engagement types and mental health scores"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            MATCH (v)-[:HAS_ENGAGEMENT]->(e:Engagement)
            RETURN e.name as engagement_type, m.name as category,
                   avg(r.score) as avg_score, count(*) as count
            ORDER BY count DESC
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def plot_temporal_trends(self, analyses, run_dir=None):
        """Plot mental health trends at different time frequencies"""
        # Use provided run directory or default to general reports directory
        reports_dir = run_dir if run_dir else 'analysis_reports'
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

    def plot_category_correlations(self, relationship_df):
        """Plot correlations between mental health categories"""
        if relationship_df.empty:
            logger.warning("No category relationships found")
            return
        
        # Create reports directory if it doesn't exist
        reports_dir = 'analysis_reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        plt.figure(figsize=(12, 8))
        
        # Create correlation matrix
        categories = pd.concat([relationship_df['category1'], relationship_df['category2']]).unique()
        corr_matrix = pd.DataFrame(0, index=categories, columns=categories, dtype=float)
        
        # Fill the correlation matrix
        for _, row in relationship_df.iterrows():
            freq = float(row['frequency'])
            corr_matrix.loc[row['category1'], row['category2']] = freq
            corr_matrix.loc[row['category2'], row['category1']] = freq
        
        # Fill diagonal with maximum frequency
        max_freq = corr_matrix.max().max()
        np.fill_diagonal(corr_matrix.values, max_freq)
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', fmt='.0f')
        plt.title('Mental Health Category Co-occurrence')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{reports_dir}/mental_health_correlations.png')
        plt.close()

    def analyze_sentiment_trajectory(self):
        """Analyze how sentiment changes over time for different mental health categories"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE r.timestamp IS NOT NULL
            RETURN m.name as category, 
                   r.sentiment as sentiment,
                   toString(r.timestamp) as timestamp_str,
                   r.score as score
            ORDER BY r.timestamp
            """
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            
            if df.empty:
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp_str'])
            return df

    def analyze_category_shifts(self):
        """Analyze how interest in different mental health categories shifts over time"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE r.timestamp IS NOT NULL
            WITH m.name as category, 
                 toString(r.timestamp) as timestamp_str
            ORDER BY r.timestamp
            WITH category, timestamp_str, count(*) as frequency
            RETURN category, collect(timestamp_str) as timestamps, 
                   collect(frequency) as frequencies
            ORDER BY size(timestamps) DESC
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def analyze_content_category_correlation(self):
        """Analyze correlation between video categories and mental health aspects"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE v.primary_category IS NOT NULL
            RETURN v.primary_category as video_category,
                   m.name as mental_health_category,
                   avg(r.score) as avg_score,
                   count(*) as count
            ORDER BY count DESC
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def create_personal_mental_health_index(self):
        """Create a composite mental health index over time"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE r.timestamp IS NOT NULL
            WITH toString(r.timestamp) as timestamp_str, 
                 avg(r.score) as avg_score,
                 count(distinct m.name) as category_diversity,
                 sum(CASE WHEN r.sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as positive_count,
                 sum(CASE WHEN r.sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as negative_count,
                 count(*) as total
            RETURN timestamp_str,
                   avg_score,
                   category_diversity,
                   toFloat(positive_count)/total as positive_ratio
            ORDER BY timestamp_str
            """
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            
            if df.empty:
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp_str'])
            
            # Create composite score - can be adjusted as needed
            df['mental_health_index'] = (
                df['avg_score'] * 0.4 + 
                df['category_diversity'] * 0.3 + 
                df['positive_ratio'] * 0.3
            )
            
            return df

    def plot_sentiment_changes(self, sentiment_data):
        """Plot sentiment changes over time for top categories"""
        if sentiment_data is None or sentiment_data.empty:
            logger.warning("No sentiment data available for plotting")
            return
        
        try:
            # Make sure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(sentiment_data['timestamp']):
                logger.warning("Timestamp column is not in datetime format")
                return
            
            # Find top categories
            category_counts = sentiment_data['category'].value_counts().head(5)
            if len(category_counts) == 0:
                logger.warning("No categories found for sentiment analysis")
                return
            
            top_categories = category_counts.index.tolist()
            
            # Prepare for plotting
            plt.figure(figsize=(15, 10))
            
            for category in top_categories:
                # Filter data for this category
                category_data = sentiment_data[sentiment_data['category'] == category].copy()
                
                if len(category_data) < 2:
                    logger.info(f"Not enough data points for category {category}")
                    continue
                
                # Create binary positive column (1 for positive, 0 for negative)
                category_data['positive'] = (category_data['sentiment'] == 'POSITIVE').astype(int)
                
                # Create weekly bins and calculate average positivity
                category_data['week'] = category_data['timestamp'].dt.to_period('W')
                weekly_positivity = category_data.groupby('week')['positive'].mean()
                
                # Convert period index to datetime for plotting
                week_dates = weekly_positivity.index.to_timestamp()
                
                if len(weekly_positivity) > 0:
                    plt.plot(week_dates, weekly_positivity.values, marker='o', label=category)
            
            plt.title('Weekly Positive Sentiment Ratio by Category')
            plt.xlabel('Date')
            plt.ylabel('Positive Sentiment Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig('sentiment_trajectory.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting sentiment changes: {str(e)}")
            # Add more detailed error information if needed
            logger.debug(f"Sentiment data columns: {sentiment_data.columns.tolist()}")
            logger.debug(f"Sentiment data types: {sentiment_data.dtypes}")

    def plot_mental_health_index(self, mh_index_data):
        """Plot mental health index over time"""
        if mh_index_data is None or mh_index_data.empty:
            return
        
        plt.figure(figsize=(15, 8))
        
        # Plot mental health index
        plt.subplot(2, 1, 1)
        plt.plot(mh_index_data['timestamp'], mh_index_data['mental_health_index'], 
                 marker='o', linewidth=2, color='blue')
        plt.title('Mental Health Index Over Time')
        plt.ylabel('Index Value')
        plt.grid(True, alpha=0.3)
        
        # Plot components
        plt.subplot(2, 1, 2)
        plt.plot(mh_index_data['timestamp'], mh_index_data['avg_score'], 
                 label='Average Score', marker='.')
        plt.plot(mh_index_data['timestamp'], mh_index_data['category_diversity'], 
                 label='Category Diversity', marker='.')
        plt.plot(mh_index_data['timestamp'], mh_index_data['positive_ratio'], 
                 label='Positive Ratio', marker='.')
        plt.xlabel('Date')
        plt.ylabel('Component Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mental_health_index.png')
        plt.close()

    def analyze_viewing_patterns(self):
        """Analyze potentially problematic viewing patterns (binge watching, late night, etc.)"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)
            WHERE v.watched_at IS NOT NULL
            WITH toString(v.watched_at) as watched_time_str, v
            
            // Extract hour and date using substring to avoid datetime operations
            WITH 
                watched_time_str,
                v.video_id as video_id,
                v.title as title,
                v.primary_category as category
            
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
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            
            if df.empty:
                return None
            
            df['date'] = pd.to_datetime(df['view_date'])
            return df

    def analyze_music_impact(self):
        """Analyze impact of music videos on mental health"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE v.primary_category = 'Music' OR v.primary_category = 'Entertainment'
            RETURN 
                v.video_id as video_id,
                v.title as title,
                v.primary_category as category,
                collect(DISTINCT m.name) as mental_health_aspects,
                avg(r.score) as avg_score,
                collect(DISTINCT r.sentiment) as sentiments
            ORDER BY avg_score DESC
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def analyze_content_addiction(self):
        """Analyze potential content addiction patterns"""
        with self.driver.session() as session:
            # Find categories with repeated high consumption and negative mental health scores
            query = """
            MATCH (v:Video)
            WHERE v.watched_at IS NOT NULL
            WITH v.primary_category as category, count(*) as view_count
            WHERE view_count > 5
            
            MATCH (v:Video)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE v.primary_category = category AND m.name IN ['Anxiety', 'Depression', 'Stress']
            WITH category, view_count, avg(r.score) as avg_negative_score, count(r) as negative_aspects
            
            RETURN 
                category, 
                view_count,
                avg_negative_score,
                negative_aspects,
                (view_count * 0.4) + (avg_negative_score * 0.6) as addiction_risk_score
            ORDER BY addiction_risk_score DESC
            LIMIT 10
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def plot_viewing_habits(self, pattern_data):
        """Plot potentially concerning viewing pattern data"""
        if pattern_data is None or pattern_data.empty:
            logger.warning("No viewing pattern data available")
            return
        
        try:
            plt.figure(figsize=(15, 12))
            
            # Plot 1: Videos watched per day
            plt.subplot(3, 1, 1)
            plt.bar(pattern_data['date'], pattern_data['videos_per_day'], color='steelblue')
            plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='High consumption threshold')
            plt.title('Daily Video Consumption')
            plt.ylabel('Videos per day')
            plt.legend()
            
            # Plot 2: Late night viewing
            plt.subplot(3, 1, 2)
            plt.bar(pattern_data['date'], pattern_data['late_night_count'], color='purple')
            plt.title('Late Night Viewing (10PM-4AM)')
            plt.ylabel('Late night videos')
            
            # Plot 3: Binge watching days
            plt.subplot(3, 1, 3)
            binge_days = pattern_data[pattern_data['binge_day'] == True]['date']
            plt.scatter(binge_days, [1] * len(binge_days), color='red', s=100, marker='x')
            plt.yticks([])
            plt.title('Binge Watching Days')
            
            plt.tight_layout()
            plt.savefig('viewing_patterns.png')
            plt.close()
            
            # Create a focused analysis on the worst week
            weekly_consumption = pattern_data.set_index('date').resample('W')['videos_per_day'].sum()
            worst_week_start = weekly_consumption.idxmax() - pd.Timedelta(days=7)
            worst_week_end = weekly_consumption.idxmax()
            
            # Filter to just the worst week
            worst_week = pattern_data[(pattern_data['date'] >= worst_week_start) & 
                                      (pattern_data['date'] <= worst_week_end)]
            
            if not worst_week.empty:
                plt.figure(figsize=(10, 6))
                plt.bar(worst_week['date'], worst_week['videos_per_day'])
                plt.title(f'Highest Consumption Week ({worst_week_start.date()} to {worst_week_end.date()})')
                plt.ylabel('Videos per day')
                plt.savefig('highest_consumption_week.png')
                plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting viewing habits: {str(e)}")
        
    def analyze_category_addiction_risk(self):
        """Analyze which video categories might present addiction risk"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)
            WHERE v.watched_at IS NOT NULL
            
            // Keep timestamps as DateTime objects for duration calculations
            WITH v.watched_at as watched_time, v
            
            // Find viewing sessions (videos watched within 30 min of each other)
            WITH watched_time, v ORDER BY watched_time
            WITH collect({time: watched_time, video: v}) as ordered_views
            UNWIND range(0, size(ordered_views)-2) as i
            WITH 
                ordered_views[i].time as current_time,
                ordered_views[i+1].time as next_time,
                ordered_views[i].video as current_video,
                ordered_views[i+1].video as next_video
            
            // Calculate minutes between views using datetime arithmetic
            WITH 
                current_video, next_video,
                duration.between(current_time, next_time).minutes as time_diff
            WHERE time_diff <= 30
            
            // Analysis of content that keeps viewers watching
            WITH current_video, next_video
            WHERE current_video.primary_category = next_video.primary_category
            
            RETURN 
                current_video.primary_category as category,
                count(*) as continuous_viewing_count
            ORDER BY continuous_viewing_count DESC
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def save_analysis_results(self, result_data, analysis_name, formats=None, run_dir=None):
        """Save analysis results to files in various formats"""
        # Default to CSV and JSON if not specified
        if formats is None:
            formats = ['csv', 'json']
        
        # Use provided run directory or default to general reports directory
        if run_dir is None:
            reports_dir = 'analysis_reports'
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        else:
            reports_dir = run_dir
            timestamp = os.path.basename(run_dir)  # Use the folder name as timestamp
        
        # Generate filename base without additional timestamp in the name
        base_filename = f"{reports_dir}/{analysis_name}"
        
        # Custom JSON encoder to handle pandas Timestamp objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (pd.Timestamp, pd.Period, datetime)):
                    return obj.isoformat()
                elif pd.isna(obj):
                    return None
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError:
                    return str(obj)
        
        # Handle different data types
        if isinstance(result_data, pd.DataFrame):
            if 'csv' in formats:
                csv_path = f"{base_filename}.csv"
                result_data.to_csv(csv_path, index=True)
                logger.info(f"Saved {analysis_name} to {csv_path}")
                
            if 'json' in formats:
                json_path = f"{base_filename}.json"
                # Convert to dict first, then use custom encoder
                if isinstance(result_data.index, pd.MultiIndex):
                    result_data_dict = result_data.reset_index().to_dict(orient='records')
                else:
                    result_data_dict = result_data.to_dict(orient='records')
                
                with open(json_path, 'w') as f:
                    json.dump(result_data_dict, f, indent=2, cls=DateTimeEncoder)
                logger.info(f"Saved {analysis_name} to {json_path}")
                
        elif isinstance(result_data, dict):
            if 'json' in formats:
                json_path = f"{base_filename}.json"
                
                # Process dict contents for JSON serialization
                serializable_data = {}
                for key, value in result_data.items():
                    if isinstance(value, pd.DataFrame):
                        if isinstance(value.index, pd.MultiIndex):
                            serializable_data[key] = value.reset_index().to_dict(orient='records')
                        else:
                            serializable_data[key] = value.to_dict(orient='records')
                    elif isinstance(value, pd.Series):
                        if isinstance(value.index, pd.MultiIndex):
                            serializable_data[key] = value.reset_index().to_dict(orient='records')
                        else:
                            serializable_data[key] = value.to_dict()
                    else:
                        serializable_data[key] = value
                
                with open(json_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2, cls=DateTimeEncoder)
                    
                logger.info(f"Saved {analysis_name} to {json_path}")
        
        return base_filename

    def analyze_title_patterns(self):
        """Analyze video title patterns for signs of addiction-promoting content"""
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)
            WHERE v.title IS NOT NULL
            
            // Extract keywords that might indicate clickbait/addictive content
            WITH v, 
                 CASE 
                     WHEN toLower(v.title) CONTAINS "shocking" OR 
                          toLower(v.title) CONTAINS "extreme" OR 
                          toLower(v.title) CONTAINS "must see" OR
                          toLower(v.title) CONTAINS "never before" OR
                          toLower(v.title) CONTAINS "you won't believe" OR
                          toLower(v.title) CONTAINS "mind blowing" OR
                          toLower(v.title) CONTAINS "insane" THEN true
                     ELSE false
                 END as contains_clickbait,
                 CASE 
                     WHEN v.title CONTAINS "!" THEN size([c in v.title WHERE c = "!"])
                     ELSE 0
                 END as exclamation_count,
                 CASE 
                     WHEN toLower(v.title) CONTAINS "part" OR 
                          toLower(v.title) CONTAINS "ep" OR 
                          toLower(v.title) CONTAINS "episode" OR
                          (toLower(v.title) =~ ".*#\\d+.*") THEN true
                     ELSE false
                 END as is_series
                 
            // Get watch statistics and mental health impact
            OPTIONAL MATCH (v)-[r:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            WHERE m.name IN ["Anxiety", "Addiction", "Depression"]
            
            WITH v, contains_clickbait, exclamation_count, is_series,
                 COUNT(r) > 0 as has_negative_impact,
                 COLLECT(DISTINCT m.name) as negative_aspects
                 
            RETURN v.video_id as video_id,
                   v.title as title,
                   v.primary_category as category,
                   contains_clickbait,
                   exclamation_count,
                   is_series,
                   has_negative_impact,
                   negative_aspects
            ORDER BY exclamation_count DESC, contains_clickbait DESC
            LIMIT 100
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def analyze_music_details(self):
        """Detailed analysis of music content and its impact"""
        with self.driver.session() as session:
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
            
            // Get watch time patterns
            // Using string functions instead of APOC
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
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def analyze_binge_triggers(self):
        """Analyze what content tends to trigger binge watching sessions"""
        with self.driver.session() as session:
            query = """
            MATCH (v1:Video)
            WHERE v1.watched_at IS NOT NULL
            
            // Find videos that start binge sessions
            WITH v1 ORDER BY v1.watched_at
            WITH collect(v1) as videos
            UNWIND range(0, size(videos)-4) as i
            
            // Get consecutive videos (4 or more videos in a short time = binge)
            WITH 
                videos[i] as first_video,
                videos[i+1] as second_video,
                videos[i+2] as third_video,
                videos[i+3] as fourth_video
            WHERE duration.between(first_video.watched_at, fourth_video.watched_at).hours < 3
            
            // Analyze what type of video started the binge
            WITH first_video
            
            MATCH (first_video)-[:HAS_MENTAL_HEALTH_ASPECT]->(m:MentalHealthCategory)
            
            RETURN first_video.video_id as trigger_video_id,
                   first_video.title as trigger_title,
                   first_video.primary_category as trigger_category,
                   collect(DISTINCT m.name) as mental_health_aspects,
                   count(*) as binge_trigger_count
            ORDER BY binge_trigger_count DESC
            """
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])

    def analyze_description_keywords(self, top_n=100):
        """Analyze video descriptions for keywords that might indicate addiction potential"""
        with self.driver.session() as session:
            # First get videos with descriptions
            query = """
            MATCH (v:Video)
            WHERE v.description IS NOT NULL
            RETURN v.video_id as video_id, v.title as title, v.description as description
            LIMIT $top_n
            """
            result = session.run(query, top_n=top_n)
            videos = pd.DataFrame([dict(record) for record in result])
            
            if videos.empty:
                return None
            
            # Process descriptions to extract keywords
            import re
            from collections import Counter
            
            # Keywords associated with addictive content
            addiction_keywords = [
                'subscribe', 'like', 'follow', 'share', 'join', 'notification',
                'viral', 'trending', 'exclusive', 'secret', 'revealed', 'expose',
                'challenge', 'giveaway', 'free', 'limited', 'official', 'premiere',
                'new', 'series', 'playlist', 'binge', 'marathon'
            ]
            
            # Process descriptions
            all_words = []
            addiction_counts = {}
            
            for _, row in videos.iterrows():
                # Clean text
                desc = str(row['description']).lower()
                desc = re.sub(r'[^\w\s]', ' ', desc)
                words = desc.split()
                
                # Count addiction keywords
                video_addiction_count = sum(1 for word in words if word in addiction_keywords)
                addiction_counts[row['video_id']] = video_addiction_count
                
                # Add to word collection
                all_words.extend(words)
            
            # Get most common words
            word_counts = Counter(all_words)
            common_words = word_counts.most_common(50)
            
            # Create results DataFrame
            videos['addiction_keyword_count'] = videos['video_id'].map(addiction_counts)
            
            # Add keyword frequency metrics
            return {
                'videos': videos.sort_values('addiction_keyword_count', ascending=False),
                'common_words': pd.DataFrame(common_words, columns=['word', 'count']),
                'addiction_keywords_found': [word for word, _ in common_words if word in addiction_keywords]
            }

    def classify_music_impact(self, music_data):
        """Classifies music content by its mental health impact"""
        if music_data is None or music_data.empty:
            return None
        
        # Process mental health impacts
        def extract_impact(impacts):
            if not impacts or len(impacts) == 0:
                return 'Unknown', 0, 'Unknown'
            
            # Get the most significant impact
            max_score = 0
            main_category = None
            sentiment = 'Unknown'
            
            for impact in impacts:
                if isinstance(impact, dict) and 'category' in impact and 'score' in impact:
                    if 'score' in impact and (main_category is None or abs(impact['score']) > abs(max_score)):
                        max_score = impact['score']
                        main_category = impact['category']
                        sentiment = impact.get('sentiment', 'Unknown')
            
            return main_category, max_score, sentiment
        
        # Apply categorization
        music_data['primary_impact'] = music_data['mental_health_impacts'].apply(
            lambda x: extract_impact(x)[0] if x else 'Unknown'
        )
        music_data['impact_score'] = music_data['mental_health_impacts'].apply(
            lambda x: extract_impact(x)[1] if x else 0
        )
        music_data['impact_sentiment'] = music_data['mental_health_impacts'].apply(
            lambda x: extract_impact(x)[2] if x else 'Unknown'
        )
        
        # Create summary by genre and energy type
        genre_impact = music_data.groupby('music_genre').agg({
            'video_id': 'count',
            'impact_score': 'mean',
            'primary_impact': lambda x: pd.Series.mode(x)[0] if not x.mode().empty else 'Unknown',
            'energy_type': lambda x: pd.Series.mode(x)[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        energy_impact = music_data.groupby('energy_type').agg({
            'video_id': 'count',
            'impact_score': 'mean',
            'primary_impact': lambda x: pd.Series.mode(x)[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        # Time of day patterns
        music_data['watch_hour'] = pd.to_numeric(music_data['watch_hour'], errors='coerce')
        time_patterns = music_data.dropna(subset=['watch_hour']).groupby(
            pd.cut(music_data['watch_hour'], 
                  bins=[0, 6, 12, 18, 24],
                  labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'])
        ).agg({
            'video_id': 'count',
            'energy_type': lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'Unknown'
        }).reset_index().rename(columns={'watch_hour': 'time_of_day'})
        
        return {
            'music_detail': music_data,
            'genre_impact': genre_impact,
            'energy_impact': energy_impact,
            'time_patterns': time_patterns
        }

    def plot_music_impact(self, music_analysis):
        """Plot music impact by genre and energy type"""
        if music_analysis is None or 'genre_impact' not in music_analysis:
            return
        
        genre_impact = music_analysis['genre_impact']
        energy_impact = music_analysis['energy_impact']
        
        # Plot genre impact
        if len(genre_impact) > 0:
            plt.figure(figsize=(12, 8))
            
            # Create bar colors based on impact score
            colors = ['red' if score < 0 else 'green' for score in genre_impact['impact_score']]
            
            bars = plt.bar(genre_impact['music_genre'], genre_impact['impact_score'], color=colors)
            plt.title('Music Genre Impact on Mental Health')
            plt.xlabel('Genre')
            plt.ylabel('Impact Score (negative = worse)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add video count annotations
            for bar, count in zip(bars, genre_impact['video_id']):
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.1 if bar.get_height() >= 0 else bar.get_height() - 0.3, 
                        f"n={count}", 
                        ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            plt.savefig('music_genre_impact.png')
            plt.close()
        
        # Plot energy type impact
        if len(energy_impact) > 0:
            plt.figure(figsize=(12, 8))
            
            # Create bar colors based on impact score
            colors = ['red' if score < 0 else 'green' for score in energy_impact['impact_score']]
            
            bars = plt.bar(energy_impact['energy_type'], energy_impact['impact_score'], color=colors)
            plt.title('Music Energy Type Impact on Mental Health')
            plt.xlabel('Energy Type')
            plt.ylabel('Impact Score (negative = worse)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add video count annotations
            for bar, count in zip(bars, energy_impact['video_id']):
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.1 if bar.get_height() >= 0 else bar.get_height() - 0.3, 
                        f"n={count}", 
                        ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            plt.savefig('music_energy_impact.png')
            plt.close()
        
        # Plot time of day patterns
        if 'time_patterns' in music_analysis and len(music_analysis['time_patterns']) > 0:
            time_patterns = music_analysis['time_patterns']
            
            plt.figure(figsize=(10, 6))
            plt.bar(time_patterns['time_of_day'], time_patterns['video_id'])
            plt.title('Music Consumption by Time of Day')
            plt.xlabel('Time of Day')
            plt.ylabel('Number of Videos')
            
            # Add energy type annotations
            for i, (_, row) in enumerate(time_patterns.iterrows()):
                plt.text(i, row['video_id'] + 0.5, f"{row['energy_type']}", 
                        ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            plt.savefig('music_time_patterns.png')
            plt.close()

    def plot_addiction_risk(self, addiction_data):
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
        
        # Plot 2: Risk factors distribution
        if 'avg_negative_score' in addiction_data.columns and 'view_count' in addiction_data.columns:
            plt.figure(figsize=(10, 8))
            plt.scatter(addiction_data['view_count'], 
                       addiction_data['avg_negative_score'], 
                       c=addiction_data['addiction_risk_score'],
                       cmap='RdYlGn_r', 
                       alpha=0.7,
                       s=100)
            
            plt.colorbar(label='Addiction Risk Score')
            plt.title('Content Addiction Risk Factors')
            plt.xlabel('View Count')
            plt.ylabel('Negative Mental Health Impact')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{reports_dir}/addiction_risk_factors.png')
            plt.close()

    def plot_binge_patterns(self, viewing_patterns):
        """Create visualizations for binge watching patterns"""
        if viewing_patterns is None or viewing_patterns.empty:
            return
        
        # Create reports directory if it doesn't exist
        reports_dir = 'analysis_reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        # Plot 1: Daily viewing counts with binge days highlighted
        plt.figure(figsize=(15, 8))
        
        # Sort by date for proper timeline
        data = viewing_patterns.sort_values('date')
        
        # Create bars with different colors for binge days
        colors = ['red' if is_binge else 'skyblue' for is_binge in data['binge_day']]
        plt.bar(data['date'], data['videos_per_day'], color=colors)
        
        # Add a horizontal line for the binge threshold
        plt.axhline(y=15, color='red', linestyle='--', alpha=0.7)
        plt.text(data['date'].iloc[0], 15.5, 'Binge Threshold (15+ videos/day)', color='red')
        
        plt.title('Daily Video Consumption with Binge Days Highlighted')
        plt.xlabel('Date')
        plt.ylabel('Videos per Day')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{reports_dir}/daily_viewing_binge_pattern.png')
        plt.close()
        
        # Plot 2: Late night viewing patterns
        if 'late_night_count' in data.columns:
            plt.figure(figsize=(15, 8))
            
            # Create a line chart of late night viewing
            plt.plot(data['date'], data['late_night_count'], 'o-', color='purple', label='Late Night Videos')
            
            # Add overall viewing as a reference
            plt.plot(data['date'], data['videos_per_day'], 'o-', color='gray', alpha=0.3, label='Total Videos')
            
            # Calculate and plot 7-day moving average
            data['late_night_ma'] = data['late_night_count'].rolling(7, min_periods=1).mean()
            plt.plot(data['date'], data['late_night_ma'], '-', color='darkred', linewidth=2, 
                    label='7-Day Moving Avg (Late Night)')
            
            plt.title('Late Night Viewing Patterns (10PM-4AM)')
            plt.xlabel('Date')
            plt.ylabel('Number of Videos')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{reports_dir}/late_night_viewing_pattern.png')
            plt.close()
            
            # Plot 3: Calendar heatmap of viewing intensity
            if len(data) >= 28:  # Only if we have enough data
                # Create calendar data
                data['yearmonth'] = data['date'].dt.strftime('%Y-%m')
                data['day'] = data['date'].dt.day
                data['weekday'] = data['date'].dt.weekday
                
                # Get unique year-months
                year_months = sorted(data['yearmonth'].unique())
                
                fig, axes = plt.subplots(len(year_months), 1, figsize=(15, 3*len(year_months)))
                if len(year_months) == 1:
                    axes = [axes]
                    
                for i, ym in enumerate(year_months):
                    month_data = data[data['yearmonth'] == ym].copy()
                    
                    # Create calendar grid with NaN values
                    calendar_data = np.zeros((7, 31)) * np.nan
                    
                    # Fill with video counts
                    for _, row in month_data.iterrows():
                        weekday = row['weekday']
                        day = row['day'] - 1  # 0-based index
                        calendar_data[weekday, day] = row['videos_per_day']
                    
                    # Plot heatmap
                    ax = axes[i]
                    im = ax.imshow(calendar_data, cmap='YlOrRd', vmin=0, vmax=data['videos_per_day'].max())
                    
                    # Set labels
                    ax.set_title(f'Viewing Intensity: {ym}')
                    ax.set_yticks(range(7))
                    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                    ax.set_xticks(range(0, 31, 1))
                    ax.set_xticklabels(range(1, 32, 1))
                    
                    # Add text annotations for binge days
                    for _, row in month_data.iterrows():
                        if row['binge_day']:
                            weekday = row['weekday']
                            day = row['day'] - 1
                            ax.text(day, weekday, 'B', ha='center', va='center', 
                                   color='white', fontweight='bold')
                
                fig.tight_layout()
                fig.colorbar(im, ax=axes, label='Videos per Day')
                plt.savefig(f'{reports_dir}/viewing_calendar_heatmap.png')
                plt.close()

    def plot_mental_health_dashboard(self, mh_index, sentiment_data, viewing_patterns):
        """Create a comprehensive mental health dashboard"""
        if mh_index is None or sentiment_data is None or viewing_patterns is None:
            return
        
        # Create reports directory if it doesn't exist
        reports_dir = 'analysis_reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        # Create a 2x2 dashboard
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Mental Health Index Trend
        ax1 = axes[0, 0]
        mh_index.plot(x='timestamp', y='mental_health_index', ax=ax1, marker='o', color='blue')
        
        # Add a trend line
        z = np.polyfit(range(len(mh_index)), mh_index['mental_health_index'], 1)
        p = np.poly1d(z)
        ax1.plot(mh_index['timestamp'], p(range(len(mh_index))), "r--", linewidth=2)
        
        ax1.set_title('Mental Health Index Trend', fontsize=14)
        ax1.set_ylabel('Index Value')
        ax1.set_xlabel('')
        ax1.grid(True, alpha=0.3)
        
        # Add improvement or decline annotation
        if len(mh_index) > 1:
            first_half = mh_index.iloc[:len(mh_index)//2]['mental_health_index'].mean()
            second_half = mh_index.iloc[len(mh_index)//2:]['mental_health_index'].mean()
            change = second_half - first_half
            trend = "improving" if change > 0 else "declining"
            ax1.annotate(f"Mental health is {trend}\nChange: {change:.2f}", 
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # 2. Sentiment Distribution
        ax2 = axes[0, 1]
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        sentiment_counts.plot.pie(autopct='%1.1f%%', ax=ax2, colors=['green', 'red', 'gray'])
        ax2.set_title('Sentiment Distribution', fontsize=14)
        ax2.set_ylabel('')
        
        # 3. Viewing Patterns and Mental Health
        ax3 = axes[1, 0]
        
        # Prepare data: resample to weekly
        viewing_weekly = viewing_patterns.set_index('date')['videos_per_day'].resample('W').mean()
        
        # Get weekly mental health score if available
        mh_weekly = mh_index.set_index('timestamp')['mental_health_index'].resample('W').mean()
        
        # Align the series
        common_idx = viewing_weekly.index.intersection(mh_weekly.index)
        
        if len(common_idx) > 0:
            viewing_aligned = viewing_weekly[common_idx]
            mh_aligned = mh_weekly[common_idx]
            
            # Create dual axis plot
            color1, color2 = 'blue', 'red'
            
            # Primary axis: viewing count
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Avg Videos per Day', color=color1)
            ax3.plot(common_idx, viewing_aligned, color=color1, marker='o', label='Videos per Day')
            ax3.tick_params(axis='y', labelcolor=color1)
            
            # Secondary axis: mental health
            ax4 = ax3.twinx()
            ax4.set_ylabel('Mental Health Index', color=color2)
            ax4.plot(common_idx, mh_aligned, color=color2, marker='s', label='Mental Health')
            ax4.tick_params(axis='y', labelcolor=color2)
            
            # Add legend
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax4.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax3.set_title('Weekly Viewing vs Mental Health', fontsize=14)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Not enough data to compare', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Weekly Viewing vs Mental Health', fontsize=14)
        
        # 4. Category Impact Heatmap
        ax4 = axes[1, 1]
        
        # Get top categories
        top_cats = sentiment_data.groupby('category')['score'].count().sort_values(ascending=False).head(8).index.tolist()
        category_data = sentiment_data[sentiment_data['category'].isin(top_cats)]
        
        # Compute impact scores (average score by category)
        impact_scores = category_data.groupby('category')['score'].mean().sort_values()
        
        # Create horizontal bar chart
        bars = ax4.barh(impact_scores.index, impact_scores.values, height=0.7)
        
        # Color bars by impact (red for negative, green for positive)
        for bar, value in zip(bars, impact_scores.values):
            if value < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax4.set_title('Mental Health Impact by Category', fontsize=14)
        ax4.set_xlabel('Average Impact Score')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{reports_dir}/mental_health_dashboard.png')
        plt.close()

def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    
    # Create a timestamp for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create base reports directory if it doesn't exist
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Create timestamp subfolder for this run
    run_dir = os.path.join(reports_dir, run_timestamp)
    os.makedirs(run_dir)
    
    # Create analyzer instance
    analyzer = MentalHealthAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Get temporal trends
        logger.info("Analyzing temporal trends...")
        temporal_analyses = analyzer.get_temporal_trends()
        if temporal_analyses:
            analyzer.plot_temporal_trends(temporal_analyses, run_dir)
            analyzer.save_analysis_results(temporal_analyses, "temporal_trends", run_dir=run_dir)
        
        # Get sentiment trajectory
        logger.info("Analyzing sentiment trajectory...")
        sentiment_data = analyzer.analyze_sentiment_trajectory()
        if sentiment_data is not None:
            analyzer.plot_sentiment_changes(sentiment_data)
            analyzer.save_analysis_results(sentiment_data, "sentiment_trajectory", run_dir=run_dir)
            
        # Get category shifts
        logger.info("Analyzing category shifts...")
        category_shifts = analyzer.analyze_category_shifts()
        if category_shifts is not None and not category_shifts.empty:
            analyzer.save_analysis_results(category_shifts, "category_shifts", run_dir=run_dir)
        
        # Get content correlations
        logger.info("Analyzing content-category correlations...")
        content_correlations = analyzer.analyze_content_category_correlation()
        if content_correlations is not None and not content_correlations.empty:
            analyzer.save_analysis_results(content_correlations, "content_correlations", run_dir=run_dir)
        
        # Create mental health index
        logger.info("Creating mental health index...")
        mh_index = analyzer.create_personal_mental_health_index()
        if mh_index is not None:
            analyzer.plot_mental_health_index(mh_index)
            analyzer.save_analysis_results(mh_index, "mental_health_index", run_dir=run_dir)
        
        # Analyze category relationships
        logger.info("Analyzing category relationships...")
        category_relationships = analyzer.analyze_category_relationships()
        if not category_relationships.empty:
            analyzer.plot_category_correlations(category_relationships)
            analyzer.save_analysis_results(category_relationships, "category_relationships", run_dir=run_dir)
        
        # Analyze engagement impact
        logger.info("Analyzing engagement impact...")
        engagement_impact = analyzer.analyze_engagement_impact()
        if engagement_impact is not None and not engagement_impact.empty:
            analyzer.save_analysis_results(engagement_impact, "engagement_impact", run_dir=run_dir)
        
        # Analyze viewing patterns
        logger.info("Analyzing viewing patterns...")
        viewing_patterns = analyzer.analyze_viewing_patterns()
        if viewing_patterns is not None:
            analyzer.plot_viewing_habits(viewing_patterns)
            analyzer.save_analysis_results(viewing_patterns, "viewing_patterns", run_dir=run_dir)
        
        # Analyze title patterns for addictive content
        logger.info("Analyzing title patterns for addictive content...")
        title_patterns = analyzer.analyze_title_patterns()
        if title_patterns is not None and not title_patterns.empty:
            analyzer.save_analysis_results(title_patterns, "title_addiction_patterns", run_dir=run_dir)
            
        # Analyze binge triggers
        logger.info("Analyzing binge triggers...")
        binge_triggers = analyzer.analyze_binge_triggers()
        if binge_triggers is not None and not binge_triggers.empty:
            analyzer.save_analysis_results(binge_triggers, "binge_triggers", run_dir=run_dir)
        
        # Analyze description keywords
        logger.info("Analyzing description keywords...")
        description_analysis = analyzer.analyze_description_keywords()
        if description_analysis is not None and 'videos' in description_analysis:
            analyzer.save_analysis_results(description_analysis['videos'], "description_keyword_analysis", run_dir=run_dir)
            analyzer.save_analysis_results(description_analysis['common_words'], "common_description_words", run_dir=run_dir)
        
        # Detailed music analysis
        logger.info("Analyzing detailed music impact...")
        music_details = analyzer.analyze_music_details()
        if music_details is not None and not music_details.empty:
            analyzer.save_analysis_results(music_details, "music_details", run_dir=run_dir)
            
            music_analysis = analyzer.classify_music_impact(music_details)
            analyzer.plot_music_impact(music_analysis)
            
            if music_analysis is not None and 'music_detail' in music_analysis:
                analyzer.save_analysis_results(music_analysis['music_detail'], "music_detailed_analysis", run_dir=run_dir)
                analyzer.save_analysis_results(music_analysis['genre_impact'], "music_genre_impact", run_dir=run_dir)
                analyzer.save_analysis_results(music_analysis['energy_impact'], "music_energy_impact", run_dir=run_dir)
        
        # Analyze content addiction patterns
        logger.info("Analyzing content addiction patterns...")
        content_addiction = analyzer.analyze_content_addiction()
        if content_addiction is not None and not content_addiction.empty:
            analyzer.save_analysis_results(content_addiction, "content_addiction", run_dir=run_dir)
        
        # Analyze category addiction risk
        logger.info("Analyzing category addiction risk...")
        category_addiction = analyzer.analyze_category_addiction_risk()
        if category_addiction is not None and not category_addiction.empty:
            analyzer.save_analysis_results(category_addiction, "category_addiction_risk", run_dir=run_dir)
        
        # Generate summary report
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{run_dir}/analysis_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("===== YouTube Mental Health Analysis Summary =====\n\n")
            f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write temporal trend info
            if temporal_analyses:
                f.write("=== Temporal Trend Analysis ===\n")
                for freq, data in temporal_analyses.items():
                    if not data.empty:
                        f.write(f"{freq.capitalize()} trends: {len(data)} data points\n")
                f.write("\n")
            
            # Write mental health index info
            if mh_index is not None and not mh_index.empty:
                f.write("=== Mental Health Index ===\n")
                stats = mh_index['mental_health_index'].describe()
                f.write(f"Average index: {stats['mean']:.2f}\n")
                
                if len(mh_index) > 1:
                    first_half = mh_index.iloc[:len(mh_index)//2]['mental_health_index'].mean()
                    second_half = mh_index.iloc[len(mh_index)//2:]['mental_health_index'].mean()
                    change = second_half - first_half
                    trend = "improving" if change > 0 else "declining"
                    f.write(f"Overall trend: Mental health appears to be {trend} (change: {change:.2f})\n")
                f.write("\n")
            
            # Write viewing pattern info
            if viewing_patterns is not None and not viewing_patterns.empty:
                f.write("=== Viewing Pattern Analysis ===\n")
                f.write(f"Total days analyzed: {len(viewing_patterns)}\n")
                f.write(f"Average videos per day: {viewing_patterns['videos_per_day'].mean():.1f}\n")
                f.write(f"Late night viewing days: {(viewing_patterns['late_night_count'] > 0).sum()}\n")
                f.write(f"Binge watching days: {viewing_patterns['binge_day'].sum()}\n")
                
                if viewing_patterns['binge_day'].sum() > 0:
                    binge_pct = (viewing_patterns['binge_day'].sum() / len(viewing_patterns)) * 100
                    f.write(f"Binge watching occurred on {binge_pct:.1f}% of days\n")
                f.write("\n")
            
            # Write music impact info
            if music_analysis is not None and 'genre_impact' in music_analysis:
                f.write("=== Music Impact Analysis ===\n")
                genre_impact = music_analysis['genre_impact']
                
                f.write("Top music genres by impact:\n")
                sorted_genres = genre_impact.sort_values('impact_score', ascending=False)
                for _, row in sorted_genres.head(3).iterrows():
                    impact_dir = "positive" if row['impact_score'] > 0 else "negative"
                    f.write(f"- {row['music_genre']}: {impact_dir} impact ({row['impact_score']:.2f})\n")
                
                f.write("\nMusic energy types by impact:\n")
                energy_impact = music_analysis['energy_impact']
                sorted_energy = energy_impact.sort_values('impact_score', ascending=False)
                for _, row in sorted_energy.head(3).iterrows():
                    impact_dir = "positive" if row['impact_score'] > 0 else "negative"
                    f.write(f"- {row['energy_type']}: {impact_dir} impact ({row['impact_score']:.2f})\n")
                f.write("\n")
            
            # Write addiction risk info
            if title_patterns is not None and not title_patterns.empty:
                f.write("=== Addiction Risk Analysis ===\n")
                clickbait_count = title_patterns['contains_clickbait'].sum()
                clickbait_pct = clickbait_count/len(title_patterns)*100
                f.write(f"Videos with clickbait titles: {clickbait_count} ({clickbait_pct:.1f}%)\n")
                
                series_count = title_patterns['is_series'].sum()
                series_pct = series_count/len(title_patterns)*100
                f.write(f"Videos part of a series: {series_count} ({series_pct:.1f}%)\n")
                
                if binge_triggers is not None and not binge_triggers.empty:
                    f.write("\nTop binge trigger categories:\n")
                    trigger_counts = binge_triggers.groupby('trigger_category')['binge_trigger_count'].sum()
                    sorted_triggers = trigger_counts.sort_values(ascending=False)
                    for category, count in sorted_triggers.head(3).items():
                        f.write(f"- {category}: {count} instances\n")
                f.write("\n")
            
            f.write("===== End of Summary =====\n")
        
        logger.info(f"Analysis summary saved to {report_path}")
        
        # Print summary statistics to console
        print("\n===== YouTube Mental Health Analysis Results =====")
        
        # Print content correlations
        if content_correlations is not None and not content_correlations.empty:
            print("\nTop Content-Mental Health Correlations:")
            print(content_correlations.head())
            
        # Print mental health index
        if mh_index is not None and not mh_index.empty:
            print("\nMental Health Index Statistics:")
            print(mh_index[['mental_health_index']].describe())
            
            # Identify trends
            if len(mh_index) > 1:
                first_half = mh_index.iloc[:len(mh_index)//2]['mental_health_index'].mean()
                second_half = mh_index.iloc[len(mh_index)//2:]['mental_health_index'].mean()
                change = second_half - first_half
                trend = "improving" if change > 0 else "declining"
                print(f"\nOverall trend: Mental health appears to be {trend} (change: {change:.2f})")
        
        # Print viewing patterns
        if viewing_patterns is not None and not viewing_patterns.empty:
            print("\nViewing Pattern Analysis:")
            print(f"Total days analyzed: {len(viewing_patterns)}")
            print(f"Average videos per day: {viewing_patterns['videos_per_day'].mean():.1f}")
            print(f"Late night viewing days: {(viewing_patterns['late_night_count'] > 0).sum()}")
            print(f"Binge watching days: {viewing_patterns['binge_day'].sum()}")
            
            if viewing_patterns['binge_day'].sum() > 0:
                print("\nPotential problematic consumption patterns detected:")
                binge_pct = (viewing_patterns['binge_day'].sum() / len(viewing_patterns)) * 100
                print(f"Binge watching occurred on {binge_pct:.1f}% of days")
        
        # Print addiction insights        
        if title_patterns is not None and not title_patterns.empty:
            print("\nPotentially Addictive Content:")
            clickbait_count = title_patterns['contains_clickbait'].sum()
            print(f"Videos with clickbait titles: {clickbait_count} ({clickbait_count/len(title_patterns)*100:.1f}%)")
            series_count = title_patterns['is_series'].sum()
            print(f"Videos part of a series: {series_count} ({series_count/len(title_patterns)*100:.1f}%)")
            
        if binge_triggers is not None and not binge_triggers.empty:
            print("\nTop Binge Trigger Categories:")
            print(binge_triggers.groupby('trigger_category')['binge_trigger_count'].sum().sort_values(ascending=False).head())
            
        # Print music analysis
        if music_analysis is not None and 'genre_impact' in music_analysis:
            print("\nMusic Genre Mental Health Impact:")
            genre_impact = music_analysis['genre_impact']
            genre_impact['impact_direction'] = genre_impact['impact_score'].apply(lambda x: "Positive" if x > 0 else "Negative")
            print(genre_impact[['music_genre', 'impact_score', 'impact_direction', 'primary_impact']].head())
            
            print("\nMusic Energy Type Mental Health Impact:")
            energy_impact = music_analysis['energy_impact']
            energy_impact['impact_direction'] = energy_impact['impact_score'].apply(lambda x: "Positive" if x > 0 else "Negative")
            print(energy_impact[['energy_type', 'impact_score', 'impact_direction', 'primary_impact']].head())
            
        if description_analysis is not None and 'addiction_keywords_found' in description_analysis:
            print("\nAddiction-Related Keywords in Descriptions:")
            print(description_analysis['addiction_keywords_found'])
            
        print(f"\nFull analysis results saved to '{run_dir}' directory")
        print("========================================================")
        
        # Plot addiction risk
        analyzer.plot_addiction_risk(content_addiction)
        
        # Plot binge patterns
        analyzer.plot_binge_patterns(viewing_patterns)
        
        # Plot mental health dashboard
        analyzer.plot_mental_health_dashboard(mh_index, sentiment_data, viewing_patterns)
        
    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}")
        raise
    finally:
        analyzer.close()

if __name__ == "__main__":
    main() 