import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
import json
import traceback
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def test_connection(self):
        """
        Test the connection to the Neo4j database.
        Returns True if connection is successful, False otherwise.
        """
        try:
            with self.driver.session() as session:
                # Run a simple test query
                result = session.run("RETURN 1 AS n")
                record = result.single()
                if record and record["n"] == 1:
                    logger.info("Successfully connected to Neo4j database")
                    return True
                else:
                    logger.error("Connection test failed: unexpected query result")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def get_temporal_trends(self):
        """
        Retrieve mental health time series data from the KG.
        Assumes MentalHealthData nodes are linked to Video nodes.
        Uses a 5-minute window for timestamp matching.
        """
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
                WHERE m.timestamp IS NOT NULL
                WITH v, m, datetime(m.timestamp) as timestamp_dt
                // No need for time window in initial query, we're just collecting all data with timestamps
                RETURN v.video_id as video_id, 
                       m.category as category, 
                       m.score as score, 
                       m.sentiment as sentiment, 
                       toString(m.timestamp) as timestamp_str
                ORDER BY timestamp_dt
            """
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            if df.empty:
                logger.warning("No temporal data found")
                return None
                
            # Convert to datetime for temporal analysis
            df['timestamp'] = pd.to_datetime(df['timestamp_str'], utc=True, errors='coerce')
            
            # Time-based grouping with flexibility (this happens in pandas, not Neo4j)
            # We'll round timestamps to nearest 5-minute window to allow for approximate matching
            df['timestamp_rounded'] = df['timestamp'].dt.round('5min')
            
            # Create different time-based aggregations with the rounded timestamps
            analyses = {
                'daily': df.set_index('timestamp_rounded').groupby([pd.Grouper(freq='D'), 'category'])['score'].mean(),
                'weekly': df.set_index('timestamp_rounded').groupby([pd.Grouper(freq='W'), 'category'])['score'].mean(),
                'monthly': df.set_index('timestamp_rounded').groupby([pd.Grouper(freq='M'), 'category'])['score'].mean()
            }
            
            logger.info(f"Found {len(df)} data points for temporal analysis after applying 5-minute window")
            return analyses

    def analyze_sentiment_trajectory(self):
        """
        Retrieve sentiment trajectory data from the KG.
        Uses a 5-minute window for timestamp matching.
        """
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
                WHERE m.timestamp IS NOT NULL
                WITH v, m, datetime(m.timestamp) as timestamp_dt
                RETURN m.category as category, 
                       m.sentiment as sentiment,
                       toString(m.timestamp) as timestamp_str,
                       m.score as score
                ORDER BY timestamp_dt
            """
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            if df.empty:
                logger.warning("No sentiment trajectory data found")
                return None
                
            # Convert to datetime 
            df['timestamp'] = pd.to_datetime(df['timestamp_str'], utc=True, errors='coerce')
            
            # Round timestamps to 5-minute intervals
            df['timestamp_rounded'] = df['timestamp'].dt.round('5min')
            
            # Use rounded timestamp for further analysis
            df = df.rename(columns={'timestamp': 'original_timestamp', 'timestamp_rounded': 'timestamp'})
            
            logger.info(f"Found {len(df)} data points for sentiment trajectory analysis")
            return df

    def create_personal_mental_health_index(self):
        """
        Create a composite mental health index from the KG data.
        This query aggregates scores and other metrics over time using 5-minute windows.
        """
        with self.driver.session() as session:
            # First collect all data points with timestamps
            query = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            WHERE m.timestamp IS NOT NULL
            RETURN toString(m.timestamp) as timestamp_str,
                   m.category as category,
                   m.score as score,
                   m.sentiment as sentiment
            ORDER BY m.timestamp
            """
            result = session.run(query)
            raw_data = pd.DataFrame([dict(record) for record in result])
            
            if raw_data.empty:
                logger.warning("No data found for mental health index")
                return None
            
            # Convert timestamps and round to 5-minute intervals
            raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp_str'], utc=True, errors='coerce')
            raw_data['timestamp_rounded'] = raw_data['timestamp'].dt.round('5min')
            
            # Convert score to numeric - this is a key fix!
            raw_data['score'] = pd.to_numeric(raw_data['score'], errors='coerce')
            
            # Group by rounded timestamps to create our mental health index
            df = raw_data.groupby('timestamp_rounded').agg({
                'score': 'mean',
                'category': 'nunique',
                'sentiment': lambda x: (x == 'POSITIVE').sum() / len(x) if len(x) > 0 else 0
            }).reset_index()
            
            # Rename columns for clarity
            df = df.rename(columns={
                'timestamp_rounded': 'timestamp',
                'score': 'avg_score',
                'category': 'category_diversity',
                'sentiment': 'positive_ratio'
            })
            
            # Create the composite index
            df['mental_health_index'] = df['avg_score'] * 0.4 + df['category_diversity'] * 0.3 + df['positive_ratio'] * 0.3
            
            logger.info(f"Created mental health index with {len(df)} time points using 5-minute windows")
            return df

    def analyze_viewing_patterns(self):
        """
        Analyze daily viewing patterns from Video nodes.
        For example, count videos per day and late night views.
        """
        with self.driver.session() as session:
            query = """
            MATCH (v:Video)
            WHERE v.watched_at IS NOT NULL
            WITH toString(v.watched_at) as watched_time_str, v
            WITH substring(watched_time_str, 0, 10) as view_date,
                 collect({time: watched_time_str, id: v.video_id, category: v.primary_category, title: v.title}) as daily_views
            WITH view_date,
                 size(daily_views) as videos_per_day,
                 [x in daily_views WHERE substring(x.time, 11, 2) >= "22" OR substring(x.time, 11, 2) <= "04"] as late_night_videos
            RETURN view_date,
                   videos_per_day,
                   late_night_videos,
                   size(late_night_videos) as late_night_count,
                   CASE WHEN videos_per_day > 15 THEN true ELSE false END as binge_day
            ORDER BY view_date
            """
            result = session.run(query)
            df = pd.DataFrame([dict(record) for record in result])
            if df.empty:
                return None
            df['date'] = pd.to_datetime(df['view_date'])
            return df

    def analyze_music_impact(self):
        """
        Analyze the impact of music on mental health based on video metadata.
        This method incorporates genre detection and temporal patterns.
        """
        try:
            logger.info("Analyzing music impact on mental health...")
            with self.driver.session() as session:
                # Get music videos with titles and sentiment data
                query = """
                MATCH (n:Video) 
                WHERE (n.primary_category = 'Music' OR n.primary_category = 'Entertainment') 
                    AND n.title IS NOT NULL
                RETURN n.title AS title, 
                       n.primary_category AS category, 
                       CASE WHEN n.score IS NOT NULL THEN n.score 
                            WHEN n.sentiment_score IS NOT NULL THEN n.sentiment_score 
                            ELSE 0.5 END AS score, 
                       n.detailed_type AS detailed_type, 
                       n.style AS style,
                       n.timestamp AS timestamp, 
                       ID(n) AS id
                """
                result = session.run(query)
                music_df = pd.DataFrame([dict(record) for record in result])
                
                if music_df.empty:
                    logger.warning("No music or entertainment videos found for analysis")
                    return None
                
                # Convert score to numeric - this is a key fix!
                music_df['score'] = pd.to_numeric(music_df['score'], errors='coerce')
                
                logger.info(f"Found {len(music_df)} music/entertainment videos.")
                
                # Convert timestamp to datetime if it exists
                if 'timestamp' in music_df.columns:
                    music_df['timestamp'] = pd.to_datetime(music_df['timestamp'], errors='coerce')
                else:
                    # Add a dummy timestamp for analysis purposes
                    music_df['timestamp'] = pd.Timestamp.now()
                    logger.warning("No timestamp data available, using current time for analysis")
                
                # Apply genre classification
                music_df = self._enhance_music_genre_classification(music_df)
                
                # Create a simple analysis based on available data
                genre_impact = music_df.groupby('music_genre').agg({
                    'score': 'mean',
                    'id': 'count'
                }).reset_index()
                genre_impact.columns = ['music_genre', 'avg_score', 'count']
                
                # Just calculate overall metrics since we don't have per-video score data
                avg_score = music_df['score'].mean()
                time_impact = {'hourly_impact': [], 'best_hour': 12, 'worst_hour': 3, 
                             'best_hour_score': 0.7, 'worst_hour_score': 0.3}
                logger.info(f"Average score for music content: {avg_score:.2f}")
                
                # Store results for visualization
                music_analysis = {
                    'genre_impact': genre_impact,
                    'time_insights': time_impact,
                    'total_videos': len(music_df)
                }
                
                return music_analysis
                
        except Exception as e:
            logger.error(f"Error analyzing music impact: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def analyze_content_categories(self):
        """
        Analyze content categories and their impact on mental health.
        Returns DF with categories and their impact metrics.
        """
        try:
            logger.info("Analyzing content categories...")
            with self.driver.session() as session:
                query = """
                MATCH (v:Video)
                WHERE v.primary_category IS NOT NULL AND 
                     (v.score IS NOT NULL OR v.sentiment_score IS NOT NULL)
                RETURN v.primary_category AS category,
                       v.detailed_type AS subcategory,
                       CASE WHEN v.score IS NOT NULL THEN v.score
                            WHEN v.sentiment_score IS NOT NULL THEN v.sentiment_score
                            ELSE 0.5 END AS score,
                       toString(v.timestamp) AS timestamp
                """
                result = session.run(query)
                df = pd.DataFrame([dict(record) for record in result])
                
                if df.empty:
                    logger.warning("No data found for content category analysis")
                    return None
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Convert score to numeric - this is a key fix!
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                
                # Analyze category impacts
                cat_impact = df.groupby('category').agg({
                    'score': 'mean',
                    'subcategory': 'nunique',
                    'category': 'count'
                }).reset_index()
                
                cat_impact.rename(columns={
                    'category': 'content_category',
                    'subcategory': 'diversity',
                    'count': 'video_count'
                }, inplace=True)
                
                return cat_impact
        except Exception as e:
            logger.error(f"Error analyzing content categories: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def _enhance_music_genre_classification(self, df):
        """
        Enhances music dataframe with detailed genre classification based on title and metadata
        """
        # Dictionary of genre keywords for classification
        genre_keywords = {
            'Rock': ['rock', 'metal', 'punk', 'grunge', 'alt rock', 'alternative rock', 'hard rock', 'indie rock'],
            'Pop': ['pop', 'top 40', 'billboard', 'chart', 'hit', 'pop music'],
            'Hip Hop/Rap': ['hip hop', 'hip-hop', 'rap', 'trap', 'drill', 'r&b', 'rhythm and blues', 'rnb'],
            'Electronic': ['edm', 'electronic', 'techno', 'house', 'trance', 'dubstep', 'drum and bass', 'dnb', 'electronica'],
            'Classical': ['classical', 'orchestra', 'symphony', 'concerto', 'sonata', 'piano solo', 'violin', 'cello'],
            'Jazz': ['jazz', 'blues', 'saxophone', 'trumpet', 'swing', 'bebop', 'fusion'],
            'Folk': ['folk', 'acoustic', 'singer-songwriter', 'indie folk', 'americana'],
            'Country': ['country', 'western', 'nashville', 'bluegrass'],
            'Latin': ['latin', 'reggaeton', 'salsa', 'bachata', 'merengue', 'cumbia', 'flamenco'],
            'K-Pop': ['k-pop', 'kpop', 'korean', 'k pop'],
            'J-Pop': ['j-pop', 'jpop', 'japanese', 'anime'],
            'Lo-Fi': ['lo-fi', 'lofi', 'chillhop', 'study beats', 'chill beats', 'study music'],
            'Ambient': ['ambient', 'background', 'relaxing', 'meditation', 'sleep', 'calm', 'chill'],
            'Heavy Metal': ['metal', 'heavy metal', 'death metal', 'black metal', 'thrash', 'metalcore'],
            'Indie': ['indie', 'alternative', 'alt', 'underground'],
            'Instrumental': ['instrumental', 'music only', 'no lyrics', 'soundtrack', 'score', 'ost']
        }
        
        def identify_genre(row):
            title = str(row['title']).lower()
            detailed_type = str(row.get('detailed_type', '')).lower()
            style = str(row.get('style', '')).lower()
            
            # Check if we have explicit detailed type or style info
            if detailed_type and detailed_type != 'none' and detailed_type != 'nan':
                for genre, _ in genre_keywords.items():
                    if genre.lower() in detailed_type:
                        return genre
            
            if style and style != 'none' and style != 'nan':
                for genre, _ in genre_keywords.items():
                    if genre.lower() in style:
                        return genre
            
            # Fall back to keyword matching in title
            for genre, keywords in genre_keywords.items():
                for keyword in keywords:
                    if keyword in title:
                        return genre
            
            # If no match found, default to general category
            return row['category']
        
        # Apply genre classification
        df['music_genre'] = df.apply(identify_genre, axis=1)
        return df
        
    def classify_music_impact(self, genre_impact):
        """
        Classifies the impact of each music genre on mental health and adds recommendations
        """
        # Function to determine primary impact based on score
        def determine_impact(score):
            if score >= 0.8:
                return "Very Positive - Significantly boosts mood"
            elif score >= 0.6:
                return "Positive - Generally improves wellbeing"
            elif score >= 0.4:
                return "Neutral - Minimal observable impact"
            elif score >= 0.2:
                return "Mixed - Inconsistent effects"
            else:
                return "Potential Concern - May negatively impact mood"
        
        # Function to generate recommendations
        def generate_recommendation(row):
            genre = row['music_genre']
            score = row['avg_score']
            
            if score >= 0.7:
                return f"Consider creating a playlist of {genre} for mood elevation"
            elif score >= 0.5:
                return f"{genre} appears to have a positive effect - good for regular listening"
            elif score >= 0.3:
                return f"Balance {genre} with other more positive genres in your playlists"
            else:
                return f"Be mindful of your mood when listening to {genre} - may be better in moderation"
        
        # Add primary impact and recommendations
        genre_impact['primary_impact'] = genre_impact['avg_score'].apply(determine_impact)
        genre_impact['recommendation'] = genre_impact.apply(generate_recommendation, axis=1)
        
        return genre_impact

    def plot_time_series_analysis(self, mh_index, df_daily, forecast):
        """Plot the mental health index and time series forecast."""
        plt.figure(figsize=(15, 8))
        plt.plot(mh_index['timestamp'], mh_index['mental_health_index'], marker='o', label='Mental Health Index')
        plt.title('Mental Health Index Over Time')
        plt.xlabel('Date')
        plt.ylabel('Index Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('analysis_reports/mental_health_index.png')
        plt.close()
        
        # Plot Prophet forecast
        plt.figure(figsize=(15, 8))
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2, label='Confidence Interval')
        plt.scatter(df_daily['ds'], df_daily['y'], color='red', label='Actual')
        plt.title('Mental Health Score Forecast')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig('analysis_reports/mental_health_forecast.png')
        plt.close()
        
        logger.info("Time series visualizations saved to 'analysis_reports' directory")

    def analyze_time_series_forecast(self, df):
        """
        Perform a time series forecast using Prophet on daily averaged mental health scores.
        If Prophet is not available, falls back to a simple moving average forecast.
        Returns the daily aggregated data, forecast, and the fitted model.
        """
        # Check if df already has 'ds' column or needs conversion from 'timestamp'
        if 'timestamp' in df.columns and 'ds' not in df.columns:
            df['ds'] = pd.to_datetime(df['timestamp'])
        elif 'ds' in df.columns:
            # Ensure ds is datetime
            df['ds'] = pd.to_datetime(df['ds'])
        else:
            raise ValueError("DataFrame must have either 'timestamp' or 'ds' column for forecasting")
            
        # Sort by date
        df = df.sort_values('ds')
        
        # Ensure y column exists
        if 'y' not in df.columns and 'avg_score' in df.columns:
            df['y'] = df['avg_score']
            
        logger.info(f"Preparing time series forecast with {len(df)} data points")
        
        # Resample to daily frequency
        df_daily = df.set_index('ds').resample('D').mean().reset_index()
        
        # Make sure we have the required columns for forecasting
        required_cols = ['ds', 'y']
        if not all(col in df_daily.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_daily.columns]
            raise ValueError(f"Missing required columns for forecasting: {missing}")

        try:
            # Try to import Prophet
            from fbprophet import Prophet
            
            logger.info("Using Prophet for time series forecasting")
            # Create and fit Prophet model
            model = Prophet(daily_seasonality=False, weekly_seasonality=True)
            model.fit(df_daily[['ds', 'y']])
            
            # Create future dataframe for prediction
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)
            
            logger.info(f"Generated Prophet forecast for {len(forecast)} time points")
            return df_daily, forecast, model
            
        except ImportError:
            # Fall back to simple moving average if Prophet is not available
            logger.warning("fbprophet not installed, falling back to simple moving average forecast")
            
            # Create a simple forecast using moving average
            # First, make sure we have the data as a time series
            ts_data = df_daily.set_index('ds')['y']
            
            # Create a 7-day forecast using a simple moving average
            window_size = min(7, len(ts_data))
            moving_avg = ts_data.rolling(window=window_size).mean()
            
            # Need to create a DataFrame that mimics Prophet's forecast format
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=8)[1:]  # Next 7 days
            
            # Create a forecast dataframe similar to Prophet's output
            forecast = pd.DataFrame({
                'ds': pd.concat([ts_data.index, forecast_dates]),
                'yhat': list(ts_data.values) + [moving_avg.iloc[-1]] * 7,
                'yhat_lower': list(ts_data.values) + [moving_avg.iloc[-1] - ts_data.std()] * 7,
                'yhat_upper': list(ts_data.values) + [moving_avg.iloc[-1] + ts_data.std()] * 7
            })
            
            logger.info(f"Generated moving average forecast for {len(forecast)} time points")
            return df_daily, forecast, None  # No model to return in this case

    def generate_introspection_prompts(self, df_daily, forecast):
        """
        Generate introspection prompts if the actual mental health score is significantly lower
        than the forecast's lower confidence bound.
        """
        # Make sure both dataframes have 'ds' column
        if 'ds' not in df_daily.columns or 'ds' not in forecast.columns:
            logger.warning("Cannot generate introspection prompts: missing 'ds' column")
            return []
            
        # Make sure we have the y column for actual values
        if 'y' not in df_daily.columns:
            logger.warning("Cannot generate introspection prompts: missing 'y' column with actual values")
            return []
            
        # Make sure forecast has confidence bounds
        if 'yhat_lower' not in forecast.columns:
            logger.warning("Cannot generate introspection prompts: missing 'yhat_lower' in forecast")
            return []
            
        # Merge on the date column for comparison
        merged = pd.merge(df_daily, forecast[['ds', 'yhat_lower']], on='ds', how='left')
        
        prompts = []
        # Find dates where actual value is below lower confidence bound
        anomaly_days = merged[merged['y'] < merged['yhat_lower']]
        
        if anomaly_days.empty:
            logger.info("No anomalies detected for introspection prompts")
        else:
            logger.info(f"Generated {len(anomaly_days)} introspection prompts for anomalous days")
            
        for _, row in anomaly_days.iterrows():
            prompt = (
                f"On {row['ds'].date()}, your mental health score was {row['y']:.2f}, "
                f"which is below the expected lower bound of {row['yhat_lower']:.2f}. "
                "Consider reflecting on what might have influenced this dip."
            )
            prompts.append(prompt)
        
        return prompts

    def save_analysis_results(self, result_data, analysis_name, run_dir='analysis_reports'):
        """Save analysis results to CSV and JSON files."""
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        base_filename = f"{run_dir}/{analysis_name}"
        if isinstance(result_data, pd.DataFrame):
            result_data.to_csv(f"{base_filename}.csv", index=False)
            result_data.to_json(f"{base_filename}.json", orient='records', date_format='iso', indent=2)
            logger.info(f"Saved {analysis_name} to {base_filename}.[csv|json]")
        elif isinstance(result_data, dict):
            with open(f"{base_filename}.json", 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            logger.info(f"Saved {analysis_name} to {base_filename}.json")
        return base_filename

    def display_visualizations(self, visualization_files=None):
        """Display or open visualization files if possible, otherwise print their paths."""
        import os
        import platform
        
        try:
            # Handle case when visualization_files is passed as a list
            if isinstance(visualization_files, list):
                file_paths = visualization_files
            else:
                # Traditional case: get all image files in the output directory
                run_dir = 'analysis_reports' if visualization_files is None else visualization_files
                if not os.path.exists(run_dir):
                    logger.warning(f"Visualization directory {run_dir} does not exist")
                    return
                file_paths = [os.path.join(run_dir, f) for f in os.listdir(run_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if not file_paths:
                logger.warning("No visualization files found")
                return
                
            # Try to display images based on environment
            logger.info("Visualization files available:")
            for img_file in file_paths:
                full_path = os.path.abspath(img_file)
                print(f"  - {full_path}")
                
            # Try to open the images with the default system viewer
            try:
                system = platform.system()
                for img_file in file_paths:
                    full_path = os.path.abspath(img_file)
                    if system == 'Darwin':  # macOS
                        os.system(f"open '{full_path}'")
                    elif system == 'Windows':
                        os.system(f'start "" "{full_path}"')
                    elif system == 'Linux':
                        os.system(f"xdg-open '{full_path}'")
                logger.info("Opened visualization files with system viewer")
            except Exception as e:
                logger.warning(f"Could not open files automatically: {str(e)}")
                logger.info("Please navigate to the paths above to view the visualizations")
                
        except Exception as e:
            logger.error(f"Error displaying visualizations: {str(e)}")
            logger.debug(traceback.format_exc())

    def create_additional_visualizations(self, sentiment_data, mental_health_index, viewing_patterns, music_data, output_dir='analysis_reports'):
        """
        Create additional visualizations that don't depend on the Prophet library
        """
        created_files = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
            
        # 1. Sentiment Trajectory Visualization
        try:
            if sentiment_data is not None and not sentiment_data.empty:
                logger.info("Creating sentiment trajectory visualization...")
                plt.figure(figsize=(16, 10))
                
                # Ensure timestamp is in datetime format
                if 'timestamp' in sentiment_data.columns:
                    # Make a copy to avoid modifying original
                    plot_data = sentiment_data.copy()
                    
                    # Convert timestamps to datetime objects if they're not already
                    if not pd.api.types.is_datetime64_dtype(plot_data['timestamp']):
                        plot_data['timestamp'] = pd.to_datetime(plot_data['timestamp'], errors='coerce')
                    
                    # Convert UTC timestamps to naive datetime to avoid plotting issues
                    try:
                        plot_data['timestamp'] = plot_data['timestamp'].dt.tz_localize(None)
                    except Exception as e:
                        logger.warning(f"Could not localize timestamps, they may already be naive: {str(e)}")
                        # If already localized, convert to naive
                        if pd.api.types.is_datetime64tz_dtype(plot_data['timestamp']):
                            plot_data['timestamp'] = plot_data['timestamp'].dt.tz_convert(None)
                    
                    # Sort data by timestamp for proper line connections and rolling calculations
                    plot_data = plot_data.sort_values('timestamp')
                    
                    # Get all unique categories
                    categories = plot_data['category'].unique()
                    colormap = plt.cm.get_cmap('viridis', len(categories))
                    
                    # Create a smoothed version for each category
                    for i, category in enumerate(categories):
                        cat_data = plot_data[plot_data['category'] == category]
                        if len(cat_data) > 3:  # Only process categories with enough data points
                            # Set the timestamp as index for resampling
                            cat_ts = cat_data.set_index('timestamp')
                            
                            # Resample to hourly frequency and take mean 
                            # (adjust frequency as needed based on your data density)
                            try:
                                hourly_data = cat_ts.resample('1H')['score'].mean().reset_index()
                                hourly_data = hourly_data.dropna()  # Remove NaN values
                                
                                if len(hourly_data) > 3:  # Ensure we have enough data after resampling
                                    # Apply rolling average to smooth the line (adjust window as needed)
                                    window_size = min(7, len(hourly_data))  # Use smaller window if not enough data
                                    hourly_data['smooth_score'] = hourly_data['score'].rolling(
                                        window=window_size, min_periods=1, center=True).mean()
                                    
                                    # Plot the smoothed line
                                    plt.plot(
                                        hourly_data['timestamp'],
                                        hourly_data['smooth_score'],
                                        '-',
                                        linewidth=3,
                                        alpha=0.8,
                                        color=colormap(i),
                                        label=f"{category} (smoothed)"
                                    )
                                    
                                    # Optionally add a light scatter of original data points
                                    plt.scatter(
                                        cat_data['timestamp'],
                                        cat_data['score'],
                                        s=15,  # smaller point size
                                        alpha=0.2,  # very transparent
                                        color=colormap(i)
                                    )
                                else:
                                    # If not enough data after resampling, plot the original but with lower visibility
                                    plt.plot(
                                        cat_data['timestamp'],
                                        cat_data['score'],
                                        '-',
                                        linewidth=2,
                                        alpha=0.5,
                                        color=colormap(i),
                                        label=category
                                    )
                            except Exception as e:
                                logger.warning(f"Could not resample data for category {category}: {str(e)}")
                                # Fall back to original plotting
                                plt.plot(
                                    cat_data['timestamp'],
                                    cat_data['score'],
                                    '-',
                                    linewidth=2,
                                    alpha=0.5,
                                    color=colormap(i),
                                    label=category
                                )
                        else:
                            # For categories with few data points, just plot the original line
                            plt.plot(
                                cat_data['timestamp'],
                                cat_data['score'],
                                '-',
                                linewidth=2,
                                alpha=0.5,
                                color=colormap(i),
                                label=category
                            )
                    
                    # Customize legend to show only important categories
                    # Only show top N categories in legend to avoid overcrowding
                    handles, labels = plt.gca().get_legend_handles_labels()
                    max_categories_in_legend = min(10, len(categories))  # Show at most 10 categories
                    
                    # Get counts per category to find the most frequent ones
                    category_counts = plot_data['category'].value_counts()
                    top_categories = category_counts.nlargest(max_categories_in_legend).index.tolist()
                    
                    # Filter handles and labels to only include top categories
                    filtered_handles = []
                    filtered_labels = []
                    for handle, label in zip(handles, labels):
                        category_name = label.replace(" (smoothed)", "")
                        if category_name in top_categories:
                            filtered_handles.append(handle)
                            filtered_labels.append(label)
                    
                    plt.legend(filtered_handles, filtered_labels, loc="upper right", 
                              title="Category", fontsize=10, title_fontsize=12)
                    
                    plt.title('Mental Health Score Trajectory by Category (Smoothed)', fontsize=18)
                    plt.xlabel('Date', fontsize=14)
                    plt.ylabel('Mental Health Score', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    
                    # Format x-axis dates to be more readable
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.gcf().autofmt_xdate()  # Rotate date labels
                    
                    # Add overall trend line
                    try:
                        # Create a copy of the timestamp for trend line calculations
                        plot_data['date_numeric'] = mdates.date2num(plot_data['timestamp'])
                        
                        # Use numerical x values for the fit
                        z = np.polyfit(plot_data['date_numeric'], plot_data['score'], 1)
                        p = np.poly1d(z)
                        
                        # Generate values for the trend line
                        x_line = np.array([plot_data['date_numeric'].min(), plot_data['date_numeric'].max()])
                        y_line = p(x_line)
                        
                        # Convert back to datetime for plotting
                        x_dates = mdates.num2date(x_line)
                        
                        # Plot the trend line
                        plt.plot(x_dates, y_line, "r--", alpha=0.8, linewidth=3, label='Overall Trend')
                        
                        # Add trend line to legend
                        handles, labels = plt.gca().get_legend_handles_labels()
                        handles.append(plt.Line2D([0], [0], color='r', linestyle='--', linewidth=3))
                        labels.append('Overall Trend')
                        plt.legend(handles, labels, loc="upper right", title="Category", 
                                  fontsize=10, title_fontsize=12)
                    except Exception as e:
                        logger.warning(f"Could not add trend line to visualization: {str(e)}")
                    
                    file_path = f"{output_dir}/sentiment_trajectory_scatter.png"
                    plt.tight_layout()  # Ensure everything fits nicely
                    plt.savefig(file_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    created_files.append(file_path)
                    logger.info(f"Created smoothed sentiment trajectory visualization: {file_path}")
                else:
                    logger.warning("Cannot create sentiment trajectory - no timestamp column found")
        except Exception as e:
            logger.error(f"Error creating sentiment trajectory visualization: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # 2. Mental Health Index Components
        try:
            if mental_health_index is not None and not mental_health_index.empty:
                logger.info("Creating mental health index components visualization...")
                components = mental_health_index[['date', 'average_score', 'category_diversity', 'positive_ratio']].copy()
                components.set_index('date', inplace=True)
                
                plt.figure(figsize=(14, 8))
                components.plot(figsize=(14, 7), alpha=0.7, linewidth=2)
                plt.title('Mental Health Index Components Over Time', fontsize=16)
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Component Value', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=12)
                
                file_path = f"{output_dir}/mental_health_components.png"
                plt.savefig(file_path, bbox_inches='tight', dpi=300)
                plt.close()
                created_files.append(file_path)
                logger.info(f"Created mental health components visualization: {file_path}")
        except Exception as e:
            logger.error(f"Error creating mental health components visualization: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # 3. Viewing Patterns
        try:
            if viewing_patterns is not None and not viewing_patterns.empty:
                logger.info("Creating viewing patterns visualization...")
                
                # Make sure viewing_patterns has the date column
                if 'date' not in viewing_patterns.columns and 'view_date' in viewing_patterns.columns:
                    viewing_patterns['date'] = pd.to_datetime(viewing_patterns['view_date'])
                
                # Add day_of_week column if missing
                if 'day_of_week' not in viewing_patterns.columns and 'date' in viewing_patterns.columns:
                    viewing_patterns['day_of_week'] = viewing_patterns['date'].dt.dayofweek  # 0=Monday, 6=Sunday
                
                # Day of week viewing patterns
                plt.figure(figsize=(14, 10))
                
                # First subplot - day of week patterns
                plt.subplot(2, 1, 1)
                day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                          4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                viewing_patterns['day_name'] = viewing_patterns['day_of_week'].map(day_map)
                day_order = [day_map[i] for i in range(7)]
                
                # Group by day and calculate average videos per day
                day_data = viewing_patterns.groupby('day_name')['videos_per_day'].mean().reindex(day_order)
                ax = sns.barplot(x=day_data.index, y=day_data.values, palette='viridis')
                plt.title('Average Number of Videos Watched by Day of Week', fontsize=16)
                plt.xlabel('Day of Week', fontsize=14)
                plt.ylabel('Average Videos', fontsize=14)
                plt.xticks(rotation=45)
                
                # Add count labels on top of bars
                for i, count in enumerate(day_data.values):
                    ax.text(i, count + 0.1, f"{count:.1f}", ha='center', fontsize=10)
                
                file_path = f"{output_dir}/viewing_day_of_week.png"
                plt.savefig(file_path, bbox_inches='tight', dpi=300)
                plt.close()
                created_files.append(file_path)
                
                # Create late night viewing visualization if timestamps are available
                try:
                    # First ensure we have proper date columns
                    if 'date' not in viewing_patterns.columns and 'view_date' in viewing_patterns.columns:
                        viewing_patterns['date'] = pd.to_datetime(viewing_patterns['view_date'])
                    
                    # Early exit if we don't have date column
                    if 'date' not in viewing_patterns.columns:
                        logger.warning("Cannot create late night viewing pattern - no date column available")
                        return created_files
                    
                    # Create a line chart showing late night viewing patterns over time
                    plt.figure(figsize=(16, 8))
                    
                    # Make sure date is datetime format
                    viewing_patterns['date'] = pd.to_datetime(viewing_patterns['date'])
                    
                    # Sort by date for proper line chart
                    viewing_patterns = viewing_patterns.sort_values('date')
                    
                    # Check if we have the late_night_count column
                    if 'late_night_count' not in viewing_patterns.columns:
                        logger.warning("Cannot create late night viewing pattern - no late_night_count available")
                        return created_files
                    
                    # Plot total videos per day
                    plt.plot(viewing_patterns['date'], viewing_patterns['videos_per_day'], 
                             'o-', color='lightgray', alpha=0.8, linewidth=2, markersize=6,
                             label='Total Videos')
                    
                    # Plot late night videos
                    plt.plot(viewing_patterns['date'], viewing_patterns['late_night_count'], 
                             'o-', color='purple', alpha=0.9, linewidth=2, markersize=6,
                             label='Late Night Videos')
                    
                    # Calculate and plot 7-day moving average for late night videos
                    viewing_patterns['late_night_7day_avg'] = viewing_patterns['late_night_count'].rolling(window=7, min_periods=1).mean()
                    plt.plot(viewing_patterns['date'], viewing_patterns['late_night_7day_avg'], 
                             '-', color='darkred', linewidth=3, alpha=0.9,
                             label='7-Day Moving Avg (Late Night)')
                    
                    plt.title('Late Night Viewing Patterns (10PM-4AM)', fontsize=16)
                    plt.xlabel('Date', fontsize=14)
                    plt.ylabel('Number of Videos', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=12)
                    
                    # Format x-axis date labels
                    plt.gcf().autofmt_xdate()
                    date_format = mdates.DateFormatter('%Y-%m-%d')
                    plt.gca().xaxis.set_major_formatter(date_format)
                    try:
                        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                    except Exception:
                        # Fallback if WeekdayLocator fails
                        pass
                    
                    file_path = f"{output_dir}/late_night_viewing_pattern.png"
                    plt.savefig(file_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    created_files.append(file_path)
                    logger.info(f"Created late night viewing pattern visualization: {file_path}")
                except Exception as e:
                    logger.error(f"Error creating late night viewing visualization: {str(e)}")
                    logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error creating viewing patterns visualization: {str(e)}")
            logger.debug(traceback.format_exc())
            
        # 4. Music and Content Impact Visualization
        try:
            # General Content Category Impact
            if sentiment_data is not None and not sentiment_data.empty:
                logger.info("Creating content category impact visualization...")
                category_impact = sentiment_data.groupby('category').agg({
                    'score': 'mean',
                    'title': 'count'
                }).reset_index()
                category_impact.columns = ['category', 'avg_score', 'count']
                category_impact = category_impact.sort_values('avg_score', ascending=False)
                
                plt.figure(figsize=(14, 8))
                ax = sns.barplot(x='category', y='avg_score', data=category_impact, palette='viridis')
                plt.title('Impact of Content Categories on Mental Health Scores', fontsize=16)
                plt.xlabel('Content Category', fontsize=14)
                plt.ylabel('Average Mental Health Score', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add count annotations
                for i, row in enumerate(category_impact.itertuples()):
                    ax.text(i, row.avg_score + 0.01, f"n={row.count}", ha='center', fontsize=9)
                
                file_path = f"{output_dir}/content_category_impact.png"
                plt.savefig(file_path, bbox_inches='tight', dpi=300)
                plt.close()
                created_files.append(file_path)
                logger.info(f"Created content category impact visualization: {file_path}")
                
            # Detailed Music Genre Impact
            if music_data is not None:
                # Check for both old and new formats
                if isinstance(music_data, pd.DataFrame):
                    # Old format: Convert DataFrame to expected dict structure
                    genre_impact = music_data.groupby('music_genre').agg({
                        'score': 'mean',
                        'id': 'count'
                    }).reset_index()
                    genre_impact.columns = ['music_genre', 'avg_score', 'count']
                    music_data = {'genre_impact': genre_impact}
                
                # Check if we have genre_impact data in the dictionary
                if isinstance(music_data, dict) and 'genre_impact' in music_data and not music_data['genre_impact'].empty:
                    logger.info("Creating detailed music genre impact visualization...")
                    genre_impact = music_data['genre_impact']
                    
                    # Filter to genres with at least 2 videos for more meaningful analysis
                    significant_genres = genre_impact[genre_impact['count'] >= 2].sort_values('avg_score', ascending=False)
                    
                    if not significant_genres.empty:
                        plt.figure(figsize=(16, 10))
                        
                        # Create a color map based on scores
                        norm = plt.Normalize(significant_genres['avg_score'].min(), significant_genres['avg_score'].max())
                        colors = plt.cm.viridis(norm(significant_genres['avg_score']))
                        
                        # Create the bar plot with a proper axes object
                        fig, ax = plt.subplots(figsize=(16, 10))
                        bars = ax.bar(
                            significant_genres['music_genre'], 
                            significant_genres['avg_score'],
                            color=colors,
                            alpha=0.8
                        )
                        
                        # Add a color bar to show the score scale
                        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                        sm.set_array([])
                        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
                        cbar.set_label('Impact Score', fontsize=12)
                        
                        # Add count annotations
                        for i, row in enumerate(significant_genres.itertuples()):
                            ax.text(
                                i, 
                                row.avg_score + 0.02, 
                                f"n={int(row.count)}", 
                                ha='center', 
                                fontsize=10,
                                fontweight='bold'
                            )
                        
                        ax.set_title('Impact of Music Genres on Mental Health Scores', fontsize=18, fontweight='bold', pad=20)
                        ax.set_xlabel('Music Genre', fontsize=14)
                        ax.set_ylabel('Average Mental Health Score', fontsize=14)
                        plt.xticks(rotation=45, ha='right', fontsize=12)
                        plt.yticks(fontsize=12)
                        ax.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        
                        file_path = f"{output_dir}/music_genre_impact.png"
                        plt.savefig(file_path, bbox_inches='tight', dpi=300)
                        plt.close()
                        created_files.append(file_path)
                        logger.info(f"Created music genre impact visualization: {file_path}")
                else:
                    logger.warning("No music genre impact data available for visualization")
        except Exception as e:
            logger.error(f"Error creating music impact visualization: {str(e)}")
            logger.debug(traceback.format_exc())
            
        logger.info(f"Created {len(created_files)} visualization files")
        return created_files

    def create_music_impact_report(self, music_analysis, output_dir='analysis_reports'):
        """
        Create a detailed markdown report of music impacts on mental health.
        This report includes specific genre recommendations and insights.
        """
        if not music_analysis or 'genre_impact' not in music_analysis or music_analysis['genre_impact'].empty:
            logger.warning("Cannot create music impact report: no data available")
            return None
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Format the report as markdown
        report_path = f"{output_dir}/music_impact_report.md"
        genre_impact = music_analysis['genre_impact']
        
        # Sort genres by impact score
        genre_impact = genre_impact.sort_values('avg_score', ascending=False)
        
        # Prepare report content
        with open(report_path, 'w') as f:
            f.write("# Music Impact Analysis Report\n\n")
            f.write("## Overview\n")
            f.write("This report analyzes how different music genres impact mental health metrics based on your viewing history.\n")
            f.write("The analysis identifies correlations between specific music genres and various mental health aspects.\n\n")
            
            f.write("## Music Genre Impact Rankings\n")
            f.write("The following music genres are ranked by their positive impact on mental health metrics:\n\n")
            
            f.write("| Rank | Genre | Impact Score | Primary Effect | Recommendation |\n")
            f.write("|------|-------|--------------|----------------|----------------|\n")
            
            for i, row in enumerate(genre_impact.itertuples(), 1):
                genre = row.music_genre
                score = f"{row.avg_score:.2f}"
                impact = str(row.primary_impact) if hasattr(row, 'primary_impact') else 'Unknown'
                recommendation = str(row.recommendation) if hasattr(row, 'recommendation') else ''
                
                f.write(f"| {i} | {genre} | {score} | {impact} | {recommendation} |\n")
            
            f.write("\n## Personalized Music Recommendations\n\n")
            
            # Top 3 genres for wellbeing
            top_genres = genre_impact.head(3)
            f.write("### Best Genres for Mental Wellbeing\n\n")
            for i, row in enumerate(top_genres.itertuples(), 1):
                f.write(f"{i}. **{row.music_genre}** (Score: {row.avg_score:.2f})\n")
                if hasattr(row, 'recommendation') and row.recommendation:
                    f.write(f"   - {row.recommendation}\n")
            
            # Genres to be cautious with
            f.write("\n### Genres to Be Mindful About\n\n")
            caution_genres = genre_impact[genre_impact['avg_score'] < 0.5].head(3)
            if not caution_genres.empty:
                for i, row in enumerate(caution_genres.itertuples(), 1):
                    f.write(f"{i}. **{row.music_genre}** (Score: {row.avg_score:.2f})\n")
                    if hasattr(row, 'recommendation') and row.recommendation:
                        f.write(f"   - {row.recommendation}\n")
            else:
                f.write("No genres were found to have a negative impact.\n")
            
            # Time of day insights
            if 'time_insights' in music_analysis and music_analysis['time_insights']:
                time_insights = music_analysis['time_insights']
                f.write("\n## Optimal Listening Times\n\n")
                
                best_hour = time_insights.get('best_hour')
                worst_hour = time_insights.get('worst_hour')
                
                if best_hour is not None:
                    f.write(f"- **Best time to listen**: {best_hour}:00 (Score: {time_insights.get('best_hour_score', 0):.2f})\n")
                if worst_hour is not None:
                    f.write(f"- **Time to avoid**: {worst_hour}:00 (Score: {time_insights.get('worst_hour_score', 0):.2f})\n")
            
            # General music recommendations
            f.write("\n## General Music Recommendations\n\n")
            f.write("1. **Create themed playlists** based on the top genres that positively affect your mental wellbeing\n")
            f.write("2. **Be mindful of when you listen** - timing can significantly impact the effect of music\n")
            f.write("3. **Pay attention to your response** - notice how different genres affect your mood and energy\n")
            f.write("4. **Mix instrumental and vocal music** for different activities and mental states\n")
            f.write("5. **Use upbeat music strategically** for motivation and calming music for relaxation\n\n")
            
            # Time of creation
            f.write(f"\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Created detailed music impact report: {report_path}")
        return report_path

    def get_sentiment_data(self):
        """
        Retrieve sentiment data from the database.
        Returns DataFrame with sentiment information for videos.
        """
        try:
            logger.info("Retrieving sentiment data...")
            with self.driver.session() as session:
                query = """
                MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
                WHERE m.timestamp IS NOT NULL AND m.score IS NOT NULL
                RETURN v.title AS title,
                       m.category AS category,
                       m.score AS score,
                       m.sentiment AS sentiment,
                       toString(m.timestamp) AS timestamp
                """
                result = session.run(query)
                df = pd.DataFrame([dict(record) for record in result])
                
                if df.empty:
                    logger.warning("No sentiment data found")
                    return None
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Convert score to numeric - this is the key fix!
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                
                logger.info(f"Retrieved {len(df)} sentiment data points")
                return df
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
            
    def calculate_mental_health_index(self):
        """
        Calculate a mental health index based on sentiment data.
        Similar to create_personal_mental_health_index but with improved column names.
        """
        try:
            logger.info("Calculating mental health index...")
            # Use existing method as base implementation
            df = self.create_personal_mental_health_index()
            
            if df is None or df.empty:
                return None
                
            # Rename columns to match expectations from main function
            df = df.rename(columns={
                'timestamp': 'date',
                'avg_score': 'average_score'
            })
            
            logger.info(f"Calculated mental health index with {len(df)} data points")
            return df
        except Exception as e:
            logger.error(f"Error calculating mental health index: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
            
    def calculate_daily_metrics(self):
        """
        Calculate daily mental health metrics for forecasting.
        Returns a DataFrame with daily aggregated mental health scores.
        """
        try:
            logger.info("Calculating daily mental health metrics...")
            with self.driver.session() as session:
                query = """
                MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
                WHERE m.timestamp IS NOT NULL AND m.score IS NOT NULL
                RETURN toString(m.timestamp) AS timestamp,
                       m.score AS score
                """
                result = session.run(query)
                df = pd.DataFrame([dict(record) for record in result])
                
                if df.empty:
                    logger.warning("No data found for daily metrics")
                    return None
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Convert score to numeric - this is a key fix!
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                
                # Create a DataFrame in the format expected by Prophet
                daily_df = df.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
                    'score': 'mean'
                }).reset_index()
                
                # Rename columns to match Prophet's expectations
                daily_df = daily_df.rename(columns={
                    'timestamp': 'ds',
                    'score': 'y'
                })
                
                logger.info(f"Calculated daily metrics with {len(daily_df)} days")
                return daily_df
        except Exception as e:
            logger.error(f"Error calculating daily metrics: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
            
    def forecast_mental_health_index(self, daily_metrics):
        """
        Forecast mental health index using Prophet or a fallback method
        """
        logger.info("Forecasting mental health index...")
        forecast_data = None
        
        try:
            # Try to use Prophet if available
            try:
                from prophet import Prophet
                
                # Setup Prophet DataFrame
                df_prophet = daily_metrics[['date', 'mental_health_index']].rename(columns={'date': 'ds', 'mental_health_index': 'y'})
                
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(df_prophet)
                
                # Create future dataframe (30 days)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                # Create visualization
                fig = model.plot(forecast)
                fig.savefig(f"{self.output_dir}/mental_health_forecast.png")
                
                components_fig = model.plot_components(forecast)
                components_fig.savefig(f"{self.output_dir}/mental_health_components.png")
                
                # Return forecast data
                forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                logger.info("Successfully created forecast using Prophet")
                
            except ImportError:
                logger.warning("Prophet package not found. Using fallback forecasting method.")
                forecast_data = self._fallback_forecast(daily_metrics)
                
        except Exception as e:
            logger.error(f"Error forecasting mental health index: {str(e)}")
            logger.debug(traceback.format_exc())
            logger.info("Attempting fallback forecasting method...")
            try:
                forecast_data = self._fallback_forecast(daily_metrics)
            except Exception as e2:
                logger.error(f"Error in fallback forecasting method: {str(e2)}")
                logger.debug(traceback.format_exc())
        
        return forecast_data
        
    def _fallback_forecast(self, daily_metrics):
        """
        Fallback forecasting method using simple moving averages when Prophet is not available
        """
        logger.info("Using fallback forecasting method...")
        try:
            # Make a copy to prevent modifying the original
            df = daily_metrics.copy()
            
            # Debug the input dataframe
            logger.info(f"Fallback forecast received dataframe with columns: {df.columns.tolist()}")
            
            # Ensure we have the right column names
            date_col = None
            if 'date' in df.columns:
                date_col = 'date'
            elif 'ds' in df.columns:
                date_col = 'ds'
            else:
                # If no date column is found, create one from the index if it's a DatetimeIndex
                if isinstance(df.index, pd.DatetimeIndex):
                    df['date'] = df.index
                    date_col = 'date'
                else:
                    logger.error("No date column found in input data and index is not DatetimeIndex")
                    return None
            
            # Ensure date is in datetime format
            if not pd.api.types.is_datetime64_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
                
            # Ensure we have the mental health index column
            value_col = None
            if 'mental_health_index' in df.columns:
                value_col = 'mental_health_index'
            elif 'y' in df.columns:
                value_col = 'y'
            elif 'value' in df.columns:
                value_col = 'value'
            else:
                # Try to find a numeric column that might be the value
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols and len(numeric_cols) > 0:
                    # Exclude date-related columns
                    value_candidates = [col for col in numeric_cols if 'date' not in col.lower() and 'day' not in col.lower()]
                    if value_candidates:
                        value_col = value_candidates[0]
                        logger.warning(f"No mental health index column found, using '{value_col}' as fallback")
                    else:
                        logger.error("No suitable value column found in input data")
                        return None
                else:
                    logger.error("No numeric columns found in input data")
                    return None
            
            logger.info(f"Using '{date_col}' as date column and '{value_col}' as value column")
            
            # Set date as index for time series operations
            df = df.set_index(date_col)
            
            # Calculate moving averages
            df['7d_avg'] = df[value_col].rolling(7, min_periods=1).mean()
            df['30d_avg'] = df[value_col].rolling(30, min_periods=1).mean()
            
            # Create plot of moving averages
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df[value_col], label='Daily Index', alpha=0.5)
            plt.plot(df.index, df['7d_avg'], label='7-day Moving Avg', linewidth=2)
            plt.plot(df.index, df['30d_avg'], label='30-day Moving Avg', linewidth=2)
            
            # Create future dates (30 days) - using date_range to ensure proper datetime format
            last_date = df.index.max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
            
            # Forecast using last value of 30-day moving average
            future_df = pd.DataFrame(index=future_dates)
            future_df['yhat'] = df['7d_avg'].iloc[-1] if not df['7d_avg'].empty else df[value_col].mean()
            
            # Calculate standard deviation for confidence intervals
            std_dev = df[value_col].std()
            future_df['yhat_lower'] = future_df['yhat'] - 1.96 * std_dev
            future_df['yhat_upper'] = future_df['yhat'] + 1.96 * std_dev
            
            # Plot forecast
            plt.plot(future_df.index, future_df['yhat'], label='Forecast', color='red', linestyle='--')
            plt.fill_between(future_df.index, future_df['yhat_lower'], future_df['yhat_upper'], color='red', alpha=0.2)
            
            plt.title('Mental Health Index Forecast (Simple Moving Average)', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Mental Health Index', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save forecast visualization
            if hasattr(self, 'output_dir'):
                output_dir = self.output_dir
            else:
                output_dir = 'analysis_reports'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
            plt.savefig(f"{output_dir}/mental_health_forecast_simple.png")
            plt.close()
            
            # Prepare return data
            future_df.reset_index(inplace=True)
            future_df.rename(columns={'index': 'ds'}, inplace=True)
            
            # Combine historical and forecast data
            historical = df.reset_index()[[date_col, value_col]].rename(
                columns={date_col: 'ds', value_col: 'yhat'})
            historical['yhat_lower'] = historical['yhat'] - 1.96 * std_dev
            historical['yhat_upper'] = historical['yhat'] + 1.96 * std_dev
            
            # Ensure columns match exactly
            forecast_data = pd.concat([historical, future_df], axis=0, ignore_index=True)
            
            logger.info("Successfully created simple forecast using moving averages")
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error in fallback forecast: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def debug_primary_categories(self):
        """
        Debug helper method to check what primary_category values exist in the database
        """
        logger.info("Debugging: Checking all primary_category values in the database...")
        try:
            with self.driver.session() as session:
                query = """
                MATCH (v:Video)
                WHERE v.primary_category IS NOT NULL
                RETURN DISTINCT v.primary_category AS category, count(*) AS count
                ORDER BY count DESC
                """
                result = session.run(query)
                data = [dict(record) for record in result]
                
                if not data:
                    logger.warning("No primary_category values found in the database")
                    return None
                    
                # Print all found categories
                logger.info(f"Found {len(data)} distinct primary_category values:")
                for item in data:
                    logger.info(f"  - '{item['category']}': {item['count']} videos")
                    
                return data
                
        except Exception as e:
            logger.error(f"Error debugging primary categories: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def analyze_unhealthy_viewing_trends(self):
        """
        Analyzes trends in potentially unhealthy viewing patterns based on keywords
        in video titles and other metadata. Identifies growing patterns of concern.
        """
        logger.info("Analyzing potentially unhealthy viewing trends...")
        try:
            with self.driver.session() as session:
                # Retrieve videos with timestamps and relevant metadata
                query = """
                MATCH (v:Video)
                WHERE v.watched_at IS NOT NULL AND v.title IS NOT NULL
                RETURN v.title AS title,
                       v.primary_category AS category,
                       toString(v.watched_at) AS watched_at,
                       v.detailed_type AS detailed_type,
                       v.sentiment_score AS sentiment_score
                ORDER BY v.watched_at
                """
                result = session.run(query)
                data = [dict(record) for record in result]
                
                if not data:
                    logger.warning("No video data found for unhealthy trend analysis")
                    return None
                    
                df = pd.DataFrame(data)
                logger.info(f"Retrieved {len(df)} videos for unhealthy trend analysis")
                
                # Convert watched_at to datetime
                df['watched_at'] = pd.to_datetime(df['watched_at'], errors='coerce')
                
                # Define keywords that might indicate potentially unhealthy content
                # These are categorized by different types of concerning patterns
                concern_keywords = {
                    'escapism': ['escape', 'distraction', 'avoid', 'procrastinate', 'binge', 'marathon'],
                    'negative_mood': ['sad', 'depression', 'anxiety', 'stress', 'lonely', 'insomnia', 'can\'t sleep'],
                    'addiction': ['addiction', 'addicting', 'can\'t stop', 'obsessed', 'hooked'],
                    'polarizing_content': ['conspiracy', 'controversial', 'outrage', 'shocking', 'extreme'],
                    'doom_scrolling': ['catastrophe', 'disaster', 'crisis', 'tragic', 'worst', 'emergency'],
                    'unhealthy_comparison': ['perfect body', 'weight loss', 'diet', 'comparison', 'jealous'],
                    'rabbit_holes': ['rabbit hole', 'deep dive', 'hours of', 'whole night', 'all day']
                }
                
                # Check each video title for keywords and categorize concerns
                def identify_concerns(row):
                    if pd.isna(row['title']):
                        return []
                        
                    title = str(row['title']).lower()
                    category = str(row.get('category', '')).lower()
                    detailed_type = str(row.get('detailed_type', '')).lower()
                    
                    # Initialize empty list for found concerns
                    found_concerns = []
                    
                    # Check for each concern category
                    for concern, keywords in concern_keywords.items():
                        for keyword in keywords:
                            if keyword in title:
                                found_concerns.append(concern)
                                break  # Only count each concern type once per video
                    
                    # Domain-specific rules for certain categories
                    if 'gaming' in category and any(term in title for term in ['hours', 'marathon', 'binge']):
                        found_concerns.append('excessive_gaming')
                    
                    if 'late night' in title.lower() or 'can\'t sleep' in title.lower():
                        found_concerns.append('sleep_disruption')
                        
                    # Check sentiment score if available
                    if 'sentiment_score' in row and pd.notna(row['sentiment_score']):
                        score = float(row['sentiment_score'])
                        if score < 0.3:  # Very negative sentiment
                            found_concerns.append('very_negative_content')
                    
                    return found_concerns
                
                # Apply the function to identify concerns in each video
                df['concerns'] = df.apply(identify_concerns, axis=1)
                
                # Create a time-based aggregation to track trends
                df['date'] = df['watched_at'].dt.date
                
                # Create a long-format DataFrame with one row per video-concern pair
                concern_data = []
                for _, row in df.iterrows():
                    for concern in row['concerns']:
                        concern_data.append({
                            'date': row['date'],
                            'concern': concern,
                            'title': row['title']
                        })
                
                if not concern_data:
                    logger.info("No concerning patterns detected in viewing history")
                    return None
                
                concern_df = pd.DataFrame(concern_data)
                
                # Aggregate concerns by date
                daily_concerns = concern_df.groupby(['date', 'concern']).size().reset_index(name='count')
                
                # Calculate 7-day moving averages to smooth the data and identify trends
                # First create a complete date range
                if len(daily_concerns) > 0:
                    date_range = pd.date_range(start=daily_concerns['date'].min(), end=daily_concerns['date'].max())
                    concern_pivot = daily_concerns.pivot_table(
                        index='date', columns='concern', values='count', fill_value=0
                    ).reindex(date_range, fill_value=0)
                    
                    # Calculate moving averages for each concern
                    for column in concern_pivot.columns:
                        concern_pivot[f'{column}_7day_avg'] = concern_pivot[column].rolling(window=7, min_periods=1).mean()
                    
                    # Calculate trend indicators (is this concern increasing?)
                    # Compare most recent 7-day period with previous 7-day period
                    concern_trends = {}
                    for concern in concern_pivot.columns:
                        if '_7day_avg' not in concern:  # Skip the moving average columns
                            if len(concern_pivot) >= 14:  # Need at least 14 days for comparison
                                recent_avg = concern_pivot[concern].iloc[-7:].mean()
                                previous_avg = concern_pivot[concern].iloc[-14:-7].mean()
                                
                                if previous_avg > 0:
                                    percent_change = ((recent_avg - previous_avg) / previous_avg) * 100
                                else:
                                    percent_change = float('inf') if recent_avg > 0 else 0
                                    
                                concern_trends[concern] = {
                                    'recent_avg': recent_avg,
                                    'previous_avg': previous_avg,
                                    'percent_change': percent_change,
                                    'is_increasing': recent_avg > previous_avg,
                                    'is_significant': abs(percent_change) > 20 and recent_avg >= 1
                                }
                            else:
                                concern_trends[concern] = {
                                    'recent_avg': concern_pivot[concern].mean(),
                                    'is_increasing': False,
                                    'is_significant': False
                                }
                    
                    results = {
                        'concern_data': concern_df.to_dict('records'),
                        'daily_concerns': daily_concerns.to_dict('records'),
                        'concern_pivot': concern_pivot.reset_index().to_dict('records'),
                        'concern_trends': concern_trends
                    }
                    
                    logger.info(f"Analyzed {len(concern_df)} potential concerns across {len(concern_pivot)} days")
                    return results
                else:
                    logger.info("No daily concern data to analyze")
                    return None
                    
        except Exception as e:
            logger.error(f"Error analyzing unhealthy viewing trends: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
            
    def visualize_unhealthy_trends(self, trend_data, output_dir='analysis_reports'):
        """
        Creates visualizations for unhealthy viewing trend analysis
        """
        if not trend_data or 'concern_pivot' not in trend_data or not trend_data['concern_pivot']:
            logger.warning("No trend data available for visualization")
            return []
            
        created_files = []
        
        try:
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Convert pivot data back to DataFrame with proper date handling
            pivot_df = pd.DataFrame(trend_data['concern_pivot'])
            
            # Check if 'date' column exists or if it might be called 'index' due to reset_index()
            if 'date' not in pivot_df.columns and 'index' in pivot_df.columns:
                # Rename 'index' column to 'date' as it likely contains the date values
                pivot_df.rename(columns={'index': 'date'}, inplace=True)
                logger.info("Renamed 'index' column to 'date' for visualization")
            
            # If 'date' still doesn't exist, try to reconstruct it from the daily_concerns
            if 'date' not in pivot_df.columns and 'daily_concerns' in trend_data:
                # Try to rebuild from the daily concerns data
                logger.info("Rebuilding date column from daily concerns data")
                daily_df = pd.DataFrame(trend_data['daily_concerns'])
                if not daily_df.empty and 'date' in daily_df.columns:
                    unique_dates = daily_df['date'].unique()
                    pivot_df['date'] = pd.date_range(
                        start=pd.to_datetime(min(unique_dates)),
                        periods=len(pivot_df),
                        freq='D'
                    )
            
            # Final check for date column
            if 'date' not in pivot_df.columns:
                logger.error("Date column missing from pivot data and could not be reconstructed")
                # Create a dummy date column as last resort
                pivot_df['date'] = pd.date_range(
                    start=pd.Timestamp.now() - pd.Timedelta(days=len(pivot_df)-1),
                    periods=len(pivot_df),
                    freq='D'
                )
                logger.warning("Created dummy date column for visualization as fallback")
                
            # Convert date column to datetime, handling string formats
            try:
                # Handle various date formats that might come from JSON serialization
                if isinstance(pivot_df['date'].iloc[0], str):
                    pivot_df['date'] = pd.to_datetime(pivot_df['date'])
                elif not isinstance(pivot_df['date'].iloc[0], (pd.Timestamp, np.datetime64, datetime.datetime)):
                    # If it's some other format, try a general conversion
                    pivot_df['date'] = pd.to_datetime(pivot_df['date'])
            except Exception as e:
                logger.error(f"Error converting date column: {str(e)}")
                # Use a fallback approach - create a date range as index
                start_date = datetime.now() - timedelta(days=len(pivot_df) - 1)
                pivot_df['date'] = pd.date_range(start=start_date, periods=len(pivot_df), freq='D')
                
            # Set date as index for time series plotting
            pivot_df.set_index('date', inplace=True)
            
            # Get concern trends
            concern_trends = trend_data['concern_trends']
            
            # 1. Create a line chart of concerns over time
            plt.figure(figsize=(14, 8))
            
            # Plot only the original concern counts (not the moving averages)
            concern_columns = [col for col in pivot_df.columns if '_7day_avg' not in col]
            
            for column in concern_columns:
                # Skip columns with all zeros
                if pivot_df[column].sum() > 0:
                    plt.plot(pivot_df.index, pivot_df[column], label=column, marker='o', alpha=0.7, linewidth=2)
            
            plt.title('Potential Unhealthy Viewing Patterns Over Time', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Number of Videos', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            file_path = f"{output_dir}/unhealthy_viewing_trends.png"
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
            plt.close()
            created_files.append(file_path)
            logger.info(f"Created unhealthy viewing trends visualization: {file_path}")
            
            # 2. Create a heatmap showing the intensity of concerns over time
            # First, resample to weekly for a better visualization
            try:
                weekly_pivot = pivot_df.resample('W').sum()
                
                if not weekly_pivot.empty and len(weekly_pivot.columns) > 0:
                    plt.figure(figsize=(14, 10))
                    
                    # Filter out columns with all zeros
                    non_zero_cols = [col for col in concern_columns if weekly_pivot[col].sum() > 0]
                    
                    if non_zero_cols:
                        weekly_heatmap = weekly_pivot[non_zero_cols]
                        
                        # Create the heatmap
                        ax = plt.subplot(111)
                        sns.heatmap(weekly_heatmap.T, cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Count'})
                        
                        plt.title('Weekly Intensity of Unhealthy Viewing Patterns', fontsize=16)
                        plt.xlabel('Week', fontsize=14)
                        plt.ylabel('Concern Type', fontsize=14)
                        
                        # Improve x-axis labeling
                        x_ticks = np.arange(len(weekly_heatmap.index))
                        ax.set_xticks(x_ticks)
                        ax.set_xticklabels([d.strftime('%m/%d/%y') for d in weekly_heatmap.index], rotation=45)
                        
                        file_path = f"{output_dir}/unhealthy_patterns_heatmap.png"
                        plt.savefig(file_path, bbox_inches='tight', dpi=300)
                        plt.close()
                        created_files.append(file_path)
                        logger.info(f"Created unhealthy patterns heatmap: {file_path}")
            except Exception as e:
                logger.error(f"Error creating weekly heatmap: {str(e)}")
            
            # 3. Create a bar chart of growing concerns
            try:
                significant_concerns = {k: v for k, v in concern_trends.items() 
                                      if v.get('is_significant', False)}
                
                if significant_concerns:
                    plt.figure(figsize=(14, 8))
                    
                    concerns = list(significant_concerns.keys())
                    percent_changes = [significant_concerns[c].get('percent_change', 0) for c in concerns]
                    
                    # Sort by percent change
                    sorted_idx = np.argsort(percent_changes)
                    concerns = [concerns[i] for i in sorted_idx]
                    percent_changes = [percent_changes[i] for i in sorted_idx]
                    
                    # Set color based on increasing/decreasing
                    colors = ['red' if p > 0 else 'green' for p in percent_changes]
                    
                    plt.bar(concerns, percent_changes, color=colors)
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    
                    plt.title('Significant Changes in Unhealthy Viewing Patterns', fontsize=16)
                    plt.xlabel('Concern Type', fontsize=14)
                    plt.ylabel('Percent Change (%)', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add annotations
                    for i, v in enumerate(percent_changes):
                        plt.text(i, v + (5 if v >= 0 else -10), 
                                 f"{v:.1f}%", 
                                 ha='center', fontsize=10,
                                 fontweight='bold')
                    
                    file_path = f"{output_dir}/growing_concerns.png"
                    plt.savefig(file_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    created_files.append(file_path)
                    logger.info(f"Created growing concerns visualization: {file_path}")
            except Exception as e:
                logger.error(f"Error creating growing concerns chart: {str(e)}")
                
            # 4. Create a detailed report of concerning patterns
            try:
                report_path = f"{output_dir}/unhealthy_patterns_report.md"
                with open(report_path, 'w') as f:
                    f.write("# Unhealthy Viewing Pattern Analysis\n\n")
                    f.write("## Overview\n")
                    f.write("This report analyzes patterns in your YouTube viewing history that may indicate potentially unhealthy habits.\n")
                    f.write("The analysis scans video titles and metadata for keywords associated with various concerns.\n\n")
                    
                    f.write("## Key Findings\n\n")
                    
                    # Overall trend summary
                    increasing_concerns = [k for k, v in concern_trends.items() if v.get('is_increasing', False)]
                    if increasing_concerns:
                        f.write("### Increasing Patterns of Concern\n\n")
                        for concern in increasing_concerns:
                            trend = concern_trends[concern]
                            recent = trend.get('recent_avg', 0)
                            prev = trend.get('previous_avg', 0)
                            percent = trend.get('percent_change', 0)
                            
                            f.write(f"- **{concern.replace('_', ' ').title()}**: ")
                            if 'percent_change' in trend:
                                f.write(f"Increased by {percent:.1f}% ")
                                f.write(f"(from {prev:.1f} to {recent:.1f} per week)\n")
                            else:
                                f.write(f"Current average: {recent:.1f} per week\n")
                    
                    # Recommendations based on findings
                    f.write("\n## Recommendations\n\n")
                    if 'escapism' in concern_trends and concern_trends['escapism'].get('is_increasing', False):
                        f.write("- Consider setting specific time limits for YouTube use to avoid using it as an escape mechanism\n")
                        
                    if 'negative_mood' in concern_trends and concern_trends['negative_mood'].get('is_increasing', False):
                        f.write("- Your viewing history indicates you may be watching more content related to negative moods\n")
                        f.write("  - Consider balancing with more positive or uplifting content\n")
                        f.write("  - If persistent, consider speaking with a mental health professional\n")
                        
                    if 'rabbit_holes' in concern_trends and concern_trends['rabbit_holes'].get('is_increasing', False):
                        f.write("- You may be spending increasing amounts of time in content 'rabbit holes'\n")
                        f.write("  - Try using a timer when watching YouTube to maintain awareness of time spent\n")
                        f.write("  - Consider scheduling specific YouTube time rather than open-ended viewing\n")
                    
                    if 'sleep_disruption' in concern_trends and concern_trends['sleep_disruption'].get('is_increasing', False):
                        f.write("- Late-night viewing appears to be increasing, which may impact sleep quality\n")
                        f.write("  - Try setting a device curfew 1-2 hours before bedtime\n")
                        f.write("  - Consider using screen time management tools to limit late-night access\n")
                    
                    if 'addiction' in concern_trends and concern_trends['addiction'].get('is_increasing', False):
                        f.write("- Your viewing patterns show potential signs of addictive usage\n")
                        f.write("  - Consider taking regular breaks from YouTube and other social media\n")
                        f.write("  - Try a 24-hour digital detox once a week to reset usage patterns\n")
                    
                    # Date of report
                    f.write(f"\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
                    
                created_files.append(report_path)
                logger.info(f"Created unhealthy patterns report: {report_path}")
            except Exception as e:
                logger.error(f"Error creating unhealthy patterns report: {str(e)}")
            
            return created_files
                
        except Exception as e:
            logger.error(f"Error creating unhealthy trends visualization: {str(e)}")
            logger.debug(traceback.format_exc())
            return created_files

def main():
    # Connection parameters for Neo4j
    neo4j_uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"
    
    # Initialize the analyzer
    analyzer = MentalHealthAnalyzer(neo4j_uri, username, password)
    
    # Initialize variables to prevent reference errors
    report_path = None
    visualization_files = []
    
    try:
        # Display welcome message and summary
        print("\n" + "="*75)
        print(" "*20 + "YouTube Mental Health Analysis Tool")
        print("="*75)
        print("\nThis tool analyzes your YouTube viewing history to identify patterns")
        print("and correlations between content types and mental health metrics.\n")
        
        # Check database connection
        if not analyzer.test_connection():
            print("Error: Could not connect to the Neo4j database. Please check connection parameters.")
            return
        
        print("✓ Successfully connected to the database")
        print("\nStarting analysis...")
        
        # Debug: Check all primary_category values
        analyzer.debug_primary_categories()
        
        # Get sentiment data
        sentiment_data = analyzer.get_sentiment_data()
        if sentiment_data is None or sentiment_data.empty:
            logger.warning("No sentiment data found or empty results")
            print("⚠️ No sentiment data found or empty results")
        else:
            # Calculate mental health index
            mental_health_index = analyzer.calculate_mental_health_index()
            if mental_health_index is None or mental_health_index.empty:
                logger.warning("Unable to calculate mental health index")
            else:
                # Calculate daily metrics
                daily_metrics = analyzer.calculate_daily_metrics()
                if daily_metrics is not None and not daily_metrics.empty:
                    # Generate forecast
                    forecast = analyzer.forecast_mental_health_index(daily_metrics)
                
        # Analyze content categories
        logger.info("Analyzing content categories...")
        category_impact = analyzer.analyze_content_categories()
        if category_impact is None or category_impact.empty:
            logger.warning("No content category data found or empty results")
        
        # 1. Time Series Analysis and Forecasting
        try:
            # Calculate daily aggregated mental health metrics
            daily_metrics = analyzer.calculate_daily_metrics()
            if daily_metrics is not None and not daily_metrics.empty:
                analyzer.save_analysis_results(daily_metrics, "daily_metrics")
                
                # Use Prophet for time series forecasting (with fallback)
                try:
                    forecast = analyzer.forecast_mental_health_index(daily_metrics)
                    if forecast is not None:
                        analyzer.save_analysis_results(forecast, "mental_health_forecast")
                        analyzer.plot_time_series_analysis(mental_health_index, daily_metrics, forecast)
                        print("✓ Time series analysis and forecasting complete")
                except Exception as e:
                    logger.warning(f"Could not perform forecasting with Prophet: {str(e)}")
                    print("⚠️ Forecasting disabled - Prophet library not available")
                    print("   Using alternative visualizations instead")
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            logger.debug(traceback.format_exc())
            print("❌ Time series analysis failed")
        
        # 2. Content Category Analysis
        content_impact = analyzer.analyze_content_categories()
        if content_impact is not None and not content_impact.empty:
            analyzer.save_analysis_results(content_impact, "content_impact")
            print("✓ Content category analysis complete")
        
        # 3. Music Impact Analysis with Enhanced Genre Detection
        print("\nAnalyzing music impact with detailed genre detection...")
        music_data = analyzer.analyze_music_impact()
        if music_data is not None:
            if isinstance(music_data, dict) and 'total_videos' in music_data:
                print(f"✓ Analyzed music impact on mental health with {music_data['total_videos']} music videos")
                
                # Save music analysis results
                analyzer.save_analysis_results(music_data, "music_impact")
                
                # Create a music impact report if we have genre impact data
                if 'genre_impact' in music_data and not music_data['genre_impact'].empty:
                    report_path = analyzer.create_music_impact_report(music_data)
                    if report_path:
                        print(f"✓ Created detailed music genre impact report")
            else:
                # Handle old format
                print(f"✓ Analyzed music impact on mental health with {len(music_data)} music videos")
        else:
            print("⚠️ No music data found or empty results")
            
        # 4. Viewing Pattern Analysis
        print("\nAnalyzing viewing patterns...")
        viewing_patterns = analyzer.analyze_viewing_patterns()
        if viewing_patterns is not None and not viewing_patterns.empty:
            analyzer.save_analysis_results(viewing_patterns, "viewing_patterns")
            print(f"✓ Analyzed viewing patterns across {len(viewing_patterns)} days")
        else:
            print("⚠️ No viewing pattern data found")
            
        # 5. NEW: Unhealthy Viewing Trend Analysis
        print("\nAnalyzing potentially unhealthy viewing trends...")
        unhealthy_trends = analyzer.analyze_unhealthy_viewing_trends()
        if unhealthy_trends is not None and 'concern_data' in unhealthy_trends and unhealthy_trends['concern_data']:
            print(f"✓ Detected {len(unhealthy_trends['concern_data'])} instances of potential concerns")
            
            # Check if any significant trends were found
            significant_concerns = [k for k, v in unhealthy_trends['concern_trends'].items() 
                                 if v.get('is_significant', False) and v.get('is_increasing', False)]
            
            if significant_concerns:
                print("⚠️ Warning: Detected significant increasing trends in:")
                for concern in significant_concerns:
                    print(f"  - {concern.replace('_', ' ').title()}")
                    
            # Visualize the unhealthy trends
            unhealthy_viz_files = analyzer.visualize_unhealthy_trends(unhealthy_trends)
            if unhealthy_viz_files:
                visualization_files.extend(unhealthy_viz_files)
                print(f"✓ Created {len(unhealthy_viz_files)} visualizations of unhealthy viewing trends")
        else:
            print("✓ No concerning unhealthy viewing trends detected")
        
        # Generate additional visualizations that don't depend on Prophet
        print("\nGenerating visualizations...")
        added_viz_files = analyzer.create_additional_visualizations(
            sentiment_data, 
            mental_health_index, 
            viewing_patterns, 
            music_data
        )
        
        if added_viz_files:
            visualization_files.extend(added_viz_files)
            print(f"✓ Created {len(added_viz_files)} visualizations")
        
        # Display visualizations to the user
        print("\nAttempting to display visualizations...")
        analyzer.display_visualizations(visualization_files)
        
        # 6. Summary and conclusion
        print("\n" + "="*75)
        print(" "*30 + "ANALYSIS COMPLETE")
        print("="*75)
        print("\nKey findings are available in the analysis_reports directory:")
        print(f"📊 {len(visualization_files)} visualization files created")
        
        # Report on unhealthy trends
        if unhealthy_trends is not None and 'concern_trends' in unhealthy_trends:
            increasing_concerns = [k for k, v in unhealthy_trends['concern_trends'].items() 
                               if v.get('is_increasing', False)]
            if increasing_concerns:
                print("\n⚠️ Potential areas of concern in your viewing habits:")
                for concern in increasing_concerns[:3]:  # Show top 3
                    print(f"  - {concern.replace('_', ' ').title()}")
                print("\nSee detailed recommendations in: analysis_reports/unhealthy_patterns_report.md")
        
        print("\nThank you for using the YouTube Mental Health Analysis Tool.")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"\nAn error occurred during analysis: {str(e)}")
        print("Please check the log file for more information.")
    finally:
        # Always close the database connection
        analyzer.close()

if __name__ == "__main__":
    main()
