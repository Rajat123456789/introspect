import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a directory for visualizations
import os
if not os.path.exists('insights'):
    os.makedirs('insights')

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.last_valid_timestamp = None  # Store the last valid timestamp
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        with self.driver.session() as session:
            logger.info("Clearing existing database...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully")

    def create_indexes(self):
        with self.driver.session() as session:
            logger.info("Creating indexes...")
            session.run("DROP INDEX video_id IF EXISTS")
            session.run("CREATE INDEX video_id IF NOT EXISTS FOR (v:Video) ON (v.video_id)")
            session.run("DROP INDEX mh_category IF EXISTS")
            session.run("CREATE INDEX mh_category IF NOT EXISTS FOR (m:MentalHealthData) ON (m.category)")
            logger.info("Indexes created successfully")

    def load_main_metadata(self, main_df):
        """Create Video nodes from main metadata CSV."""
        with self.driver.session() as session:
            for idx, row in main_df.iterrows():
                try:
                    props = row.to_dict()
                    # Validate video_id
                    if 'video_id' not in props or pd.isna(props['video_id']) or not str(props['video_id']).strip():
                        logger.warning(f"Main metadata row {idx} missing video_id. Row content: {row.to_dict()}")
                        continue
                    video_id = int(float(str(props['video_id']).strip()))
                    props['video_id'] = video_id
                    
                    # Process watched_at timestamp if present
                    if 'watched_at' in props and pd.notna(props['watched_at']):
                        try:
                            ts_str = props['watched_at']
                            if "PST" in ts_str:
                                ts_str = ts_str.replace("PST", "").strip()
                                ts = pd.to_datetime(ts_str)
                                ts = ts.tz_localize("America/Los_Angeles").tz_convert("UTC")
                            else:
                                ts = pd.to_datetime(ts_str)
                            props['watched_at'] = ts.isoformat()
                            self.last_valid_timestamp = props['watched_at']  # Store the valid timestamp
                        except Exception as e:
                            logger.warning(f"Main metadata row {idx} timestamp parsing error: {str(e)}")
                            # Use the last valid timestamp instead of None
                            props['watched_at'] = self.last_valid_timestamp
                    else:
                        props['watched_at'] = self.last_valid_timestamp  # Use last valid timestamp
                    
                    # Build the Video node query with all columns
                    query = """
                    CREATE (v:Video {
                        video_id: $video_id,
                        title: $title,
                        watched_at: $watched_at,
                        primary_category: $primary_category,
                        detailed_type: $detailed_type,
                        sentiment: $sentiment,
                        sentiment_score: $sentiment_score,
                        primary_format: $primary_format,
                        primary_purpose: $primary_purpose,
                        style: $style,
                        confidence: $confidence
                    })
                    """
                    session.run(query, **props)
                except Exception as e:
                    logger.warning(f"Error creating Video node for main metadata row {idx}: {str(e)}")
                    continue

    def load_engagement_data(self, engagement_df):
        """Create Engagement nodes and relate them to the corresponding Video node."""
        with self.driver.session() as session:
            for idx, row in engagement_df.iterrows():
                try:
                    props = row.to_dict()
                    if 'video_id' not in props or pd.isna(props['video_id']) or not str(props['video_id']).strip():
                        logger.warning(f"Engagement row {idx} missing video_id. Row content: {row.to_dict()}")
                        continue
                    video_id = int(float(str(props['video_id']).strip()))
                    props['video_id'] = video_id

                    # Process engagement timestamp if present
                    if 'timestamp' in props and pd.notna(props['timestamp']):
                        try:
                            ts_str = props['timestamp']
                            if "PST" in ts_str:
                                ts_str = ts_str.replace("PST", "").strip()
                                ts = pd.to_datetime(ts_str)
                                ts = ts.tz_localize("America/Los_Angeles").tz_convert("UTC")
                            else:
                                ts = pd.to_datetime(ts_str)
                            props['timestamp'] = ts.isoformat()
                            self.last_valid_timestamp = props['timestamp']  # Store the valid timestamp
                        except Exception as e:
                            logger.warning(f"Engagement row {idx} timestamp parsing error: {str(e)}")
                            # Use the last valid timestamp instead of None
                            props['timestamp'] = self.last_valid_timestamp
                    else:
                        props['timestamp'] = self.last_valid_timestamp  # Use last valid timestamp

                    query = """
                    MATCH (v:Video { video_id: $video_id })
                    CREATE (e:Engagement {
                        timestamp: $timestamp,
                        content_type: $content_type,
                        audience_engagement: $audience_engagement,
                        production_quality: $production_quality,
                        content_format: $content_format,
                        content_purpose: $content_purpose
                    })
                    CREATE (v)-[:HAS_ENGAGEMENT_DATA]->(e)
                    """
                    session.run(query,
                                video_id=video_id,
                                timestamp=props.get('timestamp'),
                                content_type=props.get('content_type'),
                                audience_engagement=props.get('audience_engagement'),
                                production_quality=props.get('production_quality'),
                                content_format=props.get('content_format'),
                                content_purpose=props.get('content_purpose'))
                except Exception as e:
                    logger.warning(f"Skipping malformed engagement row {idx}: {str(e)}")
                    continue

    def load_mental_health_data(self, mental_health_df):
        """Create MentalHealthData nodes and relate them to the corresponding Video node."""
        with self.driver.session() as session:
            for idx, row in mental_health_df.iterrows():
                try:
                    props = row.to_dict()
                    if 'video_id' not in props or pd.isna(props['video_id']) or not str(props['video_id']).strip():
                        logger.warning(f"Mental health row {idx} missing video_id. Row content: {row.to_dict()}")
                        continue
                    video_id = int(float(str(props['video_id']).strip()))
                    props['video_id'] = video_id

                    # Process timestamp if present
                    if 'timestamp' in props and pd.notna(props['timestamp']):
                        try:
                            ts_str = props['timestamp']
                            if "PST" in ts_str:
                                ts_str = ts_str.replace("PST", "").strip()
                                ts = pd.to_datetime(ts_str)
                                ts = ts.tz_localize("America/Los_Angeles").tz_convert("UTC")
                            else:
                                ts = pd.to_datetime(ts_str)
                            props['timestamp'] = ts.isoformat()
                            self.last_valid_timestamp = props['timestamp']  # Store the valid timestamp
                        except Exception as e:
                            logger.warning(f"Mental health row {idx} timestamp parsing error: {str(e)}")
                            # Use the last valid timestamp instead of None
                            props['timestamp'] = self.last_valid_timestamp
                    else:
                        props['timestamp'] = self.last_valid_timestamp  # Use last valid timestamp
                        
                    query = """
                    MATCH (v:Video { video_id: $video_id })
                    CREATE (m:MentalHealthData {
                        category: $category,
                        score: $score,
                        timestamp: $timestamp,
                        sentiment: $sentiment,
                        sentiment_score: $sentiment_score
                    })
                    CREATE (v)-[:HAS_MENTAL_HEALTH_DATA]->(m)
                    """
                    session.run(query,
                                video_id=video_id,
                                category=props.get('category'),
                                score=props.get('score'),
                                timestamp=props.get('timestamp'),
                                sentiment=props.get('sentiment'),
                                sentiment_score=props.get('sentiment_score'))
                except Exception as e:
                    logger.warning(f"Error processing mental health row {idx}: {str(e)}")
                    continue

    def load_patterns_data(self, patterns_df):
        """Create Pattern nodes and relate them to the corresponding Video node."""
        with self.driver.session() as session:
            for idx, row in patterns_df.iterrows():
                try:
                    props = row.to_dict()
                    if 'video_id' not in props or pd.isna(props['video_id']) or not str(props['video_id']).strip():
                        logger.warning(f"Pattern row {idx} missing video_id. Row content: {row.to_dict()}")
                        continue
                    video_id = int(float(str(props['video_id']).strip()))
                    props['video_id'] = video_id

                    # Process timestamp if present (optional: if you want to store pattern timestamp)
                    if 'timestamp' in props and pd.notna(props['timestamp']):
                        try:
                            ts_str = props['timestamp']
                            if "PST" in ts_str:
                                ts_str = ts_str.replace("PST", "").strip()
                                ts = pd.to_datetime(ts_str)
                                ts = ts.tz_localize("America/Los_Angeles").tz_convert("UTC")
                            else:
                                ts = pd.to_datetime(ts_str)
                            props['timestamp'] = ts.isoformat()
                            self.last_valid_timestamp = props['timestamp']  # Store the valid timestamp
                        except Exception as e:
                            logger.warning(f"Pattern row {idx} timestamp parsing error: {str(e)}")
                            # Use the last valid timestamp instead of None
                            props['timestamp'] = self.last_valid_timestamp
                    else:
                        props['timestamp'] = self.last_valid_timestamp  # Use last valid timestamp
                        
                    query = """
                    MATCH (v:Video { video_id: $video_id })
                    CREATE (p:Pattern {
                        pattern_type: $pattern_type,
                        pattern: $pattern,
                        timestamp: $timestamp,
                        category: $category
                    })
                    CREATE (v)-[:HAS_PATTERN_DATA]->(p)
                    """
                    session.run(query,
                                video_id=video_id,
                                pattern_type=props.get('pattern_type'),
                                pattern=props.get('pattern'),
                                timestamp=props.get('timestamp'),
                                category=props.get('category'))
                except Exception as e:
                    logger.warning(f"Error processing pattern row {idx}: {str(e)}")
                    continue

    # New methods for extracting insights
    
    def get_mental_health_trends(self):
        """Analyze mental health trends over time."""
        with self.driver.session() as session:
            logger.info("Extracting mental health trends...")
            query = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            WHERE v.watched_at IS NOT NULL AND m.timestamp IS NOT NULL
            RETURN v.watched_at AS timestamp, 
                   m.category AS category, 
                   m.score AS score,
                   m.sentiment AS sentiment,
                   m.sentiment_score AS sentiment_score
            ORDER BY timestamp
            """
            result = session.run(query)
            records = list(result)
            logger.info(f"Found {len(records)} mental health trend records")
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame([record.values() for record in records], 
                             columns=['timestamp', 'category', 'score', 'sentiment', 'sentiment_score'])
            
            # Check if we have data
            if df.empty:
                logger.warning("No mental health trend data found")
                return pd.DataFrame(columns=['timestamp', 'category', 'score', 'sentiment_score'])
            
            try:
                logger.info("Converting timestamps...")
                # Convert timestamp to datetime with errors='coerce' - invalid timestamps become NaT
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce', utc=True)
                
                # Ensure score is numeric
                logger.info("Converting scores to numeric values...")
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
                
                # Drop rows with invalid timestamps or scores
                original_len = len(df)
                df = df.dropna(subset=['timestamp', 'score'])
                if len(df) < original_len:
                    logger.warning(f"Dropped {original_len - len(df)} rows with invalid timestamps or scores")
                
                # Ensure all timestamps are timezone-naive to avoid comparison issues
                logger.info("Normalizing timezones...")
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                
                # Store the raw dataframe with valid timestamps
                df_with_valid_timestamps = df.copy()
                
                logger.info("Resampling data by day...")
                # Resample to daily averages
                daily_avg = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'category']).agg({
                    'score': 'mean',
                    'sentiment_score': 'mean'
                }).reset_index()
                
                logger.info(f"Successfully created {len(daily_avg)} daily averages")
                
                # Also create weekly and monthly averages
                logger.info("Creating additional time frequency aggregations...")
                weekly_avg = df.groupby([pd.Grouper(key='timestamp', freq='W'), 'category']).agg({
                    'score': 'mean',
                    'sentiment_score': 'mean'
                }).reset_index()
                
                monthly_avg = df.groupby([pd.Grouper(key='timestamp', freq='M'), 'category']).agg({
                    'score': 'mean',
                    'sentiment_score': 'mean'
                }).reset_index()
                
                # If enough data, create hourly averages (only needed for very large datasets)
                hourly_avg = None
                if len(df) > 1000:
                    hourly_avg = df.groupby([pd.Grouper(key='timestamp', freq='H'), 'category']).agg({
                        'score': 'mean',
                        'sentiment_score': 'mean'
                    }).reset_index()
                
                # Return all aggregations as a dictionary
                return {
                    'raw': df_with_valid_timestamps,
                    'hourly': hourly_avg,
                    'daily': daily_avg,
                    'weekly': weekly_avg,
                    'monthly': monthly_avg
                }
                
            except Exception as e:
                logger.error(f"Error in get_mental_health_trends: {str(e)}")
                return pd.DataFrame(columns=['timestamp', 'category', 'score', 'sentiment_score'])
            
    def get_content_category_mental_health_correlation(self):
        """Analyze correlation between content categories and mental health scores."""
        with self.driver.session() as session:
            logger.info("Extracting content category and mental health correlations...")
            query = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            RETURN v.primary_category AS category,
                   m.category AS mental_health_category,
                   avg(toFloat(m.score)) AS avg_score,
                   count(*) AS count
            ORDER BY avg_score DESC
            """
            result = session.run(query)
            records = list(result)
            logger.info(f"Found {len(records)} category correlation records")
            
            # Convert to DataFrame
            df = pd.DataFrame([record.values() for record in records], 
                             columns=['content_category', 'mental_health_category', 'avg_score', 'count'])
            
            # Filter for statistically significant counts
            significant_df = df[df['count'] >= 5]
            
            return significant_df
    
    def get_viewing_time_patterns(self):
        """Analyze viewing time patterns and their correlation with mental health."""
        with self.driver.session() as session:
            logger.info("Extracting viewing time patterns...")
            query = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            WHERE v.watched_at IS NOT NULL
            WITH v, m, datetime(v.watched_at) AS watch_datetime
            RETURN 
                toString(watch_datetime.hour) AS hour_of_day,
                m.category AS mental_health_category,
                avg(toFloat(m.score)) AS avg_score,
                count(*) AS count
            ORDER BY hour_of_day, mental_health_category
            """
            result = session.run(query)
            records = list(result)
            logger.info(f"Found {len(records)} time pattern records")
            
            # Convert to DataFrame
            df = pd.DataFrame([record.values() for record in records], 
                             columns=['hour_of_day', 'mental_health_category', 'avg_score', 'count'])
            
            # Convert hour to int for proper ordering
            df['hour_of_day'] = df['hour_of_day'].astype(int)
            
            return df
    
    def get_engagement_mental_health_correlation(self):
        """Analyze correlation between engagement metrics and mental health scores."""
        with self.driver.session() as session:
            logger.info("Extracting engagement and mental health correlations...")
            query = """
            MATCH (v:Video)-[:HAS_ENGAGEMENT_DATA]->(e:Engagement)
            MATCH (v)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            RETURN e.audience_engagement AS engagement_level,
                   e.production_quality AS production_quality,
                   e.content_format AS content_format,
                   m.category AS mental_health_category,
                   avg(toFloat(m.score)) AS avg_score,
                   count(*) AS count
            ORDER BY avg_score DESC
            """
            result = session.run(query)
            records = list(result)
            logger.info(f"Found {len(records)} engagement correlation records")
            
            # Convert to DataFrame
            df = pd.DataFrame([record.values() for record in records], 
                             columns=['engagement_level', 'production_quality', 'content_format', 
                                      'mental_health_category', 'avg_score', 'count'])
            
            return df
    
    def get_recurring_patterns(self):
        """Extract recurring patterns and their mental health impacts."""
        with self.driver.session() as session:
            logger.info("Extracting recurring patterns...")
            query = """
            MATCH (v:Video)-[:HAS_PATTERN_DATA]->(p:Pattern)
            MATCH (v)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            RETURN p.pattern_type AS pattern_type,
                   p.pattern AS pattern,
                   m.category AS mental_health_category,
                   avg(toFloat(m.score)) AS avg_score,
                   count(*) AS count
            ORDER BY count DESC, avg_score DESC
            """
            result = session.run(query)
            records = list(result)
            logger.info(f"Found {len(records)} pattern records")
            
            # Convert to DataFrame
            df = pd.DataFrame([record.values() for record in records], 
                             columns=['pattern_type', 'pattern', 'mental_health_category', 'avg_score', 'count'])
            
            # Filter for patterns that occur at least 3 times
            significant_patterns = df[df['count'] >= 3]
            
            return significant_patterns
            
    def get_insight_summary(self):
        """Generate a comprehensive summary of insights for the LLM."""
        insights = {}
        
        # Get all the different analyses
        insights['mental_health_trends'] = self.get_mental_health_trends()
        insights['category_correlations'] = self.get_content_category_mental_health_correlation()
        insights['viewing_time_patterns'] = self.get_viewing_time_patterns()
        insights['engagement_correlations'] = self.get_engagement_mental_health_correlation()
        insights['recurring_patterns'] = self.get_recurring_patterns()
        
        # Additional meta-analysis
        with self.driver.session() as session:
            # Get top positive impact videos
            query_positive = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            WITH v, avg(toFloat(m.score)) AS avg_score
            WHERE avg_score > 0.6
            RETURN v.title AS title, v.primary_category AS category, avg_score
            ORDER BY avg_score DESC
            LIMIT 10
            """
            result = session.run(query_positive)
            insights['top_positive_videos'] = pd.DataFrame([r.values() for r in result], 
                                                  columns=['title', 'category', 'avg_score'])
            
            # Get top negative impact videos
            query_negative = """
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            WITH v, avg(toFloat(m.score)) AS avg_score
            WHERE avg_score < 0.4
            RETURN v.title AS title, v.primary_category AS category, avg_score
            ORDER BY avg_score ASC
            LIMIT 10
            """
            result = session.run(query_negative)
            insights['top_negative_videos'] = pd.DataFrame([r.values() for r in result], 
                                                  columns=['title', 'category', 'avg_score'])
        
        return insights

    def generate_visualizations(self, insights):
        """Generate visualizations for the insights."""
        logger.info("Generating visualizations...")
        
        # 1. Mental Health Trends Over Time with multiple frequencies
        if 'mental_health_trends' in insights:
            mental_health_data = insights['mental_health_trends']
            
            # If the result is a dictionary with different frequencies
            if isinstance(mental_health_data, dict):
                # Check if there's any data to visualize
                has_data = False
                for key in ['daily', 'weekly', 'monthly', 'raw']:
                    if key in mental_health_data and isinstance(mental_health_data[key], pd.DataFrame) and not mental_health_data[key].empty:
                        has_data = True
                        break
                
                if not has_data:
                    logger.warning("No valid mental health trend data available for visualization")
                else:
                    # 1.1 Daily Trends
                    if 'daily' in mental_health_data and isinstance(mental_health_data['daily'], pd.DataFrame) and not mental_health_data['daily'].empty:
                        plt.figure(figsize=(12, 8))
                        for category in mental_health_data['daily']['category'].unique():
                            category_data = mental_health_data['daily'][mental_health_data['daily']['category'] == category]
                            plt.plot(category_data['timestamp'], category_data['score'], label=category)
                        
                        plt.title('Daily Mental Health Scores by Category')
                        plt.xlabel('Date')
                        plt.ylabel('Average Score')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig('insights/mental_health_trends_daily.png')
                        plt.close()
                    
                    # 1.2 Weekly Trends
                    if 'weekly' in mental_health_data and isinstance(mental_health_data['weekly'], pd.DataFrame) and not mental_health_data['weekly'].empty:
                        plt.figure(figsize=(12, 8))
                        for category in mental_health_data['weekly']['category'].unique():
                            category_data = mental_health_data['weekly'][mental_health_data['weekly']['category'] == category]
                            plt.plot(category_data['timestamp'], category_data['score'], label=category, marker='o')
                        
                        plt.title('Weekly Mental Health Scores by Category')
                        plt.xlabel('Week')
                        plt.ylabel('Average Score')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig('insights/mental_health_trends_weekly.png')
                        plt.close()
                    
                    # 1.3 Monthly Trends
                    if 'monthly' in mental_health_data and isinstance(mental_health_data['monthly'], pd.DataFrame) and not mental_health_data['monthly'].empty:
                        plt.figure(figsize=(12, 8))
                        for category in mental_health_data['monthly']['category'].unique():
                            category_data = mental_health_data['monthly'][mental_health_data['monthly']['category'] == category]
                            plt.plot(category_data['timestamp'], category_data['score'], label=category, marker='s', linewidth=2)
                        
                        plt.title('Monthly Mental Health Scores by Category')
                        plt.xlabel('Month')
                        plt.ylabel('Average Score')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig('insights/mental_health_trends_monthly.png')
                        plt.close()
                    
                    # 1.4 Hourly Trends (if available)
                    if 'hourly' in mental_health_data and mental_health_data['hourly'] is not None and isinstance(mental_health_data['hourly'], pd.DataFrame) and not mental_health_data['hourly'].empty:
                        # Create hourly heatmap - usually too many points for a line chart
                        plt.figure(figsize=(14, 10))
                        # Extract hour of day
                        mental_health_data['hourly']['hour'] = mental_health_data['hourly']['timestamp'].dt.hour
                        mental_health_data['hourly']['date'] = mental_health_data['hourly']['timestamp'].dt.date
                        
                        # For each category, create a heatmap of hour vs date
                        for category in mental_health_data['hourly']['category'].unique():
                            category_data = mental_health_data['hourly'][mental_health_data['hourly']['category'] == category]
                            
                            # Only use last 14 days for readability if there are many days
                            unique_dates = sorted(category_data['date'].unique())
                            if len(unique_dates) > 14:
                                recent_dates = unique_dates[-14:]
                                category_data = category_data[category_data['date'].isin(recent_dates)]
                            
                            # Create pivot table for heatmap
                            hourly_pivot = category_data.pivot_table(
                                index='hour',
                                columns='date',
                                values='score'
                            ).fillna(0)
                            
                            if not hourly_pivot.empty:
                                plt.figure(figsize=(16, 8))
                                sns.heatmap(hourly_pivot, annot=False, cmap='RdYlGn', center=0.5)
                                plt.title(f'Hourly Mental Health Scores for {category}')
                                plt.xlabel('Date')
                                plt.ylabel('Hour of Day (0-23)')
                                plt.tight_layout()
                                plt.savefig(f'insights/mental_health_trends_hourly_{category}.png')
                                plt.close()
                    
                    # 1.5 Create a combined view with all frequency trends for each category
                    for category in mental_health_data.get('daily', pd.DataFrame()).get('category', pd.Series()).unique():
                        plt.figure(figsize=(16, 12))
                        
                        # Create a 2x2 grid for different frequencies
                        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
                        fig.suptitle(f'Mental Health Trends for {category} at Different Time Scales', fontsize=16)
                        
                        # Daily trend (top left)
                        if 'daily' in mental_health_data and isinstance(mental_health_data['daily'], pd.DataFrame) and not mental_health_data['daily'].empty:
                            daily_data = mental_health_data['daily'][mental_health_data['daily']['category'] == category]
                            if not daily_data.empty:
                                axs[0, 0].plot(daily_data['timestamp'], daily_data['score'], 'b-')
                                axs[0, 0].set_title('Daily')
                                axs[0, 0].set_xlabel('Date')
                                axs[0, 0].set_ylabel('Score')
                                axs[0, 0].grid(True)
                        
                        # Weekly trend (top right)
                        if 'weekly' in mental_health_data and isinstance(mental_health_data['weekly'], pd.DataFrame) and not mental_health_data['weekly'].empty:
                            weekly_data = mental_health_data['weekly'][mental_health_data['weekly']['category'] == category]
                            if not weekly_data.empty:
                                axs[0, 1].plot(weekly_data['timestamp'], weekly_data['score'], 'g-o')
                                axs[0, 1].set_title('Weekly')
                                axs[0, 1].set_xlabel('Week')
                                axs[0, 1].set_ylabel('Score')
                                axs[0, 1].grid(True)
                        
                        # Monthly trend (bottom left)
                        if 'monthly' in mental_health_data and isinstance(mental_health_data['monthly'], pd.DataFrame) and not mental_health_data['monthly'].empty:
                            monthly_data = mental_health_data['monthly'][mental_health_data['monthly']['category'] == category]
                            if not monthly_data.empty:
                                axs[1, 0].plot(monthly_data['timestamp'], monthly_data['score'], 'r-s', linewidth=2)
                                axs[1, 0].set_title('Monthly')
                                axs[1, 0].set_xlabel('Month')
                                axs[1, 0].set_ylabel('Score')
                                axs[1, 0].grid(True)
                        
                        # Overall trend with moving average (bottom right)
                        if 'raw' in mental_health_data and isinstance(mental_health_data['raw'], pd.DataFrame) and not mental_health_data['raw'].empty:
                            raw_data = mental_health_data['raw'][mental_health_data['raw']['category'] == category]
                            if len(raw_data) > 0:
                                # Sort by timestamp
                                raw_data = raw_data.sort_values('timestamp')
                                # Calculate rolling average if enough data points
                                window_size = min(7, len(raw_data) // 3) if len(raw_data) > 9 else 1
                                if window_size > 1:
                                    raw_data['rolling_avg'] = raw_data['score'].rolling(window=window_size, min_periods=1).mean()
                                    axs[1, 1].plot(raw_data['timestamp'], raw_data['score'], 'gray', alpha=0.3, label='Raw')
                                    axs[1, 1].plot(raw_data['timestamp'], raw_data['rolling_avg'], 'purple', linewidth=2, label=f'{window_size}-point Moving Avg')
                                    axs[1, 1].legend()
                                else:
                                    axs[1, 1].plot(raw_data['timestamp'], raw_data['score'], 'purple', label='Raw Data')
                                axs[1, 1].set_title('Raw with Moving Average')
                                axs[1, 1].set_xlabel('Time')
                                axs[1, 1].set_ylabel('Score')
                                axs[1, 1].grid(True)
                        
                        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
                        plt.savefig(f'insights/mental_health_trends_combined_{category}.png')
                        plt.close()
            
            # If it's just a dataframe (for backward compatibility)
            elif not insights['mental_health_trends'].empty:
                plt.figure(figsize=(12, 8))
                # Plot each mental health category
                for category in insights['mental_health_trends']['category'].unique():
                    category_data = insights['mental_health_trends'][insights['mental_health_trends']['category'] == category]
                    plt.plot(category_data['timestamp'], category_data['score'], label=category)
                
                plt.title('Mental Health Scores Over Time by Category')
                plt.xlabel('Date')
                plt.ylabel('Average Score')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('insights/mental_health_trends.png')
                plt.close()
        
        # 2. Content Category vs Mental Health
        if not insights['category_correlations'].empty:
            plt.figure(figsize=(14, 10))
            pivot_df = insights['category_correlations'].pivot_table(
                index='content_category', 
                columns='mental_health_category', 
                values='avg_score',
                aggfunc='mean'
            ).fillna(0)
            
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.5)
            plt.title('Content Category vs Mental Health Category (Average Score)')
            plt.tight_layout()
            plt.savefig('insights/category_correlations.png')
            plt.close()
        
        # 3. Viewing Time Patterns
        if not insights['viewing_time_patterns'].empty:
            plt.figure(figsize=(14, 8))
            pivot_df = insights['viewing_time_patterns'].pivot_table(
                index='hour_of_day', 
                columns='mental_health_category', 
                values='avg_score',
                aggfunc='mean'
            ).fillna(0)
            
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.5)
            plt.title('Viewing Hour vs Mental Health Category (Average Score)')
            plt.xlabel('Mental Health Category')
            plt.ylabel('Hour of Day (0-23)')
            plt.tight_layout()
            plt.savefig('insights/viewing_time_patterns.png')
            plt.close()
            
            # Also create a count heatmap
            plt.figure(figsize=(14, 8))
            count_pivot = insights['viewing_time_patterns'].pivot_table(
                index='hour_of_day', 
                columns='mental_health_category', 
                values='count',
                aggfunc='sum'
            ).fillna(0)
            
            sns.heatmap(count_pivot, annot=True, fmt='g', cmap='Blues')
            plt.title('Number of Videos Watched by Hour and Mental Health Category')
            plt.xlabel('Mental Health Category')
            plt.ylabel('Hour of Day (0-23)')
            plt.tight_layout()
            plt.savefig('insights/viewing_time_counts.png')
            plt.close()
        
        # 4. Engagement Correlations
        if not insights['engagement_correlations'].empty:
            logger.info("Generating engagement correlation visualizations...")
            
            # Process the engagement data to make it more manageable
            engagement_df = insights['engagement_correlations'].copy()
            
            # 4.1 First, create a simplified engagement level plot using just production quality
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='mental_health_category', y='avg_score', 
                        hue='production_quality', data=engagement_df)
            plt.title('Mental Health Scores by Production Quality')
            plt.xlabel('Mental Health Category')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
            plt.legend(title='Production Quality')
            plt.tight_layout()
            plt.savefig('insights/engagement_quality_correlations.png')
            plt.close()
            
            # 4.2 Create a separate visualization for content format
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='mental_health_category', y='avg_score', 
                        hue='content_format', data=engagement_df)
            plt.title('Mental Health Scores by Content Format')
            plt.xlabel('Mental Health Category')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
            plt.legend(title='Content Format', loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig('insights/engagement_format_correlations.png')
            plt.close()
            
            # 4.3 Extract key elements from complex engagement_level strings
            if 'engagement_level' in engagement_df.columns:
                # Count the most common engagement attributes
                all_attributes = []
                for eng_str in engagement_df['engagement_level'].dropna().astype(str):
                    attributes = [attr.strip() for attr in eng_str.split(',')]
                    all_attributes.extend(attributes)
                
                from collections import Counter
                top_attributes = [item[0] for item in Counter(all_attributes).most_common(5)]
                
                # Create a binary feature for each top attribute
                for attr in top_attributes:
                    engagement_df[f'has_{attr}'] = engagement_df['engagement_level'].str.contains(attr, case=False, na=False)
                
                # Create visualizations for top engagement attributes
                plt.figure(figsize=(14, 10))
                fig, axes = plt.subplots(len(top_attributes), 1, figsize=(12, 4*len(top_attributes)), sharex=True)
                
                for i, attr in enumerate(top_attributes):
                    ax = axes[i] if len(top_attributes) > 1 else axes
                    sns.barplot(x='mental_health_category', y='avg_score', 
                               hue=f'has_{attr}', data=engagement_df, ax=ax)
                    ax.set_title(f'Mental Health Scores by Presence of "{attr}"')
                    ax.set_ylabel('Average Score')
                    ax.legend(title=f'Has {attr}')
                    
                if len(top_attributes) > 1:
                    axes[-1].set_xlabel('Mental Health Category')
                    for ax in axes:
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                else:
                    axes.set_xlabel('Mental Health Category')
                    plt.setp(axes.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig('insights/engagement_attributes_correlations.png')
                plt.close()
            
            # 4.4 Create a heatmap for a cleaner overview
            if not engagement_df.empty:
                try:
                    # Get average scores for the top engagement attributes
                    heatmap_data = pd.DataFrame()
                    
                    for attr in top_attributes:
                        attr_data = engagement_df.groupby(['mental_health_category', f'has_{attr}'])['avg_score'].mean().reset_index()
                        # Only keep rows where has_attr is True
                        attr_data = attr_data[attr_data[f'has_{attr}'] == True]
                        if not attr_data.empty:
                            heatmap_data[attr] = attr_data.set_index('mental_health_category')['avg_score']
                    
                    if not heatmap_data.empty:
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5)
                        plt.title('Impact of Engagement Attributes on Mental Health Scores')
                        plt.tight_layout()
                        plt.savefig('insights/engagement_heatmap.png')
                        plt.close()
                except Exception as e:
                    logger.warning(f"Could not create engagement heatmap: {str(e)}")
        
        # 5. Top Positive/Negative Videos
        if not insights['top_positive_videos'].empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='avg_score', y='title', data=insights['top_positive_videos'])
            plt.title('Top 10 Videos with Positive Mental Health Impact')
            plt.xlabel('Average Mental Health Score')
            plt.ylabel('Video Title')
            plt.tight_layout()
            plt.savefig('insights/top_positive_videos.png')
            plt.close()
            
        if not insights['top_negative_videos'].empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='avg_score', y='title', data=insights['top_negative_videos'])
            plt.title('Top 10 Videos with Negative Mental Health Impact')
            plt.xlabel('Average Mental Health Score')
            plt.ylabel('Video Title')
            plt.tight_layout()
            plt.savefig('insights/top_negative_videos.png')
            plt.close()
            
        logger.info("Visualizations generated and saved to 'insights' directory")
        
    def generate_insights_json(self, insights):
        """Generate a JSON file with all insights for the LLM."""
        logger.info("Generating insights JSON file...")
        
        def convert_to_serializable(obj):
            """Helper function to recursively convert all objects to JSON serializable format."""
            if isinstance(obj, pd.DataFrame):
                # Convert DataFrame to list of records
                return obj.to_dict(orient='records')
            elif isinstance(obj, dict):
                # Recursively process dictionary values
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                # Recursively process list items
                return [convert_to_serializable(item) for item in list(obj)]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                # Basic types are already serializable
                return obj
            else:
                # Try to convert other objects to string
                try:
                    return str(obj)
                except:
                    return None
        
        # Create a serializable insights dict
        serializable_insights = convert_to_serializable(insights)
                
        # Add overall statistics and interpretations
        total_videos = 0
        if 'mental_health_trends' in insights:
            mh_trends = insights['mental_health_trends']
            if isinstance(mh_trends, dict) and 'raw' in mh_trends and isinstance(mh_trends['raw'], pd.DataFrame):
                total_videos = len(mh_trends['raw']['video_id'].unique()) if 'video_id' in mh_trends['raw'].columns else len(mh_trends['raw'])
            elif isinstance(mh_trends, pd.DataFrame):
                total_videos = len(mh_trends['video_id'].unique()) if 'video_id' in mh_trends.columns else len(mh_trends)
        
        serializable_insights['meta'] = {
            'total_videos_analyzed': total_videos,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': self.generate_text_summary(insights)
        }
        
        try:
            # Save to JSON file
            with open('insights/youtube_mental_health_insights.json', 'w') as f:
                json.dump(serializable_insights, f, indent=2)
                
            logger.info("Insights JSON file generated and saved to 'insights/youtube_mental_health_insights.json'")
        except Exception as e:
            logger.error(f"Error saving insights to JSON: {str(e)}")
            # Create a simplified version if the full version fails
            try:
                simplified_insights = {
                    'meta': serializable_insights['meta'],
                    'summary': 'Full JSON serialization failed. See README.md for summary.'
                }
                with open('insights/youtube_mental_health_insights_simplified.json', 'w') as f:
                    json.dump(simplified_insights, f, indent=2)
                logger.info("Simplified insights JSON file generated due to serialization errors")
            except:
                logger.error("Failed to generate even simplified JSON insights")
        
        return serializable_insights
        
    def generate_text_summary(self, insights):
        """Generate a text summary of insights for the LLM."""
        summary = []
        
        # 1. Mental Health Trends
        if 'mental_health_trends' in insights:
            mental_health_data = insights['mental_health_trends']
            
            # Check if we have trend data in the new dictionary format
            if isinstance(mental_health_data, dict) and 'daily' in mental_health_data and isinstance(mental_health_data['daily'], pd.DataFrame) and not mental_health_data['daily'].empty:
                summary.append("MENTAL HEALTH TRENDS OVER TIME:")
                # Get the latest trend direction for each category
                daily_data = mental_health_data['daily']
                for category in daily_data['category'].unique():
                    category_data = daily_data[daily_data['category'] == category]
                    if len(category_data) >= 2:
                        # Sort by timestamp and get last two points
                        category_data = category_data.sort_values('timestamp')
                        last_points = category_data.tail(2)
                        if last_points.iloc[1]['score'] > last_points.iloc[0]['score']:
                            trend = "improving"
                        else:
                            trend = "declining"
                        summary.append(f"- {category}: {trend} (latest score: {last_points.iloc[1]['score']:.2f})")
                summary.append("")
            # For backward compatibility, check if it's a DataFrame
            elif isinstance(mental_health_data, pd.DataFrame) and not mental_health_data.empty:
                summary.append("MENTAL HEALTH TRENDS OVER TIME:")
                # Get the latest trend direction for each category
                for category in mental_health_data['category'].unique():
                    category_data = mental_health_data[mental_health_data['category'] == category]
                    if len(category_data) >= 2:
                        # Sort by timestamp and get last two points
                        category_data = category_data.sort_values('timestamp')
                        last_points = category_data.tail(2)
                        if last_points.iloc[1]['score'] > last_points.iloc[0]['score']:
                            trend = "improving"
                        else:
                            trend = "declining"
                        summary.append(f"- {category}: {trend} (latest score: {last_points.iloc[1]['score']:.2f})")
                summary.append("")
        
        # 2. Content Categories
        if 'category_correlations' in insights and isinstance(insights['category_correlations'], pd.DataFrame) and not insights['category_correlations'].empty:
            summary.append("CONTENT CATEGORIES AND MENTAL HEALTH:")
            # Get top 3 positive categories for each mental health metric
            for mh_category in insights['category_correlations']['mental_health_category'].unique():
                cat_data = insights['category_correlations'][
                    insights['category_correlations']['mental_health_category'] == mh_category
                ].sort_values('avg_score', ascending=False)
                
                if not cat_data.empty:
                    top_cats = cat_data.head(3)
                    summary.append(f"- Top content categories for {mh_category}:")
                    for _, row in top_cats.iterrows():
                        summary.append(f"  * {row['content_category']}: {row['avg_score']:.2f} (based on {row['count']} videos)")
            summary.append("")
        
        # 3. Time Patterns
        if 'viewing_time_patterns' in insights and isinstance(insights['viewing_time_patterns'], pd.DataFrame) and not insights['viewing_time_patterns'].empty:
            summary.append("VIEWING TIME PATTERNS:")
            # Get optimal and worst viewing hours
            for mh_category in insights['viewing_time_patterns']['mental_health_category'].unique():
                time_data = insights['viewing_time_patterns'][
                    insights['viewing_time_patterns']['mental_health_category'] == mh_category
                ]
                
                # Filter for hours with enough data
                significant_hours = time_data[time_data['count'] >= 3]
                if not significant_hours.empty:
                    best_hour = significant_hours.loc[significant_hours['avg_score'].idxmax()]
                    worst_hour = significant_hours.loc[significant_hours['avg_score'].idxmin()]
                    
                    summary.append(f"- {mh_category}:")
                    summary.append(f"  * Best viewing hour: {int(best_hour['hour_of_day']):02d}:00 (score: {best_hour['avg_score']:.2f})")
                    summary.append(f"  * Worst viewing hour: {int(worst_hour['hour_of_day']):02d}:00 (score: {worst_hour['avg_score']:.2f})")
            summary.append("")
        
        # 4. Top video types
        summary.append("TOP VIDEO TYPES FOR MENTAL HEALTH:")
        
        # Add positive videos summary
        if 'top_positive_videos' in insights and isinstance(insights['top_positive_videos'], pd.DataFrame) and not insights['top_positive_videos'].empty:
            summary.append("- Beneficial video types:")
            # Group by category and count occurrences
            category_counts = insights['top_positive_videos']['category'].value_counts().head(3)
            for category, count in category_counts.items():
                summary.append(f"  * {category}: {count} videos in top 10 positive impact")
        
        # Add negative videos summary
        if 'top_negative_videos' in insights and isinstance(insights['top_negative_videos'], pd.DataFrame) and not insights['top_negative_videos'].empty:
            summary.append("- Potentially harmful video types:")
            # Group by category and count occurrences
            category_counts = insights['top_negative_videos']['category'].value_counts().head(3)
            for category, count in category_counts.items():
                summary.append(f"  * {category}: {count} videos in top 10 negative impact")
                
        return "\n".join(summary)

def load_all_csv_data():
    """Load the CSV files and manually assign headers since they lack one."""
    try:
        logger.info("Loading main metadata...")
        main_df = pd.read_csv('../combining-health-and-music/Scripts/output/combined-output/youtube_analysis_COMBINED_main.csv', header=None)
        main_df.columns = ['video_id', 'title', 'watched_at', 'primary_category', 'detailed_type', 'sentiment', 'sentiment_score', 'primary_format', 'primary_purpose', 'style', 'confidence','source_file']
        # Process the full dataset instead of limiting to a sample
        logger.info(f"Main data shape: {main_df.shape}")

        logger.info("Loading mental health data...")
        mental_health_df = pd.read_csv('../combining-health-and-music/Scripts/output/combined-output/youtube_analysis_COMBINED_mental_health.csv', header=None)
        mental_health_df.columns = ['video_id', 'category', 'score', 'timestamp', 'sentiment', 'sentiment_score','source_file']
        # Limit to rows that match the video_ids in main_df
        video_ids = main_df['video_id'].unique()
        mental_health_df = mental_health_df[mental_health_df['video_id'].isin(video_ids)]
        logger.info(f"Mental health data shape: {mental_health_df.shape}")

        logger.info("Loading engagement data...")
        engagement_df = pd.read_csv('../combining-health-and-music/Scripts/output/combined-output/youtube_analysis_COMBINED_engagement.csv', header=None)
        engagement_df.columns = ['video_id', 'timestamp', 'content_type', 'audience_engagement', 'production_quality', 'content_format', 'content_purpose','source_file']
        # Limit to rows that match the video_ids in main_df
        engagement_df = engagement_df[engagement_df['video_id'].isin(video_ids)]
        logger.info(f"Engagement data shape: {engagement_df.shape}")

        logger.info("Loading patterns data...")
        patterns_df = pd.read_csv('../combining-health-and-music/Scripts/output/combined-output/youtube_analysis_COMBINED_patterns.csv', header=None)
        patterns_df.columns = ['video_id', 'pattern_type', 'pattern', 'timestamp', 'category','source_file']
        # Limit to rows that match the video_ids in main_df
        patterns_df = patterns_df[patterns_df['video_id'].isin(video_ids)]
        logger.info(f"Patterns data shape: {patterns_df.shape}")

        return main_df, mental_health_df, engagement_df, patterns_df

    except Exception as e:
        logger.error(f"Error loading data files: {str(e)}")
        raise

def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    
    try:
        main_df, mental_health_df, engagement_df, patterns_df = load_all_csv_data()
        
        # Connect to Neo4j and load data
        neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        neo4j_conn.clear_database()
        neo4j_conn.create_indexes()
        
        logger.info("Loading Video nodes from main metadata...")
        neo4j_conn.load_main_metadata(main_df)
        
        logger.info("Loading Engagement data...")
        neo4j_conn.load_engagement_data(engagement_df)
        
        logger.info("Loading Mental Health data...")
        neo4j_conn.load_mental_health_data(mental_health_df)
        
        logger.info("Loading Patterns data...")
        neo4j_conn.load_patterns_data(patterns_df)
        
        logger.info("Extracting insights from data...")
        # Get all insights from the data
        insights = neo4j_conn.get_insight_summary()
        
        # Generate visualizations
        neo4j_conn.generate_visualizations(insights)
        
        # Generate JSON insights for LLM consumption
        serialized_insights = neo4j_conn.generate_insights_json(insights)
        
        # Create a README.md file with a summary of findings
        create_readme(serialized_insights)
        
        neo4j_conn.close()
        logger.info("All data successfully processed and insights generated!")
        logger.info("Check the 'insights' directory for visualizations and detailed analysis.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def create_readme(insights):
    """Create a README.md file with insights summary."""
    logger.info("Creating README.md with insights summary...")
    
    readme_content = f"""# YouTube Viewing and Mental Health Insights

## Overview
This analysis explores the relationship between YouTube viewing habits and mental health metrics.
The analysis was performed on {insights['meta']['total_videos_analyzed']} videos, processed on {insights['meta']['analysis_timestamp']}.

## Key Findings

{insights['meta']['summary']}

## Visualizations
The following visualizations are available in the 'insights' directory:

1. **Mental Health Trends** - Shows how different mental health metrics change over time
2. **Content Category Correlations** - Shows how different content categories impact mental health
3. **Viewing Time Patterns** - Reveals optimal and suboptimal viewing hours for mental health
4. **Engagement Correlations** - Shows how engagement metrics relate to mental health
5. **Top Positive/Negative Videos** - Lists videos with the most positive/negative impact

## Using These Insights
These insights can be used by:

1. **Content Creators** - To understand what content formats are most beneficial for viewers
2. **Viewers** - To make informed choices about when and what to watch
3. **Mental Health Professionals** - To provide evidence-based recommendations for digital content consumption
4. **LLM Introspection Agents** - To provide personalized recommendations based on an individual's viewing history

## For LLM Consumption
The complete dataset is available in JSON format at `insights/youtube_mental_health_insights.json`.
This file contains detailed records that can be processed by an LLM to provide personalized insights.

## Schema
The data is organized in a graph database with the following structure:

- **Video** nodes - Core metadata about each video
- **MentalHealthData** nodes - Mental health measurements linked to videos
- **Engagement** nodes - Engagement metrics linked to videos
- **Pattern** nodes - Detected viewing patterns linked to videos

## Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    with open('insights/README.md', 'w') as f:
        f.write(readme_content)
    
    logger.info("README.md created successfully in 'insights' directory")

if __name__ == "__main__":
    main()
