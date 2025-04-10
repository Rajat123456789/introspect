import os
import re
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
from neo4j import GraphDatabase
from analyzer.mental_health_analyzer import MentalHealthAnalyzer

# Make sure we have all necessary imports at the top of the file
# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plot style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(style="darkgrid")

# Make sure matplotlib uses a backend that works without a display
plt.switch_backend('agg')

def run_analysis_for_dates(start_date, end_date, output_dir='date_range_analysis'):
    """
    Run analysis for specific date range with environment variables for Neo4j connection.
    This function is designed to be easily called by LLMs.
    
    Args:
        start_date (str): Start date in ISO format (YYYY-MM-DDThh:mm:ss+00:00)
        end_date (str): End date in ISO format (YYYY-MM-DDThh:mm:ss+00:00)
        output_dir (str): Directory to save analysis outputs
        
    Returns:
        dict: Analysis results
    """
    # Get Neo4j connection parameters from environment variables
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "12345678")  # Updated default password to match main.py
    
    # Run the analysis
    results = run_date_range_analysis(uri, user, password, start_date, end_date, output_dir) 

def detect_doom_scrolling(df, threshold=15, time_window_hours=2):
    """
    Detect doom scrolling based on high video consumption in a short time window.
    
    Args:
        df (DataFrame): DataFrame with timestamp data
        threshold (int): Minimum number of videos in time window to consider doom scrolling
        time_window_hours (int): Time window in hours to check for high consumption
        
    Returns:
        DataFrame: Original DataFrame with doom_scrolling flag column added
    """
    if df.empty or 'timestamp' not in df.columns:
        return df
    
    result_df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_dtype(result_df['timestamp']):
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], errors='coerce')
    
    # Sort by timestamp
    result_df = result_df.sort_values('timestamp')
    
    # Initialize doom_scrolling column
    result_df['pattern_doom_scrolling'] = False
    
    # Use more realistic parameters - stricter threshold and smaller time window
    threshold = 25  # Increased threshold - need more videos to consider doom scrolling
    time_window_hours = 1  # Decreased window - 1 hour is a more reasonable window
    
    # Check each video's timestamp against previous ones
    for i in range(len(result_df)):
        current_time = result_df.iloc[i]['timestamp']
        time_window_start = current_time - pd.Timedelta(hours=time_window_hours)
        
        # Count videos in the time window
        videos_in_window = result_df[
            (result_df['timestamp'] >= time_window_start) & 
            (result_df['timestamp'] <= current_time)
        ]
        
        if len(videos_in_window) >= threshold:
            # Mark all videos in this window as part of doom scrolling
            result_df.loc[videos_in_window.index, 'pattern_doom_scrolling'] = True
    
    return result_df 

def detect_rabbit_holes(df, time_column='timestamp', content_columns=['title', 'category'], 
                       min_sequence=4, max_time_gap=timedelta(hours=2)):
    """
    Detect YouTube rabbit holes (going down a specific topic/theme in sequence).
    
    Args:
        df (DataFrame): DataFrame with video data
        time_column (str): Column containing timestamp
        content_columns (list): Columns to use for identifying related content
        min_sequence (int): Minimum videos in sequence to consider a rabbit hole
        max_time_gap (timedelta): Maximum time between videos to be considered a sequence
        
    Returns:
        DataFrame: Original DataFrame with rabbit_hole flag column added
    """
    if df.empty or time_column not in df.columns:
        return df
    
    result_df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_dtype(result_df[time_column]):
        result_df[time_column] = pd.to_datetime(result_df[time_column], errors='coerce')
    
    # Sort by timestamp
    result_df = result_df.sort_values(time_column)
    
    # Extract keyword sets from content columns
    def extract_keywords(row):
        keywords = set()
        for col in content_columns:
            if col in row and isinstance(row[col], str):
                # Extract words, remove common words, keep significant terms
                words = re.findall(r'\b\w+\b', row[col].lower())
                # Remove common words (more comprehensive stop words)
                stop_words = ['this', 'that', 'with', 'from', 'have', 'has', 'had', 'the', 'and', 
                             'for', 'you', 'not', 'are', 'all', 'new', 'who', 'why', 'what', 
                             'when', 'where', 'which', 'how', 'very', 'just', 'more', 'most', 'some']
                words = [w for w in words if len(w) > 3 and w not in stop_words]
                keywords.update(words)
        return keywords
    
    result_df['content_keywords'] = result_df.apply(extract_keywords, axis=1)
    
    # Initialize rabbit hole flag
    result_df['pattern_rabbit_holes'] = False
    result_df['rabbit_hole_id'] = 0
    
    # Use stricter parameters
    min_sequence = 6  # Increased from 4 - need more videos to form a rabbit hole
    max_time_gap = timedelta(minutes=30)  # Reduced from hours to minutes
    min_keyword_overlap = 3  # Require more keyword matches to consider videos related
    
    # Identify rabbit holes by finding sequences of related videos
    current_topics = set()
    sequence_start = 0
    rabbit_hole_id = 1
    
    for i in range(len(result_df)):
        if i == 0:
            current_topics = result_df.iloc[i]['content_keywords']
            continue
            
        # Check time gap
        time_gap = result_df.iloc[i][time_column] - result_df.iloc[i-1][time_column]
        
        # Check content similarity by keyword overlap
        keywords = result_df.iloc[i]['content_keywords']
        overlap = current_topics.intersection(keywords)
        
        # If videos are related in time and content, continue the sequence
        # More strict overlap requirement
        if time_gap <= max_time_gap and len(overlap) >= min_keyword_overlap:
            # Update current topics to include new keywords
            current_topics.update(keywords)
        else:
            # Check if previous sequence qualifies as a rabbit hole
            if i - sequence_start >= min_sequence:
                result_df.loc[result_df.index[sequence_start:i], 'pattern_rabbit_holes'] = True
                result_df.loc[result_df.index[sequence_start:i], 'rabbit_hole_id'] = rabbit_hole_id
                rabbit_hole_id += 1
            
            # Start new sequence
            sequence_start = i
            current_topics = keywords
    
    # Check the last sequence
    if len(result_df) - sequence_start >= min_sequence:
        result_df.loc[result_df.index[sequence_start:], 'pattern_rabbit_holes'] = True
        result_df.loc[result_df.index[sequence_start:], 'rabbit_hole_id'] = rabbit_hole_id
    
    return result_df 

def create_daily_trend_analysis(df, output_dir):
    """
    Create daily trend analysis and visualizations for all patterns.
    
    Args:
        df (DataFrame): DataFrame with pattern flags and timestamp
        output_dir (str): Directory to save visualizations
    """
    try:
        if df.empty or 'timestamp' not in df.columns:
            logger.warning("Cannot create daily trend analysis: missing data")
            return None
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Create date column
        df['date'] = df['timestamp'].dt.date
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                          'pattern_escapism', 'pattern_negative_mood',
                          'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if not available_patterns:
            logger.warning("No pattern columns found for daily trend analysis")
            return None
        
        # Count total videos per day
        daily_counts = df.groupby('date').size().reset_index(name='total_videos')
        
        # Count pattern occurrences per day
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            pattern_daily = df[df[pattern]].groupby('date').size().reset_index(name=pattern_name)
            daily_counts = daily_counts.merge(pattern_daily, on='date', how='left')
            
        # Fill NaN with 0
        daily_counts = daily_counts.fillna(0)
        
        # Calculate percentages
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            daily_counts[f'{pattern_name}_pct'] = (daily_counts[pattern_name] / daily_counts['total_videos'] * 100).round(1)
        
        # Save to CSV
        daily_counts.to_csv(f"{output_dir}/daily_pattern_trends.csv", index=False)
        
        # Create daily trend visualization
        plt.figure(figsize=(14, 8))
        
        # Plot absolute counts
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            plt.plot(daily_counts['date'], daily_counts[pattern_name], marker='o', label=pattern_name)
        
        plt.title('Daily Pattern Counts', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_pattern_counts.png")
        plt.close()
        
        # Create percentage visualization
        plt.figure(figsize=(14, 8))
        
        # Plot percentages
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            plt.plot(daily_counts['date'], daily_counts[f'{pattern_name}_pct'], marker='o', label=pattern_name)
        
        plt.title('Daily Pattern Percentages', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Percentage of Daily Videos (%)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_pattern_percentages.png")
        plt.close()
        
        # Create stacked area chart
        plt.figure(figsize=(14, 8))
        
        # Get pattern names
        pattern_names = [p.replace('pattern_', '') for p in available_patterns]
        
        # Create stacked area chart
        plt.stackplot(daily_counts['date'], 
                     [daily_counts[name] for name in pattern_names],
                     labels=pattern_names,
                     alpha=0.7)
        
        plt.title('Daily Pattern Distribution (Stacked)', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_pattern_stacked.png")
        plt.close()
        
        # Create heatmap of daily patterns
        daily_pivot = daily_counts.copy()
        daily_pivot['weekday'] = pd.to_datetime(daily_pivot['date']).dt.day_name()
        daily_pivot['week'] = pd.to_datetime(daily_pivot['date']).dt.isocalendar().week
        
        # Create separate heatmaps for each pattern
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            
            # Skip if not enough data
            if daily_pivot[pattern_name].sum() < 5:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Pivot for heatmap
            try:
                pivot_data = daily_pivot.pivot(index='week', columns='weekday', values=pattern_name)
                
                # Ensure proper weekday order
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                pivot_data = pivot_data.reindex(columns=days_order)
                
                # Create heatmap
                sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.0f', linewidths=.5)
                
                plt.title(f'{pattern_name.replace("_", " ").title()} by Day of Week', fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/daily_{pattern_name}_heatmap.png")
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create heatmap for {pattern_name}: {str(e)}")
        
        return daily_counts
        
    except Exception as e:
        logger.error(f"Error creating daily trend analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def create_monthly_trend_analysis(df, output_dir):
    """
    Create monthly trend analysis and visualizations for all patterns.
    
    Args:
        df (DataFrame): DataFrame with pattern flags and timestamp
        output_dir (str): Directory to save visualizations
    """
    try:
        if df.empty or 'timestamp' not in df.columns:
            logger.warning("Cannot create monthly trend analysis: missing data")
            return None
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Create month column (as string for better display)
        df['month'] = df['timestamp'].dt.strftime('%Y-%m')
        
        # Check if we have multiple months of data
        unique_months = df['month'].nunique()
        if unique_months < 2:
            logger.warning("Not enough months for monthly trend analysis (found only 1)")
            # If we only have one month, use days instead
            return create_daily_trend_analysis(df, output_dir)
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                          'pattern_escapism', 'pattern_negative_mood',
                          'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if not available_patterns:
            logger.warning("No pattern columns found for monthly trend analysis")
            return None
        
        # Count total videos per month
        monthly_counts = df.groupby('month').size().reset_index(name='total_videos')
        
        # Count pattern occurrences per month
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            pattern_monthly = df[df[pattern]].groupby('month').size().reset_index(name=pattern_name)
            monthly_counts = monthly_counts.merge(pattern_monthly, on='month', how='left')
            
        # Fill NaN with 0
        monthly_counts = monthly_counts.fillna(0)
        
        # Calculate percentages
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            monthly_counts[f'{pattern_name}_pct'] = (monthly_counts[pattern_name] / monthly_counts['total_videos'] * 100).round(1)
        
        # Save to CSV
        monthly_counts.to_csv(f"{output_dir}/monthly_pattern_trends.csv", index=False)
        
        # Create monthly trend visualization
        plt.figure(figsize=(14, 8))
        
        # Plot absolute counts
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            plt.plot(monthly_counts['month'], monthly_counts[pattern_name], marker='o', label=pattern_name)
        
        plt.title('Monthly Pattern Counts', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_pattern_counts.png")
        plt.close()
        
        # Create percentage visualization
        plt.figure(figsize=(14, 8))
        
        # Plot percentages
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            plt.plot(monthly_counts['month'], monthly_counts[f'{pattern_name}_pct'], marker='o', label=pattern_name)
        
        plt.title('Monthly Pattern Percentages', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Percentage of Monthly Videos (%)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_pattern_percentages.png")
        plt.close()
        
        # Create stacked bar chart
        plt.figure(figsize=(14, 8))
        
        # Prepare data for stacked bar
        pattern_names = [p.replace('pattern_', '') for p in available_patterns]
        data = []
        for pattern in pattern_names:
            data.append(monthly_counts[pattern])
            
        # Create stacked bar
        bottom = np.zeros(len(monthly_counts))
        for i, d in enumerate(data):
            plt.bar(monthly_counts['month'], d, bottom=bottom, label=pattern_names[i], alpha=0.7)
            bottom += d
            
        plt.title('Monthly Pattern Distribution', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_pattern_stacked.png")
        plt.close()
        
        # Create pattern trend heatmap
        plt.figure(figsize=(12, 10))
        
        # Prepare data for heatmap
        heatmap_data = monthly_counts[['month'] + pattern_names].set_index('month')
        
        # Create heatmap
        sns.heatmap(heatmap_data.T, cmap='viridis', annot=True, fmt='.0f', linewidths=.5)
        
        plt.title('Monthly Pattern Distribution Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_pattern_heatmap.png")
        plt.close()
        
        return monthly_counts
        
    except Exception as e:
        logger.error(f"Error creating monthly trend analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None 

def create_doom_scrolling_visualizations(doom_dates, output_dir):
    """
    Create specialized visualizations for doom scrolling by day.
    
    Args:
        doom_dates (DataFrame): DataFrame with doom scrolling counts by date
        output_dir (str): Directory to save visualizations
    """
    try:
        if doom_dates.empty:
            logger.warning("No doom scrolling data to visualize")
            return
            
        # Make sure doom_dates is properly formatted
        doom_dates = pd.DataFrame(doom_dates)
        
        # Ensure columns are present
        required_columns = ['date', 'count', 'percent']
        for col in required_columns:
            if col not in doom_dates.columns:
                logger.warning(f"Missing required column {col} in doom_dates")
                return
                
        # Make a copy to avoid modifying the original
        plot_df = doom_dates.copy()
        
        # Convert date strings to datetime for proper sorting
        if plot_df['date'].dtype == 'object':
            plot_df['date_dt'] = pd.to_datetime(plot_df['date'])
            plot_df = plot_df.sort_values('date_dt')
        else:
            plot_df = plot_df.sort_values('date')
            
        # Log what we're about to plot
        logger.info(f"Creating doom scrolling visualizations for {len(plot_df)} days")
        logger.info(f"Sample data: {plot_df.head(3).to_dict('records')}")
        
        # Create count visualization
        plt.figure(figsize=(14, 8))
        
        # Use both seaborn and direct matplotlib for robustness
        try:
            # Try seaborn first
            ax = sns.barplot(x='date', y='count', data=plot_df)
            # Improve x-axis labels
            if 'date_dt' in plot_df.columns:
                plt.xticks(range(len(plot_df)), plot_df['date_dt'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
        except Exception as e:
            logger.warning(f"Seaborn barplot failed: {str(e)}, falling back to matplotlib")
            # Fall back to matplotlib
            plt.bar(range(len(plot_df)), plot_df['count'])
            if 'date_dt' in plot_df.columns:
                plt.xticks(range(len(plot_df)), plot_df['date_dt'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
            else:
                plt.xticks(range(len(plot_df)), plot_df['date'], rotation=45, ha='right')
        
        plt.title('Daily Doom Scrolling Instances', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Number of Doom Scrolling Videos', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        doom_scrolling_viz_path = f"{output_dir}/daily_doom_scrolling.png"
        plt.savefig(doom_scrolling_viz_path, dpi=100)
        plt.close()
        logger.info(f"Saved doom scrolling count visualization to {doom_scrolling_viz_path}")
        
        # Create percentage visualization
        plt.figure(figsize=(14, 8))
        
        # Use both seaborn and direct matplotlib for robustness
        try:
            # Try seaborn first
            ax = sns.barplot(x='date', y='percent', data=plot_df)
            # Improve x-axis labels
            if 'date_dt' in plot_df.columns:
                plt.xticks(range(len(plot_df)), plot_df['date_dt'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
        except Exception as e:
            logger.warning(f"Seaborn barplot failed: {str(e)}, falling back to matplotlib")
            # Fall back to matplotlib
            plt.bar(range(len(plot_df)), plot_df['percent'])
            if 'date_dt' in plot_df.columns:
                plt.xticks(range(len(plot_df)), plot_df['date_dt'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')
            else:
                plt.xticks(range(len(plot_df)), plot_df['date'], rotation=45, ha='right')
                
        plt.title('Daily Doom Scrolling Percentage', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Percentage of Daily Videos (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        doom_scrolling_pct_viz_path = f"{output_dir}/daily_doom_scrolling_percent.png"
        plt.savefig(doom_scrolling_pct_viz_path, dpi=100)
        plt.close()
        logger.info(f"Saved doom scrolling percentage visualization to {doom_scrolling_pct_viz_path}")
        
        return [doom_scrolling_viz_path, doom_scrolling_pct_viz_path]
        
    except Exception as e:
        logger.error(f"Error creating doom scrolling visualizations: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def run_date_range_analysis(uri, user, password, start_date=None, end_date=None, output_dir='analysis_results'):
    """
    Run a comprehensive date range analysis of YouTube viewing patterns.
    
    Args:
        uri (str): Neo4j database URI
        user (str): Neo4j username
        password (str): Neo4j password
        start_date (str): Start date for analysis (YYYY-MM-DD) or None for earliest date
        end_date (str): End date for analysis (YYYY-MM-DD) or None for latest date
        output_dir (str): Directory to save analysis results
        
    Returns:
        dict: Analysis results containing patterns, trends, and summary statistics
    """
    import logging
    import os
    from datetime import datetime, timedelta
    import json
    
    logger = logging.getLogger(__name__)
    
    # Initialize analysis results dictionary
    analysis_results = {
        'date_range': {'start': start_date, 'end': end_date},
        'summary': {},
        'patterns': {
            'doom_scrolling': {'count': 0, 'percentage': 0, 'videos': [], 'dates': []},
            'rabbit_holes': {'count': 0, 'percentage': 0, 'videos': [], 'sequences': 0},
            'negative_mood': {'count': 0, 'percentage': 0, 'videos': []},
            'addiction': {'count': 0, 'percentage': 0, 'videos': []}
        },
        'trends': {'daily': {}, 'monthly': {}},
        'visualizations': [],
        'warnings': []
    }
    
    try:
        # Initialize analyzer
        analyzer = MentalHealthAnalyzer(uri, user, password)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get video data for specified date range using the original approach
        with analyzer.driver.session() as session:
            # Use default dates if none provided
            if not start_date:
                # Default to 14 days ago
                start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%dT00:00:00+00:00')
                analysis_results['date_range']['start'] = start_date
                
            if not end_date:
                # Default to today
                end_date = datetime.now().strftime('%Y-%m-%dT23:59:59+00:00')
                analysis_results['date_range']['end'] = end_date
                
            logger.info(f"Querying data from {start_date} to {end_date}")
                
            # Comprehensive query to get all relevant video data
            query = f"""
            MATCH (v:Video)
            WHERE v.watched_at IS NOT NULL 
                AND v.watched_at >= '{start_date}' 
                AND v.watched_at <= '{end_date}'
            OPTIONAL MATCH (v)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            RETURN v.title AS title,
                   v.primary_category AS category,
                   v.detailed_type AS subcategory,
                   v.description AS description,
                   v.video_id AS video_id,
                   v.watched_at AS watched_at,
                   v.channel_name AS channel,
                   CASE WHEN m IS NOT NULL THEN m.score
                        WHEN v.score IS NOT NULL THEN v.score
                        WHEN v.sentiment_score IS NOT NULL THEN v.sentiment_score
                        ELSE 0.5 END AS score,
                   CASE WHEN m IS NOT NULL THEN m.sentiment ELSE 'NEUTRAL' END AS sentiment
            ORDER BY v.watched_at
            """
            result = session.run(query)
            df_videos = pd.DataFrame([dict(record) for record in result])
        
        if df_videos.empty:
            logger.warning("No data found for the specified date range")
            analysis_results['summary']['total_videos'] = 0
            return analysis_results
        
        # Convert timestamps
        df_videos['timestamp'] = pd.to_datetime(df_videos['watched_at'], errors='coerce')
        
        # Update date range with actual dates from data
        min_date = df_videos['timestamp'].min()
        max_date = df_videos['timestamp'].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            min_date_str = pd.to_datetime(min_date).strftime('%Y-%m-%d')
            max_date_str = pd.to_datetime(max_date).strftime('%Y-%m-%d')
            
            analysis_results['date_range']['start'] = min_date_str
            analysis_results['date_range']['end'] = max_date_str
        
        # Total videos
        total_videos = len(df_videos)
        analysis_results['summary']['total_videos'] = total_videos
        
        # Calculate date range span in days
        start_dt = pd.to_datetime(analysis_results['date_range']['start'])
        end_dt = pd.to_datetime(analysis_results['date_range']['end'])
        date_range_days = (end_dt - start_dt).days + 1
        analysis_results['summary']['date_range_days'] = date_range_days
        
        # Log important info about the analysis
        logger.info(f"Analyzing {total_videos} videos from {start_dt} to {end_dt} ({date_range_days} days)")
        
        # Check if date range is too short for reliable pattern detection
        if date_range_days < 7:
            warning = f"Date range is only {date_range_days} days, which may be too short for reliable pattern detection."
            analysis_results['warnings'].append(warning)
            logger.warning(warning)
        
        # Adjust parameters based on date range
        if date_range_days > 90:  # For long date ranges, increase thresholds
            addiction_daily_threshold = 60
            addiction_consecutive_days = 4 if date_range_days > 180 else 3
            doom_scrolling_threshold = 30 if date_range_days > 180 else 25
            rabbit_hole_min_sequence = 7 if date_range_days > 180 else 6
        else:  # For shorter date ranges
            addiction_daily_threshold = 60
            addiction_consecutive_days = 3
            doom_scrolling_threshold = 20
            rabbit_hole_min_sequence = 5
        
        logger.info(f"Using adjusted parameters based on {date_range_days} day range: " +
                    f"addiction_daily_threshold={addiction_daily_threshold}, " +
                    f"addiction_consecutive_days={addiction_consecutive_days}, " +
                    f"doom_scrolling_threshold={doom_scrolling_threshold}, " +
                    f"rabbit_hole_min_sequence={rabbit_hole_min_sequence}")
        
        # Analyze patterns
        # 1. Addiction pattern - use adjusted parameters based on date range
        df_videos = detect_addiction_pattern(
            df_videos, 
            daily_threshold=addiction_daily_threshold,
            daily_consecutive_days=addiction_consecutive_days
        )
        addiction_videos = df_videos[df_videos['pattern_addiction']].sort_values('addiction_score', ascending=False)
        addiction_count = len(addiction_videos)
        addiction_percentage = (addiction_count / total_videos) * 100 if total_videos > 0 else 0
        
        analysis_results['patterns']['addiction']['count'] = addiction_count
        analysis_results['patterns']['addiction']['percentage'] = addiction_percentage
        
        # Add top addiction videos to results
        for _, video in addiction_videos.head(10).iterrows():
            video_info = {
                'title': video.get('title', 'Unknown'),
                'channel': video.get('channel', 'Unknown'),
                'category': video.get('category', 'Unknown'),
                'score': float(video.get('addiction_score', 0))
            }
            analysis_results['patterns']['addiction']['videos'].append(video_info)
        
        # 2. Doom scrolling pattern - use adjusted threshold
        df_videos = detect_doom_scrolling(df_videos, threshold=doom_scrolling_threshold)
        doom_scrolling_videos = df_videos[df_videos['pattern_doom_scrolling']]
        doom_scrolling_count = len(doom_scrolling_videos)
        doom_scrolling_percentage = (doom_scrolling_count / total_videos) * 100 if total_videos > 0 else 0
        
        analysis_results['patterns']['doom_scrolling']['count'] = doom_scrolling_count
        analysis_results['patterns']['doom_scrolling']['percentage'] = doom_scrolling_percentage
        
        # Create a date column if it doesn't exist
        if 'date' not in df_videos.columns:
            df_videos['date'] = pd.to_datetime(df_videos['timestamp']).dt.date
            
        # Group doom scrolling videos by date to show how many days they occurred
        if not doom_scrolling_videos.empty:
            # Ensure date column exists
            doom_scrolling_videos['date'] = pd.to_datetime(doom_scrolling_videos['timestamp']).dt.date
            
            # Group by date and count doom scrolling videos per day
            doom_scrolling_by_date = doom_scrolling_videos.groupby('date').size().reset_index(name='count')
            
            # For each date with doom scrolling, calculate percentage of total videos that day
            daily_totals = df_videos.groupby('date').size().reset_index(name='total')
            doom_scrolling_by_date = pd.merge(doom_scrolling_by_date, daily_totals, on='date', how='left')
            doom_scrolling_by_date['percent'] = (doom_scrolling_by_date['count'] / doom_scrolling_by_date['total']) * 100
            
            # Sort by count descending to get days with most doom scrolling
            doom_scrolling_by_date = doom_scrolling_by_date.sort_values('count', ascending=False)
            
            # Save to CSV for debugging
            doom_scrolling_by_date_path = os.path.join(output_dir, 'doom_scrolling_by_date.csv')
            doom_scrolling_by_date.to_csv(doom_scrolling_by_date_path, index=False)
            
            # Add to results
            for _, row in doom_scrolling_by_date.iterrows():
                date_info = {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'count': int(row['count']),
                    'percent': float(row['percent'])
                }
                analysis_results['patterns']['doom_scrolling']['dates'].append(date_info)
        
        # Add top doom scrolling videos to results  
        for _, video in doom_scrolling_videos.head(10).iterrows():
            video_info = {
                'title': video.get('title', 'Unknown'),
                'channel': video.get('channel', 'Unknown'),
                'category': video.get('category', 'Unknown')
            }
            analysis_results['patterns']['doom_scrolling']['videos'].append(video_info)
        
        # 3. Rabbit hole pattern - use adjusted min_sequence
        df_videos, rabbit_hole_sequences = detect_rabbit_holes(
            df_videos, 
            min_sequence=rabbit_hole_min_sequence,
            return_sequences=True
        )
        rabbit_hole_videos = df_videos[df_videos['pattern_rabbit_holes']]
        rabbit_hole_count = len(rabbit_hole_videos)
        rabbit_hole_percentage = (rabbit_hole_count / total_videos) * 100 if total_videos > 0 else 0
        
        analysis_results['patterns']['rabbit_holes']['count'] = rabbit_hole_count
        analysis_results['patterns']['rabbit_holes']['percentage'] = rabbit_hole_percentage
        analysis_results['patterns']['rabbit_holes']['sequences'] = len(rabbit_hole_sequences)
        
        # Add top rabbit hole videos to results
        for _, video in rabbit_hole_videos.head(10).iterrows():
            video_info = {
                'title': video.get('title', 'Unknown'),
                'channel': video.get('channel', 'Unknown'),
                'category': video.get('category', 'Unknown')
            }
            analysis_results['patterns']['rabbit_holes']['videos'].append(video_info)
        
        # 4. Negative mood pattern
        df_videos = detect_negative_mood(df_videos)
        negative_mood_videos = df_videos[df_videos['pattern_negative_mood']]
        negative_mood_count = len(negative_mood_videos)
        negative_mood_percentage = (negative_mood_count / total_videos) * 100 if total_videos > 0 else 0
        
        analysis_results['patterns']['negative_mood']['count'] = negative_mood_count
        analysis_results['patterns']['negative_mood']['percentage'] = negative_mood_percentage
        
        # Add top negative mood videos to results
        for _, video in negative_mood_videos.head(10).iterrows():
            video_info = {
                'title': video.get('title', 'Unknown'),
                'channel': video.get('channel', 'Unknown'),
                'category': video.get('category', 'Unknown'),
                'sentiment_score': float(video.get('sentiment_score', 0))
            }
            analysis_results['patterns']['negative_mood']['videos'].append(video_info)
        
        # SANITY CHECK: If any pattern is detected in more than X% of videos, issue a warning
        # as this may indicate the detection parameters need adjustment
        warnings = []
        
        if addiction_percentage > 50:
            warning = f"Addiction pattern detected in {addiction_percentage:.1f}% of videos, which is unusually high."
            warnings.append(warning)
            logger.warning(warning)
            
        if doom_scrolling_percentage > 50:
            warning = f"Doom scrolling pattern detected in {doom_scrolling_percentage:.1f}% of videos, which is unusually high."
            warnings.append(warning)
            logger.warning(warning)
            
        if rabbit_hole_percentage > 60:
            warning = f"Rabbit hole pattern detected in {rabbit_hole_percentage:.1f}% of videos, which is unusually high."
            warnings.append(warning)
            logger.warning(warning)
        
        if warnings:
            warning = "These high detection rates suggest the pattern detection parameters may need adjustment."
            warnings.append(warning)
            analysis_results['warnings'].extend(warnings)
        
        # Create daily trend analysis
        daily_trends = create_daily_trend_analysis(df_videos, output_dir)
        analysis_results['trends']['daily'] = daily_trends
        
        # Create monthly trend analysis
        monthly_trends = create_monthly_trend_analysis(df_videos, output_dir)
        analysis_results['trends']['monthly'] = monthly_trends
        
        # Generate reports
        generate_plaintext_report(analysis_results, df_videos, output_dir)
        generate_json_report(analysis_results, output_dir)
        
        logger.info(f"Analysis completed successfully with {total_videos} videos.")
        
    except Exception as e:
        error_msg = f"Error in date range analysis: {str(e)}"
        logger.error(error_msg)
        analysis_results['error'] = error_msg
        import traceback
        logger.debug(traceback.format_exc())
        
    return analysis_results

def generate_plaintext_report(analysis_results, df, output_dir):
    """
    Generate a plaintext report summarizing the analysis findings.
    
    Args:
        analysis_results (dict): Analysis results dictionary
        df (DataFrame): DataFrame with video data and pattern flags
        output_dir (str): Directory to save report
    """
    try:
        report_lines = []
        
        # Report header
        report_lines.append("=" * 80)
        report_lines.append("YOUTUBE VIEWING PATTERN ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Date range
        start_date = analysis_results['date_range']['start']
        end_date = analysis_results['date_range']['end']
        report_lines.append(f"Analysis Period: {start_date} to {end_date}")
        report_lines.append("")
        
        # Include warnings at the top if they exist
        if 'warnings' in analysis_results and analysis_results['warnings']:
            report_lines.append("IMPORTANT ANALYSIS WARNINGS")
            report_lines.append("-" * 40)
            for warning in analysis_results['warnings']:
                report_lines.append(warning)
            report_lines.append("")
            report_lines.append("These warnings suggest the pattern detection parameters may need adjustment.")
            report_lines.append("Consider the following options:")
            report_lines.append("1. Increase the threshold for pattern detection")
            report_lines.append("2. Analyze a smaller date range with more typical usage")
            report_lines.append("3. Customize the parameters in the detection functions to match your usage patterns")
            report_lines.append("")
        
        # Summary statistics
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            report_lines.append("SUMMARY STATISTICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Videos Analyzed: {summary.get('total_videos', 'N/A')}")
            if 'avg_score' in summary:
                report_lines.append(f"Average Mental Health Score: {summary['avg_score']:.2f} (higher is better)")
            report_lines.append("")
        
        # Pattern analysis
        if 'patterns' in analysis_results:
            report_lines.append("PATTERN DETECTION RESULTS")
            report_lines.append("-" * 40)
            
            patterns = analysis_results['patterns']
            for pattern_name, pattern_data in patterns.items():
                count = pattern_data.get('count', 0)
                total = analysis_results['summary'].get('total_videos', 1)
                percentage = pattern_data.get('percentage', 0)
                if percentage == 0 and total > 0:
                    percentage = (count / total) * 100
                
                report_lines.append(f"{pattern_name.replace('_', ' ').title()}: {count} videos ({percentage:.1f}%)")
                
                # Add special details for each pattern type
                if pattern_name == 'doom_scrolling' and 'dates' in pattern_data and pattern_data['dates']:
                    # Enhanced doom scrolling reporting
                    doom_dates = pattern_data['dates']
                    
                    # Count total days with doom scrolling
                    report_lines.append(f"  Doom scrolling detected on {len(doom_dates)} different days")
                    
                    # Find days with highest doom scrolling
                    report_lines.append("  Top doom scrolling days:")
                    for i, date_info in enumerate(doom_dates[:5], 1):
                        date_str = date_info.get('date', 'Unknown')
                        count = date_info.get('count', 0)
                        percent = date_info.get('percent', 0)
                        report_lines.append(f"    {i}. {date_str}: {count} videos ({percent:.1f}% of daily videos)")
                    
                    # Check for unrealistic detection rate
                    if percentage > 50:
                        report_lines.append("  NOTE: The detection rate for doom scrolling is very high.")
                        report_lines.append("  This might indicate that the threshold needs adjustment.")
                        report_lines.append("  Consider increasing the detection threshold for more realistic results.")
                    
                    # Calculate frequency metrics
                    total_days = len(set(pd.to_datetime(df['timestamp']).dt.date))
                    frequency = (len(doom_dates) / total_days) * 100 if total_days > 0 else 0
                    report_lines.append(f"  Doom scrolling frequency: {frequency:.1f}% of days show doom scrolling behavior")
                    
                    # Calculate average daily doom scrolling
                    if doom_dates:
                        avg_videos = sum(date_info.get('count', 0) for date_info in doom_dates) / len(doom_dates)
                        report_lines.append(f"  Average videos per doom scrolling day: {avg_videos:.1f}")
                    
                    # Calculate day of week frequency for doom scrolling
                    try:
                        dow_counts = {}
                        for date_info in doom_dates:
                            date_str = date_info.get('date', '')
                            if date_str:
                                day_name = pd.to_datetime(date_str).day_name()
                                if day_name not in dow_counts:
                                    dow_counts[day_name] = 0
                                dow_counts[day_name] += 1
                        
                        if dow_counts:
                            max_day = max(dow_counts.items(), key=lambda x: x[1])
                            report_lines.append(f"  Most common day for doom scrolling: {max_day[0]} ({max_day[1]} occurrences)")
                    except Exception as e:
                        logger.warning(f"Error analyzing doom scrolling day of week: {str(e)}")
                
                elif pattern_name == 'addiction' and percentage > 50:
                    report_lines.append("  NOTE: The detection rate for addiction is very high.")
                    report_lines.append("  This might indicate that the threshold parameters need adjustment.")
                    report_lines.append("  Consider increasing daily_threshold or daily_consecutive_days in detect_addiction_pattern.")
                    report_lines.append("  Typical values: daily_threshold=20-30, daily_consecutive_days=3-5")
                
                elif pattern_name == 'rabbit_holes' and 'sequences' in pattern_data:
                    report_lines.append(f"  Identified {pattern_data['sequences']} distinct rabbit hole sequences")
                    if percentage > 60:
                        report_lines.append("  NOTE: The detection rate for rabbit holes is very high.")
                        report_lines.append("  Consider increasing min_sequence or decreasing max_time_gap in detect_rabbit_holes.")
                
                # Add example videos for each pattern
                if 'videos' in pattern_data and pattern_data['videos']:
                    report_lines.append("  Example videos:")
                    for i, video in enumerate(pattern_data['videos'][:3], 1):
                        title = video.get('title', 'Unknown')
                        category = video.get('category', 'Unknown')
                        report_lines.append(f"  {i}. {title} ({category})")
                
                report_lines.append("")
        
        # Daily Trend Analysis
        if 'trends' in analysis_results and 'daily' in analysis_results['trends'] and analysis_results['trends']['daily']:
            report_lines.append("DAILY VIEWING PATTERN TRENDS")
            report_lines.append("-" * 40)
            
            # Get daily trend data
            daily_trends = analysis_results['trends']['daily']
            if daily_trends:
                # Find days with highest pattern activity
                pattern_peaks = {}
                for record in daily_trends:
                    for key, value in record.items():
                        if key.endswith('_pct') and isinstance(value, (int, float)) and value > 0:
                            pattern_name = key.replace('_pct', '')
                            if pattern_name not in pattern_peaks or value > pattern_peaks[pattern_name]['percentage']:
                                pattern_peaks[pattern_name] = {
                                    'date': record.get('date', 'Unknown'),
                                    'percentage': value,
                                    'count': record.get(pattern_name, 0)
                                }
                
                # Report peak days
                if pattern_peaks:
                    report_lines.append("Peak pattern days:")
                    for pattern_name, peak_data in pattern_peaks.items():
                        date_str = peak_data['date']
                        if isinstance(date_str, str) and date_str.startswith('20'):  # Simple date format check
                            try:
                                date_str = pd.to_datetime(date_str).strftime('%Y-%m-%d')
                            except:
                                pass
                        report_lines.append(f"  - {pattern_name.replace('_', ' ').title()}: {date_str} - {peak_data['count']} videos ({peak_data['percentage']:.1f}%)")
                
                # Calculate day of week trends
                try:
                    dow_trends = {}
                    for record in daily_trends:
                        date_str = record.get('date', '')
                        if date_str:
                            try:
                                date = pd.to_datetime(date_str)
                                day_name = date.day_name()
                                
                                if day_name not in dow_trends:
                                    dow_trends[day_name] = {'total': 0}
                                    
                                for key, value in record.items():
                                    if not key.endswith('_pct') and key not in ['date', 'total_videos'] and isinstance(value, (int, float)):
                                        if key not in dow_trends[day_name]:
                                            dow_trends[day_name][key] = 0
                                        dow_trends[day_name][key] += value
                                        dow_trends[day_name]['total'] += value
                            except:
                                pass
                    
                    # Find day with most pattern activity
                    if dow_trends:
                        max_day = max(dow_trends.items(), key=lambda x: x[1]['total'])
                        report_lines.append(f"\nDay of week with most pattern activity: {max_day[0]}")
                        
                        # List top patterns for that day
                        day_patterns = {k: v for k, v in max_day[1].items() if k != 'total' and v > 0}
                        if day_patterns:
                            sorted_patterns = sorted(day_patterns.items(), key=lambda x: x[1], reverse=True)
                            report_lines.append("  Top patterns:")
                            for pattern, count in sorted_patterns[:3]:
                                report_lines.append(f"    - {pattern.replace('_', ' ').title()}: {count} videos")
                except Exception as e:
                    logger.warning(f"Error analyzing day of week trends: {str(e)}")
                
                report_lines.append("")
        
        # Monthly Trend Analysis
        if 'trends' in analysis_results and 'monthly' in analysis_results['trends'] and analysis_results['trends']['monthly']:
            report_lines.append("MONTHLY VIEWING PATTERN TRENDS")
            report_lines.append("-" * 40)
            
            # Get monthly trend data
            monthly_trends = analysis_results['trends']['monthly']
            if monthly_trends:
                # Find months with highest pattern activity
                pattern_peaks = {}
                for record in monthly_trends:
                    for key, value in record.items():
                        if key.endswith('_pct') and isinstance(value, (int, float)) and value > 0:
                            pattern_name = key.replace('_pct', '')
                            if pattern_name not in pattern_peaks or value > pattern_peaks[pattern_name]['percentage']:
                                pattern_peaks[pattern_name] = {
                                    'month': record.get('month', 'Unknown'),
                                    'percentage': value,
                                    'count': record.get(pattern_name, 0)
                                }
                
                # Report peak months
                if pattern_peaks:
                    report_lines.append("Peak pattern months:")
                    for pattern_name, peak_data in pattern_peaks.items():
                        report_lines.append(f"  - {pattern_name.replace('_', ' ').title()}: {peak_data['month']} - {peak_data['count']} videos ({peak_data['percentage']:.1f}%)")
                
                # Calculate trend direction (increasing/decreasing)
                if len(monthly_trends) >= 2:
                    try:
                        # Sort by month
                        sorted_months = sorted(monthly_trends, key=lambda x: x.get('month', ''))
                        if len(sorted_months) >= 2:
                            first_month = sorted_months[0]
                            last_month = sorted_months[-1]
                            
                            report_lines.append("\nPattern trends over time:")
                            for pattern_name in [p.replace('pattern_', '') for p in ['pattern_addiction', 'pattern_doom_scrolling', 
                                                                               'pattern_escapism', 'pattern_negative_mood',
                                                                               'pattern_unhealthy_comparison', 'pattern_rabbit_holes']]:
                                if pattern_name in first_month and pattern_name in last_month:
                                    first_count = first_month.get(pattern_name, 0)
                                    last_count = last_month.get(pattern_name, 0)
                                    
                                    if first_count > 0 or last_count > 0:
                                        if last_count > first_count * 1.2:
                                            trend = "🔺 Increasing"
                                        elif last_count < first_count * 0.8:
                                            trend = "🔻 Decreasing"
                                        else:
                                            trend = "➡️ Stable"
                                            
                                        report_lines.append(f"  - {pattern_name.replace('_', ' ').title()}: {trend} ({first_count} → {last_count})")
                    except Exception as e:
                        logger.warning(f"Error analyzing monthly trend direction: {str(e)}")
                
                report_lines.append("")
        
        # Pattern correlations
        correlation_text = analyze_pattern_correlations(df)
        if correlation_text:
            report_lines.append("PATTERN CORRELATIONS")
            report_lines.append("-" * 40)
            report_lines.extend(correlation_text)
            report_lines.append("")
        
        # Pattern insights
        time_insights = analyze_time_patterns(df)
        if time_insights:
            report_lines.append("TIME-BASED INSIGHTS")
            report_lines.append("-" * 40)
            report_lines.extend(time_insights)
            report_lines.append("")
        
        # Recommendations
        recommendations = generate_recommendations(analysis_results, df)
        if recommendations:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            report_lines.extend(recommendations)
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("Analysis generated using YouTube Mental Health Analysis Tool")
        report_lines.append(f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        # Write report to file
        with open(f"{output_dir}/pattern_analysis_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))
        
    except Exception as e:
        logger.error(f"Error generating plaintext report: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc()) 

def detect_addiction_pattern(df, time_column='timestamp', daily_threshold=60, 
                           daily_consecutive_days=3, weekly_average_threshold=15):
    """
    Detect patterns indicative of potential YouTube addiction:
    - High daily consumption over consecutive days
    - Consistent usage at unusual hours (late night/early morning)
    
    Args:
        df (DataFrame): DataFrame with video data
        time_column (str): Column containing timestamp data
        daily_threshold (int): Number of videos per day to consider high consumption
        daily_consecutive_days (int): Number of consecutive days with high consumption required
        weekly_average_threshold (int): Weekly average number of videos to consider high
        
    Returns:
        DataFrame: Input DataFrame with addiction pattern flags and scores
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Clone the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Ensure the time column is datetime
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Add date column
    df['date'] = df[time_column].dt.date
    
    # Initialize pattern columns
    df['pattern_addiction'] = False
    df['addiction_score'] = 0.0
    
    # Group by date to count videos per day
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts['is_high_consumption'] = daily_counts['count'] >= daily_threshold
    
    # Find dates with consecutive days of high consumption
    high_consumption_dates = set()
    
    # Log the daily threshold being used
    logger.info(f"Using daily_threshold={daily_threshold} and daily_consecutive_days={daily_consecutive_days}")
    logger.info(f"Found {len(daily_counts)} unique dates in the data")
    logger.info(f"Days with high consumption (>={daily_threshold} videos): {daily_counts['is_high_consumption'].sum()}")
    
    # Identify consecutive days with high consumption
    # Sort by date first to ensure proper sequence detection
    daily_counts = daily_counts.sort_values('date')
    
    # Reset consecutive count for each new day
    consecutive_count = 0
    last_date = None
    
    for _, row in daily_counts.iterrows():
        # Check if this is a high consumption day
        if row['is_high_consumption']:
            # Check if it's consecutive with the last high consumption day
            if last_date is not None and row['date'] == last_date + pd.Timedelta(days=1):
                consecutive_count += 1
            else:
                # Reset counter for new sequence
                consecutive_count = 1
            
            # If we've reached the required consecutive days
            if consecutive_count >= daily_consecutive_days:
                # Add this date
                high_consumption_dates.add(row['date'])
                # Add the previous days in the sequence too
                for i in range(1, daily_consecutive_days):
                    prev_date = row['date'] - pd.Timedelta(days=i)
                    high_consumption_dates.add(prev_date)
        
        # Update last date if this was a high consumption day
        if row['is_high_consumption']:
            last_date = row['date']
    
    logger.info(f"Found {len(high_consumption_dates)} dates that are part of high consumption consecutive sequences")
    
    # Only mark videos on the specific high consumption dates
    if high_consumption_dates:
        df['pattern_addiction'] = df['date'].isin(high_consumption_dates)
    
    # Add late night viewing pattern - only if not already marked
    # Check for late night/early morning usage (12am-5am)
    df['hour'] = df[time_column].dt.hour
    df['is_late_night'] = (df['hour'] >= 0) & (df['hour'] < 5)
    
    # Group late night videos by date to find dates with significant late-night usage
    # Higher threshold for late-night usage (8 videos instead of 5)
    late_night_threshold = 8
    
    if df['is_late_night'].any():
        late_night_counts = df[df['is_late_night']].groupby('date').size().reset_index(name='late_night_count')
        late_night_dates = set(late_night_counts[late_night_counts['late_night_count'] >= late_night_threshold]['date'])
        
        logger.info(f"Found {len(late_night_dates)} dates with significant late-night usage (>={late_night_threshold} videos)")
        
        # Only mark videos that aren't already marked
        if late_night_dates:
            # Only mark the specific late night videos, not all videos on that date
            late_night_mask = df['date'].isin(late_night_dates) & df['is_late_night']
            df.loc[late_night_mask, 'pattern_addiction'] = True
    
    # Calculate addiction score for videos with the pattern
    addiction_mask = df['pattern_addiction']
    for idx, row in df[addiction_mask].iterrows():
        # Base score based on daily consumption relative to threshold
        daily_count = daily_counts[daily_counts['date'] == row['date']]['count'].iloc[0]
        consumption_factor = min(2.0, daily_count / daily_threshold)
        
        # Time of day factor (higher for late night)
        hour = row['hour']
        if 0 <= hour < 5:  # Late night (12am-5am)
            time_factor = 1.5
        elif 5 <= hour < 8:  # Early morning (5am-8am)
            time_factor = 1.2
        elif 22 <= hour < 24:  # Late evening (10pm-12am)
            time_factor = 1.3
        else:
            time_factor = 1.0
        
        # Combine factors for final score
        addiction_score = consumption_factor * time_factor
        df.loc[idx, 'addiction_score'] = addiction_score
    
    addiction_count = df['pattern_addiction'].sum()
    total_videos = len(df)
    addiction_percentage = (addiction_count / total_videos) * 100 if total_videos > 0 else 0
    logger.info(f"Found {addiction_count} videos with addiction pattern ({addiction_percentage:.1f}% of total)")
    
    return df

def detect_escapism(df, time_column='timestamp', content_column='category',
                  entertainment_categories=['Entertainment', 'Comedy', 'Gaming'],
                  minimum_daily_videos=8, entertainment_ratio=0.7):
    """
    Detect escapism patterns in video consumption:
    - High consumption of entertainment content, especially during work hours
    - Consistent patterns of using YouTube as an escape
    
    Args:
        df (DataFrame): DataFrame with video data
        time_column (str): Column containing timestamp
        content_column (str): Column containing content category
        entertainment_categories (list): Categories considered entertainment/escapism
        minimum_daily_videos (int): Minimum videos per day to consider
        entertainment_ratio (float): Ratio of entertainment videos to flag as escapism
        
    Returns:
        DataFrame: Original DataFrame with escapism flag and score columns
    """
    if df.empty or time_column not in df.columns or content_column not in df.columns:
        return df
    
    result_df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_dtype(result_df[time_column]):
        result_df[time_column] = pd.to_datetime(result_df[time_column], errors='coerce')
    
    # Extract date and hour components
    result_df['date'] = result_df[time_column].dt.date
    result_df['hour'] = result_df[time_column].dt.hour
    result_df['weekday'] = result_df[time_column].dt.weekday  # 0=Monday, 6=Sunday
    
    # Initialize escapism flag and score
    result_df['pattern_escapism'] = False
    result_df['escapism_score'] = 0.0
    
    # Identify entertainment content
    result_df['is_entertainment'] = result_df[content_column].isin(entertainment_categories)
    
    # Identify work hours (9am-5pm on weekdays)
    result_df['is_work_hours'] = (
        (result_df['weekday'] < 5) &  # Weekday
        (result_df['hour'] >= 9) & (result_df['hour'] < 17)  # 9am-5pm
    )
    
    # Group by date
    daily_stats = result_df.groupby('date').agg({
        'is_entertainment': 'sum',  # Count of entertainment videos
        'is_work_hours': 'sum',     # Count during work hours
        time_column: 'count'        # Total videos that day
    }).reset_index()
    
    daily_stats.columns = ['date', 'entertainment_count', 'work_hours_count', 'total_count']
    
    # Calculate entertainment ratio
    daily_stats['entertainment_ratio'] = daily_stats['entertainment_count'] / daily_stats['total_count']
    
    # Calculate work hour entertainment ratio
    work_hours_stats = result_df[result_df['is_work_hours']].groupby('date').agg({
        'is_entertainment': 'sum',  # Entertainment during work hours
        time_column: 'count'        # Total during work hours
    }).reset_index()
    
    if not work_hours_stats.empty:
        work_hours_stats.columns = ['date', 'work_entertainment_count', 'work_total_count']
        work_hours_stats['work_entertainment_ratio'] = (
            work_hours_stats['work_entertainment_count'] / work_hours_stats['work_total_count']
        )
        daily_stats = pd.merge(daily_stats, work_hours_stats, on='date', how='left')
        daily_stats = daily_stats.fillna(0)
    else:
        daily_stats['work_entertainment_count'] = 0
        daily_stats['work_total_count'] = 0
        daily_stats['work_entertainment_ratio'] = 0
    
    # Identify days with escapism patterns
    escapism_dates = daily_stats[
        (daily_stats['total_count'] >= minimum_daily_videos) &  # Sufficient video count
        (daily_stats['entertainment_ratio'] >= entertainment_ratio)  # High entertainment ratio
    ]['date'].tolist()
    
    # Additional check for work hours escapism
    work_escapism_dates = daily_stats[
        (daily_stats['work_total_count'] >= 5) &  # At least 5 videos during work hours
        (daily_stats['work_entertainment_ratio'] >= 0.6)  # High entertainment during work
    ]['date'].tolist()
    
    # Combine date lists
    all_escapism_dates = list(set(escapism_dates + work_escapism_dates))
    
    if all_escapism_dates:
        # Mark videos on those dates
        result_df.loc[result_df['date'].isin(all_escapism_dates), 'pattern_escapism'] = True
    
    # Calculate escapism score
    daily_scores = pd.merge(
        result_df, 
        daily_stats[['date', 'entertainment_ratio', 'work_entertainment_ratio']], 
        on='date', 
        how='left'
    )
    
    # Score calculation:
    # - Entertainment ratio contributes 50%
    # - Work hour entertainment ratio contributes 30%
    # - Entertainment content type contributes 20%
    result_df['escapism_score'] = (
        (daily_scores['entertainment_ratio'] * 0.5) +
        (daily_scores['work_entertainment_ratio'] * 0.3) +
        (result_df['is_entertainment'].astype(float) * 0.2)
    ).clip(0, 1.0)
    
    return result_df

def detect_negative_mood(df, content_columns=['title', 'description'], 
                      negative_keywords=['sad', 'depressed', 'anxiety', 'angry', 'frustrated', 
                                        'unhappy', 'miserable', 'pain', 'lonely', 'tired',
                                        'exhausted', 'struggling', 'stress', 'worried', 'fear'],
                      negative_threshold=0.2):
    """
    Detect negative mood patterns based on content of videos watched:
    - Content with negative emotional keywords
    - Clusters of negatively themed videos
    
    Args:
        df (DataFrame): DataFrame with video data
        content_columns (list): Columns to analyze for mood indicators
        negative_keywords (list): Keywords indicating negative mood
        negative_threshold (float): Threshold ratio to flag negative mood
        
    Returns:
        DataFrame: Original DataFrame with negative_mood flag column added
    """
    if df.empty:
        return df
    
    # Check if any content columns exist
    available_columns = [col for col in content_columns if col in df.columns]
    if not available_columns:
        return df
    
    result_df = df.copy()
    
    # Initialize negative mood flag
    result_df['pattern_negative_mood'] = False
    
    # Function to detect negative keywords in text
    def contains_negative(text):
        if not isinstance(text, str):
            return False
        
        text = text.lower()
        return any(keyword in text for keyword in negative_keywords)
    
    # Check each content column for negative keywords
    for column in available_columns:
        result_df[f'{column}_negative'] = result_df[column].apply(contains_negative)
    
    # Determine overall negativity based on any column containing negative keywords
    negative_columns = [f'{column}_negative' for column in available_columns]
    result_df['any_negative'] = result_df[negative_columns].any(axis=1)
    
    # If timestamp is available, do time-based clustering
    if 'timestamp' in result_df.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(result_df['timestamp']):
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], errors='coerce')
        
        # Sort by timestamp
        result_df = result_df.sort_values('timestamp')
        
        # Extract date
        result_df['date'] = result_df['timestamp'].dt.date
        
        # Calculate daily negative ratio
        daily_stats = result_df.groupby('date').agg({
            'any_negative': 'sum',  # Count of negative videos
            'timestamp': 'count'    # Total videos that day
        }).reset_index()
        
        daily_stats.columns = ['date', 'negative_count', 'total_count']
        daily_stats['negative_ratio'] = daily_stats['negative_count'] / daily_stats['total_count']
        
        # Identify days with high negative content
        negative_days = daily_stats[
            (daily_stats['total_count'] >= 5) &  # At least 5 videos that day
            (daily_stats['negative_ratio'] > negative_threshold)  # Exceeds threshold
        ]['date'].tolist()
        
        if negative_days:
            # Mark all videos on negative days
            result_df.loc[result_df['date'].isin(negative_days), 'pattern_negative_mood'] = True
    
    # Mark individual videos with multiple negative indicators
    multi_negative = result_df[result_df[negative_columns].sum(axis=1) >= 2].index
    result_df.loc[multi_negative, 'pattern_negative_mood'] = True
    
    return result_df

def detect_unhealthy_comparison(df, content_columns=['title', 'description'], 
                            comparison_keywords=['best', 'better', 'top', 'perfect', 'ideal', 
                                               'goals', 'ultimate', 'flawless', 'perfection',
                                               'beauty', 'beautiful', 'attractive', 'hot', 'sexy'],
                            categories_of_concern=['Beauty', 'Lifestyle', 'Fashion', 'Fitness']):
    """
    Detect patterns of potential unhealthy comparison:
    - Content focused on ideals, perfection, competition
    - Clusters of beauty/lifestyle/comparison content
    
    Args:
        df (DataFrame): DataFrame with video data
        content_columns (list): Columns to analyze for comparison indicators
        comparison_keywords (list): Keywords indicating comparison
        categories_of_concern (list): Categories with higher risk of comparison
        
    Returns:
        DataFrame: Original DataFrame with unhealthy_comparison flag and score
    """
    if df.empty:
        return df
    
    # Check if any content columns exist
    available_columns = [col for col in content_columns if col in df.columns]
    if not available_columns and 'category' not in df.columns:
        return df
    
    result_df = df.copy()
    
    # Initialize comparison flag and score
    result_df['pattern_unhealthy_comparison'] = False
    result_df['comparison_score'] = 0.0
    
    # Function to detect comparison keywords in text
    def contains_comparison(text):
        if not isinstance(text, str):
            return 0
        
        text = text.lower()
        count = sum(1 for keyword in comparison_keywords if keyword in text)
        return min(count, 5)  # Cap at 5 for scoring
    
    # Check each content column for comparison keywords
    for column in available_columns:
        result_df[f'{column}_comparison'] = result_df[column].apply(contains_comparison)
    
    # Calculate base comparison score from content
    comparison_columns = [f'{column}_comparison' for column in available_columns]
    if comparison_columns:
        result_df['content_comparison_score'] = result_df[comparison_columns].sum(axis=1) / (len(comparison_columns) * 5)
    else:
        result_df['content_comparison_score'] = 0
    
    # Category score
    if 'category' in df.columns:
        result_df['category_score'] = result_df['category'].apply(
            lambda x: 0.8 if x in categories_of_concern else 0
        )
    else:
        result_df['category_score'] = 0
    
    # Calculate overall comparison score
    result_df['comparison_score'] = (
        (result_df['content_comparison_score'] * 0.7) + 
        (result_df['category_score'] * 0.3)
    ).clip(0, 1.0)
    
    # Mark videos with high comparison score
    result_df.loc[result_df['comparison_score'] > 0.4, 'pattern_unhealthy_comparison'] = True
    
    # If timestamp is available, do time-based clustering
    if 'timestamp' in result_df.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(result_df['timestamp']):
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], errors='coerce')
        
        # Extract date
        result_df['date'] = result_df['timestamp'].dt.date
        
        # Identify clusters: days with multiple high-score videos
        high_score_videos = result_df[result_df['comparison_score'] > 0.3]
        
        if not high_score_videos.empty:
            # Count high-score videos per day
            daily_high_scores = high_score_videos.groupby('date').size().reset_index(name='high_score_count')
            
            # Find days with significant high-score clusters
            cluster_days = daily_high_scores[daily_high_scores['high_score_count'] >= 3]['date']
            
            if not cluster_days.empty:
                # Mark all videos on these days
                result_df.loc[result_df['date'].isin(cluster_days), 'pattern_unhealthy_comparison'] = True
    
    return result_df 

def create_pattern_time_series(df, output_dir):
    """
    Create time series visualizations for all patterns.
    
    Args:
        df (DataFrame): DataFrame with pattern flags and timestamp
        output_dir (str): Directory to save visualizations
    """
    try:
        if df.empty or 'timestamp' not in df.columns:
            logger.warning("Cannot create pattern time series: missing data")
            return None
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                          'pattern_escapism', 'pattern_negative_mood',
                          'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if not available_patterns:
            logger.warning("No pattern columns found for time series analysis")
            return None
        
        # Create an hourly time series
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Aggregate by hour
        hourly_data = []
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            # Count videos with this pattern per hour
            hourly_pattern = df[df[pattern]].groupby('hour').size()
            hourly_pattern.name = pattern_name
            hourly_data.append(hourly_pattern)
        
        # Combine all patterns into one DataFrame
        if hourly_data:
            hourly_combined = pd.concat(hourly_data, axis=1).fillna(0)
            
            # Create time series plot
            plt.figure(figsize=(15, 8))
            
            for pattern in hourly_combined.columns:
                plt.plot(hourly_combined.index, hourly_combined[pattern], label=pattern, linewidth=2)
            
            plt.title('Pattern Occurrence by Hour', fontsize=16)
            plt.xlabel('Date/Hour', fontsize=14)
            plt.ylabel('Number of Videos', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/pattern_time_series.png")
            plt.close()
            
            # Save hourly data
            hourly_combined.to_csv(f"{output_dir}/hourly_pattern_data.csv")
        
        # Create a smoothed time series (rolling average)
        # First create a complete time range
        if not df.empty:
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            date_range = pd.date_range(
                start=min_date.floor('D'),
                end=max_date.ceil('D'),
                freq='H'
            )
            
            # Create a DataFrame with the complete date range
            complete_range = pd.DataFrame({'hour': date_range})
            
            # Merge with hourly data
            merged_data = []
            for pattern in available_patterns:
                pattern_name = pattern.replace('pattern_', '')
                pattern_data = df[df[pattern]].groupby('hour').size().reset_index()
                pattern_data.columns = ['hour', pattern_name]
                
                # Merge with complete range
                complete_pattern = pd.merge(complete_range, pattern_data, on='hour', how='left').fillna(0)
                
                # Calculate rolling average (24-hour window)
                complete_pattern[f'{pattern_name}_rolling'] = complete_pattern[pattern_name].rolling(24, min_periods=1).mean()
                
                merged_data.append(complete_pattern[[
                    'hour', pattern_name, f'{pattern_name}_rolling'
                ]])
            
            if merged_data:
                # Merge all pattern data
                merged_df = merged_data[0]
                for i in range(1, len(merged_data)):
                    merged_df = pd.merge(merged_df, merged_data[i], on='hour')
                
                # Save combined data
                merged_df.to_csv(f"{output_dir}/pattern_time_series_data.csv", index=False)
                
                # Create smoothed time series plot
                plt.figure(figsize=(15, 8))
                
                for pattern in available_patterns:
                    pattern_name = pattern.replace('pattern_', '')
                    plt.plot(
                        merged_df['hour'], 
                        merged_df[f'{pattern_name}_rolling'], 
                        label=f"{pattern_name} (24h avg)",
                        linewidth=2
                    )
                
                plt.title('Pattern Trends (24-hour Rolling Average)', fontsize=16)
                plt.xlabel('Date/Hour', fontsize=14)
                plt.ylabel('Average Number of Videos', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/pattern_time_series_smoothed.png")
                plt.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating pattern time series: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def create_pattern_summary(df, output_dir):
    """
    Create summary visualizations for all patterns.
    
    Args:
        df (DataFrame): DataFrame with pattern flags
        output_dir (str): Directory to save visualizations
    """
    try:
        if df.empty:
            logger.warning("Cannot create pattern summary: missing data")
            return None
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                          'pattern_escapism', 'pattern_negative_mood',
                          'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if not available_patterns:
            logger.warning("No pattern columns found for summary")
            return None
        
        # Calculate counts for each pattern
        pattern_counts = []
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            count = df[pattern].sum()
            percentage = round((count / len(df)) * 100, 2)
            pattern_counts.append({
                'pattern': pattern_name,
                'count': count,
                'percentage': percentage
            })
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(pattern_counts)
        
        # Save summary data
        summary_df.to_csv(f"{output_dir}/pattern_summary.csv", index=False)
        
        # Create bar chart of pattern counts
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(
            summary_df['pattern'], 
            summary_df['count'],
            color=sns.color_palette('viridis', len(summary_df))
        )
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 5,
                f'{int(height)}',
                ha='center', 
                va='bottom',
                fontsize=11
            )
        
        plt.title('Number of Videos by Pattern', fontsize=16)
        plt.xlabel('Pattern', fontsize=14)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pattern_count_summary.png")
        plt.close()
        
        # Create percentage bar chart
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(
            summary_df['pattern'], 
            summary_df['percentage'],
            color=sns.color_palette('viridis', len(summary_df))
        )
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{height}%',
                ha='center', 
                va='bottom',
                fontsize=11
            )
        
        plt.title('Percentage of Videos by Pattern', fontsize=16)
        plt.xlabel('Pattern', fontsize=14)
        plt.ylabel('Percentage of Videos (%)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pattern_percentage_summary.png")
        plt.close()
        
        # Create pie chart
        plt.figure(figsize=(12, 10))
        
        plt.pie(
            summary_df['count'],
            labels=summary_df['pattern'],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('viridis', len(summary_df)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        
        plt.title('Distribution of Patterns', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pattern_distribution_pie.png")
        plt.close()
        
        return summary_df
        
    except Exception as e:
        logger.error(f"Error creating pattern summary: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def create_day_of_week_patterns(df, output_dir):
    """
    Create analysis of patterns by day of week.
    
    Args:
        df (DataFrame): DataFrame with pattern flags and timestamp
        output_dir (str): Directory to save visualizations
    """
    try:
        if df.empty or 'timestamp' not in df.columns:
            logger.warning("Cannot create day of week analysis: missing data")
            return None
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Extract day of week
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                          'pattern_escapism', 'pattern_negative_mood',
                          'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if not available_patterns:
            logger.warning("No pattern columns found for day of week analysis")
            return None
        
        # Total videos by day of week
        total_by_day = df.groupby('day_of_week').size().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        # Patterns by day of week
        patterns_by_day = {}
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            pattern_by_day = df[df[pattern]].groupby('day_of_week').size().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]).fillna(0)
            patterns_by_day[pattern_name] = pattern_by_day
        
        # Create combined DataFrame
        day_of_week_df = pd.DataFrame({'total_videos': total_by_day})
        for pattern_name, pattern_data in patterns_by_day.items():
            day_of_week_df[pattern_name] = pattern_data
            day_of_week_df[f'{pattern_name}_pct'] = (pattern_data / total_by_day * 100).round(1)
        
        # Save data
        day_of_week_df.to_csv(f"{output_dir}/day_of_week_patterns.csv")
        
        # Create bar chart
        plt.figure(figsize=(14, 8))
        
        bar_width = 0.7 / len(patterns_by_day)
        index = np.arange(7)  # 7 days of week
        
        for i, (pattern_name, pattern_data) in enumerate(patterns_by_day.items()):
            plt.bar(
                index + i * bar_width - 0.3,
                pattern_data,
                bar_width,
                label=pattern_name
            )
        
        plt.title('Pattern Occurrence by Day of Week', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.xticks(index, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/day_of_week_patterns.png")
        plt.close()
        
        # Create percentage bar chart
        plt.figure(figsize=(14, 8))
        
        for i, pattern_name in enumerate(patterns_by_day.keys()):
            plt.bar(
                index + i * bar_width - 0.3,
                day_of_week_df[f'{pattern_name}_pct'],
                bar_width,
                label=pattern_name
            )
        
        plt.title('Pattern Percentage by Day of Week', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Percentage of Videos (%)', fontsize=14)
        plt.xticks(index, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/day_of_week_patterns_pct.png")
        plt.close()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        heatmap_data = day_of_week_df[[f'{name}_pct' for name in patterns_by_day.keys()]]
        heatmap_data.columns = list(patterns_by_day.keys())
        
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt='.1f', 
            cmap='viridis',
            linewidths=.5,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title('Pattern Percentage Heatmap by Day of Week', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/day_of_week_heatmap.png")
        plt.close()
        
        return day_of_week_df
        
    except Exception as e:
        logger.error(f"Error creating day of week analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def create_hour_of_day_patterns(df, output_dir):
    """
    Create analysis of patterns by hour of day.
    
    Args:
        df (DataFrame): DataFrame with pattern flags and timestamp
        output_dir (str): Directory to save visualizations
    """
    try:
        if df.empty or 'timestamp' not in df.columns:
            logger.warning("Cannot create hour of day analysis: missing data")
            return None
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Extract hour of day
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                          'pattern_escapism', 'pattern_negative_mood',
                          'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if not available_patterns:
            logger.warning("No pattern columns found for hour of day analysis")
            return None
        
        # Total videos by hour of day
        total_by_hour = df.groupby('hour_of_day').size()
        
        # Patterns by hour of day
        patterns_by_hour = {}
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            pattern_by_hour = df[df[pattern]].groupby('hour_of_day').size().reindex(
                range(24)
            ).fillna(0)
            patterns_by_hour[pattern_name] = pattern_by_hour
        
        # Create combined DataFrame
        hour_of_day_df = pd.DataFrame({'total_videos': total_by_hour})
        for pattern_name, pattern_data in patterns_by_hour.items():
            hour_of_day_df[pattern_name] = pattern_data
            hour_of_day_df[f'{pattern_name}_pct'] = (pattern_data / total_by_hour * 100).round(1)
        
        # Save data
        hour_of_day_df.to_csv(f"{output_dir}/hour_of_day_patterns.csv")
        
        # Create line chart
        plt.figure(figsize=(14, 8))
        
        for pattern_name, pattern_data in patterns_by_hour.items():
            plt.plot(
                pattern_data.index,
                pattern_data,
                'o-',
                linewidth=2,
                label=pattern_name
            )
        
        plt.title('Pattern Occurrence by Hour of Day', fontsize=16)
        plt.xlabel('Hour (24-hour format)', fontsize=14)
        plt.ylabel('Number of Videos', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(0, 24, 2))  # Every 2 hours
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hour_of_day_patterns.png")
        plt.close()
        
        # Create percentage line chart
        plt.figure(figsize=(14, 8))
        
        for pattern_name in patterns_by_hour.keys():
            plt.plot(
                hour_of_day_df.index,
                hour_of_day_df[f'{pattern_name}_pct'],
                'o-',
                linewidth=2,
                label=pattern_name
            )
        
        plt.title('Pattern Percentage by Hour of Day', fontsize=16)
        plt.xlabel('Hour (24-hour format)', fontsize=14)
        plt.ylabel('Percentage of Videos (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(0, 24, 2))  # Every 2 hours
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hour_of_day_patterns_pct.png")
        plt.close()
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        
        heatmap_data = hour_of_day_df[[f'{name}_pct' for name in patterns_by_hour.keys()]]
        heatmap_data.columns = list(patterns_by_hour.keys())
        
        # Reshape for better visualization
        heatmap_reshaped = heatmap_data.transpose()
        
        sns.heatmap(
            heatmap_reshaped, 
            annot=True, 
            fmt='.1f', 
            cmap='viridis',
            linewidths=.5,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title('Pattern Percentage Heatmap by Hour of Day', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Pattern', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hour_of_day_heatmap.png")
        plt.close()
        
        return hour_of_day_df
        
    except Exception as e:
        logger.error(f"Error creating hour of day analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def create_pattern_correlation(df, output_dir):
    """
    Create correlation analysis between different patterns.
    
    Args:
        df (DataFrame): DataFrame with pattern flags
        output_dir (str): Directory to save visualizations
    """
    try:
        if df.empty:
            logger.warning("Cannot create pattern correlation: missing data")
            return None
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                          'pattern_escapism', 'pattern_negative_mood',
                          'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if len(available_patterns) < 2:
            logger.warning("Not enough pattern columns found for correlation analysis")
            return None
        
        # Calculate correlation matrix
        correlation_matrix = df[available_patterns].corr()
        
        # Save correlation data
        correlation_matrix.to_csv(f"{output_dir}/pattern_correlation.csv")
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create heatmap with better labels
        pattern_labels = [p.replace('pattern_', '') for p in available_patterns]
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            mask=mask,
            vmin=-1, vmax=1,
            linewidths=.5,
            xticklabels=pattern_labels,
            yticklabels=pattern_labels,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title('Pattern Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pattern_correlation_heatmap.png")
        plt.close()
        
        # Create scatter plots for highly correlated patterns
        high_correlations = []
        for i in range(len(available_patterns)):
            for j in range(i+1, len(available_patterns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.3:  # Only show significant correlations
                    high_correlations.append({
                        'pattern1': available_patterns[i],
                        'pattern2': available_patterns[j],
                        'correlation': corr
                    })
        
        # Create scatter plots for top correlations
        for corr_info in high_correlations:
            pattern1 = corr_info['pattern1']
            pattern2 = corr_info['pattern2']
            corr = corr_info['correlation']
            
            plt.figure(figsize=(10, 8))
            
            # If both are binary flags, add jitter for better visualization
            if df[pattern1].isin([0, 1]).all() and df[pattern2].isin([0, 1]).all():
                x = df[pattern1] + np.random.normal(0, 0.05, len(df))
                y = df[pattern2] + np.random.normal(0, 0.05, len(df))
                plt.scatter(x, y, alpha=0.6)
            else:
                plt.scatter(df[pattern1], df[pattern2], alpha=0.6)
            
            pattern1_label = pattern1.replace('pattern_', '')
            pattern2_label = pattern2.replace('pattern_', '')
            
            plt.title(f'Correlation between {pattern1_label} and {pattern2_label}: {corr:.2f}', 
                     fontsize=14)
            plt.xlabel(pattern1_label, fontsize=12)
            plt.ylabel(pattern2_label, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_{pattern1_label}_{pattern2_label}.png")
            plt.close()
        
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error creating pattern correlation: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def analyze_pattern_correlations(df):
    """
    Analyze correlations between patterns and generate insights.
    
    Args:
        df (DataFrame): DataFrame with pattern flags
        
    Returns:
        list: List of insights about pattern correlations
    """
    insights = []
    
    try:
        if df.empty:
            return ["No data available for pattern correlation analysis."]
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                         'pattern_escapism', 'pattern_negative_mood',
                         'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if len(available_patterns) < 2:
            return ["Not enough pattern types detected for correlation analysis."]
        
        # Calculate correlation matrix
        correlation_matrix = df[available_patterns].corr()
        
        # Generate insights based on correlation strengths
        for i in range(len(available_patterns)):
            for j in range(i+1, len(available_patterns)):
                pattern1 = available_patterns[i].replace('pattern_', '')
                pattern2 = available_patterns[j].replace('pattern_', '')
                corr = correlation_matrix.iloc[i, j]
                
                if abs(corr) > 0.7:
                    strength = "very strong"
                elif abs(corr) > 0.5:
                    strength = "strong"
                elif abs(corr) > 0.3:
                    strength = "moderate"
                elif abs(corr) > 0.1:
                    strength = "weak"
                else:
                    continue  # Skip insignificant correlations
                
                direction = "positive" if corr > 0 else "negative"
                
                insight = f"There is a {strength} {direction} correlation ({corr:.2f}) between {pattern1} and {pattern2} patterns."
                
                # Add interpretation for strong correlations
                if abs(corr) > 0.5:
                    if corr > 0:
                        insight += f" This suggests that users who display {pattern1} behavior are likely to also display {pattern2} behavior."
                    else:
                        insight += f" This suggests that users who display {pattern1} behavior are unlikely to display {pattern2} behavior."
                
                insights.append(insight)
        
        # Add summary insight
        if not insights:
            insights.append("No significant correlations were found between different patterns.")
        else:
            # Find strongest correlation
            max_corr = 0
            max_pair = ""
            for i in range(len(available_patterns)):
                for j in range(i+1, len(available_patterns)):
                    corr = abs(correlation_matrix.iloc[i, j])
                    if corr > max_corr:
                        max_corr = corr
                        pattern1 = available_patterns[i].replace('pattern_', '')
                        pattern2 = available_patterns[j].replace('pattern_', '')
                        max_pair = f"{pattern1} and {pattern2}"
            
            if max_corr > 0.3:
                insights.append(f"The strongest relationship is between {max_pair} with a correlation of {max_corr:.2f}.")
            
            # Add overall assessment
            high_corrs = sum(1 for i in range(len(available_patterns)) 
                           for j in range(i+1, len(available_patterns)) 
                           if abs(correlation_matrix.iloc[i, j]) > 0.5)
            
            if high_corrs > 2:
                insights.append("Multiple strong correlations suggest that mental health patterns tend to occur together rather than in isolation.")
        
    except Exception as e:
        insights.append(f"Error analyzing pattern correlations: {str(e)}")
    
    return insights

def analyze_time_patterns(df):
    """
    Analyze time-based patterns and generate insights.
    
    Args:
        df (DataFrame): DataFrame with pattern flags and timestamp
        
    Returns:
        list: List of insights about time-based patterns
    """
    insights = []
    
    try:
        if df.empty or 'timestamp' not in df.columns:
            return ["No time data available for time pattern analysis."]
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday and Sunday
        df['month'] = df['timestamp'].dt.month
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                         'pattern_escapism', 'pattern_negative_mood',
                         'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available
        available_patterns = [col for col in pattern_columns if col in df.columns]
        if not available_patterns:
            return ["No pattern data available for time pattern analysis."]
        
        # Generate insights for each pattern
        for pattern in available_patterns:
            pattern_name = pattern.replace('pattern_', '')
            pattern_df = df[df[pattern] == True]
            
            if len(pattern_df) < 5:
                insights.append(f"Not enough data points for {pattern_name} time pattern analysis.")
                continue
            
            # Time of day analysis
            hour_counts = pattern_df.groupby('hour').size()
            total_hour_counts = df.groupby('hour').size()
            hour_percentages = (hour_counts / total_hour_counts * 100).fillna(0)
            
            # Find peak hours (top 3)
            peak_hours = hour_percentages.nlargest(3)
            peak_hour_str = ", ".join([f"{hour}:00 ({pct:.1f}%)" 
                                      for hour, pct in peak_hours.items()])
            insights.append(f"{pattern_name} pattern peaks during these hours: {peak_hour_str}")
            
            # Find periods with 3+ consecutive hours of high activity
            high_hour_threshold = hour_percentages.mean() + hour_percentages.std()
            high_hours = hour_percentages[hour_percentages > high_hour_threshold].index.tolist()
            
            # Check for 3+ consecutive hours
            for i in range(22):  # Up to hour 21 (to check 21, 22, 23)
                if i in high_hours and i+1 in high_hours and i+2 in high_hours:
                    insights.append(f"{pattern_name} shows sustained high activity from {i}:00 to {i+2}:59.")
                    break
            
            # Day of week analysis
            day_counts = pattern_df.groupby('day_of_week').size()
            total_day_counts = df.groupby('day_of_week').size()
            day_percentages = (day_counts / total_day_counts * 100).fillna(0)
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Find peak days
            peak_day_idx = day_percentages.idxmax()
            peak_day = day_names[peak_day_idx]
            peak_pct = day_percentages[peak_day_idx]
            
            insights.append(f"{pattern_name} pattern is most prevalent on {peak_day} ({peak_pct:.1f}%).")
            
            # Weekend vs weekday
            weekend_df = pattern_df[pattern_df['is_weekend']]
            weekday_df = pattern_df[~pattern_df['is_weekend']]
            
            weekend_pct = len(weekend_df) / len(pattern_df) * 100 if len(pattern_df) > 0 else 0
            
            if weekend_pct > 60:
                insights.append(f"{pattern_name} occurs predominantly on weekends ({weekend_pct:.1f}% of occurrences).")
            elif weekend_pct < 20:
                insights.append(f"{pattern_name} occurs predominantly on weekdays ({100-weekend_pct:.1f}% of occurrences).")
            
            # Monthly trends (if data spans multiple months)
            month_counts = pattern_df.groupby('month').size()
            
            if len(month_counts) > 1:
                # Calculate month-to-month changes
                month_changes = month_counts.pct_change() * 100
                
                significant_changes = []
                for month, change in month_changes.items():
                    if pd.notnull(change):
                        if change > 20:
                            month_name = datetime(2020, month, 1).strftime('%B')
                            significant_changes.append(f"increased by {change:.1f}% in {month_name}")
                        elif change < -20:
                            month_name = datetime(2020, month, 1).strftime('%B')
                            significant_changes.append(f"decreased by {abs(change):.1f}% in {month_name}")
                
                if significant_changes:
                    insights.append(f"{pattern_name} pattern {' and '.join(significant_changes)}.")
        
        # Overall time insights
        total_videos = len(df)
        
        # Check if there's a significant peak time for overall consumption
        hour_totals = df.groupby('hour').size()
        peak_hour = hour_totals.idxmax()
        peak_hour_pct = hour_totals[peak_hour] / total_videos * 100
        
        if peak_hour_pct > 15:  # Significant peak if >15% of videos in one hour
            insights.append(f"Overall video consumption peaks significantly at {peak_hour}:00 ({peak_hour_pct:.1f}% of all videos).")
        
        # Check if weekends significantly different from weekdays
        weekend_videos = df[df['is_weekend']].shape[0]
        weekend_pct = weekend_videos / total_videos * 100
        
        if weekend_pct > 50:
            insights.append(f"Overall, {weekend_pct:.1f}% of video consumption occurs on weekends, showing a weekend-heavy usage pattern.")
        elif weekend_pct < 20:
            insights.append(f"Only {weekend_pct:.1f}% of video consumption occurs on weekends, showing a strongly weekday-focused usage pattern.")
        
    except Exception as e:
        insights.append(f"Error analyzing time patterns: {str(e)}")
    
    return insights

def generate_recommendations(analysis_results, df):
    """
    Generate recommendations based on pattern analysis.
    
    Args:
        analysis_results (dict): Results from previous analyses
        df (DataFrame): DataFrame with pattern flags
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    try:
        if df.empty:
            return ["Insufficient data for generating recommendations."]
        
        # Pattern columns
        pattern_columns = ['pattern_addiction', 'pattern_doom_scrolling', 
                         'pattern_escapism', 'pattern_negative_mood',
                         'pattern_unhealthy_comparison', 'pattern_rabbit_holes']
        
        # Check which patterns are available and calculate prevalence
        pattern_prevalence = {}
        for pattern in pattern_columns:
            if pattern in df.columns:
                pattern_name = pattern.replace('pattern_', '')
                pattern_count = df[pattern].sum()
                pattern_pct = (pattern_count / len(df)) * 100
                pattern_prevalence[pattern_name] = {
                    'count': pattern_count,
                    'percentage': pattern_pct
                }
        
        if not pattern_prevalence:
            return ["No patterns detected for generating recommendations."]
        
        # General recommendation based on overall analysis
        recommendations.append("Based on the analysis of your YouTube viewing patterns, here are recommendations to improve your digital wellbeing:")
        
        # Recommendations based on pattern prevalence
        high_patterns = [p for p, stats in pattern_prevalence.items() 
                       if stats['percentage'] > 15]  # Patterns occurring in >15% of videos
        
        moderate_patterns = [p for p, stats in pattern_prevalence.items() 
                           if 5 <= stats['percentage'] <= 15]  # Patterns in 5-15% of videos
        
        # Add specific recommendations for high-prevalence patterns
        if high_patterns:
            recommendations.append("\nPriority areas to address:")
            
            if 'addiction' in high_patterns:
                recommendations.append("1. Consider using YouTube's built-in tools to manage watch time:")
                recommendations.append("   - Set up 'Take a Break' reminders in the YouTube app")
                recommendations.append("   - Use the 'Time Watched' feature to monitor your usage")
                recommendations.append("   - Schedule specific times for YouTube use rather than continuous browsing")
            
            if 'doom_scrolling' in high_patterns:
                recommendations.append("2. Break the doom scrolling cycle:")
                recommendations.append("   - Set a timer before starting YouTube sessions")
                recommendations.append("   - Use browser extensions that limit infinite scrolling")
                recommendations.append("   - Practice the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds")
            
            if 'rabbit_holes' in high_patterns:
                recommendations.append("3. Avoid falling down rabbit holes:")
                recommendations.append("   - Disable autoplay in your YouTube settings")
                recommendations.append("   - Create focused playlists instead of following recommendations")
                recommendations.append("   - Set clear learning objectives before starting educational content")
            
            if 'escapism' in high_patterns:
                recommendations.append("4. Manage escapism tendencies:")
                recommendations.append("   - Schedule dedicated relaxation time separate from work/study")
                recommendations.append("   - Try alternative relaxation methods like reading, walking, or meditation")
                recommendations.append("   - Use the Pomodoro technique to balance work with short breaks")
            
            if 'negative_mood' in high_patterns:
                recommendations.append("5. Improve content for mental wellbeing:")
                recommendations.append("   - Curate your feed by clicking 'Not Interested' on negative content")
                recommendations.append("   - Subscribe to channels that focus on personal growth and positivity")
                recommendations.append("   - Consider using mood tracking alongside YouTube use to identify correlations")
            
            if 'unhealthy_comparison' in high_patterns:
                recommendations.append("6. Address unhealthy comparison:")
                recommendations.append("   - Follow more diverse creators who promote realistic expectations")
                recommendations.append("   - Remember that most content is curated to show ideal circumstances")
                recommendations.append("   - Practice media literacy to critically evaluate the content you consume")
        
        # Add recommendations for moderate-prevalence patterns
        if moderate_patterns:
            recommendations.append("\nAdditional areas to monitor:")
            
            for pattern in moderate_patterns:
                if pattern == 'addiction' and 'addiction' not in high_patterns:
                    recommendations.append("- Be mindful of potential addictive patterns by setting time limits for YouTube use")
                
                elif pattern == 'doom_scrolling' and 'doom_scrolling' not in high_patterns:
                    recommendations.append("- Watch for extended viewing sessions and take regular breaks to avoid doom scrolling")
                
                elif pattern == 'rabbit_holes' and 'rabbit_holes' not in high_patterns:
                    recommendations.append("- Notice when you're following a chain of recommended videos and consciously decide if it's valuable")
                
                elif pattern == 'escapism' and 'escapism' not in high_patterns:
                    recommendations.append("- Balance entertainment content with productive activities, especially during work hours")
                
                elif pattern == 'negative_mood' and 'negative_mood' not in high_patterns:
                    recommendations.append("- Pay attention to how content affects your mood and adjust your subscriptions accordingly")
                
                elif pattern == 'unhealthy_comparison' and 'unhealthy_comparison' not in high_patterns:
                    recommendations.append("- Be aware of lifestyle, beauty or fitness content that may trigger unhealthy comparisons")
        
        # Time-based recommendations
        if 'timestamp' in df.columns:
            # Check for late night usage
            try:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                late_night = df[(df['hour'] >= 22) | (df['hour'] <= 5)].shape[0]
                late_night_pct = (late_night / len(df)) * 100
                
                if late_night_pct > 25:  # Significant late night usage
                    recommendations.append("\nTime management recommendations:")
                    recommendations.append("- Consider reducing late night YouTube use (10pm-5am), which accounts for {:.1f}% of your viewing".format(late_night_pct))
                    recommendations.append("- Enable Night Mode to reduce blue light exposure")
                    recommendations.append("- Try using Do Not Disturb mode on your devices after a certain hour")
            except:
                pass  # Skip if timestamp processing fails
        
        # Add healthy habits recommendations
        recommendations.append("\nHealthy YouTube habits to develop:")
        recommendations.append("1. Quality over quantity: Focus on content that truly adds value to your life")
        recommendations.append("2. Active vs. passive viewing: Take notes or engage with educational content")
        recommendations.append("3. Social viewing: Share and discuss videos with friends rather than isolated viewing")
        recommendations.append("4. Content creation: Consider creating content yourself as a more active engagement")
        recommendations.append("5. Digital detox: Schedule regular breaks from all digital media")
        
        # Add technical recommendations
        recommendations.append("\nTechnical solutions to try:")
        recommendations.append("1. Browser extensions like 'Unhook' or 'DF YouTube' to remove distracting elements")
        recommendations.append("2. Screen time management apps like 'Digital Wellbeing' (Android) or 'Screen Time' (iOS)")
        recommendations.append("3. YouTube Premium for ad-free viewing (reduces temptation to keep watching during ads)")
        recommendations.append("4. Set up specific user profiles for different types of content consumption")
        
    except Exception as e:
        recommendations.append(f"Error generating recommendations: {str(e)}")
    
    return recommendations

def generate_json_report(analysis_results, output_dir):
    """
    Generate a JSON report containing all analysis results.
    
    Args:
        analysis_results (dict): Analysis results dictionary
        output_dir (str): Directory to save report
    """
    import json
    import os
    
    json_file_path = os.path.join(output_dir, 'analysis_results.json')
    
    try:
        # Convert any non-serializable objects to strings
        def json_serial(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            return str(obj)
        
        with open(json_file_path, 'w') as json_file:
            json.dump(analysis_results, json_file, indent=2, default=json_serial)
        
        logger.info(f"JSON report generated: {json_file_path}")
    except Exception as e:
        logger.error(f"Error generating JSON report: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze YouTube viewing patterns for a date range')
    parser.add_argument('--uri', type=str, default="bolt://localhost:7687", help='Neo4j URI')
    parser.add_argument('--user', type=str, default="neo4j", help='Neo4j username')
    parser.add_argument('--password', type=str, default="12345678", help='Neo4j password')
    parser.add_argument('--start_date', type=str, help='Start date in ISO format (YYYY-MM-DDThh:mm:ss+00:00)')
    parser.add_argument('--end_date', type=str, help='End date in ISO format (YYYY-MM-DDThh:mm:ss+00:00)')
    parser.add_argument('--output_dir', type=str, default="pattern_analysis", help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Run the analysis
    results = run_date_range_analysis(
        args.uri, args.user, args.password,
        args.start_date, args.end_date, args.output_dir
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir} directory.")
    print("You can view the following files:")
    print(f" - {args.output_dir}/report.txt - Plain text report with key findings")
    print(f" - {args.output_dir}/daily_pattern_trends.csv - CSV file with daily pattern data")
    print(f" - {args.output_dir}/monthly_pattern_trends.csv - CSV file with monthly pattern data")
    print(f" - {args.output_dir}/*.png - Visualizations of patterns and trends")