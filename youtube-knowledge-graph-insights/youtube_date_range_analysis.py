import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from youtube_mental_health_analysis_test import MentalHealthAnalyzer, logger

# Define the date range
end_date = '2025-02-13T04:39:20+00:00'
start_date = '2025-02-08T04:39:20+00:00'  # 5 days before

def run_date_range_analysis(uri, user, password, start_date, end_date):
    """
    Run analysis for a specific date range and generate visualizations
    """
    # Initialize the analyzer
    analyzer = MentalHealthAnalyzer(uri, user, password)
    
    # Create output directory
    output_dir = 'date_range_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set custom output directory
    analyzer.output_dir = output_dir
    
    try:
        # 1. Get viewing pattern data for the specific date range
        with analyzer.driver.session() as session:
            query = f"""
            MATCH (v:Video)
            WHERE v.watched_at IS NOT NULL 
                AND v.watched_at >= '{start_date}' 
                AND v.watched_at <= '{end_date}'
            WITH toString(v.watched_at) as watched_time_str, v
            WITH substring(watched_time_str, 0, 10) as view_date,
                 collect({{time: watched_time_str, id: v.video_id, category: v.primary_category, title: v.title}}) as daily_views
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
            df_viewing = pd.DataFrame([dict(record) for record in result])
            
            if not df_viewing.empty:
                # Convert date
                df_viewing['date'] = pd.to_datetime(df_viewing['view_date'])
                
                # Save viewing patterns
                df_viewing.to_csv(f"{output_dir}/viewing_patterns.csv", index=False)
                
                # Create visualizations for viewing patterns
                create_viewing_pattern_visualizations(df_viewing, output_dir)
        
        # 2. Get sentiment data for the specific date range
        with analyzer.driver.session() as session:
            query = f"""
            MATCH (v:Video)-[:HAS_MENTAL_HEALTH_DATA]->(m:MentalHealthData)
            WHERE m.timestamp IS NOT NULL AND m.score IS NOT NULL
                AND m.timestamp >= '{start_date}' 
                AND m.timestamp <= '{end_date}'
            RETURN v.title AS title,
                   m.category AS category,
                   m.score AS score,
                   m.sentiment AS sentiment,
                   toString(m.timestamp) AS timestamp
            """
            result = session.run(query)
            df_sentiment = pd.DataFrame([dict(record) for record in result])
            
            if not df_sentiment.empty:
                # Convert timestamp to datetime
                df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'], errors='coerce')
                
                # Convert score to numeric
                df_sentiment['score'] = pd.to_numeric(df_sentiment['score'], errors='coerce')
                
                # Save sentiment data
                df_sentiment.to_csv(f"{output_dir}/sentiment_data.csv", index=False)
                
                # Create sentiment visualizations
                create_sentiment_visualizations(df_sentiment, output_dir)
        
        # 3. Get music data for the specific date range
        with analyzer.driver.session() as session:
            query = f"""
            MATCH (n:Video) 
            WHERE (n.primary_category = 'Music' OR n.primary_category = 'Entertainment') 
                AND n.title IS NOT NULL
                AND n.timestamp >= '{start_date}' 
                AND n.timestamp <= '{end_date}'
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
            df_music = pd.DataFrame([dict(record) for record in result])
            
            if not df_music.empty:
                # Convert timestamp and score
                df_music['timestamp'] = pd.to_datetime(df_music['timestamp'], errors='coerce')
                df_music['score'] = pd.to_numeric(df_music['score'], errors='coerce')
                
                # Save music data
                df_music.to_csv(f"{output_dir}/music_data.csv", index=False)
                
                # Apply genre classification and create visualizations
                df_music = analyzer._enhance_music_genre_classification(df_music)
                create_music_visualizations(df_music, output_dir)
        
        # 4. Get category data for the specific date range
        with analyzer.driver.session() as session:
            query = f"""
            MATCH (v:Video)
            WHERE v.primary_category IS NOT NULL 
                 AND (v.score IS NOT NULL OR v.sentiment_score IS NOT NULL)
                 AND v.timestamp >= '{start_date}' 
                 AND v.timestamp <= '{end_date}'
            RETURN v.primary_category AS category,
                   v.detailed_type AS subcategory,
                   CASE WHEN v.score IS NOT NULL THEN v.score
                        WHEN v.sentiment_score IS NOT NULL THEN v.sentiment_score
                        ELSE 0.5 END AS score,
                   toString(v.timestamp) AS timestamp
            """
            result = session.run(query)
            df_categories = pd.DataFrame([dict(record) for record in result])
            
            if not df_categories.empty:
                # Convert timestamp and score
                df_categories['timestamp'] = pd.to_datetime(df_categories['timestamp'], errors='coerce')
                df_categories['score'] = pd.to_numeric(df_categories['score'], errors='coerce')
                
                # Save category data
                df_categories.to_csv(f"{output_dir}/category_data.csv", index=False)
                
                # Create category visualizations
                create_category_visualizations(df_categories, output_dir)
        
        # List all created visualizations
        vis_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"Created {len(vis_files)} visualizations for date range {start_date} to {end_date}:")
        for file in vis_files:
            print(f"  - {os.path.join(output_dir, file)}")
        
        return vis_files
            
    except Exception as e:
        logger.error(f"Error in date range analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        analyzer.close()


def create_viewing_pattern_visualizations(df, output_dir):
    """Create visualizations for viewing patterns"""
    try:
        if df.empty:
            return
            
        # Create day of week histogram
        plt.figure(figsize=(12, 6))
        # Extract day of week
        df['day_of_week'] = df['date'].dt.day_name()
        # Group by day and calculate total videos
        day_data = df.groupby('day_of_week')['videos_per_day'].sum()
        # Order days properly
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_data = day_data.reindex(days_order)
        # Create bar chart
        ax = sns.barplot(x=day_data.index, y=day_data.values, palette='viridis')
        plt.title('Videos Watched by Day of Week', fontsize=16)
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Number of Videos', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/viewing_day_of_week.png")
        plt.close()
        
        # Create late night viewing pattern 
        plt.figure(figsize=(12, 6))
        # Create data for late night viewing
        late_night_data = df.groupby('day_of_week')['late_night_count'].sum()
        late_night_data = late_night_data.reindex(days_order)
        ax = sns.barplot(x=late_night_data.index, y=late_night_data.values, palette='plasma')
        plt.title('Late Night Viewing Pattern (10 PM - 4 AM)', fontsize=16)
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Number of Late Night Videos', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/late_night_viewing_pattern.png")
        plt.close()
        
        # Create binge watching pattern
        if 'binge_day' in df.columns:
            plt.figure(figsize=(12, 6))
            binge_days = df[df['binge_day'] == True]['view_date'].tolist()
            if binge_days:
                binge_df = pd.DataFrame({'date': binge_days})
                binge_df['date'] = pd.to_datetime(binge_df['date'])
                binge_df['day_of_week'] = binge_df['date'].dt.day_name()
                binge_counts = binge_df['day_of_week'].value_counts().reindex(days_order, fill_value=0)
                ax = sns.barplot(x=binge_counts.index, y=binge_counts.values, palette='rocket')
                plt.title('Binge Watching Days (15+ videos)', fontsize=16)
                plt.xlabel('Day of Week', fontsize=12)
                plt.ylabel('Number of Binge Days', fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/binge_watching_pattern.png")
                plt.close()
    except Exception as e:
        logger.error(f"Error creating viewing pattern visualizations: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())


def create_sentiment_visualizations(df, output_dir):
    """Create visualizations for sentiment data"""
    try:
        if df.empty:
            return
            
        # Create sentiment over time scatter plot
        plt.figure(figsize=(14, 7))
        # Sort by timestamp
        df = df.sort_values('timestamp')
        # Create scatter with trendline
        sns.regplot(x=df.index, y='score', data=df, scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
        plt.title('Sentiment Score Trajectory', fontsize=16)
        plt.xlabel('Video Index (Chronological)', fontsize=12)
        plt.ylabel('Sentiment Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/sentiment_trajectory.png")
        plt.close()
        
        # Create sentiment by category
        if 'category' in df.columns:
            plt.figure(figsize=(14, 7))
            # Group by category and get mean scores
            cat_scores = df.groupby('category')['score'].mean().sort_values(ascending=False)
            # Only show top 15 categories
            if len(cat_scores) > 15:
                cat_scores = cat_scores[:15]
            # Create bar chart
            sns.barplot(x=cat_scores.index, y=cat_scores.values, palette='viridis')
            plt.title('Average Sentiment Score by Content Category', fontsize=16)
            plt.xlabel('Content Category', fontsize=12)
            plt.ylabel('Average Sentiment Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sentiment_by_category.png")
            plt.close()
            
        # Create hourly sentiment pattern
        plt.figure(figsize=(14, 7))
        # Extract hour from timestamp
        df['hour'] = df['timestamp'].dt.hour
        # Group by hour and calculate mean sentiment
        hourly_sentiment = df.groupby('hour')['score'].mean()
        # Create bar chart
        sns.barplot(x=hourly_sentiment.index, y=hourly_sentiment.values, palette='coolwarm')
        plt.title('Average Sentiment Score by Hour of Day', fontsize=16)
        plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hourly_sentiment_pattern.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error creating sentiment visualizations: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())


def create_music_visualizations(df, output_dir):
    """Create visualizations for music data"""
    try:
        if df.empty:
            return
            
        # Create genre impact visualization
        plt.figure(figsize=(14, 8))
        # Group by music genre
        genre_impact = df.groupby('music_genre').agg({
            'score': 'mean',
            'id': 'count'
        }).reset_index()
        genre_impact.columns = ['music_genre', 'avg_score', 'count']
        # Sort by count (popularity)
        genre_impact = genre_impact.sort_values('count', ascending=False)
        # Only show top 15 genres
        if len(genre_impact) > 15:
            genre_impact = genre_impact[:15]
        # Create bar chart
        ax = sns.barplot(x='music_genre', y='avg_score', data=genre_impact, palette='viridis')
        # Add count as text on bars
        for i, row in enumerate(genre_impact.itertuples()):
            ax.text(i, row.avg_score/2, f'n={row.count}', 
                    ha='center', color='white', fontweight='bold')
        plt.title('Impact of Music Genres on Mental Health', fontsize=16)
        plt.xlabel('Music Genre', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/music_genre_impact.png")
        plt.close()
        
        # Create hourly music impact
        plt.figure(figsize=(14, 7))
        # Extract hour from timestamp
        df['hour'] = df['timestamp'].dt.hour
        # Group by hour and calculate mean sentiment
        hourly_music = df.groupby('hour')['score'].mean()
        # Create bar chart
        sns.barplot(x=hourly_music.index, y=hourly_music.values, palette='plasma')
        plt.title('Music Impact by Hour of Day', fontsize=16)
        plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hourly_music_impact.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error creating music visualizations: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())


def create_category_visualizations(df, output_dir):
    """Create visualizations for content categories"""
    try:
        if df.empty:
            return
            
        # Create category impact visualization
        plt.figure(figsize=(14, 8))
        # Group by category
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
        # Sort by count (popularity)
        cat_impact = cat_impact.sort_values('video_count', ascending=False)
        # Only show top 15 categories
        if len(cat_impact) > 15:
            cat_impact = cat_impact[:15]
        # Create bar chart
        ax = sns.barplot(x='content_category', y='score', data=cat_impact, palette='viridis')
        # Add count as text on bars
        for i, row in enumerate(cat_impact.itertuples()):
            ax.text(i, row.score/2, f'n={row.video_count}', 
                    ha='center', color='white', fontweight='bold')
        plt.title('Impact of Content Categories on Mental Health', fontsize=16)
        plt.xlabel('Content Category', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_impact.png")
        plt.close()
        
        # Create diversity vs impact visualization
        plt.figure(figsize=(12, 8))
        # Create scatter plot of diversity vs score
        sns.scatterplot(x='diversity', y='score', size='video_count', 
                        sizes=(50, 500), alpha=0.7, data=cat_impact)
        plt.title('Content Diversity vs Mental Health Impact', fontsize=16)
        plt.xlabel('Content Diversity (Number of Subcategories)', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        # Add category labels to points
        for i, row in cat_impact.iterrows():
            plt.text(row['diversity'], row['score'], row['content_category'], 
                    ha='center', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_diversity_impact.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error creating category visualizations: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    # Get Neo4j connection parameters
    import os
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    
    # Run the analysis
    run_date_range_analysis(uri, user, password, start_date, end_date)
    
    print(f"\nAnalysis complete for date range: {start_date} to {end_date}")
    print(f"Visualizations saved to the 'date_range_analysis' directory") 