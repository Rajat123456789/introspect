import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from matplotlib.colors import LinearSegmentedColormap

def load_data(timestamp_folder):
    """Load all CSV and JSON files from a timestamp folder into a dictionary"""
    data = {}
    file_paths = {}  # Store the original file paths
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(timestamp_folder, "*.csv"))
    for file_path in csv_files:
        file_name = os.path.basename(file_path).replace(".csv", "")
        data[file_name] = pd.read_csv(file_path)
        file_paths[file_name] = file_path  # Store the file path
        print(f"Loaded {file_name} from CSV")
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(timestamp_folder, "*.json"))
    for file_path in json_files:
        file_name = os.path.basename(file_path).replace(".json", "")
        
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                
            # If JSON contains a list of records, convert to DataFrame
            if isinstance(json_data, list):
                data[file_name] = pd.DataFrame(json_data)
            else:
                data[file_name] = json_data
                
            file_paths[file_name] = file_path  # Store the file path
            print(f"Loaded {file_name} from JSON")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return data, file_paths

def visualize_music_impact(data, file_paths):
    """Create visualizations for music impact data"""
    visualizations = []
    
    if 'music_energy_impact' in data and 'music_energy_impact' in file_paths:
        df = data['music_energy_impact']
        
        # 1. Energy Type Impact Chart
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in df['impact_score']]
        
        plt.bar(df['energy_type'], df['impact_score'], color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Mental Health Impact by Music Energy Type')
        plt.xlabel('Energy Type')
        plt.ylabel('Impact Score (Negative → Positive)')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_dir = os.path.dirname(file_paths['music_energy_impact'])
        vis_path = os.path.join(output_dir, 'music_energy_impact_viz.png')
        plt.savefig(vis_path)
        plt.close()
        visualizations.append(vis_path)
        
    if 'music_genre_impact' in data and 'music_genre_impact' in file_paths:
        df = data['music_genre_impact']
        
        # 2. Genre Impact Chart
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'green' for x in df['impact_score']]
        
        bars = plt.barh(df['music_genre'], df['impact_score'], color=colors)
        
        # Add count annotations
        if 'count' in df.columns:
            for i, bar in enumerate(bars):
                count = df.iloc[i]['count']
                plt.text(
                    bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height()/2,
                    f"n={count}",
                    va='center',
                    alpha=0.7
                )
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Mental Health Impact by Music Genre')
        plt.xlabel('Impact Score (Negative → Positive)')
        plt.ylabel('Music Genre')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        output_dir = os.path.dirname(file_paths['music_genre_impact'])
        vis_path = os.path.join(output_dir, 'music_genre_impact_viz.png')
        plt.savefig(vis_path)
        plt.close()
        visualizations.append(vis_path)
    
    return visualizations

def visualize_addiction_risk(data, file_paths):
    """Create visualizations for addiction risk data"""
    visualizations = []
    
    if 'category_addiction_risk' in data and 'category_addiction_risk' in file_paths:
        df = data['category_addiction_risk']
        
        # Debug: Print column names
        print(f"Columns in category_addiction_risk: {df.columns.tolist()}")
        
        # Check if required columns exist
        if 'addiction_risk_score' not in df.columns:
            print("Warning: 'addiction_risk_score' column not found in category_addiction_risk data")
            
            # Try to visualize what we have - looking for alternative column name
            score_columns = [col for col in df.columns if 'score' in col.lower() or 'risk' in col.lower()]
            if score_columns:
                print(f"Found alternative score column: {score_columns[0]}")
                score_column = score_columns[0]
            else:
                # If no score column is available, skip this visualization
                print("No suitable score column found, skipping addiction risk visualization")
                return visualizations
        else:
            score_column = 'addiction_risk_score'
        
        # 1. Category Addiction Risk
        plt.figure(figsize=(12, 8))
        
        # Create custom colormap for risk levels
        cmap = LinearSegmentedColormap.from_list('risk_cmap', ['green', 'yellow', 'orange', 'red'])
        
        # Sort by score column
        df_sorted = df.sort_values(score_column, ascending=False)
        
        # Check if 'category' column exists
        if 'category' not in df.columns:
            print("Warning: 'category' column not found, using first string column as category")
            str_columns = [col for col in df.columns if df[col].dtype == 'object']
            if str_columns:
                category_column = str_columns[0]
            else:
                print("No suitable category column found, skipping visualization")
                return visualizations
        else:
            category_column = 'category'
        
        # Create bars with color based on risk
        bars = plt.barh(df_sorted[category_column], df_sorted[score_column], 
                       color=cmap(df_sorted[score_column]))
        
        # Add risk level text
        plt.title('Content Category Addiction Risk')
        plt.xlabel(f'{score_column} (0-1)')
        plt.grid(axis='x', alpha=0.3)
        
        # Add watch count annotations if available
        if 'video_count' in df.columns:
            for i, bar in enumerate(bars):
                count = df_sorted.iloc[i]['video_count']
                plt.text(
                    bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f"n={count}",
                    va='center',
                    alpha=0.7
                )
        
        plt.tight_layout()
        
        output_dir = os.path.dirname(file_paths['category_addiction_risk'])
        vis_path = os.path.join(output_dir, 'category_addiction_risk_viz.png')
        plt.savefig(vis_path)
        plt.close()
        visualizations.append(vis_path)
        
        # 2. Category Risk Bubble Chart
        if all(col in df.columns for col in ['video_count', 'avg_negative_score']):
            plt.figure(figsize=(12, 10))
            
            # Create bubble chart with:
            # - x: average negative score magnitude
            # - y: video count
            # - size: addiction risk score
            
            plt.scatter(
                df['avg_negative_score'].abs(),  # Use absolute value for plotting
                df['video_count'],
                s=df['addiction_risk_score'] * 1000,  # Scale for visibility
                c=df['addiction_risk_score'],
                cmap=cmap,
                alpha=0.7
            )
            
            # Add category labels
            for i, row in df.iterrows():
                plt.annotate(
                    row['category'],
                    (row['avg_negative_score'].abs(), row['video_count']),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
                
            plt.colorbar(label='Addiction Risk Score')
            plt.title('Content Addiction Risk Factors')
            plt.xlabel('Average Negative Mental Health Impact')
            plt.ylabel('Number of Videos Watched')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['category_addiction_risk'])
            vis_path = os.path.join(output_dir, 'addiction_risk_bubble_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
    
    return visualizations

def visualize_music_details(data, file_paths):
    """Create visualizations for detailed music analysis"""
    visualizations = []
    
    if 'music_detailed_analysis' in data and 'music_detailed_analysis' in file_paths:
        df = data['music_detailed_analysis']
        
        # 1. Watch Hour Distribution by Genre
        if 'watch_hour' in df.columns and 'music_genre' in df.columns:
            # Filter out null watch_hours
            hour_data = df.dropna(subset=['watch_hour'])
            
            # Convert watch_hour to numeric if it's not
            if not pd.api.types.is_numeric_dtype(hour_data['watch_hour']):
                hour_data['watch_hour'] = pd.to_numeric(hour_data['watch_hour'], errors='coerce')
                hour_data = hour_data.dropna(subset=['watch_hour'])
            
            # Group by genre and hour
            hour_counts = hour_data.groupby(['music_genre', 'watch_hour']).size().reset_index(name='count')
            
            # Pivot for heatmap
            pivot_data = hour_counts.pivot(index='music_genre', columns='watch_hour', values='count')
            pivot_data = pivot_data.fillna(0)
            
            plt.figure(figsize=(15, 8))
            sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Number of Videos'})
            plt.title('Music Genre Viewing by Hour of Day')
            plt.xlabel('Hour of Day (24h)')
            plt.ylabel('Music Genre')
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['music_detailed_analysis'])
            vis_path = os.path.join(output_dir, 'music_genre_by_hour_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
        
        # 2. Energy Type vs Genre Distribution
        if 'energy_type' in df.columns and 'music_genre' in df.columns:
            # Count combinations of genre and energy
            genre_energy = df.groupby(['music_genre', 'energy_type']).size().reset_index(name='count')
            
            # Pivot for heatmap
            pivot_data = genre_energy.pivot(index='music_genre', columns='energy_type', values='count')
            pivot_data = pivot_data.fillna(0)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.0f', cbar_kws={'label': 'Count'})
            plt.title('Music Genre by Energy Type Distribution')
            plt.xlabel('Energy Type')
            plt.ylabel('Music Genre')
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['music_detailed_analysis'])
            vis_path = os.path.join(output_dir, 'genre_energy_distribution_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
    
    return visualizations

def visualize_sentiment_trajectory(data, file_paths):
    """Create visualizations for sentiment trajectory data"""
    visualizations = []
    
    if 'sentiment_trajectory' in data and 'sentiment_trajectory' in file_paths:
        df = data['sentiment_trajectory']
        
        # 1. Sentiment Distribution Pie Chart
        if 'sentiment' in df.columns:
            plt.figure(figsize=(10, 8))
            sentiment_counts = df['sentiment'].value_counts()
            
            # Define colors for sentiments
            colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            color_map = [colors.get(s, 'blue') for s in sentiment_counts.index]
            
            sentiment_counts.plot.pie(
                autopct='%1.1f%%', 
                colors=color_map,
                shadow=True,
                startangle=90,
                explode=[0.05] * len(sentiment_counts)
            )
            plt.title('Sentiment Distribution Across Videos')
            plt.ylabel('')
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['sentiment_trajectory'])
            vis_path = os.path.join(output_dir, 'sentiment_distribution_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
        
        # 2. Sentiment Score by Category
        if all(col in df.columns for col in ['score', 'category']):
            plt.figure(figsize=(12, 8))
            
            # Group by category and get average score
            category_scores = df.groupby('category')['score'].agg(['mean', 'count']).reset_index()
            category_scores = category_scores.sort_values('mean')
            
            # Create bars with color based on score
            colors = ['red' if x < 0 else 'green' for x in category_scores['mean']]
            bars = plt.barh(category_scores['category'], category_scores['mean'], color=colors)
            
            # Add count annotations
            for i, bar in enumerate(bars):
                count = category_scores.iloc[i]['count']
                plt.text(
                    bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height()/2,
                    f"n={count}",
                    va='center',
                    alpha=0.7
                )
            
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Average Sentiment Score by Content Category')
            plt.xlabel('Sentiment Score (Negative → Positive)')
            plt.ylabel('Category')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['sentiment_trajectory'])
            vis_path = os.path.join(output_dir, 'category_sentiment_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
        
        # 3. Sentiment Timeline if timestamp data is available
        if 'timestamp' in df.columns:
            plt.figure(figsize=(15, 8))
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            
            # Plot sentiment score over time
            plt.scatter(df_sorted['timestamp'], df_sorted['score'], 
                       c=df_sorted['score'], cmap='RdYlGn', alpha=0.7)
            
            # Add trend line
            if len(df_sorted) > 1:
                z = np.polyfit(range(len(df_sorted)), df_sorted['score'], 1)
                p = np.poly1d(z)
                plt.plot(df_sorted['timestamp'], p(range(len(df_sorted))), "r--", linewidth=2)
            
            plt.title('Sentiment Score Trajectory Over Time')
            plt.xlabel('Date')
            plt.ylabel('Sentiment Score')
            plt.grid(True, alpha=0.3)
            plt.colorbar(label='Sentiment Score')
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['sentiment_trajectory'])
            vis_path = os.path.join(output_dir, 'sentiment_timeline_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
    
    return visualizations

def visualize_viewing_patterns(data, file_paths):
    """Create visualizations for viewing pattern data"""
    visualizations = []
    
    if 'viewing_patterns' in data and 'viewing_patterns' in file_paths:
        df = data['viewing_patterns']
        
        # 1. Videos per day distribution
        if 'videos_per_day' in df.columns:
            plt.figure(figsize=(12, 6))
            
            # Create histogram of videos per day
            bins = range(0, int(df['videos_per_day'].max()) + 5, 5)
            plt.hist(df['videos_per_day'], bins=bins, color='skyblue', edgecolor='black')
            
            # Add a vertical line for binge threshold (e.g. 15 videos)
            plt.axvline(x=15, color='red', linestyle='--', label='Binge Threshold')
            
            plt.title('Distribution of Videos Watched Per Day')
            plt.xlabel('Number of Videos')
            plt.ylabel('Frequency (Days)')
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['viewing_patterns'])
            vis_path = os.path.join(output_dir, 'videos_per_day_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
        
        # 2. Late night viewing calendar
        if all(col in df.columns for col in ['date', 'late_night_count']):
            # Ensure date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Create a calendar heatmap
            plt.figure(figsize=(16, 8))
            
            # Convert data to a format suitable for a calendar heatmap
            # We'll create a simple heatmap by month and day of week
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # Create pivot table: months as columns, days of week as rows
            pivot_data = df.pivot_table(
                index='day_of_week', 
                columns='month', 
                values='late_night_count',
                aggfunc='sum'
            ).fillna(0)
            
            # Day of week labels
            day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Create heatmap
            sns.heatmap(
                pivot_data, 
                cmap='YlOrRd',
                linewidths=0.5,
                annot=True,
                fmt='.0f',
                cbar_kws={'label': 'Late Night Videos Count'}
            )
            
            plt.title('Late Night Viewing Pattern by Day of Week and Month')
            plt.xlabel('Month')
            plt.ylabel('Day of Week')
            plt.yticks(np.arange(0.5, len(day_labels), 1), day_labels)
            
            # Only show month labels for columns that exist in the data
            visible_months = sorted(df['month'].unique())
            plt.xticks(
                np.arange(0.5, len(visible_months), 1),
                [month_labels[m-1] for m in visible_months]
            )
            
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['viewing_patterns'])
            vis_path = os.path.join(output_dir, 'late_night_viewing_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
        
        # 3. Binge watching trends
        if all(col in df.columns for col in ['date', 'binge_day']):
            plt.figure(figsize=(14, 7))
            
            # Group by month and count binge days
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
            df['year_month'] = df['date'].dt.to_period('M')
            monthly_binges = df.groupby('year_month')['binge_day'].sum().reset_index()
            monthly_days = df.groupby('year_month').size().reset_index(name='total_days')
            
            # Calculate percentage
            monthly_stats = pd.merge(monthly_binges, monthly_days)
            monthly_stats['binge_percentage'] = monthly_stats['binge_day'] / monthly_stats['total_days'] * 100
            
            # Create a bar chart
            plt.bar(
                monthly_stats['year_month'].astype(str),
                monthly_stats['binge_percentage'],
                color='orange'
            )
            
            plt.title('Monthly Binge Watching Trends')
            plt.xlabel('Month')
            plt.ylabel('Percentage of Days with Binge Watching')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['viewing_patterns'])
            vis_path = os.path.join(output_dir, 'binge_trends_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
    
    return visualizations

def visualize_title_patterns(data, file_paths):
    """Create visualizations for title pattern analysis data"""
    visualizations = []
    
    if 'title_addiction_patterns' in data and 'title_addiction_patterns' in file_paths:
        df = data['title_addiction_patterns']
        
        # Debug info
        print(f"Columns in title_addiction_patterns: {df.columns.tolist()}")
        print(f"Shape of title_addiction_patterns data: {df.shape}")
        
        # 1. Clickbait Title Analysis
        clickbait_cols = [col for col in df.columns if 'clickbait' in col.lower()]
        if clickbait_cols:
            clickbait_col = clickbait_cols[0]
            
            # Check if it's boolean or numeric
            if df[clickbait_col].dtype == bool or set(df[clickbait_col].unique()).issubset({0, 1}):
                # Create pie chart of clickbait distribution
                plt.figure(figsize=(10, 8))
                clickbait_counts = df[clickbait_col].value_counts()
                
                # Convert to strings for the pie chart
                if clickbait_counts.index.dtype != 'object':
                    clickbait_counts.index = clickbait_counts.index.map({1: 'Clickbait', 0: 'Normal'})
                
                colors = ['red', 'green']
                clickbait_counts.plot.pie(
                    autopct='%1.1f%%',
                    colors=colors,
                    labels=None  # Turn off the terrible default labels
                )
                plt.legend(clickbait_counts.index, loc='upper right')
                plt.title('Clickbait Title Distribution')
                plt.ylabel('')
                plt.tight_layout()
                
                output_dir = os.path.dirname(file_paths['title_addiction_patterns'])
                vis_path = os.path.join(output_dir, 'clickbait_distribution_viz.png')
                plt.savefig(vis_path)
                plt.close()
                visualizations.append(vis_path)
        
        # 2. Series Content Analysis
        series_cols = [col for col in df.columns if 'series' in col.lower()]
        if series_cols:
            series_col = series_cols[0]
            
            # Create pie chart of series distribution
            plt.figure(figsize=(10, 8))
            series_counts = df[series_col].value_counts()
            
            # Convert to strings for the pie chart
            if series_counts.index.dtype != 'object':
                series_counts.index = series_counts.index.map({1: 'Series Content', 0: 'Standalone Content'})
            
            colors = ['orange', 'blue']
            series_counts.plot.pie(
                autopct='%1.1f%%',
                colors=colors,
                labels=None
            )
            plt.legend(series_counts.index, loc='upper right')
            plt.title('Series vs. Standalone Content Distribution')
            plt.ylabel('')
            plt.tight_layout()
            
            output_dir = os.path.dirname(file_paths['title_addiction_patterns'])
            vis_path = os.path.join(output_dir, 'series_distribution_viz.png')
            plt.savefig(vis_path)
            plt.close()
            visualizations.append(vis_path)
        
        # 3. Word count distribution in titles
        title_cols = [col for col in df.columns if 'title' in col.lower()]
        for col in title_cols:
            if df[col].dtype == 'object':  # If it's a string column
                # Calculate word counts
                df['word_count'] = df[col].str.split().str.len()
                
                plt.figure(figsize=(12, 6))
                plt.hist(df['word_count'], bins=range(1, 16), color='skyblue', edgecolor='black')
                plt.title('Word Count Distribution in Video Titles')
                plt.xlabel('Number of Words')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                output_dir = os.path.dirname(file_paths['title_addiction_patterns'])
                vis_path = os.path.join(output_dir, 'title_word_count_viz.png')
                plt.savefig(vis_path)
                plt.close()
                visualizations.append(vis_path)
                break  # Only process the first title column
    
    return visualizations

def main():
    # Get all timestamp folders
    reports_dir = 'analysis_reports'
    if not os.path.exists(reports_dir):
        print(f"Reports directory '{reports_dir}' not found")
        return
    
    # Find all timestamp folders
    timestamp_folders = [os.path.join(reports_dir, d) for d in os.listdir(reports_dir) 
                        if os.path.isdir(os.path.join(reports_dir, d))]
    
    if not timestamp_folders:
        print("No timestamp folders found in the reports directory")
        return
    
    # Sort folders by timestamp (newest first)
    timestamp_folders.sort(reverse=True)
    
    # Ask user which folder to visualize
    print("\nAvailable analysis runs:")
    for i, folder in enumerate(timestamp_folders):
        timestamp = os.path.basename(folder)
        print(f"{i+1}. {timestamp}")
    
    try:
        choice = int(input("\nEnter the number of the analysis run to visualize (or 0 for newest): "))
        if choice == 0:
            folder_to_visualize = timestamp_folders[0]
        else:
            folder_to_visualize = timestamp_folders[choice-1]
    except (ValueError, IndexError):
        print("Invalid choice, using the most recent analysis")
        folder_to_visualize = timestamp_folders[0]
    
    print(f"\nVisualizing data from: {os.path.basename(folder_to_visualize)}")
    
    # Load data from the selected folder
    data, file_paths = load_data(folder_to_visualize)
    
    if not data:
        print("No data files found in the selected folder")
        return
    
    # Visualize different aspects of the data
    visualizations = []
    visualizations.extend(visualize_sentiment_trajectory(data, file_paths))
    visualizations.extend(visualize_viewing_patterns(data, file_paths))
    visualizations.extend(visualize_music_impact(data, file_paths))
    visualizations.extend(visualize_title_patterns(data, file_paths))
    visualizations.extend(visualize_addiction_risk(data, file_paths))
    visualizations.extend(visualize_music_details(data, file_paths))
    
    # Report on generated visualizations
    if visualizations:
        print(f"\nGenerated {len(visualizations)} new visualizations:")
        for vis in visualizations:
            print(f"- {os.path.basename(vis)}")
    else:
        print("No visualizations were generated. The data may not match expected formats.")

if __name__ == "__main__":
    main() 