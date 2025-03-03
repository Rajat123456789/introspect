"""
Basic Plotters Module

This module contains functions for creating basic visualizations of Spotify data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicPlotters:
    """
    Class for creating basic visualizations of Spotify data.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the plotter.
        
        Args:
            output_dir (str, optional): Directory to save plots to.
        """
        self.output_dir = output_dir
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
    def save_figure(self, fig, filename):
        """
        Save a figure to the output directory.
        
        Args:
            fig: The figure to save.
            filename (str): The filename to save to.
        """
        if self.output_dir:
            filepath = Path(self.output_dir) / filename
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
            logger.info(f"Saved figure to {filepath}")
            
    def plot_listening_by_time(self, data, time_col='hour', title=None, save_as=None):
        """
        Plot listening patterns by time.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            time_col (str): The column to use for time (e.g., 'hour', 'day_of_week').
            title (str, optional): The title for the plot.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if time_col not in data.columns:
            logger.warning(f"Column {time_col} not found in data")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if time_col == 'hour':
            # For hours, we want to ensure all 24 hours are shown
            hour_counts = data['hour'].value_counts().sort_index()
            all_hours = pd.Series(0, index=range(24))
            hour_counts = hour_counts.add(all_hours, fill_value=0).sort_index()
            
            sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax)
            ax.set_xlabel('Hour of Day')
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
            
            if not title:
                title = 'Listening Activity by Hour of Day'
                
        elif time_col == 'day_of_week':
            # For days, ensure correct order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = data['day_of_week'].value_counts().reindex(day_order)
            
            sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax)
            ax.set_xlabel('Day of Week')
            
            if not title:
                title = 'Listening Activity by Day of Week'
        
        ax.set_ylabel('Number of Tracks')
        ax.set_title(title)
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45 if time_col == 'day_of_week' else 0)
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_top_items(self, data, item_col, title=None, top_n=10, save_as=None):
        """
        Plot top items (artists, genres, etc.).
        
        Args:
            data (pandas.DataFrame): The data to plot.
            item_col (str): The column containing the items to count.
            title (str, optional): The title for the plot.
            top_n (int): Number of top items to show.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if item_col not in data.columns:
            logger.warning(f"Column {item_col} not found in data")
            return None
            
        # Get top items
        top_items = data[item_col].value_counts().head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        bars = sns.barplot(x=top_items.values, y=top_items.index, ax=ax)
        
        # Add count labels to bars
        for i, v in enumerate(top_items.values):
            ax.text(v + 0.1, i, str(v), va='center')
        
        # Set labels and title
        ax.set_xlabel('Count')
        ax.set_ylabel(item_col.replace('_', ' ').title())
        
        if not title:
            title = f'Top {top_n} {item_col.replace("_", " ").title()}'
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_audio_features_distribution(self, data, features=None, save_as=None):
        """
        Plot distribution of audio features.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            features (list, optional): List of features to plot. If None, uses default audio features.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if features is None:
            features = ['danceability', 'energy', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in features if col in data.columns]
        
        if not available_features:
            logger.warning("No audio features found in the data")
            return None
            
        # Create figure with subplots
        n_features = len(available_features)
        n_cols = 2
        n_rows = (n_features + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
        axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(available_features):
            if i < len(axes):
                sns.histplot(data[feature].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}')
                axes[i].set_xlabel(feature.replace('_', ' ').title())
                axes[i].set_ylabel('Count')
        
        # Hide any unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_feature_correlations(self, data, features=None, save_as=None):
        """
        Plot correlation matrix of audio features.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            features (list, optional): List of features to include. If None, uses default audio features.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if features is None:
            features = ['danceability', 'energy', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in features if col in data.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough audio features found in the data for correlation analysis")
            return None
            
        # Calculate correlation matrix
        corr_matrix = data[available_features].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Correlation Between Audio Features')
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_genre_wordcloud(self, data, genre_col='track_genre', save_as=None):
        """
        Create a word cloud of genres.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            genre_col (str): The column containing genre information.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if genre_col not in data.columns:
            logger.warning(f"Column {genre_col} not found in data")
            return None
            
        # Count genres
        genre_counts = data[genre_col].value_counts().to_dict()
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             colormap='viridis', max_words=100).generate_from_frequencies(genre_counts)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Genre Word Cloud')
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_listening_timeline(self, data, time_unit='D', save_as=None):
        """
        Plot listening activity over time.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            time_unit (str): Time unit for resampling ('D' for daily, 'W' for weekly, 'M' for monthly).
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if 'end_time' not in data.columns:
            logger.warning("Column 'end_time' not found in data")
            return None
            
        # Ensure end_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['end_time']):
            try:
                data['end_time'] = pd.to_datetime(data['end_time'])
            except Exception as e:
                logger.error(f"Error converting end_time to datetime: {e}")
                return None
        
        # Count tracks by time period
        time_counts = data.set_index('end_time').resample(time_unit).size()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot timeline
        time_counts.plot(ax=ax)
        
        # Set labels and title
        time_unit_name = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[time_unit]
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Tracks')
        ax.set_title(f'{time_unit_name} Listening Activity')
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig 