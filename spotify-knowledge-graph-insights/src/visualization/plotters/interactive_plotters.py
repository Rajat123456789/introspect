"""
Interactive Plotters Module

This module contains functions for creating interactive visualizations of Spotify data using Plotly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractivePlotters:
    """
    Class for creating interactive visualizations of Spotify data.
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
        
        # Set default template
        pio.templates.default = "plotly_white"
        
    def save_figure(self, fig, filename):
        """
        Save a figure to the output directory.
        
        Args:
            fig: The figure to save.
            filename (str): The filename to save to.
        """
        if self.output_dir:
            filepath = Path(self.output_dir) / filename
            fig.write_html(filepath)
            logger.info(f"Saved interactive figure to {filepath}")
    
    def plot_listening_patterns_heatmap(self, data, save_as=None):
        """
        Create a heatmap of listening patterns by day of week and hour.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            plotly.graph_objects.Figure: The created figure.
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
        
        # Extract day of week and hour
        df = data.copy()
        df['day_of_week'] = df['end_time'].dt.day_name()
        df['hour'] = df['end_time'].dt.hour
        
        # Create pivot table
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pd.pivot_table(
            df, 
            values='track_id', 
            index='day_of_week',
            columns='hour',
            aggfunc='count',
            fill_value=0
        ).reindex(day_order)
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Number of Tracks"),
            x=[f"{h:02d}:00" for h in range(24)],
            y=day_order,
            color_continuous_scale="Viridis",
            title="Listening Patterns by Day and Hour"
        )
        
        fig.update_layout(
            width=1000,
            height=600,
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            coloraxis_colorbar=dict(title="Track Count")
        )
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_audio_features_radar(self, data, track_ids=None, n_tracks=5, save_as=None):
        """
        Create an interactive radar chart of audio features for tracks.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            track_ids (list, optional): List of track IDs to include. If None, uses top tracks.
            n_tracks (int): Number of top tracks to include if track_ids is None.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            plotly.graph_objects.Figure: The created figure.
        """
        # Define audio features to include
        features = ['danceability', 'energy', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in features if col in data.columns]
        
        if not available_features:
            logger.warning("No audio features found in the data")
            return None
            
        if 'track_name_x' not in data.columns and 'track_name_y' not in data.columns:
            logger.warning("Track name column not found in data")
            return None
            
        # Use the available track name column
        track_name_col = 'track_name_x' if 'track_name_x' in data.columns else 'track_name_y'
        
        # Filter tracks
        if track_ids is not None:
            if 'track_id' not in data.columns:
                logger.warning("Column 'track_id' not found in data")
                return None
                
            tracks_data = data[data['track_id'].isin(track_ids)].drop_duplicates(subset=['track_id'])
        else:
            # Get top tracks by play count
            top_tracks = data[track_name_col].value_counts().head(n_tracks).index
            tracks_data = data[data[track_name_col].isin(top_tracks)].drop_duplicates(subset=[track_name_col])
        
        if len(tracks_data) == 0:
            logger.warning("No data available for selected tracks")
            return None
            
        # Create figure
        fig = go.Figure()
        
        # Add one trace per track
        for _, track in tracks_data.iterrows():
            values = [track[feature] for feature in available_features]
            values.append(values[0])  # Close the loop
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_features + [available_features[0]],  # Close the loop
                fill='toself',
                name=track[track_name_col]
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Audio Features Comparison",
            width=800,
            height=800
        )
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_genre_distribution_sunburst(self, data, save_as=None):
        """
        Create a sunburst chart of genre distribution.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            plotly.graph_objects.Figure: The created figure.
        """
        if 'track_genre' not in data.columns:
            logger.warning("Column 'track_genre' not found in data")
            return None
            
        # Process genres to create hierarchy
        df = data.copy()
        
        # Split genres into main and sub-genres
        df['main_genre'] = df['track_genre'].str.split('-').str[0]
        df['sub_genre'] = df['track_genre'].str.split('-').str[1]
        
        # Replace NaN in sub_genre with empty string
        df['sub_genre'] = df['sub_genre'].fillna('')
        
        # Count tracks by genre
        genre_counts = df.groupby(['main_genre', 'sub_genre']).size().reset_index(name='count')
        
        # Create sunburst chart
        fig = px.sunburst(
            genre_counts,
            path=['main_genre', 'sub_genre'],
            values='count',
            title="Genre Distribution",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            width=800,
            height=800
        )
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_listening_timeline_interactive(self, data, time_unit='D', save_as=None):
        """
        Create an interactive timeline of listening activity.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            time_unit (str): Time unit for resampling ('D' for daily, 'W' for weekly, 'M' for monthly).
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            plotly.graph_objects.Figure: The created figure.
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
        time_counts = data.set_index('end_time').resample(time_unit).size().reset_index(name='count')
        
        # Create time series plot
        time_unit_name = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[time_unit]
        
        fig = px.line(
            time_counts,
            x='end_time',
            y='count',
            title=f"{time_unit_name} Listening Activity",
            labels={'end_time': 'Date', 'count': 'Number of Tracks'}
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            width=1000,
            height=600
        )
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_top_artists_bar(self, data, top_n=10, save_as=None):
        """
        Create an interactive bar chart of top artists.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            top_n (int): Number of top artists to include.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            plotly.graph_objects.Figure: The created figure.
        """
        if 'artist_name' not in data.columns:
            logger.warning("Column 'artist_name' not found in data")
            return None
            
        # Count tracks by artist
        artist_counts = data['artist_name'].value_counts().head(top_n).reset_index()
        artist_counts.columns = ['artist', 'count']
        
        # Sort by count
        artist_counts = artist_counts.sort_values('count')
        
        # Create bar chart
        fig = px.bar(
            artist_counts,
            y='artist',
            x='count',
            title=f"Top {top_n} Artists",
            labels={'artist': 'Artist', 'count': 'Number of Tracks'},
            orientation='h',
            color='count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            width=900,
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_audio_features_scatter(self, data, x_feature='danceability', y_feature='energy', 
                                   color_by='track_genre', size_by='popularity', save_as=None):
        """
        Create an interactive scatter plot of audio features.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            x_feature (str): Feature to plot on x-axis.
            y_feature (str): Feature to plot on y-axis.
            color_by (str): Column to use for coloring points.
            size_by (str): Column to use for sizing points.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            plotly.graph_objects.Figure: The created figure.
        """
        # Check if required columns exist
        required_cols = [x_feature, y_feature]
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in data")
                return None
        
        # Check optional columns
        if color_by and color_by not in data.columns:
            logger.warning(f"Column {color_by} not found in data, using default coloring")
            color_by = None
            
        if size_by and size_by not in data.columns:
            logger.warning(f"Column {size_by} not found in data, using default sizing")
            size_by = None
        
        # Prepare data
        df = data.copy()
        
        # If color_by is a categorical column with too many categories, limit to top N
        if color_by and df[color_by].nunique() > 10:
            top_categories = df[color_by].value_counts().head(10).index
            df.loc[~df[color_by].isin(top_categories), color_by] = 'Other'
        
        # Create hover text
        if 'track_name_x' in df.columns:
            track_name_col = 'track_name_x'
        elif 'track_name_y' in df.columns:
            track_name_col = 'track_name_y'
        else:
            track_name_col = None
            
        if 'artist_name' in df.columns and track_name_col:
            df['hover_text'] = df[track_name_col] + ' - ' + df['artist_name']
        elif track_name_col:
            df['hover_text'] = df[track_name_col]
        else:
            df['hover_text'] = ''
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color=color_by if color_by else None,
            size=size_by if size_by else None,
            hover_name='hover_text',
            title=f"{y_feature.title()} vs {x_feature.title()}",
            labels={
                x_feature: x_feature.replace('_', ' ').title(),
                y_feature: y_feature.replace('_', ' ').title(),
                color_by: color_by.replace('_', ' ').title() if color_by else None,
                size_by: size_by.replace('_', ' ').title() if size_by else None
            },
            opacity=0.7
        )
        
        # Update layout
        fig.update_layout(
            width=1000,
            height=700,
            xaxis=dict(range=[0, 1]) if x_feature in ['danceability', 'energy', 'speechiness', 
                                                     'acousticness', 'instrumentalness', 
                                                     'liveness', 'valence'] else {},
            yaxis=dict(range=[0, 1]) if y_feature in ['danceability', 'energy', 'speechiness', 
                                                     'acousticness', 'instrumentalness', 
                                                     'liveness', 'valence'] else {}
        )
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig 