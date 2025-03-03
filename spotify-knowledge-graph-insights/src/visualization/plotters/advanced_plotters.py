"""
Advanced Plotters Module

This module contains functions for creating advanced visualizations of Spotify data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPlotters:
    """
    Class for creating advanced visualizations of Spotify data.
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
    
    def plot_feature_radar_chart(self, data, features=None, n_tracks=5, save_as=None):
        """
        Create a radar chart of audio features for top tracks.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            features (list, optional): List of features to include. If None, uses default audio features.
            n_tracks (int): Number of top tracks to include.
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
            
        if 'track_name_x' not in data.columns and 'track_name_y' not in data.columns:
            logger.warning("Track name column not found in data")
            return None
            
        # Use the available track name column
        track_name_col = 'track_name_x' if 'track_name_x' in data.columns else 'track_name_y'
        
        # Get top tracks by play count
        top_tracks = data[track_name_col].value_counts().head(n_tracks).index
        
        # Filter data to include only top tracks
        top_tracks_data = data[data[track_name_col].isin(top_tracks)].drop_duplicates(subset=[track_name_col])
        
        if len(top_tracks_data) == 0:
            logger.warning("No data available for top tracks")
            return None
            
        # Number of variables
        N = len(available_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Set angle for each feature
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add feature labels
        plt.xticks(angles[:-1], [f.replace('_', ' ').title() for f in available_features], size=12)
        
        # Draw one line per track and fill area
        for i, track in enumerate(top_tracks_data[track_name_col]):
            values = top_tracks_data[top_tracks_data[track_name_col] == track][available_features].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=track)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Audio Features Comparison for Top Tracks', size=15)
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_pca_analysis(self, data, features=None, color_by='track_genre', save_as=None):
        """
        Perform PCA analysis on audio features and visualize.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            features (list, optional): List of features to include. If None, uses default audio features.
            color_by (str): Column to use for coloring points.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if features is None:
            features = ['danceability', 'energy', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in features if col in data.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough audio features found in the data for PCA analysis")
            return None
            
        # Check if color column exists
        if color_by not in data.columns:
            logger.warning(f"Column {color_by} not found in data, using default coloring")
            color_by = None
            
        # Prepare data for PCA
        X = data[available_features].dropna()
        
        if len(X) == 0:
            logger.warning("No valid data for PCA analysis after removing NaN values")
            return None
            
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        
        # Create DataFrame with principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        
        # Add color column if available
        if color_by:
            pca_df[color_by] = data.loc[X.index, color_by].values
            
            # Limit to top categories if there are too many
            if pca_df[color_by].nunique() > 10:
                top_categories = data[color_by].value_counts().head(10).index
                pca_df.loc[~pca_df[color_by].isin(top_categories), color_by] = 'Other'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot PCA results
        if color_by:
            scatter = sns.scatterplot(x='PC1', y='PC2', hue=color_by, data=pca_df, ax=ax)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            scatter = sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax)
        
        # Add explained variance as labels
        explained_variance = pca.explained_variance_ratio_
        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
        
        ax.set_title('PCA of Audio Features')
        
        # Add feature vectors
        feature_vectors = pca.components_.T
        for i, feature in enumerate(available_features):
            plt.arrow(0, 0, feature_vectors[i, 0], feature_vectors[i, 1], 
                     head_width=0.05, head_length=0.05, fc='red', ec='red')
            plt.text(feature_vectors[i, 0] * 1.15, feature_vectors[i, 1] * 1.15, 
                    feature.replace('_', ' ').title(), color='red')
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_clustering_analysis(self, data, features=None, n_clusters=5, save_as=None):
        """
        Perform clustering analysis on audio features and visualize.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            features (list, optional): List of features to include. If None, uses default audio features.
            n_clusters (int): Number of clusters to create.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            tuple: (matplotlib.figure.Figure, pandas.DataFrame) The created figure and data with cluster labels.
        """
        if features is None:
            features = ['danceability', 'energy', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in features if col in data.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough audio features found in the data for clustering analysis")
            return None, None
            
        # Prepare data for clustering
        X = data[available_features].dropna()
        
        if len(X) == 0:
            logger.warning("No valid data for clustering analysis after removing NaN values")
            return None, None
            
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to original data
        clustered_data = data.copy()
        clustered_data.loc[X.index, 'cluster'] = cluster_labels
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        
        # Create DataFrame with principal components and cluster labels
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['cluster'] = cluster_labels
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot clusters
        scatter = sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='viridis', 
                                 data=pca_df, ax=ax)
        
        # Add cluster centers
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5, marker='X')
        
        # Add explained variance as labels
        explained_variance = pca.explained_variance_ratio_
        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
        
        ax.set_title(f'KMeans Clustering of Tracks (k={n_clusters})')
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig, clustered_data
    
    def plot_feature_by_time(self, data, feature='valence', time_unit='M', save_as=None):
        """
        Plot average audio feature values over time.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            feature (str): Audio feature to plot.
            time_unit (str): Time unit for resampling ('D' for daily, 'W' for weekly, 'M' for monthly).
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if feature not in data.columns:
            logger.warning(f"Feature {feature} not found in data")
            return None
            
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
        
        # Calculate average feature value by time period
        feature_by_time = data.set_index('end_time').resample(time_unit)[feature].mean()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot feature over time
        feature_by_time.plot(ax=ax)
        
        # Add trend line
        try:
            z = np.polyfit(range(len(feature_by_time)), feature_by_time.values, 1)
            p = np.poly1d(z)
            ax.plot(feature_by_time.index, p(range(len(feature_by_time))), "r--", alpha=0.8)
        except Exception as e:
            logger.warning(f"Could not add trend line: {e}")
        
        # Set labels and title
        time_unit_name = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[time_unit]
        ax.set_xlabel('Date')
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(f'{time_unit_name} Average {feature.replace("_", " ").title()} Over Time')
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig
    
    def plot_feature_comparison_by_genre(self, data, features=None, top_n_genres=5, save_as=None):
        """
        Compare audio features across different genres.
        
        Args:
            data (pandas.DataFrame): The data to plot.
            features (list, optional): List of features to include. If None, uses default audio features.
            top_n_genres (int): Number of top genres to include.
            save_as (str, optional): Filename to save the plot as.
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if 'track_genre' not in data.columns:
            logger.warning("Column 'track_genre' not found in data")
            return None
            
        if features is None:
            features = ['danceability', 'energy', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in features if col in data.columns]
        
        if not available_features:
            logger.warning("No audio features found in the data")
            return None
            
        # Get top genres
        top_genres = data['track_genre'].value_counts().head(top_n_genres).index
        
        # Filter data to include only top genres
        genre_data = data[data['track_genre'].isin(top_genres)]
        
        if len(genre_data) == 0:
            logger.warning("No data available for top genres")
            return None
            
        # Create figure with subplots
        n_features = len(available_features)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
        
        if n_features == 1:
            axes = [axes]
        
        # Plot each feature
        for i, feature in enumerate(available_features):
            sns.boxplot(x='track_genre', y=feature, data=genre_data, ax=axes[i])
            axes[i].set_title(f'{feature.replace("_", " ").title()} by Genre')
            axes[i].set_xlabel('')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        axes[-1].set_xlabel('Genre')
        
        plt.tight_layout()
        
        if save_as:
            self.save_figure(fig, save_as)
            
        return fig 