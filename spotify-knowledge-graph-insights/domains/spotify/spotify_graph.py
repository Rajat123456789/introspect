#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify knowledge graph module.
This module provides a specialized knowledge graph implementation for Spotify data.
"""

import logging
from typing import Dict, List, Any, Optional

from knowledge_graph.connection import Neo4jConnection
from knowledge_graph.entities import Node, Relationship
from knowledge_graph.graph import KnowledgeGraph

# Get logger
logger = logging.getLogger(__name__)


class SpotifyKnowledgeGraph(KnowledgeGraph):
    """
    Specialized knowledge graph for Spotify data.
    """
    
    def __init__(self, connection: Neo4jConnection):
        """Initialize the Spotify knowledge graph."""
        super().__init__(connection)
        # Create necessary indexes
        self.create_indexes()
    
    def create_indexes(self):
        """Create indexes for Spotify entities."""
        try:
            self.create_node_index("Artist", "id")
            self.create_node_index("Track", "id")
            self.create_node_index("Album", "id")
            self.create_node_index("Genre", "name")
            self.create_node_index("User", "id")
            self.create_node_index("Playlist", "id")
        except Exception as e:
            logger.warning(f"Error creating indexes: {str(e)}")
    
    def add_artist(self, artist_id: str, name: str, popularity: int = None, 
                  followers: int = None, genres: List[str] = None, **kwargs) -> Node:
        """
        Add an artist node to the knowledge graph.
        
        Args:
            artist_id: Spotify artist ID
            name: Artist name
            popularity: Artist popularity score
            followers: Number of followers
            genres: List of genres
            **kwargs: Additional artist properties
            
        Returns:
            The created Artist node
        """
        properties = {
            "id": artist_id,
            "name": name,
            **kwargs
        }
        
        if popularity is not None:
            properties["popularity"] = popularity
        
        if followers is not None:
            properties["followers"] = followers
        
        artist_node = Node(labels=["Artist"], properties=properties)
        self.add_node(artist_node)
        
        # Add genres if provided
        if genres:
            for genre in genres:
                genre_node = self.add_genre(genre)
                rel = Relationship(
                    rel_type="BELONGS_TO_GENRE",
                    source_node=artist_node,
                    target_node=genre_node
                )
                self.add_relationship(rel)
        
        return artist_node
    
    def add_track(self, track_id: str, name: str, duration_ms: int = None,
                 popularity: int = None, explicit: bool = None, **kwargs) -> Node:
        """
        Add a track node to the knowledge graph.
        
        Args:
            track_id: Spotify track ID
            name: Track name
            duration_ms: Track duration in milliseconds
            popularity: Track popularity score
            explicit: Whether the track has explicit content
            **kwargs: Additional track properties
            
        Returns:
            The created Track node
        """
        properties = {
            "id": track_id,
            "name": name,
            **kwargs
        }
        
        if duration_ms is not None:
            properties["duration_ms"] = duration_ms
        
        if popularity is not None:
            properties["popularity"] = popularity
            
        if explicit is not None:
            properties["explicit"] = explicit
        
        track_node = Node(labels=["Track"], properties=properties)
        self.add_node(track_node)
        return track_node
    
    def add_album(self, album_id: str, name: str, release_date: str = None,
                 album_type: str = None, total_tracks: int = None, **kwargs) -> Node:
        """
        Add an album node to the knowledge graph.
        
        Args:
            album_id: Spotify album ID
            name: Album name
            release_date: Album release date
            album_type: Album type (album, single, etc.)
            total_tracks: Number of tracks in the album
            **kwargs: Additional album properties
            
        Returns:
            The created Album node
        """
        properties = {
            "id": album_id,
            "name": name,
            **kwargs
        }
        
        if release_date is not None:
            properties["release_date"] = release_date
        
        if album_type is not None:
            properties["album_type"] = album_type
            
        if total_tracks is not None:
            properties["total_tracks"] = total_tracks
        
        album_node = Node(labels=["Album"], properties=properties)
        self.add_node(album_node)
        return album_node
    
    def add_genre(self, name: str) -> Node:
        """
        Add a genre node to the knowledge graph.
        
        Args:
            name: Genre name
            
        Returns:
            The created Genre node
        """
        # Check if genre already exists in our local nodes
        for node in self.nodes:
            if "Genre" in node.labels and node.properties.get("name") == name:
                return node
        
        # Create new genre node
        genre_node = Node(labels=["Genre"], properties={"name": name})
        self.add_node(genre_node)
        return genre_node
    
    def add_artist_track_relationship(self, artist_id: str, track_id: str) -> None:
        """
        Create a relationship between an artist and a track.
        
        Args:
            artist_id: Spotify artist ID
            track_id: Spotify track ID
        """
        rel = Relationship(
            rel_type="PERFORMED",
            source_node=artist_id,
            target_node=track_id
        )
        self.add_relationship(rel)
    
    def add_album_track_relationship(self, album_id: str, track_id: str, 
                                    track_number: int = None) -> None:
        """
        Create a relationship between an album and a track.
        
        Args:
            album_id: Spotify album ID
            track_id: Spotify track ID
            track_number: Position of the track in the album
        """
        properties = {}
        if track_number is not None:
            properties["track_number"] = track_number
            
        rel = Relationship(
            rel_type="CONTAINS",
            source_node=album_id,
            target_node=track_id,
            properties=properties
        )
        self.add_relationship(rel)
    
    def add_artist_album_relationship(self, artist_id: str, album_id: str) -> None:
        """
        Create a relationship between an artist and an album.
        
        Args:
            artist_id: Spotify artist ID
            album_id: Spotify album ID
        """
        rel = Relationship(
            rel_type="RELEASED",
            source_node=artist_id,
            target_node=album_id
        )
        self.add_relationship(rel)
    
    def get_artist_recommendations(self, artist_id: str, limit: int = 10) -> List[Dict]:
        """
        Get artist recommendations based on common genres and collaborations.
        
        Args:
            artist_id: Spotify artist ID
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended artists with similarity scores
        """
        query = """
            MATCH (a:Artist {id: $artist_id})-[:BELONGS_TO_GENRE]->(g:Genre)<-[:BELONGS_TO_GENRE]-(rec:Artist)
            WHERE a <> rec
            WITH rec, count(g) AS genreOverlap
            
            OPTIONAL MATCH (a:Artist {id: $artist_id})-[:PERFORMED]->(:Track)<-[:PERFORMED]-(rec)
            WITH rec, genreOverlap, count(rec) AS collaborations
            
            RETURN rec.id AS id, rec.name AS name, rec.popularity AS popularity,
                   genreOverlap * 2 + collaborations * 3 AS score
            ORDER BY score DESC
            LIMIT $limit
        """
        
        params = {"artist_id": artist_id, "limit": limit}
        return self.query(query, params)
    
    def get_track_similarities(self, track_id: str, limit: int = 10) -> List[Dict]:
        """
        Get similar tracks based on common artists and genres.
        
        Args:
            track_id: Spotify track ID
            limit: Maximum number of similar tracks
            
        Returns:
            List of similar tracks with similarity scores
        """
        query = """
            MATCH (t:Track {id: $track_id})<-[:PERFORMED]-(a:Artist)
            MATCH (a)-[:PERFORMED]->(similar:Track)
            WHERE t <> similar
            WITH similar, count(a) AS artistOverlap
            
            MATCH (t:Track {id: $track_id})<-[:PERFORMED]-(a:Artist)-[:BELONGS_TO_GENRE]->(g:Genre)
            MATCH (g)<-[:BELONGS_TO_GENRE]-(other:Artist)-[:PERFORMED]->(similar:Track)
            WHERE t <> similar AND NOT (a)-[:PERFORMED]->(similar)
            
            WITH similar, artistOverlap, count(DISTINCT g) AS genreOverlap
            
            RETURN similar.id AS id, similar.name AS name, similar.popularity AS popularity,
                   artistOverlap * 3 + genreOverlap AS score
            ORDER BY score DESC
            LIMIT $limit
        """
        
        params = {"track_id": track_id, "limit": limit}
        return self.query(query, params) 