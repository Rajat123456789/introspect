#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Builder and Neo4j Representation
This module provides classes and utilities to build, manage, and visualize
knowledge graphs using Neo4j as the backend database.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Neo4jConnection:
    """
    A class to handle Neo4j database connection and query execution.
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection parameters.
        
        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687')
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default: 'neo4j')
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            logger.info(f"Connected to Neo4j database at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """
        Execute a Cypher query against the Neo4j database.
        
        Args:
            query: Cypher query string
            params: Dictionary of query parameters
            
        Returns:
            List of records as dictionaries
        """
        if not params:
            params = {}
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise


class KnowledgeGraphEntity:
    """Base class for all knowledge graph entities (nodes and relationships)."""
    
    def __init__(self, properties: Dict[str, Any]):
        """
        Initialize entity with properties.
        
        Args:
            properties: Dictionary of entity properties
        """
        self.properties = properties
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return self.properties


class Node(KnowledgeGraphEntity):
    """
    Represents a node (vertex) in the knowledge graph.
    """
    
    def __init__(self, labels: List[str], properties: Dict[str, Any], node_id: Optional[str] = None):
        """
        Initialize a node with labels and properties.
        
        Args:
            labels: List of node labels (types)
            properties: Dictionary of node properties
            node_id: Optional unique identifier for the node
        """
        super().__init__(properties)
        self.labels = labels
        if node_id:
            self.properties['id'] = node_id
    
    def to_cypher_create(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher CREATE statement for the node.
        
        Returns:
            Tuple of (query_string, parameters)
        """
        labels_str = ':'.join(self.labels)
        param_key = f"props_{hash(tuple(sorted(self.properties.items())))}"
        
        query = f"CREATE (n:{labels_str} ${param_key}) RETURN n"
        params = {param_key: self.properties}
        
        return query, params


class Relationship(KnowledgeGraphEntity):
    """
    Represents a relationship (edge) in the knowledge graph.
    """
    
    def __init__(self, 
                 rel_type: str, 
                 source_node: Union[Node, str], 
                 target_node: Union[Node, str],
                 properties: Dict[str, Any] = None):
        """
        Initialize a relationship between nodes.
        
        Args:
            rel_type: Type of the relationship
            source_node: Source node or its ID
            target_node: Target node or its ID
            properties: Dictionary of relationship properties
        """
        super().__init__(properties or {})
        self.rel_type = rel_type
        self.source_node = source_node
        self.target_node = target_node
    
    def to_cypher_create(self, source_id_field: str = 'id', target_id_field: str = 'id') -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher CREATE statement for the relationship.
        
        Args:
            source_id_field: Property name to match source node
            target_id_field: Property name to match target node
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Get source and target node IDs
        if isinstance(self.source_node, Node):
            source_id = self.source_node.properties.get(source_id_field)
        else:
            source_id = self.source_node
            
        if isinstance(self.target_node, Node):
            target_id = self.target_node.properties.get(target_id_field)
        else:
            target_id = self.target_node
        
        # Create relationship query
        param_key = f"rel_props_{hash(tuple(sorted(self.properties.items())))}"
        source_param = f"source_{hash(source_id)}"
        target_param = f"target_{hash(target_id)}"
        
        query = f"""
            MATCH (a) WHERE a.{source_id_field} = ${source_param}
            MATCH (b) WHERE b.{target_id_field} = ${target_param}
            CREATE (a)-[r:{self.rel_type} ${param_key}]->(b)
            RETURN r
        """
        
        params = {
            source_param: source_id,
            target_param: target_id,
            param_key: self.properties
        }
        
        return query, params


class KnowledgeGraph:
    """
    Manages the creation, querying, and visualization of a knowledge graph.
    """
    
    def __init__(self, connection: Neo4jConnection):
        """
        Initialize the knowledge graph manager.
        
        Args:
            connection: Neo4jConnection instance
        """
        self.connection = connection
        self.nodes: List[Node] = []
        self.relationships: List[Relationship] = []
        # For keeping track of nodes/relationships already added to Neo4j
        self.added_node_ids = set()
        self.added_relationship_ids = set()
    
    def add_node(self, node: Node) -> None:
        """
        Add a node to the knowledge graph.
        
        Args:
            node: Node instance to add
        """
        self.nodes.append(node)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the knowledge graph.
        
        Args:
            relationship: Relationship instance to add
        """
        self.relationships.append(relationship)
    
    def create_node_index(self, label: str, property_name: str) -> None:
        """
        Create an index on a node property for faster lookups.
        
        Args:
            label: Node label to index
            property_name: Property name to index
        """
        query = f"CREATE INDEX ON :{label}({property_name})"
        self.connection.execute_query(query)
        logger.info(f"Created index on :{label}({property_name})")
    
    def clear_database(self) -> None:
        """Delete all nodes and relationships in the database."""
        query = "MATCH (n) DETACH DELETE n"
        self.connection.execute_query(query)
        logger.info("Database cleared")
        self.added_node_ids.clear()
        self.added_relationship_ids.clear()
    
    def commit_to_neo4j(self, batch_size: int = 100) -> None:
        """
        Commit all pending nodes and relationships to Neo4j.
        
        Args:
            batch_size: Number of operations to commit in a single transaction
        """
        # Add nodes in batches
        for i in range(0, len(self.nodes), batch_size):
            batch = self.nodes[i:i+batch_size]
            node_ids = []
            
            # Create batch query
            queries = []
            params = {}
            
            for node in batch:
                # Skip if already added
                node_id = node.properties.get('id')
                if node_id and node_id in self.added_node_ids:
                    continue
                    
                query, query_params = node.to_cypher_create()
                queries.append(query)
                params.update(query_params)
                
                if node_id:
                    node_ids.append(node_id)
            
            if queries:
                query = " UNION ALL ".join(queries)
                self.connection.execute_query(query, params)
                # Mark nodes as added
                self.added_node_ids.update(node_ids)
                logger.info(f"Added {len(queries)} nodes to Neo4j")
        
        # Add relationships in batches
        for i in range(0, len(self.relationships), batch_size):
            batch = self.relationships[i:i+batch_size]
            
            # Create batch query
            for rel in batch:
                query, params = rel.to_cypher_create()
                # Create relationship in Neo4j
                try:
                    self.connection.execute_query(query, params)
                except Exception as e:
                    logger.warning(f"Failed to create relationship: {str(e)}")
            
            logger.info(f"Added {len(batch)} relationships to Neo4j")
    
    def query(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        """
        Execute a Cypher query against the knowledge graph.
        
        Args:
            cypher_query: Cypher query string
            params: Dictionary of query parameters
            
        Returns:
            List of records as dictionaries
        """
        return self.connection.execute_query(cypher_query, params)
    
    def to_networkx(self, limit: int = 1000) -> nx.DiGraph:
        """
        Convert the knowledge graph to a NetworkX graph for visualization.
        
        Args:
            limit: Maximum number of nodes to retrieve
            
        Returns:
            NetworkX DiGraph representation of the knowledge graph
        """
        # Query nodes
        node_query = f"MATCH (n) RETURN n LIMIT {limit}"
        nodes = self.connection.execute_query(node_query)
        
        # Query relationships
        rel_query = f"""
            MATCH (a)-[r]->(b) 
            WHERE id(a) in $node_ids AND id(b) in $node_ids
            RETURN a, r, b
        """
        
        node_ids = [node['n'].id for node in nodes]
        relationships = self.connection.execute_query(rel_query, {"node_ids": node_ids})
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_record in nodes:
            node = node_record['n']
            node_id = node.id
            node_labels = list(node.labels)
            node_properties = dict(node)
            
            G.add_node(node_id, labels=node_labels, **node_properties)
        
        # Add edges
        for rel_record in relationships:
            source_id = rel_record['a'].id
            target_id = rel_record['b'].id
            rel = rel_record['r']
            rel_type = type(rel).__name__
            rel_properties = dict(rel)
            
            G.add_edge(source_id, target_id, type=rel_type, **rel_properties)
        
        return G
    
    def visualize(self, graph: Optional[nx.DiGraph] = None, 
                 figsize: Tuple[int, int] = (12, 10),
                 title: str = "Knowledge Graph Visualization") -> None:
        """
        Visualize the knowledge graph using NetworkX and Matplotlib.
        
        Args:
            graph: NetworkX graph to visualize (if None, calls to_networkx)
            figsize: Figure size as (width, height)
            title: Plot title
        """
        if graph is None:
            graph = self.to_networkx()
            
        plt.figure(figsize=figsize)
        plt.title(title)
        
        # Positions for nodes
        pos = nx.spring_layout(graph, k=0.5, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, width=1, alpha=0.5, arrows=True)
        
        # Draw labels
        labels = {node: data.get('name', str(node)[:10]) 
                 for node, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export the knowledge graph to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        # Query all nodes
        nodes = self.connection.execute_query("MATCH (n) RETURN n")
        
        # Query all relationships
        rels = self.connection.execute_query("MATCH ()-[r]->() RETURN r")
        
        # Prepare data for export
        export_data = {
            "nodes": [dict(node['n']) for node in nodes],
            "relationships": [dict(rel['r']) for rel in rels]
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Knowledge graph exported to {filepath}")


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


# Example usage
if __name__ == "__main__":
    # Connect to Neo4j
    connection = Neo4jConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"  # Replace with your password
    )
    
    # Create a Spotify knowledge graph
    spotify_kg = SpotifyKnowledgeGraph(connection)
    
    # Clear existing data
    spotify_kg.clear_database()
    
    # Add sample data
    artist1 = spotify_kg.add_artist(
        artist_id="1vCWHaC5f2uS3yhpwWbIA6",
        name="Avicii",
        popularity=85,
        genres=["edm", "dance pop", "progressive house"]
    )
    
    artist2 = spotify_kg.add_artist(
        artist_id="4tZwfgrHOc3mvqYlEYSvVi",
        name="Daft Punk",
        popularity=88,
        genres=["electronic", "french house", "disco"]
    )
    
    album1 = spotify_kg.add_album(
        album_id="7dqftJ3kas6D0VdoYJ1HeP",
        name="True",
        release_date="2013-09-13",
        album_type="album",
        total_tracks=10
    )
    
    track1 = spotify_kg.add_track(
        track_id="4h8VwCb1MTGoLKueQ1WgbD",
        name="Wake Me Up",
        duration_ms=247426,
        popularity=90
    )
    
    track2 = spotify_kg.add_track(
        track_id="2Oehrcv4Kov0SuIgWyQY9e",
        name="Hey Brother",
        duration_ms=255093,
        popularity=85
    )
    
    # Add relationships
    spotify_kg.add_artist_album_relationship(
        artist_id="1vCWHaC5f2uS3yhpwWbIA6",
        album_id="7dqftJ3kas6D0VdoYJ1HeP"
    )
    
    spotify_kg.add_artist_track_relationship(
        artist_id="1vCWHaC5f2uS3yhpwWbIA6",
        track_id="4h8VwCb1MTGoLKueQ1WgbD"
    )
    
    spotify_kg.add_artist_track_relationship(
        artist_id="1vCWHaC5f2uS3yhpwWbIA6",
        track_id="2Oehrcv4Kov0SuIgWyQY9e"
    )
    
    spotify_kg.add_album_track_relationship(
        album_id="7dqftJ3kas6D0VdoYJ1HeP",
        track_id="4h8VwCb1MTGoLKueQ1WgbD",
        track_number=1
    )
    
    spotify_kg.add_album_track_relationship(
        album_id="7dqftJ3kas6D0VdoYJ1HeP",
        track_id="2Oehrcv4Kov0SuIgWyQY9e",
        track_number=2
    )
    
    # Commit data to Neo4j
    spotify_kg.commit_to_neo4j()
    
    # Run sample query
    artist_recs = spotify_kg.get_artist_recommendations("1vCWHaC5f2uS3yhpwWbIA6")
    print("Artist recommendations:")
    for rec in artist_recs:
        print(f"{rec['name']} (Score: {rec['score']})")
    
    # Close connection
    connection.close()
