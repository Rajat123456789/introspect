#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge graph core module.
This module provides the base KnowledgeGraph class for managing graph operations.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set

from .connection import Neo4jConnection
from .entities import Node, Relationship

# Get logger
logger = logging.getLogger(__name__)


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