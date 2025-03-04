#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge graph entity classes.
This module provides classes for nodes and relationships in a knowledge graph.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Get logger
logger = logging.getLogger(__name__)


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