#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge graph visualization module.
This module provides functions to visualize knowledge graphs using NetworkX and Matplotlib.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import networkx as nx
import matplotlib.pyplot as plt

from .connection import Neo4jConnection

# Get logger
logger = logging.getLogger(__name__)


def graph_to_networkx(connection: Neo4jConnection, limit: int = 1000) -> nx.DiGraph:
    """
    Convert a Neo4j graph to a NetworkX graph for visualization.
    
    Args:
        connection: Neo4jConnection instance
        limit: Maximum number of nodes to retrieve
        
    Returns:
        NetworkX DiGraph representation of the knowledge graph
    """
    # Query nodes
    node_query = f"MATCH (n) RETURN n LIMIT {limit}"
    nodes = connection.execute_query(node_query)
    
    # Query relationships
    rel_query = f"""
        MATCH (a)-[r]->(b) 
        WHERE id(a) in $node_ids AND id(b) in $node_ids
        RETURN a, r, b
    """
    
    node_ids = [node['n'].id for node in nodes]
    relationships = connection.execute_query(rel_query, {"node_ids": node_ids})
    
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


def visualize_graph(graph: nx.DiGraph, 
                   figsize: Tuple[int, int] = (12, 10),
                   title: str = "Knowledge Graph Visualization") -> None:
    """
    Visualize a NetworkX graph using Matplotlib.
    
    Args:
        graph: NetworkX graph to visualize
        figsize: Figure size as (width, height)
        title: Plot title
    """
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


def visualize_knowledge_graph(connection: Neo4jConnection,
                             limit: int = 1000,
                             figsize: Tuple[int, int] = (12, 10),
                             title: str = "Knowledge Graph Visualization") -> None:
    """
    Convenience function to visualize a Neo4j knowledge graph directly.
    
    Args:
        connection: Neo4jConnection instance
        limit: Maximum number of nodes to retrieve
        figsize: Figure size as (width, height)
        title: Plot title
    """
    graph = graph_to_networkx(connection, limit)
    visualize_graph(graph, figsize, title) 