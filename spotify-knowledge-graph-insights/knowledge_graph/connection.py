#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neo4j connection management module.
"""

import logging
from typing import Dict, List, Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Get logger
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