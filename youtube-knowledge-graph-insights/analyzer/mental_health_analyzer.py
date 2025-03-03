"""
Core analyzer class that handles Neo4j connection and basic operations
"""
import pandas as pd
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class MentalHealthAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

    def execute_query(self, query):
        """Execute a Cypher query and return results as DataFrame"""
        with self.driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])
            
    def save_analysis_results(self, result_data, analysis_name, formats=None):
        """Save analysis results to files in various formats"""
        # Import the save functionality from utils
        from analyzer.utils import save_dataframe_to_files
        return save_dataframe_to_files(result_data, analysis_name, formats) 