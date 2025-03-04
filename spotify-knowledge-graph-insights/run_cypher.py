#!/usr/bin/env python3
"""
Run Cypher queries against the Spotify Neo4j knowledge graph from the command line.
Example:
    python run_cypher.py "MATCH (n) RETURN labels(n) as label, count(n) as count GROUP BY label"
"""

import sys
import pandas as pd
from neo4j import GraphDatabase
import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_query(uri, username, password, query, output_format="table", output_file=None):
    """Run a Cypher query and return the results in the specified format."""
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Run the query
        with driver.session() as session:
            result = list(session.run(query))
        
        # Close the connection
        driver.close()
        
        # If no results, return early
        if not result:
            logger.info("Query returned no results.")
            return
        
        # Convert to DataFrame
        keys = result[0].keys()
        data = {key: [record[key] for record in result] for key in keys}
        df = pd.DataFrame(data)
        
        # Output based on format
        if output_format == "json":
            output = df.to_json(orient="records", indent=2)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(output)
            else:
                print(output)
        elif output_format == "csv":
            if output_file:
                df.to_csv(output_file, index=False)
            else:
                print(df.to_csv(index=False))
        else:  # default to table
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(df.to_string())
            else:
                print(df.to_string())
        
        logger.info(f"Query returned {len(df)} rows.")
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Cypher queries against the Spotify knowledge graph.")
    parser.add_argument("query", help="The Cypher query to execute")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI (default: bolt://localhost:7687)")
    parser.add_argument("--username", default="neo4j", help="Neo4j username (default: neo4j)")
    parser.add_argument("--password", default="password", help="Neo4j password (default: password)")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table", help="Output format (default: table)")
    parser.add_argument("--output", help="Output file (if not specified, prints to stdout)")
    
    args = parser.parse_args()
    
    # Run the query
    run_query(args.uri, args.username, args.password, args.query, args.format, args.output)

if __name__ == "__main__":
    main() 