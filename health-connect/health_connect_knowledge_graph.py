import os
import glob
import pandas as pd
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthConnectKG:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def create_indexes(self):
        with self.driver.session(database=self.database) as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Metric) ON (m.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:DataPoint) ON (d.timestamp)")
    
    def build_graph(self, data_dir):
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        for file in csv_files:
            metric_name = os.path.splitext(os.path.basename(file))[0]
            df = pd.read_csv(file)
            logger.info(f"Processing {metric_name} from file: {file}")
            # Create a Metric node for this metric
            with self.driver.session(database=self.database) as session:
                session.run(
                    "MERGE (m:Metric {name: $metric_name})",
                    metric_name=metric_name
                )
            # Iterate over rows to add DataPoint nodes and relationships
            for _, row in df.iterrows():
                timestamp = row.get("timestamp")
                value = row.get("value")
                if pd.isna(timestamp) or pd.isna(value):
                    continue
                with self.driver.session(database=self.database) as session:
                    session.run(
                        """
                        MATCH (m:Metric {name: $metric_name})
                        CREATE (d:DataPoint {timestamp: $timestamp, value: $value})
                        CREATE (m)-[:HAS_DATAPOINT]->(d)
                        """,
                        metric_name=metric_name,
                        timestamp=timestamp,
                        value=float(value)
                    )

def main():
    # Update these with your Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    # Change 'your_username' to the appropriate username for your data folder
    data_dir = os.path.join("Data", "your_username", "Cleaned")
    kg = HealthConnectKG(uri, user, password)
    kg.create_indexes()
    kg.build_graph(data_dir)
    kg.close()
    logger.info("Knowledge graph created successfully.")

if __name__ == "__main__":
    main()