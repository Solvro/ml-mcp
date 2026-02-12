import os

from langchain_neo4j import Neo4jGraph
from prefect import get_run_logger, task


class GraphPopulator:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not username or not password:
            raise ValueError("NEO4J connection settings are required")

        logger = get_run_logger()
        logger.info(f"Connecting to Neo4j at {uri} as {username}")
        self.graph_db = Neo4jGraph(url=uri, username=username, password=password)

    def execute_cypher(self, query: str):
        logger = get_run_logger()
        if not query or not query.strip():
            logger.error("Empty Cypher query")
            return
        try:
            logger.info("Executing Cypher query: %s", query)
            self.graph_db.query(query)
            logger.info("Cypher executed successfully")
        except Exception as e:
            logger.error("Failed to execute cypher: %s", e)
            raise


@task
def populate_graph(cypher_query: str):
    """Execute a cypher query against the configured Neo4j instance."""
    logger = get_run_logger()
    logger.info("populate_graph task received query of length %d", len(cypher_query or ""))
    pop = GraphPopulator()
    pop.execute_cypher(cypher_query)
