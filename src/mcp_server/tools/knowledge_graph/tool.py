from typing import Any, Dict, List

from langchain_neo4j import Neo4jGraph


class Tool:
    """Tool interface for querying Neo4j knowledge graph."""
    
    def __init__(
        self,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        max_results: int = 5,
    ):
        """
        Initialize the knowledge graph tool.

        Args:
            neo4j_url: Neo4j database connection URL
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            max_results: Maximum number of results to return (default: 5)
        """
        self.database = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database="neo4j",
        )
        self.max_results = max_results

    def invoke(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the Neo4j database.

        Args:
            cypher_query: Cypher query string to execute

        Returns:
            List of result dictionaries from the query
        """
        try:
            # Add LIMIT if not present
            if "LIMIT" not in cypher_query.upper():
                cypher_query = f"{cypher_query.rstrip(';')} LIMIT {self.max_results}"

            response = self.database.query(cypher_query)
            return response if isinstance(response, list) else [response]

        except Exception as e:
            error_msg = str(e)
            print(f"[Query Error] {error_msg}")
            return []
        # return [{'answer': 'Nagroda dziekana to nagroda dla najlepszeych studentów. Top 10% studnetów z danego kiernuku na danym roku studiów otrzymują nagrodę dziekana. W zależności od rocznika i kierunku studiów nagroda może być różna. W tym roku nagroda wynosi 950 zł.'}]

    async def ainvoke(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Async version of invoke for better performance in concurrent scenarios.

        Args:
            cypher_query: Cypher query string to execute

        Returns:
            List of result dictionaries from the query
        """
        # Run sync query in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, cypher_query)
