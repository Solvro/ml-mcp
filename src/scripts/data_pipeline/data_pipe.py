import logging
import os
from typing import List

from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import get_config

from .llm_pipe import LLMPipe
from .pdf_loader import PDFLoader


class DataPipe:
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        api_key: str = None,
        nodes: List[str] = None,
        relations: List[str] = None,
        max_chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        config = get_config()

        # Apply config defaults
        if max_chunk_size is None:
            max_chunk_size = config.data_pipeline.max_chunk_size
        if chunk_overlap is None:
            chunk_overlap = config.data_pipeline.chunk_overlap

        self.docs_data = []

        if not url:
            raise ValueError("Neo4j URL is required")
        if not username:
            raise ValueError("Neo4j username is required")
        if not password:
            raise ValueError("Neo4j password is required")

        logging.info(f"Connecting to Neo4j at: {url}")
        logging.info(f"Username: {username}")

        try:
            self.llm_pipe = LLMPipe(api_key=api_key)  # , nodes=nodes, relations=relations)
            self.graph_db = Neo4jGraph(url=url, username=username, password=password)

            self.graph_db.query("RETURN 1 as test")
            logging.info("Successfully connected to Neo4j database")

        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {str(e)}")
            logging.error(f"URL: {url}")
            logging.error(f"Username: {username}")
            raise ConnectionError(f"Could not connect to Neo4j database: {str(e)}")

        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _load_data(self, file_path: str) -> None:
        """Load data from a file and append it to the docs_data list."""
        try:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return

            if os.path.getsize(file_path) == 0:
                logging.error(f"Empty file: {file_path}")
                return

            logging.info(f"Loading file: {file_path}")
            loader = PDFLoader(file_path)
            content = loader.load_document()

            if content.startswith("ERROR:"):
                logging.error(content)
                return

            if len(content) > self.max_chunk_size:
                logging.info(f"Document is large ({len(content)} chars), splitting into chunks")
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    filename = os.path.basename(file_path)
                    part_info = f"[Part {i + 1} of {len(chunks)} from {filename}]"
                    self.docs_data.append(f"{part_info} {chunk}")
                    logging.info(f"Added chunk {i + 1} of {len(chunks)} from {file_path}")
            else:
                self.docs_data.append(content)

            logging.info(f"Successfully loaded: {file_path}")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")

    def load_data_from_directory(self, directory_path: str) -> None:
        """Load data from all files in a given directory."""
        if not os.path.exists(directory_path):
            logging.error(f"Directory not found: {directory_path}")
            return

        logging.info(f"Loading files from directory: {directory_path}")
        file_count = 0

        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf") or filename.endswith(".txt") or filename.endswith(".docx"):
                self._load_data(os.path.join(directory_path, filename))
                file_count += 1

        logging.info(f"Loaded {len(self.docs_data)} documents/chunks from {file_count} files")

    def clear_database(self) -> None:
        """Clear the Neo4j database."""
        try:
            self.execute_cypher("MATCH (n) DETACH DELETE n")
            logging.info("Database cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing database: {str(e)}")

    def execute_cypher(self, query: str) -> None:
        """Execute a Cypher query on the Neo4j database."""
        if not query or not query.strip():
            logging.error("Empty Cypher query")
            return

        try:
            self.graph_db.query(query)
            logging.info("Cypher query executed successfully")
        except Exception as e:
            logging.error(f"Error executing Cypher query: {str(e)}")
            logging.error(f"Query: {query}")
            raise

    def process_documents(self):
        """Process all loaded documents through the LLM pipe."""
        all_results = []

        for i, doc in enumerate(self.docs_data):
            try:
                logging.info(f"Processing document chunk {i + 1}/{len(self.docs_data)}")
                char_count = len(doc)
                logging.info(f"Document chunk size: {char_count} characters")

                if char_count > self.max_chunk_size * 2:
                    logging.warning(
                        f"Document chunk may be too large for model ({char_count} chars)"
                    )

                cypher_code = "".join([code for code in self.llm_pipe.run(doc)])

                logging.info(f"Generated Cypher code of length {len(cypher_code)}")
                logging.info(
                    "Executing Cypher for document: " + doc[:50] + "..." if len(doc) > 50 else doc
                )

                try:
                    self.execute_cypher(cypher_code)
                    all_results.append(cypher_code)
                except Exception as e:
                    logging.error(f"Failed to execute Cypher: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing document chunk {i + 1}: {str(e)}")

        return all_results
