import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from .data_pipe import DataPipe


def load_config_file(file_path: str) -> dict:
    """Load and validate configuration from JSON file."""
    try:
        with open(file_path) as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in configuration file: {file_path}")
        raise


def process_chunk(chunk: str, pipe: DataPipe) -> str:
    """Process a single document chunk."""
    try:
        pipe_response = "".join(
            pipe_element for pipe_element in pipe.llm_pipe.run(chunk.strip("|"))
        )
        pipe.execute_cypher(pipe_response)
        return pipe_response
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return ""


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python main.py <input_dir> <graph_config.json> <num_threads> [--clear-db]")
        sys.exit(1)

    input_dir = sys.argv[1]
    config_path = sys.argv[2]
    try:
        num_threads = int(sys.argv[3])
        if num_threads < 1:
            raise ValueError("Number of threads must be positive")
    except ValueError as e:
        print(f"Invalid number of threads: {e}")
        sys.exit(1)

    clear_db = "--clear-db" in sys.argv

    load_dotenv()

    try:
        config = load_config_file(config_path)
        nodes = config.get("nodes", [])
        relations = config.get("relationships", [])
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        return

    try:
        pipe = DataPipe(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            nodes=nodes,
            relations=relations,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
    except Exception as e:
        logging.error(f"Failed to initialize DataPipe: {str(e)}")
        return

    if clear_db:
        try:
            pipe.clear_database()
            logging.info("Database cleared successfully")
        except Exception as e:
            logging.error(f"Failed to clear database: {str(e)}")
            return

    try:
        pipe.load_data_from_directory(input_dir)
        if not pipe.docs_data:
            logging.error("No documents were loaded from the input directory")
            return

        logging.info(f"Processing {len(pipe.docs_data)} chunks using {num_threads} threads")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for chunk in pipe.docs_data:
                futures.append(executor.submit(process_chunk, chunk, pipe))

            successful_queries = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful_queries += 1
                except Exception as e:
                    logging.error(f"Error in thread: {str(e)}")

        logging.info(
            f"Successfully processed {successful_queries} out of {len(pipe.docs_data)} chunks"
        )

    except Exception as e:
        logging.error(f"Error during document processing: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
