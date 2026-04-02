from dotenv import load_dotenv
from prefect import flow, get_run_logger

from src.data_pipeline.flows.data_acquisition import acquire_data
from src.data_pipeline.flows.graph_populating import populate_graph
from src.data_pipeline.flows.llm_cypher_generation import generate_cypher_queries
from src.data_pipeline.flows.schema_reflection import reflect_on_schema


@flow(log_prints=True)
def data_pipeline_flow():
    """Agentic graph extraction loop.

    For each page of source content:
      1. Extract text via OCR/text loader.
      2. Generate Cypher MERGE statements with the LLM, injecting the current
         graph schema as context so the model stays consistent with what is
         already stored.
      3. Populate the Neo4j graph with the generated statements.
      4. Reflect on the updated schema to produce a concise summary for the
         next iteration.

    The loop runs until all pages are processed, then the graph is ready for
    querying by the RAG pipeline.
    """
    load_dotenv()
    logger = get_run_logger()

    pages = acquire_data()

    # Normalise: acquire_data may return a single string or a list of strings
    if isinstance(pages, str):
        pages = [pages]

    schema_context: str = ""

    for i, page_content in enumerate(pages):
        logger.info("Processing page %d / %d", i + 1, len(pages))

        if not page_content or not page_content.strip():
            logger.warning("Page %d is empty — skipping", i + 1)
            continue

        cypher_query = generate_cypher_queries(page_content, schema_context)

        if cypher_query:
            populate_graph(cypher_query)
        else:
            logger.warning("No Cypher generated for page %d — skipping populate", i + 1)

        # Reflect on the graph after each page to update schema context
        schema_context = reflect_on_schema()

    logger.info("Pipeline complete. Graph is ready for querying.")


if __name__ == "__main__":
    data_pipeline_flow()
