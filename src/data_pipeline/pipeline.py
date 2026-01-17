from dotenv import load_dotenv
from prefect import flow

from src.data_pipeline.flows.data_acquisition import acquire_data
from src.data_pipeline.flows.graph_populating import populate_graph
from src.data_pipeline.flows.llm_cypher_generation import generate_cypher_queries
from src.data_pipeline.flows.ocr_extraction import ocr_extraction


@flow(log_prints=True)
def data_pipeline_flow():
    load_dotenv()

    acquire_data()

    extracted_text = ocr_extraction()

    cypher_query = generate_cypher_queries(extracted_text)

    populate_graph(cypher_query)


if __name__ == "__main__":
    data_pipeline_flow()
