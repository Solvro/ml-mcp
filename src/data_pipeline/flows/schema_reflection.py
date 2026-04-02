import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai.chat_models.base import BaseChatOpenAI
from prefect import get_run_logger, task
from pydantic import SecretStr

from src.config.config import get_config


@task
def reflect_on_schema() -> str:
    """Query Neo4j for the current graph schema and produce an LLM summary.

    Returns:
        A concise schema summary string to inject as context into the next
        extraction pass, or an empty string if the graph is still empty.
    """
    logger = get_run_logger()

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        logger.warning("Neo4j credentials not set — skipping schema reflection")
        return ""

    graph = Neo4jGraph(url=uri, username=username, password=password)
    schema = graph.get_schema

    if not schema or not schema.strip():
        logger.info("Graph is empty — no schema to reflect on")
        return ""

    config = get_config()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""

    model = BaseChatOpenAI(
        model=config.llm.fast_model.name,
        api_key=SecretStr(api_key),
        temperature=config.llm.fast_model.temperature,
    )

    template = PromptTemplate(
        input_variables=["schema"],
        template=config.prompts.schema_reflection,
    )

    chain = template | model | StrOutputParser()

    try:
        summary = chain.invoke({"schema": schema})
        logger.info("Schema reflection completed (%d chars)", len(summary))
        logger.debug("Schema summary: %s", summary[:500])
        return summary
    except Exception as exc:
        logger.error("Schema reflection LLM call failed: %s", exc)
        return schema
