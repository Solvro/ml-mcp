import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from prefect import get_run_logger, task
from pydantic import SecretStr

from src.config.config import get_config


class PipeState(MessagesState):
    context: str
    schema_context: str
    generated_cypher: List[str]


class LLMPipe:
    def __init__(self):
        config = get_config()
        self.model = BaseChatOpenAI(
            model=config.llm.accurate_model.name,
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
            temperature=config.llm.accurate_model.temperature,
        )
        self.generate_template = PromptTemplate(
            input_variables=["context", "schema_context"],
            template=config.prompts.cypher_insert,
        )
        self._build_pipe_graph()

    def _build_pipe_graph(self) -> None:
        builder = StateGraph(PipeState)
        builder.add_node("generate", self.generate_cypher)
        builder.add_edge(START, "generate")
        builder.add_edge("generate", END)
        self.graph = builder.compile()

    def generate_cypher(self, state: PipeState) -> dict:
        logger = get_run_logger()

        chain = self.generate_template | self.model | StrOutputParser()

        payload = {
            "context": state["context"],
            "schema_context": state.get("schema_context") or "(empty — first pass)",
        }

        logger.debug(
            "Invoking LLM generate_cypher with context length %d",
            len(str(payload["context"])),
        )

        try:
            cypher_code = chain.invoke(payload)
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            return {"generated_cypher": []}

        logger.debug("Raw LLM output: %r", cypher_code)

        parts = [part.strip() for part in (cypher_code or "").split("|") if part and part.strip()]

        if not parts:
            logger.warning(
                "LLM returned no usable Cypher parts (raw output length=%d)",
                len(str(cypher_code or "")),
            )

        return {"generated_cypher": parts}

    def run(self, context: str, schema_context: str = "") -> List[str]:
        result = self.graph.invoke(
            {"context": context, "schema_context": schema_context, "generated_cypher": []},
            config={"configurable": {"thread_id": 1}},
        )
        return result["generated_cypher"]


@task
def generate_cypher_queries(extracted_text: str, schema_context: str = "") -> str:
    """Generate cypher statements from text using LLMPipe.

    Args:
        extracted_text: Raw text content to extract knowledge from.
        schema_context: Current graph schema summary from the reflection step.

    Returns:
        A single string with Cypher statements separated by pipe ``|``.
    """
    load_dotenv()
    logger = get_run_logger()
    llm = LLMPipe()
    parts = llm.run(extracted_text, schema_context)

    try:
        logger.info("LLM returned %d parts", len(parts))
        for i, p in enumerate(parts[:10]):
            logger.info("LLM part %d: %s", i, (p[:400] + "...") if len(p) > 400 else p)
    except Exception:
        logger.debug("Failed to log LLM parts")

    if not parts:
        logger.warning("LLM produced no cypher parts; returning empty string")
        return ""

    return "|".join(parts)
