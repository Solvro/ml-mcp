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
from src.data_pipeline.canonical_nodes import rewrite_merge_to_canonical_key
from src.data_pipeline.label_vocabulary import LabelVocabulary, render_allowed_labels
from src.text_normalization import fold_diacritics, normalize_cypher_string_literals


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
            input_variables=[
                "context",
                "schema_context",
                "node_labels",
                "relationship_types",
            ],
            template=config.prompts.cypher_insert,
        )
        self.node_labels = render_allowed_labels(config.graph_schema)
        self.relationship_types = ", ".join(config.graph_schema.relationship_types)
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
            "node_labels": self.node_labels,
            "relationship_types": self.relationship_types,
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


def _canonicalize_labels(parts: List[str], logger) -> List[str]:
    """Force every generated node label into the configured vocabulary.

    The prompt already lists the allowed labels, but a label it invents anyway is invisible in
    the output and splits an entity across two nodes, so the set is enforced here as well.

    Args:
        parts: Generated Cypher statements
        logger: Prefect run logger used to report the drift that was corrected

    Returns:
        The statements with every node label resolved to a configured label
    """
    vocabulary = LabelVocabulary(get_config().graph_schema)
    canonical_parts: List[str] = []
    all_rewrites: dict[str, str] = {}

    for part in parts:
        rewritten, rewrites = vocabulary.canonicalize_statement(part)
        canonical_parts.append(rewritten)
        all_rewrites.update(rewrites)

    if all_rewrites:
        logger.info(
            "Rewrote %d off-vocabulary label(s): %s",
            len(all_rewrites),
            ", ".join(
                f"{found} -> {canonical}" for found, canonical in sorted(all_rewrites.items())
            ),
        )

    return canonical_parts


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
    parts = [normalize_cypher_string_literals(part, normalizer=fold_diacritics) for part in parts]
    parts = _canonicalize_labels(parts, logger)
    parts = [rewrite_merge_to_canonical_key(part) for part in parts]

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
