import os
import threading
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
from src.data_pipeline.completeness import extract_list_rows, rows_missing_from_cypher
from src.data_pipeline.label_vocabulary import LabelVocabulary, render_allowed_labels
from src.text_normalization import fold_diacritics, normalize_cypher_string_literals

# The second extraction pass is one extra model call per page that lost rows. It only fires on a
# detected miss and is capped at one pass per page, but on a run of list-heavy pages that is a
# real cost, so a run can bound it and always reports what it spent.
_missed_row_passes = 0
_missed_row_passes_lock = threading.Lock()


def _get_missed_row_pass_budget() -> int:
    """Read the per-run cap on extra extraction passes; 0 means unlimited."""
    raw_value = os.getenv("DATA_PIPELINE_MAX_MISSED_ROW_PASSES", "0").strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        return 0
    return max(0, parsed)


def reset_missed_row_passes() -> None:
    """Start a fresh budget. Called once by the flow before any page is submitted."""
    global _missed_row_passes
    with _missed_row_passes_lock:
        _missed_row_passes = 0


def missed_row_passes_used() -> int:
    """Number of extra extraction passes spent so far in this run."""
    with _missed_row_passes_lock:
        return _missed_row_passes


def _claim_missed_row_pass() -> bool:
    """Take one extra pass from the run's budget.

    Returns:
        True when the pass may run; False when the run has spent its budget
    """
    global _missed_row_passes
    budget = _get_missed_row_pass_budget()
    with _missed_row_passes_lock:
        if budget and _missed_row_passes >= budget:
            return False
        _missed_row_passes += 1
        return True


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
        self.missing_rows_template = PromptTemplate(
            input_variables=["rows", "context", "node_labels", "relationship_types"],
            template=config.prompts.cypher_insert_missing_rows,
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

    def run_missing_rows(self, context: str, rows: List[str]) -> List[str]:
        """Extract rows the first pass skipped.

        Args:
            context: The page the rows came from, for wording and relationships
            rows: Rows that are not represented in the first pass output

        Returns:
            Cypher statements for the missed rows, or an empty list when the call fails
        """
        logger = get_run_logger()
        chain = self.missing_rows_template | self.model | StrOutputParser()

        try:
            cypher_code = chain.invoke(
                {
                    "rows": "\n".join(f"- {row}" for row in rows),
                    "context": context,
                    "node_labels": self.node_labels,
                    "relationship_types": self.relationship_types,
                }
            )
        except Exception as exc:
            logger.error("Missed-row extraction failed: %s", exc)
            return []

        return [part.strip() for part in (cypher_code or "").split("|") if part and part.strip()]


def _recover_missed_rows(
    llm: "LLMPipe", extracted_text: str, parts: List[str], logger
) -> List[str]:
    """Extract the list and table rows the first pass left out.

    The prompt already forbids skipping rows, but a page of dates is exactly where a miss is
    invisible: the entries that survive look like a complete answer. Rows are therefore counted
    from the source and checked against what was generated, and one extra pass collects whatever
    is missing.

    Args:
        llm: The pipe that produced the first pass
        extracted_text: Page text the statements were generated from
        parts: Statements from the first pass
        logger: Prefect run logger

    Returns:
        The first pass statements followed by any recovered ones
    """
    rows = extract_list_rows(extracted_text)
    if not rows:
        return parts

    missing = rows_missing_from_cypher(rows, parts)
    if not missing:
        logger.info("Completeness check: all %d list/table row(s) extracted", len(rows))
        return parts

    logger.warning(
        "Completeness check: %d of %d row(s) missing from the first pass: %s",
        len(missing),
        len(rows),
        "; ".join(missing[:10]),
    )

    if not _claim_missed_row_pass():
        logger.warning(
            "Missed-row extraction budget spent (DATA_PIPELINE_MAX_MISSED_ROW_PASSES=%d); "
            "%d row(s) stay absent",
            _get_missed_row_pass_budget(),
            len(missing),
        )
        return parts

    recovered = llm.run_missing_rows(extracted_text, missing)
    if not recovered:
        logger.warning(
            "Missed-row extraction returned nothing; %d row(s) stay absent", len(missing)
        )
        return parts

    still_missing = rows_missing_from_cypher(missing, recovered)
    logger.info(
        "Missed-row extraction recovered %d of %d row(s) in %d statement(s)",
        len(missing) - len(still_missing),
        len(missing),
        len(recovered),
    )
    return parts + recovered


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
    parts = _recover_missed_rows(llm, extracted_text, parts, logger)
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
