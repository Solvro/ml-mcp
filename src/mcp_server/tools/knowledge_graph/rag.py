import asyncio
import json
import logging
import os
import re
import threading
import time
from enum import Enum
from typing import Any, Dict, List, NamedTuple

from google.genai.errors import APIError as GoogleAPIError
from google.genai.errors import ServerError as GoogleServerError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_neo4j import Neo4jGraph
from langchain_openai.chat_models.base import BaseChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from neo4j import READ_ACCESS
from neo4j.exceptions import (
    AuthError,
    ClientError,
    DatabaseError,
    DriverError,
    TransientError,
)
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from openai import APIError as OpenAIAPIError

from ....config.config import get_config
from ....config.messages import (
    GRAPH_PIPELINE_TIMEOUT_MESSAGE,
    NO_GRAPH_DATA_MESSAGE,
    OFF_TOPIC_MESSAGE,
)
from ....config.system_labels import SYSTEM_LABELS
from ....config.timeouts import (
    get_graph_timeout_seconds,
    get_llm_timeout_seconds,
    get_neo4j_connection_timeout_seconds,
    get_neo4j_max_transaction_retry_seconds,
    get_neo4j_query_timeout_seconds,
    get_schema_refresh_seconds,
    get_schema_version_probe_seconds,
)
from ....text_normalization import (
    POLISH_FUNCTION_WORDS,
    ensure_case_insensitive_fuzzy_matching,
    fold_diacritics,
    normalize_cypher_string_literals,
    normalize_search_text,
)
from .cypher_guardrails import (
    UnsafeCypherQueryError,
    ensure_limit,
    strip_code_fences,
    trailing_limit,
    validate_read_only,
)
from .graph_visualizer import GraphVisualizer
from .question_analysis import (
    build_lucene_query,
    extract_search_phrases,
    strip_question_literal_filters,
)
from .schema_visibility import hide_system_labels
from .state import State

logger = logging.getLogger(__name__)

PROVIDER_FALLBACK_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    GoogleServerError,
)
SINGLE_PROVIDER_RETRY_EXCEPTIONS = (APIConnectionError, InternalServerError, GoogleServerError)
PROVIDER_EXCEPTIONS = (OpenAIAPIError, GoogleAPIError, ChatGoogleGenerativeAIError)


def _is_worth_one_retry(exc: Exception) -> bool:
    """Report whether a failure is a blip a single-provider setup should repeat."""
    if isinstance(exc, APITimeoutError):
        return False
    return isinstance(exc, SINGLE_PROVIDER_RETRY_EXCEPTIONS)


GUARDRAIL_DECISION_ALIASES = {
    "generate": "generate_cypher",
    "generate_cypher": "generate_cypher",
    "end": "end",
}

# Label-agnostic rescue for a generated query that matched nothing. The graph carries many
# overlapping labels for the same kind of thing, so an answer stored under a label the model did
# not pick is still reachable by searching titles regardless of label.
#
# The search is Lucene-backed rather than a CONTAINS scan: an index-backed lookup does not walk
# every node, and it returns a relevance score, which turns "is this row good enough" into a
# number instead of a judgement call.
FULLTEXT_INDEX_NAME = "entity_search"
# The pipeline's own bookkeeping nodes carry no answer for a user question.
FULLTEXT_EXCLUDED_LABELS = SYSTEM_LABELS
FULLTEXT_SEARCH_PROCEDURE = "db.index.fulltext.queryNodes"
# The only procedure any retrieval query may call. Generated Cypher is validated with an
# empty allowlist, so this never widens what the model is allowed to do.
ALLOWED_RETRIEVAL_PROCEDURES = frozenset({FULLTEXT_SEARCH_PROCEDURE})

# How many characters of each candidate row the grader is shown. Enough to judge relevance,
# short enough that a wide fallback result stays one cheap call.
GRADER_ROW_CHARS = 400

FALLBACK_SEARCH_CYPHER = """CALL db.index.fulltext.queryNodes($index_name, $lucene_query)
YIELD node, score
WHERE score >= $min_score
  AND node.title IS NOT NULL
WITH node, score
ORDER BY score DESC, node.title
LIMIT {max_nodes}
OPTIONAL MATCH (node)-[relation]->(neighbour)
WHERE neighbour.title IS NOT NULL
RETURN labels(node) AS labels,
       node.title AS title,
       node.context AS context,
       score,
       collect(DISTINCT type(relation) + ': ' + neighbour.title) AS related
ORDER BY score DESC, title"""

SHOW_FULLTEXT_INDEX_CYPHER = """SHOW INDEXES YIELD name, type, labelsOrTypes, properties
WHERE name = $index_name
RETURN labelsOrTypes AS labels, properties AS properties"""

# Whether the schema needs re-reading is a question the graph can answer for the price of a
# label scan. Every terminal path of data_pipeline_flow stamps a PipelineRun
# (graph_populating.record_pipeline_run / record_restore_run), so a moved marker means new
# labels and properties, and an unmoved one means the cached schema still describes the graph.
# toString keeps the comparison a plain string rather than a neo4j.time.DateTime.
GRAPH_VERSION_CYPHER = """MATCH (pr:PipelineRun)
RETURN toString(max(pr.run_at)) AS version"""
# Retrieval never ran, so its zero rows say nothing about what the graph holds. Classified by
# exception type, not error code: the driver's hierarchy already draws this line, and a prefix
# list silently reclassifies whatever Neo4j adds next. OSError covers gaierror/ConnectionError.
# The remaining ClientError codes are deliberately absent. Statement.*, Procedure.* and Schema.*
# come from a database that is demonstrably up and mean this query could not run. A missing
# full-text index arrives as Procedure.ProcedureCallFailed, and the reindex-and-retry rescue in
# _search_every_label only happens while that stays a recoverable "found nothing".
#
# Transaction.TransactionTimedOut* is the one ClientError that escalates anyway, through
# _is_neo4j_query_timeout rather than this tuple. It says nothing about the query and everything
# about how long it was given, and neo4j_query_timeout_seconds is what gives it.
NEO4J_INFRASTRUCTURE_EXCEPTIONS = (
    DriverError,
    TransientError,
    AuthError,
    DatabaseError,
    OSError,
)
NEO4J_QUERY_TIMEOUT_CODE_PREFIX = "Neo.ClientError.Transaction.TransactionTimedOut"


def _is_neo4j_query_timeout(exc: Exception) -> bool:
    """Report whether Neo4j stopped a query for outliving the per-query timeout."""
    return isinstance(exc, ClientError) and str(getattr(exc, "code", "")).startswith(
        NEO4J_QUERY_TIMEOUT_CODE_PREFIX
    )


# Neo4j refused the statement itself: a syntax error, an undefined variable, a type mismatch. It
# says the model wrote bad Cypher and nothing about the graph, which is why such a failure is
# escalated to the label-agnostic search rather than reported (issue #3). Schema.* and
# Procedure.* codes are not in this family - they name something missing in the database.
NEO4J_STATEMENT_ERROR_CODE_PREFIX = "Neo.ClientError.Statement."


def _is_neo4j_statement_error(exc: Exception) -> bool:
    """Report whether Neo4j rejected the generated statement as malformed."""
    return isinstance(exc, ClientError) and str(getattr(exc, "code", "")).startswith(
        NEO4J_STATEMENT_ERROR_CODE_PREFIX
    )


class LLMProvider(Enum):
    """Available LLM providers for the runtime fallback chain."""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"


class RetrievalStrategy(Enum):
    """Which attempt in the retrieval escalation produced the context."""

    PRIMARY = "primary"
    REPAIRED_LITERALS = "repaired_literals"
    LABEL_AGNOSTIC_PHRASES = "label_agnostic_phrases"
    # The model's Cypher was rejected by Neo4j and the full-text search answered instead. Kept
    # distinct from LABEL_AGNOSTIC_PHRASES so a log or trace shows that no primary query ran.
    LABEL_AGNOSTIC_AFTER_ERROR = "label_agnostic_after_error"
    # The model's query returned rows, the grader rejected all of them, and the full-text search
    # answered instead (issue #99). Distinct for the same reason as the one above: a trace has to
    # show that the model's query ran and matched the wrong thing.
    LABEL_AGNOSTIC_AFTER_GRADING = "label_agnostic_after_grading"
    GRADED_OUT = "graded_out"
    EMPTY = "empty"


# Rows that came back from a traversal the Cypher model wrote. When the grader rejects every one
# of them the traversal was wrong, and unlike an empty result nothing else has been tried yet, so
# the run goes on to the label-agnostic search instead of ending in "no data" (issue #99).
MODEL_QUERY_STRATEGIES = frozenset(
    {RetrievalStrategy.PRIMARY.value, RetrievalStrategy.REPAIRED_LITERALS.value}
)


def _matched_anything(rows: List[Any]) -> bool:
    """
    Report whether a model query's rows hold anything at all.

    The Cypher prompt asks for a list as ``collect()`` so ``LIMIT`` cannot cut it (issue #106),
    and an aggregate with no grouping key answers even when nothing matched:
    ``RETURN collect(i.title) AS items`` comes back as one row, ``{items: []}``. That is the
    zero-row result the escalation exists for. A 0 or a false is an answer and counts; only a
    null, an empty string and an empty list or map do not.

    That leaves a known limit: "Ile jest ..." over a filter that matched nothing returns
    ``count(*) = 0``, which is reported as the answer rather than escalated. It behaved the same
    before this check, and nothing in the row tells a real zero from a filter that missed.
    """
    return any(
        value not in (None, "", [], {})
        for row in rows
        for value in (row.values() if isinstance(row, dict) else [row])
    )


# A word this short carries no name of its own ("dr", "i", "w"), so an anchor need not repeat it.
ANCHOR_MIN_WORD_CHARS = 4
# How many leading characters two words must share to be the same Polish word. A case ending
# changes the tail ("dydaktyczna", "dydaktyczne"), and a short word keeps one letter of slack for
# it ("wolne", "wolny").
ANCHOR_STEM_CHARS = 5


class GraderVerdict(NamedTuple):
    """What the grader said about one batch of rows."""

    kept: List[int]
    entity: str | None
    anchor: str | None


def _words(text: str) -> List[str]:
    """Split text into case- and diacritic-folded words, dropping quotes and punctuation."""
    return re.findall(r"\w+", normalize_search_text(text))


def _contains_phrase(words: List[str], phrase: List[str]) -> bool:
    """Report whether ``phrase`` occurs in ``words`` as whole, consecutive words."""
    return f" {' '.join(phrase)} " in f" {' '.join(words)} "


def _same_word(left: str, right: str) -> bool:
    """Report whether two folded words are one Polish word in different cases."""
    if left == right:
        return True
    if left.isdigit() or right.isdigit():
        return False
    stem = min(ANCHOR_STEM_CHARS, len(left) - 1, len(right) - 1)
    return stem >= 3 and left[:stem] == right[:stem]


def _is_code(word: str) -> bool:
    """Report whether a word is a code like "R2" or "W4": short, but a name in its own right."""
    return any(character.isdigit() for character in word)


def _required_words(entity: str) -> List[str]:
    """The words of the entity an anchor has to account for."""
    return [
        word
        for word in _words(entity)
        if (len(word) >= ANCHOR_MIN_WORD_CHARS or _is_code(word))
        and word not in POLISH_FUNCTION_WORDS
    ]


def _anchor_covers_entity(entity: str, anchor_words: List[str]) -> bool:
    """
    Report whether an anchor names the whole entity rather than a word that resembles part of it.

    Every content word of the entity has to reappear in the anchor, in any case ending. That is
    the check "dydaktyczne" fails for "działalność dydaktyczna": the adjective matches and
    nothing in it names the activity (issue #99 review). A code counts however short it is:
    "R1" and "R2" are two entities one character apart.
    """
    return all(
        any(_same_word(word, candidate) for candidate in anchor_words)
        for word in _required_words(entity)
    )


# The graph describes one institution, so its name qualifies nothing: the backend's answer agent
# appends "na Politechnice Wrocławskiej" to most questions and it must not read as a name the
# anchor left out. Folded stems, compared with ``_same_word``.
INSTITUTION_WORDS = ("politechnika", "wroclawska", "pwr")


def _is_institution(word: str) -> bool:
    return any(_same_word(word, name) for name in INSTITUTION_WORDS)


def _anchor_names_part_of_entity(entity: str, anchor_words: List[str]) -> bool:
    """
    Report whether the anchor is the specific part of an entity phrase wider than a name.

    The grader names the entity as the question's noun phrase more often than as the name in
    it — "kryteria w kategorii Dorobek naukowy", "kompetencje pożądane dla naukowca R2" — and
    then points at the name. A Polish noun phrase puts its head first and what
    makes it specific after, so the anchor may leave out the head before it and nothing after
    it but function words: "kursy prowadzone" inside "kursy prowadzone przez dr Jan Kowalski"
    is the head with the name left out, and so is a name the anchor skips anywhere
    — a capitalised word or a code, except a capital at word 0, which is how the grader
    happened to spell the JSON. The anchor itself has to be more than one lowercase word:
    two or more content words, a code, or a capitalised word — "dydaktyczne" sits inside
    "działalność dydaktyczna" too and is exactly the #99 leak.
    """
    folded = _words(entity)
    spelled = re.findall(r"\w+", entity)
    if len(spelled) != len(folded):
        spelled = folded
    content_words = [word for word in anchor_words if word not in POLISH_FUNCTION_WORDS]
    width = len(anchor_words)
    for start in range(len(folded) - width + 1):
        if not all(_same_word(folded[start + i], anchor_words[i]) for i in range(width)):
            continue
        matched = spelled[start : start + width]
        named = len(content_words) >= 2 or any(
            _is_code(word) or (index > 0 and word[:1].isupper())
            for index, word in enumerate(matched, start)
        )
        if not named:
            continue
        left_out = [
            (index, folded[index], spelled[index])
            for index in range(len(folded))
            if not start <= index < start + width and not _is_institution(folded[index])
        ]
        if any(index > start and word not in POLISH_FUNCTION_WORDS for index, word, _ in left_out):
            continue
        if any(
            _is_code(word) or (index > 0 and text[:1].isupper()) for index, word, text in left_out
        ):
            continue
        return True
    return False


def _is_label_in(anchor: str, cypher: str) -> bool:
    """Report whether the anchor is a label or relationship type the query matches on."""
    name = anchor.strip().strip("`:")
    if not name:
        return False
    return re.search(rf"[:|]\s*`?{re.escape(name)}`?(?![\w`])", cypher, re.IGNORECASE) is not None


def _row_values(rows: List[Any]) -> List[str]:
    """Render every value of every row as text an anchor can be looked up in."""
    values: List[str] = []
    for row in rows:
        for value in row.values() if isinstance(row, dict) else [row]:
            values.append(value if isinstance(value, str) else json.dumps(value, default=str))
    return values


def _locate_candidate(
    candidate: str | None, entity: str | None, cypher: str, rows: List[Any]
) -> str | None:
    """Say where one candidate anchor sits, if it holds the entity at all."""
    anchor_words = _words(candidate or "")
    if not anchor_words:
        return None
    if _is_label_in(candidate, cypher):
        return "query"
    covers = not entity or _anchor_covers_entity(entity, anchor_words)
    if _contains_phrase(_words(cypher), anchor_words):
        # A filter the query itself applies only has to name the specific part of the entity;
        # an anchor found in a row has to cover all of it, since that is where the leak was.
        if covers or _anchor_names_part_of_entity(entity, anchor_words):
            return "query"
        return None
    if covers and any(_contains_phrase(_words(value), anchor_words) for value in _row_values(rows)):
        return "rows"
    return None


def locate_anchor(verdict: GraderVerdict, cypher: str, rows: List[Any]) -> str | None:
    """
    Find where the anchor the grader named actually sits.

    The grader has to point at the text that holds the question's entity, and this checks that
    the text is really there and really is the entity, so a grader that talks itself into a
    similar word cannot keep the rows. A label is taken on the grader's word: it names a kind of
    thing in English, which no Polish entity spells out, and it is the only anchor a question
    like "Jakie są dni wolne?" has.

    The entity itself is tried second. Against the fast model the grader named the entity
    right and still answered a null anchor about one run in six, with that exact phrase sitting
    in the query's filter; the filter is evidence, the null is not.

    How much of the entity the anchor has to name depends on where it sits. A filter the query
    applies selects what it returns, so it only has to be the specific part of the entity —
    the grader writes "kryteria w kategorii Dorobek naukowy" and points at "Dorobek naukowy"
    (issue #27). An anchor found only in a row has to cover the whole entity, since a row
    holding one word of it is how "dydaktyczne" once stood in for "działalność dydaktyczna".

    Args:
        verdict: The grader's reply
        cypher: The query that returned the rows
        rows: The rows the grader was shown

    Returns:
        "query" when the query filters on or matches the entity, "rows" when only a row holds
        it, and None when nothing does
    """
    for candidate in (verdict.anchor, verdict.entity):
        located = _locate_candidate(candidate, verdict.entity, cypher, rows)
        if located is not None:
            return located
    return None


class KnowledgeGraphUnavailableError(RuntimeError):
    """Raised when Neo4j cannot be consulted to answer a retrieval query."""


class KnowledgeGraphQueryError(RuntimeError):
    """Raised when the generated Cypher could not be executed on a database that is up.

    Either the read-only guardrail refused it or Neo4j rejected it. Both are failures of this
    system, not facts about the graph: the question was never asked of the data, so neither
    "no data" nor the error text may be handed back as an answer. The server turns this into a
    ToolError with a fixed message; the Cypher and Neo4j's reason are logged for the operator.
    """

    def __init__(self, reason: str, *, cypher: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cypher = cypher


class LLMUnavailableError(RuntimeError):
    """Raised when required LLM calls could not be completed."""

    def __init__(self, reason: str, *, timed_out: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.timed_out = timed_out


class RAG:
    """Retrieval-Augmented Generation system with Neo4j graph database backend."""

    def __init__(
        self,
        api_key: str,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        enable_debug: bool = None,
        max_results: int = None,
        llm_timeout_sec: float | None = None,
        graph_timeout_sec: float | None = None,
        neo4j_query_timeout_sec: float | None = None,
        neo4j_connection_timeout_sec: float | None = None,
        neo4j_max_transaction_retry_sec: float | None = None,
    ):
        """
        Initialize RAG system with API keys and database credentials.

        Args:
            api_key: OpenAI/DeepSeek API key
            neo4j_url: Neo4j database connection URL
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            enable_debug: Force DEBUG logging for this pipeline, whatever LOG_LEVEL says
            max_results: Maximum number of results from Neo4j (default: 5)
            llm_timeout_sec: Per-call HTTP timeout for each LLM client
            graph_timeout_sec: Wall-clock budget for the whole RAG run
            neo4j_query_timeout_sec: Timeout for one Neo4j query execution
            neo4j_connection_timeout_sec: Timeout for establishing a Neo4j connection
            neo4j_max_transaction_retry_sec: Driver retry budget for transient failures
        """
        config = get_config()

        self.api_key = api_key
        self.config = config
        self.llm_timeout_sec = (
            llm_timeout_sec if llm_timeout_sec is not None else get_llm_timeout_seconds()
        )
        self.graph_timeout_sec = (
            graph_timeout_sec if graph_timeout_sec is not None else get_graph_timeout_seconds()
        )
        self.enable_debug = enable_debug if enable_debug is not None else config.rag.enable_debug
        if self.enable_debug:
            logger.setLevel(logging.DEBUG)
        self.max_results = max_results if max_results is not None else config.rag.max_results
        self.enable_fallback_search = config.rag.enable_fallback_search
        self.fallback_min_score = config.rag.fallback_min_score
        configured_neo4j_query_timeout = (
            neo4j_query_timeout_sec
            if neo4j_query_timeout_sec is not None
            else get_neo4j_query_timeout_seconds()
        )
        if configured_neo4j_query_timeout > self.graph_timeout_sec:
            logger.warning(
                "Neo4j query timeout %.1fs exceeds graph timeout %.1fs; capping to graph timeout",
                configured_neo4j_query_timeout,
                self.graph_timeout_sec,
            )
        self.neo4j_query_timeout_sec = min(configured_neo4j_query_timeout, self.graph_timeout_sec)
        self.neo4j_connection_timeout_sec = (
            neo4j_connection_timeout_sec
            if neo4j_connection_timeout_sec is not None
            else get_neo4j_connection_timeout_seconds()
        )
        configured_retry_budget = (
            neo4j_max_transaction_retry_sec
            if neo4j_max_transaction_retry_sec is not None
            else get_neo4j_max_transaction_retry_seconds()
        )
        self.neo4j_max_transaction_retry_sec = configured_retry_budget
        configured_connection_timeout = self.neo4j_connection_timeout_sec
        configured_total = configured_connection_timeout + configured_retry_budget
        if configured_total > self.graph_timeout_sec > 0:
            scale = self.graph_timeout_sec / configured_total
            self.neo4j_connection_timeout_sec = configured_connection_timeout * scale
            self.neo4j_max_transaction_retry_sec = configured_retry_budget * scale
            logger.warning(
                "Neo4j connection timeout %.1fs plus retry budget %.1fs would outlast the %.1fs "
                "graph timeout; scaling both to %.1fs and %.1fs",
                configured_connection_timeout,
                configured_retry_budget,
                self.graph_timeout_sec,
                self.neo4j_connection_timeout_sec,
                self.neo4j_max_transaction_retry_sec,
            )

        self.single_provider = len(self._get_configured_providers()) == 1
        self.fast_llm = self._build_llm_with_fallback(use_accurate=False)
        self.cypher_llm = self._build_llm_with_fallback(use_accurate=True)

        self._initialize_prompt_templates()

        self.database = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=config.database.name,
            driver_config={
                "connection_timeout": self.neo4j_connection_timeout_sec,
                "max_transaction_retry_time": self.neo4j_max_transaction_retry_sec,
                "notifications_disabled_classifications": ["UNRECOGNIZED"],
            },
            enhanced_schema=True,
        )
        self.database.timeout = self.neo4j_query_timeout_sec

        self._init_schema_cache()

        if self.enable_fallback_search:
            try:
                self.ensure_fulltext_index()
            except KnowledgeGraphUnavailableError as exc:
                logger.warning("Could not build %s at startup: %s", FULLTEXT_INDEX_NAME, exc)

        self.visualizer = GraphVisualizer()
        self.graph = self._build_processing_graph()

    def ping_database(self) -> None:
        """
        Prove the graph connection can still serve a query.

        Blocking, like every other call through ``Neo4jGraph``. Raises the driver's own error
        rather than returning a bool, so a caller can report why the graph is unreachable and
        not merely that it is.

        Raises:
            Exception: Whatever the Neo4j driver raises when the query cannot run
        """
        self.database.query("RETURN 1 AS ok")

    def close(self) -> None:
        """
        Release the Neo4j driver opened in the constructor.

        Idempotent: `Neo4jGraph.close` drops its driver reference, so a second call does
        nothing. A RAG built without a database (tests) closes cleanly too.
        """
        database = getattr(self, "database", None)
        if database is None:
            return
        database.close()

    def _get_invoke_config(
        self,
        trace_id: str,
        tags: list,
        run_name: str,
        handler=None,
        session_id: str = None,
    ) -> dict:
        """
        Build invoke config with optional callbacks.

        Args:
        trace_id: Trace identifier for this single request
        tags: Langfuse tags applied to the spans
        run_name: Human-readable name for the span in Langfuse
        handler: Optional CallbackHandler
        session_id: Conversation session identifier used as Langfuse session_id

        """
        config = {
            "run_name": run_name,
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_tags": tags,
            },
        }
        if handler is not None:
            config["callbacks"] = [handler]
        return config

    @staticmethod
    def _available_provider_keys() -> Dict[LLMProvider, str]:
        """Return providers that have a non-empty API key in the environment."""
        key_by_provider = {
            LLMProvider.OPENAI: os.environ.get("OPENAI_API_KEY", "").strip(),
            LLMProvider.DEEPSEEK: os.environ.get("DEEPSEEK_API_KEY", "").strip(),
            LLMProvider.GOOGLE: os.environ.get("GOOGLE_API_KEY", "").strip(),
        }
        return {provider: key for provider, key in key_by_provider.items() if key}

    def _get_configured_providers(self) -> List[LLMProvider]:
        """Read provider fallback order from config and filter by available API keys."""
        available = self._available_provider_keys()

        providers: List[LLMProvider] = []
        for provider_name in self.config.llm.provider_fallback_order:
            name = str(provider_name).strip().lower()
            try:
                provider = LLMProvider(name)
            except ValueError:
                logger.warning("Unknown provider in config: %r; skipping", provider_name)
                continue
            if provider in available and provider not in providers:
                providers.append(provider)

        if providers:
            return providers
        return [provider for provider in LLMProvider if provider in available]

    def _build_chat_model(self, provider: LLMProvider, *, use_accurate: bool = False):
        """Create a single LLM client for the specified provider."""
        model_cfg = self.config.llm.accurate_model if use_accurate else self.config.llm.fast_model

        if provider == LLMProvider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            return BaseChatOpenAI(
                model=model_cfg.name,
                api_key=api_key,
                temperature=model_cfg.temperature,
                timeout=self.llm_timeout_sec,
                max_retries=0,
            )

        if provider == LLMProvider.DEEPSEEK:
            deepseek_cfg = self.config.llm.deepseek
            api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or self.api_key
            return BaseChatOpenAI(
                model=deepseek_cfg.accurate_model if use_accurate else deepseek_cfg.fast_model,
                api_key=api_key,
                base_url=self.config.llm.deepseek.base_url,
                temperature=model_cfg.temperature,
                timeout=self.llm_timeout_sec,
                max_retries=0,
            )

        if provider == LLMProvider.GOOGLE:
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip() or self.api_key
            return ChatGoogleGenerativeAI(
                model=self.config.llm.gemini.name,
                google_api_key=api_key,
                temperature=model_cfg.temperature,
                timeout=self.llm_timeout_sec,
                max_retries=0,
            )

        raise ValueError(f"Unknown provider: {provider}")

    def _build_llm_with_fallback(self, *, use_accurate: bool = False):
        """
        Build primary LLM client with provider fallbacks via LangChain with_fallbacks.

        Resilience is provider switching (OpenAI → DeepSeek → Google), not same-provider
        retries. Only provider/network failures listed in PROVIDER_FALLBACK_EXCEPTIONS
        trigger a switch.
        """
        providers = self._get_configured_providers()
        if not providers:
            raise RuntimeError(
                "No LLM provider available. Set OPENAI_API_KEY, "
                "DEEPSEEK_API_KEY, or GOOGLE_API_KEY."
            )

        models = [
            self._build_chat_model(provider, use_accurate=use_accurate) for provider in providers
        ]
        primary, *secondaries = models
        if not secondaries:
            return primary

        return primary.with_fallbacks(
            secondaries,
            exceptions_to_handle=PROVIDER_FALLBACK_EXCEPTIONS,
        )

    def _invoke_with_single_provider_retry(
        self,
        chain: Any,
        payload: dict[str, Any],
        invoke_config: dict[str, Any],
        *,
        operation_name: str,
    ) -> str:
        """
        Run one required LLM call, repeating a blip when nothing else could have answered.

        Args:
            chain: Prompt-to-string runnable for this call
            payload: Prompt variables
            invoke_config: LangChain invoke config carrying the Langfuse context
            operation_name: Name of the call, for the log line

        Returns:
            The model's reply

        Raises:
            LLMUnavailableError: The model call failed and the run cannot continue. Anything
                outside PROVIDER_EXCEPTIONS propagates untouched - it is a bug, not an outage
        """
        try:
            return chain.invoke(payload, config=invoke_config)
        except PROVIDER_EXCEPTIONS as exc:
            if self.single_provider and _is_worth_one_retry(exc):
                logger.warning(
                    "%s hit a transient LLM error with a single provider configured; "
                    "retrying once: %s",
                    operation_name,
                    exc,
                )
                try:
                    return chain.invoke(payload, config=invoke_config)
                except PROVIDER_EXCEPTIONS as retry_exc:
                    logger.error(
                        "%s failed after single-provider retry: %s", operation_name, retry_exc
                    )
                    raise LLMUnavailableError(
                        f"{operation_name}: {retry_exc}",
                        timed_out=isinstance(retry_exc, APITimeoutError),
                    ) from retry_exc

            logger.error("%s failed: %s", operation_name, exc)
            raise LLMUnavailableError(
                f"{operation_name}: {exc}",
                timed_out=isinstance(exc, APITimeoutError),
            ) from exc

    def _init_schema_cache(self) -> None:
        """
        Reset everything the schema cache is made of.

        Kept separate from ``__init__`` so a bare instance built for a test starts from the
        same state a real one does.
        """
        self._cached_schema: str | None = None
        self._schema_fetched_at: float = 0.0
        self._schema_ttl_sec: float = get_schema_refresh_seconds()
        self._schema_lock = threading.RLock()
        self._known_labels: frozenset[str] | None = None
        self._graph_version_seen: str | None = None
        self._version_probed_at: float = 0.0
        self._version_probe_interval_sec: float = get_schema_version_probe_seconds()

    @staticmethod
    def _schema_is_empty(db_schema: str | None) -> bool:
        """Report whether a schema string describes a graph that holds nothing."""
        stripped = (db_schema or "").strip()
        headers_only = (
            "Node properties:" in stripped
            and "Relationship properties:" in stripped
            and "The relationships:" in stripped
            and stripped.replace("Node properties:", "")
            .replace("Relationship properties:", "")
            .replace("The relationships:", "")
            .strip()
            == ""
        )
        return not stripped or headers_only

    def invalidate_schema_cache(self) -> None:
        """Force the next schema read to re-query Neo4j."""
        with self._schema_lock:
            self._cached_schema = None
            self._schema_fetched_at = 0.0
            self._version_probed_at = 0.0
            self._graph_version_seen = None

    def _graph_version(self) -> str | None:
        """
        Read the marker the pipeline stamps at the end of every run.

        Returns:
            The latest run timestamp, an empty string when nothing has been ingested yet, or
            None when the probe itself failed. The last two are deliberately different: "no
            runs" is a fact about the graph, "probe failed" is an absence of information and
            must not be allowed to look like an unchanged graph.
        """
        try:
            rows = self.database.query(GRAPH_VERSION_CYPHER)
        except Exception as exc:
            logger.warning("Could not read the graph version marker: %s", exc)
            return None

        return str(rows[0].get("version") or "") if rows else ""

    def _cached_schema_is_fresh(self) -> bool:
        """
        Report whether the cached schema may still be served without re-reading it.

        Three gates, cheapest first. The TTL is the backstop rather than the mechanism: it
        exists for whatever writes to the graph without recording a run - ``uv run
        dedup-graph`` relabels nodes across the whole graph and stamps no PipelineRun, and
        neither does a hand-run Cypher fix. Everything the pipeline itself does is caught by
        the marker, within one probe interval instead of one TTL.
        """
        if not self._cached_schema:
            return False

        now = time.monotonic()
        if now - self._schema_fetched_at >= self._schema_ttl_sec:
            return False

        if now - self._version_probed_at < self._version_probe_interval_sec:
            return True

        version = self._graph_version()
        self._version_probed_at = now

        if version is None:
            # The probe failed, so nothing is known about the graph. The cached schema is the
            # last thing known to be true about it; the next question probes again.
            return True

        if version != self._graph_version_seen:
            logger.info(
                "Graph version moved (%s -> %s); re-reading the schema",
                self._graph_version_seen or "(none)",
                version or "(none)",
            )
            return False

        return True

    def _fetch_schema(self) -> str:
        """
        Re-read the schema from Neo4j and cache it when it describes a populated graph.

        ``Neo4jGraph.get_schema`` is a stored string, not a query: only ``refresh_schema()``
        goes back to the database. Reading the property alone returns whatever the graph looked
        like when the driver was constructed, which for any deployment that starts before
        ingestion finishes is an empty graph, forever.

        A failed refresh keeps the last good schema. Serving a slightly stale schema is a far
        smaller problem than generating Cypher against nothing.

        What comes back describes every label the database holds, including the ones the
        pipeline keeps its own books in, so ``hide_system_labels`` takes those out before
        anything caches or reads the text.

        Raises:
            KnowledgeGraphUnavailableError: The refresh failed and there is no cached schema to
                fall back on, so nothing is known about the graph.

        Returns:
            The schema text, or an empty string when the graph holds nothing
        """
        # Read before refreshing, never after: a run that lands while the refresh is in flight
        # would otherwise be recorded as already seen, and its labels would stay invisible
        # until the run after it. Reading first can only cost one redundant refresh.
        version = self._graph_version()

        try:
            self.database.refresh_schema()
        except Exception as exc:
            if self._cached_schema:
                logger.warning("Could not refresh the Neo4j schema; keeping cached schema: %s", exc)
                return self._cached_schema
            logger.error("Neo4j could not be consulted while refreshing schema: %s", exc)
            raise KnowledgeGraphUnavailableError(str(exc)) from exc

        # Filtered before the emptiness check, not after: a graph holding nothing but the
        # pipeline's own bookkeeping has nothing to answer a question with, and reading as
        # empty is what makes the run abstain instead of querying provenance rows.
        db_schema = hide_system_labels(self.database.get_schema)

        if self._schema_is_empty(db_schema):
            # Not cached, and no timestamp written, so the next question tries again rather
            # than waiting out the TTL on a graph that was merely mid-ingestion.
            self._cached_schema = None
            logger.warning("Graph is empty; schema will be re-fetched on the next call")
            return ""

        self._cached_schema = db_schema
        self._schema_fetched_at = time.monotonic()
        self._graph_version_seen = version
        self._version_probed_at = self._schema_fetched_at
        logger.info("Fetched %d chars of schema from Neo4j", len(db_schema))
        return db_schema

    @property
    def schema(self) -> str:
        """
        Database schema for Cypher generation, re-read from Neo4j when the cache goes stale.

        Cached for ``rag.schema_refresh_seconds`` rather than for the process lifetime: the
        pipeline adds labels and properties on every nightly run, and a serving process that
        never looks again generates queries against a graph that no longer exists. An empty
        result is never cached, so a temporary empty database at startup cannot poison it.
        """
        if self._cached_schema_is_fresh():
            return self._cached_schema or ""

        fetched_at = self._schema_fetched_at
        with self._schema_lock:
            if self._cached_schema and self._schema_fetched_at != fetched_at:
                # Another thread refreshed while this one waited for the lock. Re-running the
                # freshness check here instead would meet the probe stamp that check had just
                # written and conclude the cache is fine, skipping the refresh entirely.
                return self._cached_schema
            return self._fetch_schema()

    def get_graph(self):
        """Return graph visualizer with Mermaid capabilities"""
        return self.visualizer

    def _initialize_prompt_templates(self):
        """Initialize all prompt templates used in the RAG pipeline."""
        config = get_config()

        self.generate_cypher_template = PromptTemplate(
            input_variables=["user_question", "normalized_question", "schema"],
            template=config.prompts.cypher_search,
        )

        self.guard_rails_template = PromptTemplate(
            input_variables=["user_question"], template=config.prompts.guardrails
        )

        self.context_grader_template = PromptTemplate(
            input_variables=["user_question", "retrieval", "candidates"],
            template=config.prompts.context_grader,
        )

    def _parse_guardrail_output(self, raw_output: str) -> Dict[str, str]:
        """Parse guardrail JSON and normalize the decision with a safe fallback."""
        cleaned_output = strip_code_fences(raw_output)

        try:
            start = cleaned_output.index("{")
            payload, _ = json.JSONDecoder().raw_decode(cleaned_output[start:])
        except (ValueError, json.JSONDecodeError) as exc:
            logger.debug("Guardrail output was not valid JSON: %s; raw=%s", exc, raw_output)
            return {"decision": "end"}

        decision = str(payload.get("decision", "")).strip().lower()
        normalized_decision = GUARDRAIL_DECISION_ALIASES.get(decision)

        if normalized_decision is None:
            logger.debug("Guardrail decision %r is unknown; raw=%s", decision, raw_output)
            return {"decision": "end"}

        return {"decision": normalized_decision}

    @staticmethod
    def _render_grader_candidates(context: List[Dict[str, Any]]) -> str:
        """Render retrieved rows as a numbered list the grader can refer to by index."""
        lines = []
        for position, row in enumerate(context, start=1):
            rendered = json.dumps(row, ensure_ascii=False, default=str)
            if len(rendered) > GRADER_ROW_CHARS:
                rendered = f"{rendered[:GRADER_ROW_CHARS]}..."
            lines.append(f"{position}. {rendered}")
        return "\n".join(lines)

    def _parse_grader_output(self, raw_output: str, row_count: int) -> GraderVerdict | None:
        """
        Read the row numbers the grader kept, and the entity and anchor it named.

        Args:
            raw_output: Raw grader reply
            row_count: How many rows the grader was shown

        Returns:
            Zero-based indices of the rows to keep with the entity and anchor, or None when the
            reply is unusable. A missing or non-text entity or anchor comes back as None.
        """
        cleaned_output = strip_code_fences(raw_output)

        try:
            start = cleaned_output.index("{")
            payload, _ = json.JSONDecoder().raw_decode(cleaned_output[start:])
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Context grader returned unusable JSON: %s", exc)
            return None

        relevant = payload.get("relevant")
        if not isinstance(relevant, list):
            logger.warning("Context grader reply had no 'relevant' list: %r", raw_output)
            return None

        kept: List[int] = []
        for entry in relevant:
            try:
                position = int(entry)
            except (TypeError, ValueError):
                continue
            if 1 <= position <= row_count and position - 1 not in kept:
                kept.append(position - 1)

        entity = payload.get("entity")
        anchor = payload.get("anchor")
        return GraderVerdict(
            kept=kept,
            entity=entity if isinstance(entity, str) else None,
            anchor=anchor if isinstance(anchor, str) else None,
        )

    @staticmethod
    def _describe_retrieval(state: State) -> str:
        """
        Tell the grader how the rows were found.

        A row from the model's query holds only the columns that query returned, so a course
        title from a traversal that started at the teacher the question names does not repeat
        the teacher. Without the query the grader sees a bare title and cannot tell it answers
        the question.
        """
        if state.get("retrieval_strategy") in MODEL_QUERY_STRATEGIES:
            return f"The Cypher query that returned them:\n{state.get('generated_cypher') or ''}"
        return (
            "A full-text search for words from the question across every kind of entity, so a "
            "row may only share a word with it."
        )

    def grade_context(self, state: State):
        """
        Drop retrieved rows that do not answer the question.

        Whether to answer is decided here, on retrieval evidence, rather than left to the
        answering model: rows recovered by a widened search are candidates, not an answer, and a
        row that only shares a word with the question is what produces a confident wrong answer.

        Rows from the model's own query are graded too, because a query that followed the wrong
        relationship returns rows as confidently as a right one (issue #99). None of them is
        kept unless the grader points at an anchor, text in the query or a row that holds the
        entity the question is about, and ``locate_anchor`` confirms it: a row that only uses a
        similar word is how a grader that was merely asked "is this relevant" kept the wrong
        category. A primary query that filters on the entity itself keeps its whole list, since
        everything it returned sits under that entity, whatever the grader said about single
        rows; one whose anchor is only found in the rows mixed entities, and keeps just the rows
        the grader picked. When nothing is kept,
        the run moves on to the label-agnostic search, since a wrong result must get the same
        second chance an empty one does.

        A grader that fails or replies with nonsense leaves the rows untouched - a model outage
        must not be indistinguishable from an empty graph.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with the rows that survived grading, and where the run goes next
        """
        context = state.get("context") or []
        strategy = state.get("retrieval_strategy")
        kept_as_retrieved = {"context_graded": False, "next_node": "end"}

        if not context:
            return kept_as_retrieved

        grader_chain = self.context_grader_template | self.fast_llm | StrOutputParser()

        try:
            grader_output = grader_chain.invoke(
                {
                    "user_question": state["user_question"],
                    "retrieval": self._describe_retrieval(state),
                    "candidates": self._render_grader_candidates(context),
                },
                config=self._get_invoke_config(
                    trace_id=state.get("trace_id"),
                    tags=["knowledge_graph", "context_grader"],
                    run_name="Context Grader",
                    handler=state.get("callback_handler"),
                    session_id=state.get("session_id"),
                ),
            )
        except Exception as exc:
            logger.warning("Context grading failed; keeping retrieved rows: %s", exc)
            return kept_as_retrieved

        verdict = self._parse_grader_output(grader_output, len(context))
        if verdict is None:
            return kept_as_retrieved

        kept = verdict.kept
        logger.debug(
            "Context grader kept %d of %d %s row(s); entity=%r anchor=%r",
            len(kept),
            len(context),
            strategy,
            verdict.entity,
            verdict.anchor,
        )

        anchored_in = None
        if strategy in MODEL_QUERY_STRATEGIES:
            anchored_in = locate_anchor(verdict, state.get("generated_cypher") or "", context)
            if anchored_in is None and kept:
                logger.info(
                    "Context grader kept %d %s row(s) but nothing holds %r (anchor %r); "
                    "treating them as rejected",
                    len(kept),
                    strategy,
                    verdict.entity,
                    verdict.anchor,
                )
                kept = []

        if strategy == RetrievalStrategy.PRIMARY.value and anchored_in == "query":
            # The anchor is the check here, not the row list: measured against the fast model,
            # it found the teacher filter every time and still dropped the course titles under
            # it, because none of them repeats the teacher's name.
            return {"context_graded": True, "next_node": "end"}

        if not kept:
            logger.info("Context grader rejected all %d %s row(s)", len(context), strategy)
            return {
                "context": [],
                "context_graded": True,
                "retrieval_strategy": RetrievalStrategy.GRADED_OUT.value,
                "next_node": (
                    "search_after_grading" if strategy in MODEL_QUERY_STRATEGIES else "end"
                ),
            }

        return {
            "context": [context[index] for index in kept],
            "context_graded": True,
            "next_node": "end",
        }

    def search_after_grading(self, state: State):
        """
        Search every label once the grader has rejected every row of the model's query.

        Retrieval only escalates on zero rows, so a query that matched the wrong entity used to
        end the run on rows the answering model then, correctly, declined to use. Asked again,
        the same question could come back empty and be answered from the full-text index, which
        made the outcome depend on whether the bad query happened to match anything (issue #99).

        The rows found here go back through ``grade_context`` like any other rescue. They are
        never escalated again, so the run cannot loop.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with the recovered context, or nothing when the search found nothing
            and the run stays graded out
        """
        logger.debug("Graded-out query:\n%s", state.get("generated_cypher"))

        fallback = self._search_every_label(state.get("user_question") or "")
        if fallback is None:
            logger.info("Label-agnostic search after a graded-out query found nothing")
            return {}

        logger.info(
            "Label-agnostic search after a graded-out query found %d row(s)",
            len(fallback["context"]),
        )
        return {
            **fallback,
            "retrieval_strategy": RetrievalStrategy.LABEL_AGNOSTIC_AFTER_GRADING.value,
        }

    def _build_processing_graph(self):
        """Construct the state machine graph for the RAG pipeline."""
        builder = StateGraph(State)
        visualizer = self.visualizer

        nodes = [
            ("guardrails_system", self.guardrails_system),
            ("generate_cypher", self.generate_cypher),
            ("retrieve", self.retrieve),
            ("grade_context", self.grade_context),
            ("search_after_grading", self.search_after_grading),
            ("return_none", self.return_none),
        ]

        for node_name, node_func in nodes:
            builder.add_node(node_name, node_func)
            visualizer.add_node(node_name)

        builder.add_edge(START, "guardrails_system")
        visualizer.add_edge(START, "guardrails_system")

        guardrail_edges = {
            "generate_cypher": "generate_cypher",
            "end": "return_none",
        }

        builder.add_conditional_edges(
            "guardrails_system", lambda state: state["next_node"], guardrail_edges
        )
        visualizer.add_conditional_edges("guardrails_system", guardrail_edges)

        cypher_edges = {
            "retrieve": "retrieve",
            "end": END,
        }

        builder.add_conditional_edges(
            "generate_cypher", lambda state: state["next_node"], cypher_edges
        )
        visualizer.add_conditional_edges("generate_cypher", cypher_edges)

        builder.add_edge("return_none", END)
        visualizer.add_edge("return_none", END)

        builder.add_edge("retrieve", "grade_context")
        visualizer.add_edge("retrieve", "grade_context")

        grading_edges = {
            "search_after_grading": "search_after_grading",
            "end": END,
        }

        builder.add_conditional_edges(
            "grade_context", lambda state: state["next_node"], grading_edges
        )
        visualizer.add_conditional_edges("grade_context", grading_edges)

        builder.add_edge("search_after_grading", "grade_context")
        visualizer.add_edge("search_after_grading", "grade_context")

        return builder.compile()

    @staticmethod
    def _build_cypher_prompt_payload(user_question: str, schema: str) -> dict[str, str]:
        """Provide both natural-language and canonical search forms to the LLM."""
        return {
            "user_question": user_question,
            "normalized_question": normalize_search_text(user_question),
            "schema": schema,
        }

    def generate_cypher(self, state: State):
        """
        Generate CYPHER query from user question using database schema.
        Uses accurate model with provider fallback (OpenAI → DeepSeek → Google).

        Args:
            state: Current pipeline state

        Returns:
            Updated state with generated CYPHER query
        """
        schema = self.schema
        logger.debug(
            "Schema used for Cypher generation (%d chars):\n%s", len(schema), schema or "(empty)"
        )

        if not schema:
            # Neo4j has just told us it holds nothing. Two model calls cannot retrieve from an
            # empty graph, and a query written against no schema is exactly the kind of
            # invented-property answer the abstention path exists to prevent.
            logger.warning("Neo4j reports an empty schema; abstaining instead of generating Cypher")
            return {
                "generated_cypher": None,
                "context": [],
                "retrieval_strategy": RetrievalStrategy.EMPTY.value,
                "next_node": "end",
            }

        chain = self.generate_cypher_template | self.cypher_llm | StrOutputParser()
        generated_cypher = self._invoke_with_single_provider_retry(
            chain=chain,
            payload=self._build_cypher_prompt_payload(state["user_question"], schema),
            invoke_config=self._get_invoke_config(
                trace_id=state.get("trace_id"),
                tags=["knowledge_graph", "generated_cypher"],
                run_name="Generate Cypher",
                handler=state.get("callback_handler"),
                session_id=state.get("session_id"),
            ),
            operation_name="Generate Cypher",
        )

        return {"generated_cypher": generated_cypher, "next_node": "retrieve"}

    def retrieve(self, state: State):
        """
        Execute CYPHER query against Neo4j database and retrieve results.

        A query that executes but matches nothing is escalated rather than reported as missing
        data, because the two most common Text2Cypher mistakes both surface as zero rows. See
        _escalate_empty_retrieval.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with retrieved context and the strategy that produced it
        """
        cypher_query = state.get("generated_cypher", "")
        user_question = state.get("user_question") or ""

        try:
            cypher_query = strip_code_fences(cypher_query)
            cypher_query = normalize_cypher_string_literals(
                cypher_query,
                normalizer=fold_diacritics,
            )
            cypher_query = ensure_case_insensitive_fuzzy_matching(cypher_query)
            validate_read_only(cypher_query)
            cypher_query = ensure_limit(cypher_query, self.max_results)

            response = self._read_query(cypher_query)
            if _matched_anything(response):
                return {
                    "context": response,
                    "generated_cypher": cypher_query,
                    "retrieval_strategy": RetrievalStrategy.PRIMARY.value,
                    "rows_truncated": self._rows_were_capped(response, cypher_query),
                }

            return self._escalate_empty_retrieval(cypher_query, user_question)

        except UnsafeCypherQueryError as exc:
            logger.warning("Cypher blocked: %s", exc)
            raise KnowledgeGraphQueryError(f"blocked: {exc}", cypher=cypher_query) from exc

        except KnowledgeGraphUnavailableError:
            raise

        except NEO4J_INFRASTRUCTURE_EXCEPTIONS as exc:
            logger.error("Neo4j could not be consulted: %s", exc)
            raise KnowledgeGraphUnavailableError(str(exc)) from exc

        except Exception as exc:
            if _is_neo4j_query_timeout(exc):
                logger.error("Neo4j did not finish the query in time: %s", exc)
                raise KnowledgeGraphUnavailableError(str(exc)) from exc

            if _is_neo4j_statement_error(exc):
                return self._recover_from_statement_error(cypher_query, user_question, exc)

            logger.warning("Cypher execution failed: %s", exc)
            raise KnowledgeGraphQueryError(str(exc), cypher=cypher_query) from exc

    def _recover_from_statement_error(
        self, cypher_query: str, user_question: str, error: Exception
    ) -> dict[str, Any]:
        if not self._fallback_search_is_possible(user_question):
            logger.warning("Cypher rejected by Neo4j and no search possible: %s", error)
            raise KnowledgeGraphQueryError(str(error), cypher=cypher_query) from error

        logger.warning("Cypher rejected by Neo4j; searching every label instead: %s", error)
        fallback = self._search_every_label(user_question)
        if fallback is None:
            logger.info("Label-agnostic search after a rejected statement found nothing")
            return {
                "context": [],
                "generated_cypher": cypher_query,
                "retrieval_strategy": RetrievalStrategy.EMPTY.value,
            }

        return {
            **fallback,
            "retrieval_strategy": RetrievalStrategy.LABEL_AGNOSTIC_AFTER_ERROR.value,
        }

    def _rows_were_capped(self, rows: List[Dict[str, Any]], cypher_query: str) -> bool:
        """
        Report whether the result filled its row cap, so the graph may hold more of the answer.

        A count taken from the rows themselves is then a lower bound and not a total: "Ile jest
        kryteriów w kategorii Działalność dydaktyczna?" ran a row-per-criterion query over a
        category holding 11 and handed the answering model the 10 the cap allowed (issue #107).
        The rows do not say they were cut, so nothing downstream could tell that count from a
        real one.

        A full result is only evidence that it *might* have been cut - a category with exactly
        10 criteria fills the cap and lost nothing. That is the right way round: the flag makes
        the answer hedge a count it cannot verify, and a hedge on a complete list is cheaper
        than a wrong total.

        Args:
            rows: What the query returned
            cypher_query: The query as it was executed, after ``ensure_limit``

        Returns:
            True when the query returned as many rows as it was allowed to
        """
        cap = trailing_limit(cypher_query)
        if cap is None:
            # The full-text search caps its nodes before collecting neighbours, so its LIMIT is
            # not the trailing clause; what bounds its rows is the same max_results.
            cap = self.max_results
        return len(rows) >= cap

    def _fallback_search_is_possible(self, user_question: str) -> bool:
        """Report whether the label-agnostic search has something it could run for a question."""
        if not self.enable_fallback_search or not user_question:
            return False
        phrases = extract_search_phrases(user_question)
        return bool(phrases) and bool(build_lucene_query(phrases))

    def _escalate_empty_retrieval(self, executed_cypher: str, user_question: str) -> Dict[str, Any]:
        """
        Retry a query that executed successfully but matched no rows.

        Escalation runs only after a successful execution, so a blocked or failing query is
        still reported as such. Two retries are attempted in order:

        1. drop fuzzy predicates whose literal is copied question text, keeping the traversal
           the model wrote but without the filter that could never match;
        2. search every label for the question's noun phrases, which recovers an answer stored
           under a label the model did not pick.

        Args:
            executed_cypher: The query that ran and returned no rows
            user_question: The question the query was generated from

        Returns:
            Updated state with whatever context the retries recovered
        """
        repaired, dropped_literals = strip_question_literal_filters(executed_cypher, user_question)
        if dropped_literals:
            logger.debug("Retrieval retry dropped question literals: %s", dropped_literals)
            response = self._run_recovery_query(repaired, "question-literal repair")
            if _matched_anything(response):
                return {
                    "context": response,
                    "generated_cypher": repaired,
                    "retrieval_strategy": RetrievalStrategy.REPAIRED_LITERALS.value,
                    "rows_truncated": self._rows_were_capped(response, repaired),
                }

        fallback = self._search_every_label(user_question)
        if fallback is not None:
            return fallback

        return {
            "context": [],
            "generated_cypher": executed_cypher,
            "retrieval_strategy": RetrievalStrategy.EMPTY.value,
        }

    def _read_query(
        self, cypher_query: str, params: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Run a retrieval query in a transaction the database will not let write.

        Every query that reaches here was written by the Cypher model or is aimed at the
        question the model was given, and ``validate_read_only`` has already refused the write
        keywords. This is the second line: the guardrail is a regex over text, while the access
        mode is the database's own rule, and only the database knows what a statement really
        does once a procedure is involved (issue #85).

        ``Neo4jGraph.query`` exposes no routing argument, so the access mode goes in through
        ``session_params``. That takes the driver's implicit-transaction path, which trades the
        managed transaction's retry on a transient error for the explicit read mode. Against the
        single instance this deploys on the retry has nothing to recover - a leader switch is
        what it is for - and a retrieval failure already escalates through ``retrieve`` rather
        than being repeated here. ``ping_database`` and the schema probe keep the managed path,
        so the outage bound measured for the health signal is unchanged.

        Args:
            cypher_query: Query to execute
            params: Optional Cypher parameters

        Returns:
            The rows the query returned
        """
        # A fresh dict per call: Neo4jGraph.query writes the database name into whatever it is
        # handed.
        session_params = {"default_access_mode": READ_ACCESS}
        if params is None:
            return self.database.query(cypher_query, session_params=session_params)
        return self.database.query(cypher_query, params=params, session_params=session_params)

    def _run_recovery_query(
        self, cypher_query: str, description: str, params: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Run a retry query, treating a rejected or failed query as "recovered nothing".

        A retry exists to improve on an empty result, so a query the database refuses - a bad
        statement, a missing index - must not turn that empty result into an error. Only an
        unreachable graph or a timed-out retry query propagates.

        Args:
            cypher_query: Query to execute
            description: Retry name used in the warning log
            params: Optional Cypher parameters

        Returns:
            Retrieved rows, or an empty list when the retry was rejected or failed
        """
        try:
            validate_read_only(cypher_query, allowed_procedures=ALLOWED_RETRIEVAL_PROCEDURES)
            return self._read_query(cypher_query, params)
        except NEO4J_INFRASTRUCTURE_EXCEPTIONS as exc:
            raise KnowledgeGraphUnavailableError(str(exc)) from exc

        except Exception as exc:
            if _is_neo4j_query_timeout(exc):
                raise KnowledgeGraphUnavailableError(str(exc)) from exc
            logger.warning("Retrieval retry (%s) failed: %s", description, exc)
            return []

    def ensure_fulltext_index(self) -> bool:
        """
        Create or refresh the full-text index the label-agnostic search reads.

        The index has to name its labels, so it is built from the labels the database actually
        holds and rebuilt when that set changes. Ingestion can introduce a label between
        restarts; the search re-checks the index when a lookup fails, so a new label is picked
        up without waiting for a redeploy.

        Raises:
            KnowledgeGraphUnavailableError: The graph could not be consulted at all, so False
                would claim the index is unavailable when the whole database is. Whether that
                is fatal is the caller's call; __init__ is the one place that tolerates it.

        Returns:
            True when the index exists and covers the current labels
        """
        try:
            labels = sorted(
                row["label"]
                for row in self.database.query("CALL db.labels() YIELD label RETURN label")
                if row["label"] not in FULLTEXT_EXCLUDED_LABELS
            )
        except NEO4J_INFRASTRUCTURE_EXCEPTIONS as exc:
            raise KnowledgeGraphUnavailableError(str(exc)) from exc
        except Exception as exc:
            if _is_neo4j_query_timeout(exc):
                raise KnowledgeGraphUnavailableError(str(exc)) from exc
            logger.warning("Could not read graph labels for the full-text index: %s", exc)
            return False

        if not labels:
            return False

        label_set = frozenset(labels)
        if self._known_labels is not None and label_set != self._known_labels:
            # New labels carry new properties, and the Cypher model can only use what the
            # schema shows it. Whoever noticed the label set moved is the cheapest place to
            # notice the schema did too.
            logger.info("Graph labels changed; dropping the cached Neo4j schema")
            self.invalidate_schema_cache()
        self._known_labels = label_set

        try:
            existing = self.database.query(
                SHOW_FULLTEXT_INDEX_CYPHER, params={"index_name": FULLTEXT_INDEX_NAME}
            )
            if existing and sorted(existing[0].get("labels") or []) == labels:
                return True

            if existing:
                logger.info("Graph labels changed; rebuilding the %s index", FULLTEXT_INDEX_NAME)
                self.database.query(f"DROP INDEX {FULLTEXT_INDEX_NAME} IF EXISTS")

            label_spec = "|".join(f"`{label}`" for label in labels)
            self.database.query(
                f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS "
                f"FOR (n:{label_spec}) ON EACH [n.title, n.context]"
            )
            logger.info("Full-text index %s covers %d labels", FULLTEXT_INDEX_NAME, len(labels))
            return True
        except NEO4J_INFRASTRUCTURE_EXCEPTIONS as exc:
            raise KnowledgeGraphUnavailableError(str(exc)) from exc
        except Exception as exc:
            if _is_neo4j_query_timeout(exc):
                raise KnowledgeGraphUnavailableError(str(exc)) from exc
            logger.warning("Could not create the %s full-text index: %s", FULLTEXT_INDEX_NAME, exc)
            return False

    def _search_every_label(self, user_question: str) -> Dict[str, Any] | None:
        """
        Search every label's titles for the question's noun phrases, ranked by relevance.

        Rows scoring below the configured threshold are dropped here rather than being passed on
        as weak candidates, so the decision to abstain is made on retrieval evidence.

        Args:
            user_question: User's natural language question

        Returns:
            Updated state with the recovered context, or None when the search is disabled,
            has nothing to search for, or found nothing above the score threshold
        """
        if not self.enable_fallback_search:
            return None

        phrases = extract_search_phrases(user_question)
        if not phrases:
            return None

        lucene_query = build_lucene_query(phrases)
        if not lucene_query:
            return None

        cypher_query = FALLBACK_SEARCH_CYPHER.format(max_nodes=self.max_results)
        params = {
            "index_name": FULLTEXT_INDEX_NAME,
            "lucene_query": lucene_query,
            "min_score": self.fallback_min_score,
        }
        logger.debug("Retrieval retry with label-agnostic search: %s", lucene_query)

        response = self._run_recovery_query(cypher_query, "label-agnostic search", params=params)
        if not response and self.ensure_fulltext_index():
            # The index may be missing entirely, or stale after ingestion added a label.
            response = self._run_recovery_query(
                cypher_query, "label-agnostic search (reindexed)", params=params
            )
        if not response:
            return None

        return {
            "context": response,
            "generated_cypher": cypher_query,
            "retrieval_strategy": RetrievalStrategy.LABEL_AGNOSTIC_PHRASES.value,
            "rows_truncated": self._rows_were_capped(response, cypher_query),
        }

    def guardrails_system(self, state: State):
        """
        Decide whether to use graph retrieval or general LLM knowledge.
        Uses fast model with provider fallback (OpenAI → DeepSeek → Google).
        Expects JSON response with decision field ("generate" or "end").


        Args:
            state: Current pipeline state

        Returns:
            Updated state with next node decision
        """
        guardrails_chain = self.guard_rails_template | self.fast_llm | StrOutputParser()
        guardrail_output = self._invoke_with_single_provider_retry(
            chain=guardrails_chain,
            payload={"user_question": state["user_question"]},
            invoke_config=self._get_invoke_config(
                trace_id=state.get("trace_id"),
                tags=["knowledge_graph", "guardrails"],
                run_name="Guardrails",
                handler=state.get("callback_handler"),
                session_id=state.get("session_id"),
            ),
            operation_name="Guardrails",
        )
        guardrail_result = self._parse_guardrail_output(guardrail_output)

        return {
            "next_node": guardrail_result["decision"],
            "guardrail_decision": guardrail_result["decision"],
        }

    def return_none(self, state: State):
        """
        Report that the question was routed away from graph retrieval.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with the off-topic answer and no context
        """
        return {
            "answer": OFF_TOPIC_MESSAGE,
            "context": [],
            "generated_cypher": None,
            "retrieval_strategy": RetrievalStrategy.EMPTY.value,
        }

    @staticmethod
    def _format_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Turn terminal graph state into the tool response.

        An empty retrieval is reported as an explicit "no data" sentence rather than an empty
        JSON list, so the answering model has something unambiguous to abstain on instead of a
        gap it can fill from its own knowledge.

        Args:
            result: Terminal state of the LangGraph run

        Returns:
            Dictionary with the answer and retrieval metadata
        """
        context_data = result.get("context") or []
        strategy = result.get("retrieval_strategy")
        metadata = {
            "guardrail_decision": result.get("guardrail_decision"),
            "cypher_query": result.get("generated_cypher"),
            "retrieval_strategy": strategy,
            "context_graded": result.get("context_graded"),
            "context": context_data,
        }

        if result.get("answer") == OFF_TOPIC_MESSAGE:
            return {
                "answer": OFF_TOPIC_MESSAGE,
                "metadata": {**metadata, "cypher_query": None, "context": []},
            }

        if not context_data:
            return {"answer": NO_GRAPH_DATA_MESSAGE, "metadata": metadata}

        # The strategy travels with the rows: how they were found is what says whether they are
        # an answer or a candidate, and the answering model cannot tell the two apart otherwise.
        payload = {
            "retrieval_strategy": strategy,
            "context_graded": bool(result.get("context_graded")),
            # The rows cannot say they were cut at the cap, and a count read off a cut list is
            # a wrong total rather than a missing one (issue #107).
            "rows_truncated": bool(result.get("rows_truncated")),
            "rows": context_data,
        }
        # default=str: Neo4j hands back its own temporal and spatial types (neo4j.time.DateTime
        # for any `datetime()` property, which the pipeline's provenance nodes all carry), and
        # json has no encoder for them. Without this, a row that merely touches one turns the
        # whole call into a ToolError. The grader's row rendering does the same.
        return {
            "answer": json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            "metadata": metadata,
        }

    def invoke(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute the RAG pipeline with user message.

        Args:
            message: User's question/input
            session_id: Session identifier for tracking

        Returns:
            Dictionary with graph context, or an explicit off-topic/no-data answer
        """
        result = self.graph.invoke({"user_question": message})

        return self._format_result(result)

    async def ainvoke(
        self,
        message: str,
        session_id: str = "default",
        trace_id: str = "default",
        callback_handler: CallbackHandler = None,
    ) -> Dict[str, Any]:
        """
        Async version of invoke for better performance in concurrent scenarios.

        Args:
            message: User's question/input
            session_id: Session identifier for tracking
            trace_id: Trace identifier for this single chat turn
            callback_handler: Optional Langfuse CallbackHandler scoped to this request

        Returns:
            Dictionary with graph context, or an explicit off-topic/no-data answer
        """
        try:
            result = await asyncio.wait_for(
                self.graph.ainvoke(
                    {
                        "user_question": message,
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "callback_handler": callback_handler,
                    }
                ),
                timeout=self.graph_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(GRAPH_PIPELINE_TIMEOUT_MESSAGE) from exc

        return self._format_result(result)
