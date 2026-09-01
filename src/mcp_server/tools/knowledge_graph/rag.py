import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Dict, List

from google.genai.errors import ServerError as GoogleServerError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_openai.chat_models.base import BaseChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from ....config.config import get_config
from ....config.messages import (
    GRAPH_PIPELINE_TIMEOUT_MESSAGE,
    NO_GRAPH_DATA_MESSAGE,
    OFF_TOPIC_MESSAGE,
)
from ....config.timeouts import get_graph_timeout_seconds, get_llm_timeout_seconds
from ....text_normalization import (
    ensure_case_insensitive_fuzzy_matching,
    fold_diacritics,
    normalize_cypher_string_literals,
    normalize_search_text,
)
from .cypher_guardrails import (
    UnsafeCypherQueryError,
    ensure_limit,
    strip_code_fences,
    validate_read_only,
)
from .graph_visualizer import GraphVisualizer
from .question_analysis import (
    build_lucene_query,
    extract_search_phrases,
    strip_question_literal_filters,
)
from .state import State

logger = logging.getLogger(__name__)

PROVIDER_FALLBACK_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    GoogleServerError,
)

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
FULLTEXT_EXCLUDED_LABELS = frozenset({"ProcessedDocument", "PipelineRun"})
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
    GRADED_OUT = "graded_out"
    EMPTY = "empty"


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
    ):
        """
        Initialize RAG system with API keys and database credentials.

        Args:
            api_key: OpenAI/DeepSeek API key
            neo4j_url: Neo4j database connection URL
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            enable_debug: Enable debug output (default: False)
            max_results: Maximum number of results from Neo4j (default: 5)
            llm_timeout_sec: Per-call HTTP timeout for each LLM client
            graph_timeout_sec: Wall-clock budget for the whole RAG run
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
        self.max_results = max_results if max_results is not None else config.rag.max_results
        self.enable_fallback_search = config.rag.enable_fallback_search
        self.fallback_min_score = config.rag.fallback_min_score

        self.fast_llm = self._build_llm_with_fallback(use_accurate=False)
        self.cypher_llm = self._build_llm_with_fallback(use_accurate=True)

        self._initialize_prompt_templates()

        self.database = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database=config.database.name,
            enhanced_schema=True,
        )

        self._cached_schema = None

        if self.enable_fallback_search:
            self.ensure_fulltext_index()

        self.visualizer = GraphVisualizer()
        self.graph = self._build_processing_graph()

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

    @property
    def schema(self):
        """Cached database schema to avoid repeated fetches.

        Only caches when a non-empty schema is found so that a temporary empty
        database at startup does not permanently poison the cache.
        """
        if not self._cached_schema:
            db_schema = self.database.get_schema

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

            is_empty = not stripped or headers_only

            if not is_empty:
                self._cached_schema = db_schema
                print(f"[Schema] fetched {len(db_schema)} chars from Neo4j")
            else:
                print("[Schema] database is empty — schema will be re-fetched on next call")

        return self._cached_schema or ""

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
            input_variables=["user_question", "candidates"],
            template=config.prompts.context_grader,
        )

    def _parse_guardrail_output(self, raw_output: str) -> Dict[str, str]:
        """Parse guardrail JSON and normalize the decision with a safe fallback."""
        cleaned_output = strip_code_fences(raw_output)

        try:
            start = cleaned_output.index("{")
            payload, _ = json.JSONDecoder().raw_decode(cleaned_output[start:])
        except (ValueError, json.JSONDecodeError) as exc:
            if self.enable_debug:
                print(f"[Guardrails Parse Error] invalid JSON: {exc}; raw={raw_output}")
            return {"decision": "end"}

        decision = str(payload.get("decision", "")).strip().lower()
        normalized_decision = GUARDRAIL_DECISION_ALIASES.get(decision)

        if normalized_decision is None:
            if self.enable_debug:
                print(f"[Guardrails Parse Error] invalid decision={decision!r}; raw={raw_output}")
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

    def _parse_grader_output(self, raw_output: str, row_count: int) -> List[int] | None:
        """
        Read the row numbers the grader kept.

        Args:
            raw_output: Raw grader reply
            row_count: How many rows the grader was shown

        Returns:
            Zero-based indices of the rows to keep, or None when the reply is unusable
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

        return kept

    def grade_context(self, state: State):
        """
        Drop retrieved rows that do not answer the question.

        Whether to answer is decided here, on retrieval evidence, rather than left to the
        answering model: rows recovered by a widened search are candidates, not an answer, and a
        row that only shares a word with the question is what produces a confident wrong answer.

        A query that ran as the model wrote it is trusted and skips grading. A grader that fails
        or replies with nonsense leaves the rows untouched - a model outage must not be
        indistinguishable from an empty graph.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with the rows that survived grading
        """
        context = state.get("context") or []
        strategy = state.get("retrieval_strategy")

        if not context or strategy == RetrievalStrategy.PRIMARY.value:
            return {"context_graded": False}

        grader_chain = self.context_grader_template | self.fast_llm | StrOutputParser()

        try:
            grader_output = grader_chain.invoke(
                {
                    "user_question": state["user_question"],
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
            return {"context_graded": False}

        kept = self._parse_grader_output(grader_output, len(context))
        if kept is None:
            return {"context_graded": False}

        graded_context = [context[index] for index in kept]

        if self.enable_debug:
            print(f"[Context Grader] kept {len(graded_context)} of {len(context)} row(s)")

        if not graded_context:
            logger.info("Context grader rejected all %d retrieved row(s)", len(context))
            return {
                "context": [],
                "context_graded": True,
                "retrieval_strategy": RetrievalStrategy.GRADED_OUT.value,
            }

        return {"context": graded_context, "context_graded": True}

    def _build_processing_graph(self):
        """Construct the state machine graph for the RAG pipeline."""
        builder = StateGraph(State)
        visualizer = self.visualizer

        nodes = [
            ("guardrails_system", self.guardrails_system),
            ("generate_cypher", self.generate_cypher),
            ("retrieve", self.retrieve),
            ("grade_context", self.grade_context),
            ("return_none", self.return_none),
        ]

        if self.enable_debug:
            nodes.append(("debug_print", self.debug_print))

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

        builder.add_edge("generate_cypher", "retrieve")
        visualizer.add_edge("generate_cypher", "retrieve")

        builder.add_edge("return_none", END)
        visualizer.add_edge("return_none", END)

        builder.add_edge("retrieve", "grade_context")
        visualizer.add_edge("retrieve", "grade_context")

        builder.add_edge("grade_context", END)
        visualizer.add_edge("grade_context", END)

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
        print(f"[Schema used for Cypher generation] ({len(schema)} chars):\n{schema or '(empty)'}")

        chain = self.generate_cypher_template | self.cypher_llm | StrOutputParser()
        generated_cypher = chain.invoke(
            self._build_cypher_prompt_payload(state["user_question"], schema),
            config=self._get_invoke_config(
                trace_id=state.get("trace_id"),
                tags=["knowledge_graph", "generated_cypher"],
                run_name="Generate Cypher",
                handler=state.get("callback_handler"),
                session_id=state.get("session_id"),
            ),
        )

        return {"generated_cypher": generated_cypher}

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

            response = self.database.query(cypher_query)
            if response:
                return {
                    "context": response,
                    "generated_cypher": cypher_query,
                    "retrieval_strategy": RetrievalStrategy.PRIMARY.value,
                }

            return self._escalate_empty_retrieval(cypher_query, user_question)

        except UnsafeCypherQueryError as e:
            error_msg = f"Blocked unsafe Cypher: {e}"
            if self.enable_debug:
                print(f"[Cypher Blocked] {error_msg}")
            return {
                "context": [],
                "generated_cypher": error_msg,
                "retrieval_strategy": RetrievalStrategy.EMPTY.value,
            }

        except Exception as e:
            error_msg = str(e)

            if self.enable_debug:
                print(f"[Query Error] {error_msg}")

            return {
                "context": [],
                "generated_cypher": f"Query failed: {error_msg}",
                "retrieval_strategy": RetrievalStrategy.EMPTY.value,
            }

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
            if self.enable_debug:
                print(f"[Retrieval Retry] dropped question literals: {dropped_literals}")
            response = self._run_recovery_query(repaired, "question-literal repair")
            if response:
                return {
                    "context": response,
                    "generated_cypher": repaired,
                    "retrieval_strategy": RetrievalStrategy.REPAIRED_LITERALS.value,
                }

        fallback = self._search_every_label(user_question)
        if fallback is not None:
            return fallback

        return {
            "context": [],
            "generated_cypher": executed_cypher,
            "retrieval_strategy": RetrievalStrategy.EMPTY.value,
        }

    def _run_recovery_query(
        self, cypher_query: str, description: str, params: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Run a retry query, treating any failure as "recovered nothing".

        A retry exists to improve on an empty result, so it must never turn one into an error.

        Args:
            cypher_query: Query to execute
            description: Retry name used in the warning log
            params: Optional Cypher parameters

        Returns:
            Retrieved rows, or an empty list when the retry was rejected or failed
        """
        try:
            validate_read_only(cypher_query, allowed_procedures=ALLOWED_RETRIEVAL_PROCEDURES)
            if params is None:
                return self.database.query(cypher_query)
            return self.database.query(cypher_query, params=params)
        except Exception as exc:
            logger.warning("Retrieval retry (%s) failed: %s", description, exc)
            return []

    def ensure_fulltext_index(self) -> bool:
        """
        Create or refresh the full-text index the label-agnostic search reads.

        The index has to name its labels, so it is built from the labels the database actually
        holds and rebuilt when that set changes. Ingestion can introduce a label between
        restarts; the search re-checks the index when a lookup fails, so a new label is picked
        up without waiting for a redeploy.

        Returns:
            True when the index exists and covers the current labels
        """
        try:
            labels = sorted(
                row["label"]
                for row in self.database.query("CALL db.labels() YIELD label RETURN label")
                if row["label"] not in FULLTEXT_EXCLUDED_LABELS
            )
        except Exception as exc:
            logger.warning("Could not read graph labels for the full-text index: %s", exc)
            return False

        if not labels:
            return False

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
        except Exception as exc:
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
        if self.enable_debug:
            print(f"[Retrieval Retry] label-agnostic search: {lucene_query}")

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

        guardrail_output = guardrails_chain.invoke(
            {"user_question": state["user_question"]},
            config=self._get_invoke_config(
                trace_id=state.get("trace_id"),
                tags=["knowledge_graph", "guardrails"],
                run_name="Guardrails",
                handler=state.get("callback_handler"),
                session_id=state.get("session_id"),
            ),
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
            "rows": context_data,
        }
        return {
            "answer": json.dumps(payload, ensure_ascii=False, indent=2),
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
