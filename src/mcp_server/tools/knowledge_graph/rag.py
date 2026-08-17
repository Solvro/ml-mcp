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
from ....config.messages import GRAPH_PIPELINE_TIMEOUT_MESSAGE
from ....config.timeouts import get_graph_timeout_seconds, get_llm_timeout_seconds
from ....text_normalization import (
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


class LLMProvider(Enum):
    """Available LLM providers for the runtime fallback chain."""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"


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

    def _build_processing_graph(self):
        """Construct the state machine graph for the RAG pipeline."""
        builder = StateGraph(State)
        visualizer = self.visualizer

        nodes = [
            ("guardrails_system", self.guardrails_system),
            ("generate_cypher", self.generate_cypher),
            ("retrieve", self.retrieve),
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

        builder.add_edge("retrieve", END)
        visualizer.add_edge("retrieve", END)

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
        If query fails, return empty context and use general knowledge.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with retrieved context
        """
        cypher_query = state.get("generated_cypher", "")

        try:
            cypher_query = strip_code_fences(cypher_query)
            cypher_query = normalize_cypher_string_literals(
                cypher_query,
                normalizer=fold_diacritics,
            )
            validate_read_only(cypher_query)
            cypher_query = ensure_limit(cypher_query, self.max_results)

            response = self.database.query(cypher_query)

            return {"context": response, "generated_cypher": cypher_query}

        except UnsafeCypherQueryError as e:
            error_msg = f"Blocked unsafe Cypher: {e}"
            if self.enable_debug:
                print(f"[Cypher Blocked] {error_msg}")
            return {"context": [], "generated_cypher": error_msg}

        except Exception as e:
            error_msg = str(e)

            if self.enable_debug:
                print(f"[Query Error] {error_msg}")

            return {"context": [], "generated_cypher": f"Query failed: {error_msg}"}

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
        next_node = guardrail_result["decision"]
        return {
            "next_node": next_node,
            "guardrail_decision": guardrail_result["decision"],
        }
        guardrail_result = self._parse_guardrail_output(guardrail_output)
        next_node = guardrail_result["decision"]

        return {
            "next_node": next_node,
            "guardrail_decision": guardrail_result["decision"],
        }

    def return_none(self, state: State):
        """
        Return 'W bazie danych nie ma informacji' when question is not
        related to university studies.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with answer set to None
        """
        return {
            "answer": "W bazie danych nie ma informacji",
            "context": [],
            "generated_cypher": None,
        }

    def invoke(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute the RAG pipeline with user message.

        Args:
            message: User's question/input
            session_id: Session identifier for tracking

        Returns:
            Dictionary with context from graph or "W bazie danych nie ma informacji"
        """
        result = self.graph.invoke({"user_question": message})

        if result.get("answer") == "W bazie danych nie ma informacji":
            return {
                "answer": "W bazie danych nie ma informacji",
                "metadata": {
                    "guardrail_decision": result.get("guardrail_decision"),
                    "cypher_query": None,
                    "context": [],
                },
            }

        context_data = result.get("context", [])
        context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

        return {
            "answer": context_json,
            "metadata": {
                "guardrail_decision": result.get("guardrail_decision"),
                "cypher_query": result.get("generated_cypher"),
                "context": context_data,
            },
        }

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
            Dictionary with context from graph or "W bazie danych nie ma informacji"
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

        if result.get("answer") == "W bazie danych nie ma informacji":
            return {
                "answer": "W bazie danych nie ma informacji",
                "metadata": {
                    "guardrail_decision": result.get("guardrail_decision"),
                    "cypher_query": None,
                    "context": [],
                },
            }

        context_data = result.get("context", [])
        context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

        return {
            "answer": context_json,
            "metadata": {
                "guardrail_decision": result.get("guardrail_decision"),
                "cypher_query": result.get("generated_cypher"),
                "context": context_data,
            },
        }
