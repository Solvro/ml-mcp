import asyncio
import json
import os
import re
from enum import Enum
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_openai.chat_models.base import BaseChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random

try:
    from openai import APIError, APITimeoutError, RateLimitError
except ImportError:
    APIError = Exception
    APITimeoutError = TimeoutError
    RateLimitError = Exception

from ....config.config import get_config
from .graph_visualizer import GraphVisualizer
from .state import State

FORBIDDEN_CLAUSES = frozenset({"CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP"})
GUARDRAIL_DECISION_ALIASES = {
    "generate": "generate_cypher",
    "generate_cypher": "generate_cypher",
    "end": "end",
}

RETRYABLE_EXCEPTIONS = (APIError, APITimeoutError, RateLimitError, TimeoutError)


class LLMProvider(Enum):
    """Available LLM providers for fallback chain."""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"


class RAG:
    """Retrieval-Augmented Generation system with Neo4j graph database backend."""

    GRAPH_PIPELINE_TIMEOUT_MESSAGE = (
        "The knowledge graph pipeline exceeded the maximum allowed wait time."
    )
    LLM_CALL_TIMEOUT_MESSAGE = "The language model request exceeded the maximum allowed wait time."

    def __init__(
        self,
        api_key: str,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        enable_debug: bool = None,
        max_results: int = None,
        llm_timeout_sec: float = 30.0,
        graph_timeout_sec: float = 90.0,
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
            graph_timeout_sec: Max wall time for the full async graph in ainvoke()
        """
        config = get_config()

        self.graph_timeout_sec = graph_timeout_sec
        self.llm_timeout_sec = llm_timeout_sec
        self.api_key = api_key
        self.config = config
        self.enable_debug = enable_debug if enable_debug is not None else config.rag.enable_debug
        self.max_results = max_results if max_results is not None else config.rag.max_results

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
        handler: CallbackHandler = None,
    ) -> dict:
        """Build invoke config with optional callbacks."""
        config = {
            "metadata": {
                "langfuse_session_id": trace_id,
                "langfuse_tags": tags,
                "run_name": run_name,
            },
        }
        if handler is not None:
            config["callbacks"] = [handler]
        return config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10) + wait_random(0, 1),
    )
    async def _ainvoke_with_retry(self, chain: Any, inputs: Dict[str, Any], config: dict) -> Any:
        try:
            return await chain.ainvoke(inputs, config=config)
        except TimeoutError as exc:
            raise TimeoutError(RAG.LLM_CALL_TIMEOUT_MESSAGE) from exc

    def _get_configured_providers(self) -> List[LLMProvider]:
        """Read provider fallback order from config and filter by available API keys."""
        config = self.config
        fallback_order = getattr(config.llm, "provider_fallback_order", [])
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        google_key = os.environ.get("GOOGLE_API_KEY", "").strip()

        providers = []
        for provider_name in fallback_order:
            provider_name_lower = str(provider_name).lower().strip()
            if provider_name_lower == "openai":
                if openai_key:
                    providers.append(LLMProvider.OPENAI)
            elif provider_name_lower == "deepseek":
                if deepseek_key:
                    providers.append(LLMProvider.DEEPSEEK)
            elif provider_name_lower == "google":
                if google_key:
                    providers.append(LLMProvider.GOOGLE)

        return providers if providers else [LLMProvider.OPENAI]  # Fallback to OpenAI

    def _get_model(self, provider: LLMProvider):
        """Create LLM client for the specified provider."""
        if provider == LLMProvider.OPENAI:
            return BaseChatOpenAI(
                model=self.config.llm.fast_model.name,
                api_key=self.api_key,
                temperature=self.config.llm.fast_model.temperature,
                timeout=self.llm_timeout_sec,
            )
        elif provider == LLMProvider.DEEPSEEK:
            # TODO: Implement DeepSeek client initialization when DEEPSEEK_API_KEY is available
            raise NotImplementedError("DeepSeek provider not yet implemented")
        elif provider == LLMProvider.GOOGLE:
            # TODO: Implement Google Gemini client initialization when GOOGLE_API_KEY is available
            raise NotImplementedError("Google provider not yet implemented")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _ainvoke_with_provider_fallback(
        self,
        template: PromptTemplate,
        inputs: Dict[str, Any],
        config: dict,
    ) -> str:
        """
        Invoke LLM chain with provider fallback (OpenAI → DeepSeek → Google).
        Each provider gets up to 3 retries with exponential backoff.
        Only retries on network/API errors; other errors fail fast.

        Args:
            template: PromptTemplate to use
            inputs: Input dict for the chain
            config: Invoke config with callbacks/metadata

        Returns:
            LLM output string

        Raises:
            RuntimeError: When all configured providers are exhausted
        """
        providers = self._get_configured_providers()
        last_error = None

        for provider in providers:
            try:
                if self.enable_debug:
                    print(f"[Fallback] Trying provider: {provider.value}")

                model = self._get_model(provider)
                chain = template | model | StrOutputParser()

                result = await self._ainvoke_with_retry(chain, inputs, config)

                if self.enable_debug:
                    print(f"[Fallback] Success with {provider.value}")
                return result

            except RETRYABLE_EXCEPTIONS as exc:
                # Network/API errors → try next provider
                last_error = exc
                if self.enable_debug:
                    print(f"[Fallback] {provider.value} failed (retryable): {exc}; trying next...")
                continue

            except Exception as exc:
                # Other errors (config, not-implemented, etc) → fail fast
                if self.enable_debug:
                    print(f"[Fallback] {provider.value} failed (non-retryable): {exc}")
                raise

        # All providers exhausted
        raise RuntimeError(
            f"All LLM providers exhausted. Last error: {last_error}. "
            f"Configured providers: {[p.value for p in providers]}"
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
            input_variables=["user_question", "schema"],
            template=config.prompts.cypher_search,
        )

        self.guard_rails_template = PromptTemplate(
            input_variables=["user_question"], template=config.prompts.guardrails
        )

    def _validate_cypher_readonly(self, query: str) -> None:
        """Raise if generated Cypher contains write-operation keywords."""
        clean = re.sub(r"//[^\n]*", "", query)
        clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)

        tokens = set(re.findall(r"\b[A-Z]+\b", clean.upper()))
        blocked = tokens & FORBIDDEN_CLAUSES

        if blocked:
            raise ValueError(f"Cypher zawiera niedozwolone operacje: {blocked}")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences that may wrap JSON output."""
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _parse_guardrail_output(self, raw_output: str) -> Dict[str, str]:
        """Parse guardrail JSON and normalize the decision with a safe fallback."""
        cleaned_output = self._strip_code_fences(raw_output)
        json_match = re.search(r"\{.*\}", cleaned_output, flags=re.DOTALL)
        payload_text = json_match.group(0) if json_match else cleaned_output

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
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

    async def generate_cypher(self, state: State):
        """
        Generate CYPHER query from user question using database schema.
        Uses better model (gpt-5-mini) for complex Cypher generation.
        Tries providers in configured fallback order.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with generated CYPHER query
        """
        schema = self.schema
        print(f"[Schema used for Cypher generation] ({len(schema)} chars):\n{schema or '(empty)'}")

        generated_cypher = await self._ainvoke_with_provider_fallback(
                self.generate_cypher_template,
            {
                "user_question": state["user_question"],
                "schema": schema,
            },
            config=self._get_invoke_config(
                trace_id=state["trace_id"],
                tags=["knowledge_graph", "generated_cypher"],
                run_name="Generate Cypher",
                handler=state.get("callback_handler"),
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
            self._validate_cypher_readonly(cypher_query)

            if "LIMIT" not in cypher_query.upper():
                cypher_query = f"{cypher_query.rstrip(';')} LIMIT {self.max_results}"

            response = self.database.query(cypher_query)

            return {"context": response}

        except Exception as e:
            error_msg = str(e)

            if self.enable_debug:
                print(f"[Query Error] {error_msg}")

            return {"context": [], "generated_cypher": f"Query failed: {error_msg}"}

    async def guardrails_system(self, state: State):
        """
        Decide whether to use graph retrieval or general LLM knowledge.
        Uses fast model (gpt-5-nano) for quick decision.
        Expects JSON response with decision field ("generate" or "end").
        Tries providers in configured fallback order.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with next node decision
        """
        guardrail_output = await self._ainvoke_with_provider_fallback(
                self.guard_rails_template,
            {"user_question": state["user_question"]},
            config=self._get_invoke_config(
                trace_id=state["trace_id"],
                tags=["knowledge_graph", "guardrails"],
                run_name="Guardrails",
                handler=state.get("callback_handler"),
            ),
        )
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

    def _format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert graph output into the public response payload."""
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

    def invoke(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute the RAG pipeline with user message.

        Args:
            message: User's question/input
            session_id: Session identifier for tracking

        Returns:
            Dictionary with context from graph or "W bazie danych nie ma informacji"
        """
        """Synchronous wrapper for CLI/tests only; do not call from an active event loop."""
        return asyncio.run(self.ainvoke(message, session_id=session_id))

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

        Returns:
            Dictionary with context from graph or "W bazie danych nie ma informacji"
        """

        try:
            result = await asyncio.wait_for(
                self.graph.ainvoke(
                    {
                        "user_question": message,
                        "trace_id": trace_id,
                        "callback_handler": callback_handler,
                    }
                ),
                timeout=self.graph_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(RAG.GRAPH_PIPELINE_TIMEOUT_MESSAGE) from exc

        return self._format_response(result)
