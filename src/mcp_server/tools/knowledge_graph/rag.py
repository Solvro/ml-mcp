import asyncio
import json
import os
import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_openai.chat_models.base import BaseChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from ....config.config import get_config
from ....llm_hardening import get_llm_timeout_seconds
from .graph_visualizer import GraphVisualizer
from .state import State

WRITE_CYPHER_KEYWORDS = frozenset(
    {
        "CREATE",
        "DELETE",
        "DETACH",
        "DROP",
        "FOREACH",
        "LOAD",
        "MERGE",
        "REMOVE",
        "SET",
    }
)
READ_ONLY_START_RE = re.compile(r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|CALL|UNWIND)\b", re.IGNORECASE)
STRING_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)


class UnsafeCypherQueryError(ValueError):
    """Raised when generated Cypher contains a mutating operation."""


def _scrub_cypher_for_validation(cypher_query: str) -> str:
    without_comments = COMMENT_RE.sub(" ", cypher_query)
    return STRING_LITERAL_RE.sub(" ", without_comments)


def validate_read_only_cypher(cypher_query: str) -> None:
    """Reject Cypher that can mutate the graph before Neo4j execution."""
    scrubbed_query = _scrub_cypher_for_validation(cypher_query).strip()
    if not scrubbed_query:
        raise UnsafeCypherQueryError("generated Cypher query is empty")

    if ";" in scrubbed_query.rstrip(";"):
        raise UnsafeCypherQueryError("multiple Cypher statements are not allowed")

    if not READ_ONLY_START_RE.search(scrubbed_query):
        raise UnsafeCypherQueryError("Cypher must start with a read-only clause")

    normalized_query = scrubbed_query.upper()
    for keyword in WRITE_CYPHER_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized_query):
            raise UnsafeCypherQueryError(f"blocked mutating Cypher keyword: {keyword}")

    if not re.search(r"\bRETURN\b", normalized_query):
        raise UnsafeCypherQueryError("read-only Cypher must return data")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _invoke_llm_chain_with_retry(chain, payload: dict, invoke_config: dict) -> str:
    return chain.invoke(payload, config=invoke_config)


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
        """
        config = get_config()

        self.api_key = api_key
        self.enable_debug = enable_debug if enable_debug is not None else config.rag.enable_debug
        self.max_results = max_results if max_results is not None else config.rag.max_results
        self.llm_timeout_seconds = get_llm_timeout_seconds()

        # Check for non-empty API keys
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

        if openai_key or deepseek_key:
            self.fast_llm = BaseChatOpenAI(
                model=config.llm.fast_model.name,
                api_key=api_key,
                temperature=config.llm.fast_model.temperature,
                timeout=self.llm_timeout_seconds,
            )

            self.cypher_llm = BaseChatOpenAI(
                model=config.llm.accurate_model.name,
                api_key=api_key,
                temperature=config.llm.accurate_model.temperature,
                timeout=self.llm_timeout_seconds,
            )
        else:
            self.fast_llm = ChatGoogleGenerativeAI(
                model=config.llm.gemini.name,
                google_api_key=api_key,
                temperature=1.0,
                request_timeout=self.llm_timeout_seconds,
            )
            self.cypher_llm = ChatGoogleGenerativeAI(
                model=config.llm.gemini.name,
                google_api_key=api_key,
                temperature=1.0,
                request_timeout=self.llm_timeout_seconds,
            )

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

        self.handler = None

    def _get_invoke_config(self, trace_id: str, tags: list, run_name: str) -> dict:
        """Build invoke config with optional callbacks."""
        config = {
            "metadata": {
                "langfuse_session_id": trace_id,
                "langfuse_tags": tags,
                "run_name": run_name,
            },
        }
        if self.handler is not None:
            config["callbacks"] = [self.handler]
        return config

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

    def generate_cypher(self, state: State):
        """
        Generate CYPHER query from user question using database schema.
        Uses better model (gpt-5-mini) for complex Cypher generation.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with generated CYPHER query
        """
        schema = self.schema
        print(f"[Schema used for Cypher generation] ({len(schema)} chars):\n{schema or '(empty)'}")

        chain = self.generate_cypher_template | self.cypher_llm | StrOutputParser()
        generated_cypher = _invoke_llm_chain_with_retry(
            chain,
            {
                "user_question": state["user_question"],
                "schema": schema,
            },
            self._get_invoke_config(
                trace_id=state["trace_id"],
                tags=["knowledge_graph", "generated_cypher"],
                run_name="Generate Cypher",
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
            if "LIMIT" not in cypher_query.upper():
                cypher_query = f"{cypher_query.rstrip(';')} LIMIT {self.max_results}"

            validate_read_only_cypher(cypher_query)
            response = self.database.query(cypher_query)

            return {"context": response}

        except UnsafeCypherQueryError as e:
            error_msg = f"Blocked unsafe Cypher query: {e}"
            if self.enable_debug:
                print(f"[Query Guardrail] {error_msg}")
            return {"context": [], "generated_cypher": error_msg}

        except Exception as e:
            error_msg = str(e)

            if self.enable_debug:
                print(f"[Query Error] {error_msg}")

            return {"context": [], "generated_cypher": f"Query failed: {error_msg}"}

    def guardrails_system(self, state: State):
        """
        Decide whether to use graph retrieval or general LLM knowledge.
        Uses fast model (gpt-5-nano) for quick decision.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with next node decision
        """
        guardrails_chain = self.guard_rails_template | self.fast_llm | StrOutputParser()

        guardrail_output = (
            _invoke_llm_chain_with_retry(
                guardrails_chain,
                {"user_question": state["user_question"]},
                self._get_invoke_config(
                    trace_id=state["trace_id"],
                    tags=["knowledge_graph", "guardrails"],
                    run_name="Guardrails",
                ),
            )
            .strip()
            .lower()
        )

        next_node = "generate_cypher" if "generate" in guardrail_output else "end"

        return {
            "next_node": next_node,
            "guardrail_decision": guardrail_output,
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

        Returns:
            Dictionary with context from graph or "W bazie danych nie ma informacji"
        """
        self.handler = callback_handler

        result = await asyncio.wait_for(
            self.graph.ainvoke({"user_question": message, "trace_id": trace_id}),
            timeout=self.llm_timeout_seconds,
        )

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
