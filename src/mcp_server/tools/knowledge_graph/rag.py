import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai.chat_models.base import BaseChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .graph_visualizer import GraphVisualizer
from .state import State


class RAG:
    """Retrieval-Augmented Generation system with Neo4j graph database backend."""

    def __init__(
        self,
        api_key: str,
        neo4j_url: str,
        neo4j_username: str,
        neo4j_password: str,
        enable_debug: bool = False,
        max_results: int = 5,
    ) -> None:
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
        self.api_key = api_key
        self.enable_debug = enable_debug
        self.max_results = max_results

        self.fast_llm = BaseChatOpenAI(
            model="gpt-5-nano",
            api_key=api_key,
            temperature=0.1,
        )

        self.cypher_llm = BaseChatOpenAI(
            model="gpt-5-mini",
            api_key=api_key,
            temperature=0,
        )

        self._initialize_prompt_templates()

        self.database = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
            database="neo4j",
        )

        self._cached_schema = None

        self.visualizer = GraphVisualizer()
        self.graph = self._build_processing_graph()

        self.handler = None

    @property
    def schema(self) -> str:
        """Cached database schema to avoid repeated fetches"""
        if self._cached_schema is None:
            self._cached_schema = self.database.get_schema
        return self._cached_schema

    def get_graph(self) -> GraphVisualizer:
        """Return graph visualizer with Mermaid capabilities"""
        return self.visualizer

    def _initialize_prompt_templates(self) -> None:
        """Initialize all prompt templates used in the RAG pipeline."""

        self.generate_cypher_template = PromptTemplate(
            input_variables=["user_question", "schema"],
            template="""Generate ONLY valid Cypher query. No explanations.

            Schema: {schema}
            Question: {user_question}

            Cypher:""",
        )

        self.guard_rails_template = PromptTemplate(
            input_variables=["user_question"],
            template="""Is this about Wroclaw University of Science and Technology
                    (or university at all) or about another topic?
                    Answer ONLY: "generate_cypher" or "end"

                    Question: {user_question}
                    Answer:
                    """,
        )

    def _build_processing_graph(self) -> CompiledStateGraph:
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

    def generate_cypher(self, state: State) -> dict[str, str]:
        """
        Generate CYPHER query from user question using database schema.
        Uses better model (gpt-5-mini) for complex Cypher generation.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with generated CYPHER query
        """
        chain = self.generate_cypher_template | self.cypher_llm | StrOutputParser()
        generated_cypher = chain.invoke(
            {
                "user_question": state["user_question"],
                "schema": self.schema,
            },
            config={
                "callbacks": [self.handler],
                "metadata": {
                    "langfuse_session_id": state["trace_id"],
                    "langfuse_tags": ["knowledge_graph", "generated_cypher"],
                    "run_name": "Generate Cypher",
                },
            },
        )

        return {"generated_cypher": generated_cypher}

    def retrieve(self, state: State) -> dict[str, Any]:
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

            response = self.database.query(cypher_query)

            return {"context": response}

        except Exception as e:
            error_msg = str(e)

            if self.enable_debug:
                print(f"[Query Error] {error_msg}")

            return {"context": [], "generated_cypher": f"Query failed: {error_msg}"}

    def guardrails_system(self, state: State) -> dict[str, str]:
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
            guardrails_chain.invoke(
                {"user_question": state["user_question"]},
                config={
                    "callbacks": [self.handler],
                    "metadata": {
                        "langfuse_session_id": state["trace_id"],
                        "langfuse_tags": ["knowledge_graph", "guardrails"],
                        "run_name": "Guardrails",
                    },
                },
            )
            .strip()
            .lower()
        )

        next_node = "generate_cypher" if "generate" in guardrail_output else "end"

        return {
            "next_node": next_node,
            "guardrail_decision": guardrail_output,
        }

    def return_none(self, state: State) -> dict[str, Any]:
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

    def invoke(self, message: str, session_id: str = "default") -> dict[str, Any]:
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
        callback_handler: CallbackHandler | None = None,
    ) -> dict[str, Any]:
        """
        Async version of invoke for better performance in concurrent scenarios.

        Args:
            message: User's question/input
            session_id: Session identifier for tracking

        Returns:
            Dictionary with context from graph or "W bazie danych nie ma informacji"
        """
        self.handler = callback_handler

        result = await self.graph.ainvoke({"user_question": message, "trace_id": trace_id})

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
