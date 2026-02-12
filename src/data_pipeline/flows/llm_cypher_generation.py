import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from prefect import get_run_logger, task
from pydantic import SecretStr


class PipeState(MessagesState):
    context: str
    generated_cypher: List[str]


class LLMPipe:
    def __init__(
        self,
        nodes: List[str] | None = None,
        relations: List[str] | None = None,
    ):
        self.model = BaseChatOpenAI(
            model="gpt-5.2",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
            temperature=0,
        )
        self._initialize_prompt_templates()
        self.nodes = nodes or []
        self.relations = relations or []
        self._build_pipe_graph()

    def _initialize_prompt_templates(self) -> None:
        self.generate_template = PromptTemplate(
            input_variables=["context", "nodes", "relations"],
            template="""
                Generate Neo4j Cypher statements based EXCLUSIVELY on the provided context.
                Use ONLY the allowed node types and relation types.
                DO NOT include any additional text or explanations.

                CONTEXT: {context}

                STRICT RULES:
                1. OUTPUT MUST:
                - Contain ONLY executable Cypher statements
                - Begin with "MERGE"
                - Separate multiple statements with NEWLINE character
                - Use UNIQUE variable names (node1, node2, etc. - never reused)
                - LIMIT TOKENS TO 65536 TO AVOID ERRORS WITH DEEPSEEK API

                2. FOR NODES:
                - MERGE each node with unique variable name
                - Include 'title' and 'context' properties
                - Replace Polish characters (ą→a, ć→c, ę→e, ł→l, ń→n, ó→o, ś→s, ź→z, ż→z)
                - Use ONLY ASCII characters
                - Escape single quotes in text with backslash (\')

                3. FOR RELATIONSHIPS:
                - MERGE between existing node variables
                - Use ONLY allowed relationship types
                - Direction matters (A→B ≠ B→A)

                4. VARIABLE NAMES:
                - Must be UNIQUE across entire query
                - Recommended pattern: (node1), (node2), (person1), (dept1), etc.
                - NEVER reuse variables - this causes errors

                EXAMPLE OUTPUT:
                MERGE (node1:Person {{title: 'John Smith', context: 'Professor at UW'}})
                MERGE (node2:Department {{title: 'Computer Science', context: 'CS department'}})
                MERGE (node1)-[:works_in]->(node2)

                OUTPUT MUST BE EXACTLY IN THIS FORMAT:
                MERGE (...)
                MERGE (...)
                MERGE (...)-[:...]->(...)

                Each MERGE statement on its own line.
                NO OTHER TEXT OR CHARACTERS ALLOWED!

            """,
        )

    def _build_pipe_graph(self) -> None:
        builder = StateGraph(PipeState)

        nodes = [("generate", self.generate_cypher)]

        for node_name, node_func in nodes:
            builder.add_node(node_name, node_func)

        builder.add_edge(START, "generate")
        builder.add_edge("generate", END)

        self.graph = builder.compile()

    def generate_cypher(self, state: PipeState) -> List[str]:
        logger = get_run_logger()

        chain = self.generate_template | self.model | StrOutputParser()

        payload = {
            "context": state["context"],
            "nodes": self.nodes,
            "relations": self.relations,
        }

        logger.debug(
            "Invoking LLM generate_cypher with context length %d",
            len(str(payload.get("context") or "")),
        )

        try:
            cypher_code = chain.invoke(payload)
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            return {"generated_cypher": []}

        logger.debug("Raw LLM output: %r", cypher_code)

        # Split on pipe and remove empty/whitespace-only parts
        parts = [part.strip() for part in (cypher_code or "").split("|") if part and part.strip()]

        if not parts:
            logger.warning(
                "LLM returned no usable Cypher parts (raw output length=%d)",
                len(str(cypher_code or "")),
            )

        return {"generated_cypher": parts}

    def run(self, context: str) -> List[str]:
        result = self.graph.invoke(
            {"context": context, "generated_cypher": []},
            config={"configurable": {"thread_id": 1}},
        )
        return result["generated_cypher"]


@task
def generate_cypher_queries(extracted_text: str) -> str:
    """Generate cypher statements from text using `LLMPipe`.

    Returns a single string with statements separated by pipe `|`.
    """
    load_dotenv()
    logger = get_run_logger()
    llm = LLMPipe()
    parts = llm.run(extracted_text)

    # Log raw results for debugging (visible in Prefect UI)
    try:
        logger.info("LLM returned %d parts", len(parts))
        for i, p in enumerate(parts[:10]):
            logger.info("LLM part %d: %s", i, (p[:400] + "...") if len(p) > 400 else p)
    except Exception:
        # Avoid failing the task due to logging
        logger.debug("Failed to log LLM parts")

    if not parts:
        logger.warning("LLM produced no cypher parts; returning empty string")
        return ""

    return "|".join(parts)
