from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph


class PipeState(MessagesState):
    context: str
    generated_cypher: List[str]


class LLMPipe:
    def __init__(
        self, api_key: str = None, nodes: List[str] = None, relations: List[str] = None
    ):
        BaseChatOpenAI(
            model="gpt-5-mini",
            api_key=api_key,
            temperature=0,
        )
        self._initialize_prompt_templates()
        self.nodes = nodes
        self.relations = relations
        self._build_pipe_graph()

    def _initialize_prompt_templates(self) -> None:
        """Initialize all prompt templates used in the RAG pipeline."""

        self.generate_template = PromptTemplate(
            input_variables=["context", "nodes", "relations"],
            template="""
                Generate Neo4j Cypher statements based EXCLUSIVELY on the provided context.
                Use ONLY the allowed node types and relation types.
                DO NOT include any additional text or explanations.

                CONTEXT: {context}

                ALLOWED NODE LABELS: {nodes}
                ALLOWED RELATIONSHIP TYPES: {relations}

                STRICT RULES:
                1. OUTPUT MUST:
                   - Contain ONLY executable Cypher statements
                   - Begin with "MERGE"
                   - Separate multiple statements with PIPE character (|)
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
                MERGE (node1:Person {{title: 'John Smith',
                context: 'Professor at UW'}})|MERGE (node2:Department
                {{title: 'Computer Science', context: 'CS department'}})|MERGE
                (node1)-[:works_in]->(node2)

                OUTPUT MUST BE EXACTLY IN THIS FORMAT:
                MERGE (...) [|MERGE (...)]* [|MERGE (...)-[:...]->(...)]*
                NO OTHER TEXT OR CHARACTERS ALLOWED!

            """,
        )

    def _build_pipe_graph(self) -> None:
        """Build the pipeline graph for the RAG process."""

        builder = StateGraph(PipeState)

        nodes = [
            ("generate", self.generate_cypher),
        ]

        for node_name, node_func in nodes:
            builder.add_node(node_name, node_func)

        builder.add_edge(START, "generate")
        builder.add_edge("generate", END)

        self.graph = builder.compile()

    def generate_cypher(self, state: PipeState) -> List[str]:
        chain = self.generate_template | self.model | StrOutputParser()

        cypher_code = chain.invoke(
            {
                "context": state["context"],
                "nodes": self.nodes,
                "relations": self.relations,
            }
        )

        return {
            "generated_cypher": [code_part for code_part in cypher_code.split("|")],
        }

    def run(self, context: str) -> List[str]:
        """Run the pipeline with the given context."""
        result = self.graph.invoke(
            {
                "context": context,
                "generated_cypher": [],
            },
            config={"configurable": {"thread_id": 1}},
        )
        return result["generated_cypher"]
