from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from config.config import get_config


class PipeState(MessagesState):
    context: str
    generated_cypher: List[str]


class LLMPipe:
    def __init__(self, api_key: str = None, nodes: List[str] = None, relations: List[str] = None):
        config = get_config()
        BaseChatOpenAI(
            model=config.llm.accurate_model.name,
            api_key=api_key,
            temperature=config.llm.accurate_model.temperature,
        )
        self._initialize_prompt_templates()
        self.nodes = nodes
        self.relations = relations
        self._build_pipe_graph()

    def _initialize_prompt_templates(self) -> None:
        """Initialize all prompt templates used in the RAG pipeline."""
        config = get_config()
        template_str = config.prompts.cypher_search

        self.generate_template = PromptTemplate(
            input_variables=["context", "nodes", "relations"],
            template=template_str,
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
