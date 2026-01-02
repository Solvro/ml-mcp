from langgraph.graph import END, START


class GraphVisualizer:
    """Helper class for building Mermaid diagrams from graph structure"""

    def __init__(self) -> None:
        self.nodes = set()
        self.edges = set()

    def add_node(self, node_name: str) -> None:
        self.nodes.add(node_name)

    def add_edge(self, from_node: str, to_node: str) -> None:
        from_str = "start_node" if from_node is START else str(from_node)
        to_str = "end_node" if to_node is END else str(to_node)
        self.edges.add((from_str, to_str, None))

    def add_conditional_edges(self, source_node: str, edges_dict: dict[str, str]) -> None:
        source_str = "start_node" if source_node is START else str(source_node)
        for condition, target_node in edges_dict.items():
            target_str = "end_node" if target_node is END else str(target_node)
            self.edges.add((source_str, target_str, condition))

    def draw_mermaid(self) -> str:
        """Generate Mermaid JS code representing the graph"""
        mermaid = ["%%{init: {'flowchart': {'curve': 'linear'}}}%%", "graph TD;"]

        mermaid.append("    start_node([<p>start</p>]):::first;")
        mermaid.append("    end_node([<p>end</p>]):::last;")

        for node in sorted(self.nodes - {"start_node", "end_node"}):
            mermaid.append(f"    {node}({node})")

        for from_node, to_node, condition in self.edges:
            arrow = "-.->" if condition else "-->"
            if condition:
                mermaid.append(f"    {from_node} {arrow} |{condition}| {to_node};")
            else:
                mermaid.append(f"    {from_node} {arrow} {to_node};")

        mermaid.append("    classDef default fill:#f2f0ff,line-height:1.2;")
        mermaid.append("    classDef first fill-opacity:0;")
        mermaid.append("    classDef last fill:#bfb6fc;")

        return "\n".join(mermaid)
