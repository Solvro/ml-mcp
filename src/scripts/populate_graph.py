"""Populate Neo4j with synthetic data for Politechnika Wrocławska (PWr).

Run with:
    uv run python -m src.scripts.populate_graph

Requires NEO4J_URI, NEO4J_USER (or NEO4J_USERNAME), and NEO4J_PASSWORD
environment variables (loaded from .env automatically).
"""

import os
import sys

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

# ---------------------------------------------------------------------------
# Synthetic Cypher statements — each is a self-contained MERGE block
# ---------------------------------------------------------------------------

STATEMENTS = [
    # ------------------------------------------------------------------
    # Departments
    # ------------------------------------------------------------------
    """
    MERGE (d1:Department {title: 'Faculty of Computer Science and Management', context: 'W8 faculty at PWr, covers CS, AI, and management'})
    MERGE (d2:Department {title: 'Faculty of Electronics', context: 'W4 faculty at PWr, covers electronics, telecommunications, cybersecurity'})
    MERGE (d3:Department {title: 'Faculty of Mathematics', context: 'W13 faculty at PWr, covers pure and applied mathematics'})
    MERGE (d4:Department {title: 'Faculty of Mechanical Engineering', context: 'W10 faculty at PWr, covers mechanical engineering and robotics'})
    MERGE (d5:Department {title: 'Faculty of Civil Engineering', context: 'W2 faculty at PWr, covers structural and environmental engineering'})
    """,

    # ------------------------------------------------------------------
    # Professors
    # ------------------------------------------------------------------
    """
    MERGE (p1:Professor {title: 'Prof. Jan Kowalski', context: 'Full professor in artificial intelligence and machine learning at W8'})
    MERGE (p2:Professor {title: 'Prof. Anna Wiśniewska', context: 'Associate professor specialising in algorithms and complexity theory at W8'})
    MERGE (p3:Professor {title: 'Prof. Marek Zielinski', context: 'Full professor in signal processing and embedded systems at W4'})
    MERGE (p4:Professor {title: 'Prof. Katarzyna Nowak', context: 'Associate professor in numerical methods and scientific computing at W13'})
    MERGE (p5:Professor {title: 'Prof. Tomasz Lewandowski', context: 'Full professor in robotics and mechatronics at W10'})
    MERGE (p6:Professor {title: 'Prof. Ewa Dabrowska', context: 'Associate professor in structural mechanics at W2'})
    MERGE (p7:Professor {title: 'Dr. Piotr Szymanski', context: 'Assistant professor in cybersecurity and cryptography at W4'})
    MERGE (p8:Professor {title: 'Dr. Monika Wojcik', context: 'Assistant professor in data engineering and databases at W8'})
    """,

    # ------------------------------------------------------------------
    # Courses
    # ------------------------------------------------------------------
    """
    MERGE (c1:Course {title: 'Introduction to Artificial Intelligence', context: 'Undergraduate course covering search, ML basics, neural networks — 5 ECTS'})
    MERGE (c2:Course {title: 'Algorithms and Data Structures', context: 'Core undergraduate CS course on sorting, graphs, complexity — 6 ECTS'})
    MERGE (c3:Course {title: 'Digital Signal Processing', context: 'Undergraduate electronics course on Fourier analysis, filters — 5 ECTS'})
    MERGE (c4:Course {title: 'Mathematical Analysis', context: 'First-year course on calculus, series, differential equations — 8 ECTS'})
    MERGE (c5:Course {title: 'Numerical Methods', context: 'Applied mathematics course on interpolation, integration, ODEs — 5 ECTS'})
    MERGE (c6:Course {title: 'Robotics and Automation', context: 'Graduate course on kinematics, trajectory planning, ROS — 6 ECTS'})
    MERGE (c7:Course {title: 'Database Systems', context: 'Undergraduate CS course on relational and NoSQL databases — 5 ECTS'})
    MERGE (c8:Course {title: 'Cybersecurity Fundamentals', context: 'Undergraduate electronics course covering threat modelling, PKI, network security — 4 ECTS'})
    MERGE (c9:Course {title: 'Machine Learning', context: 'Graduate CS course on supervised/unsupervised learning, deep learning — 6 ECTS'})
    MERGE (c10:Course {title: 'Computer Networks', context: 'Undergraduate CS course on TCP/IP, routing, network programming — 5 ECTS'})
    """,

    # ------------------------------------------------------------------
    # Programs / Degrees
    # ------------------------------------------------------------------
    """
    MERGE (pr1:Program {title: 'Computer Science BSc', context: '3.5-year undergraduate program at W8 leading to Bachelor of Science'})
    MERGE (pr2:Program {title: 'Computer Science MSc', context: '1.5-year graduate program at W8 leading to Master of Science'})
    MERGE (pr3:Program {title: 'Electronics and Telecommunications BSc', context: '3.5-year undergraduate program at W4'})
    MERGE (pr4:Program {title: 'Robotics MSc', context: '1.5-year graduate program at W10'})
    MERGE (pr5:Program {title: 'Applied Mathematics BSc', context: '3.5-year undergraduate program at W13'})
    """,

    # ------------------------------------------------------------------
    # Buildings / Rooms
    # ------------------------------------------------------------------
    """
    MERGE (b1:Building {title: 'Building C-3', context: 'Main CS and Management faculty building on PWr campus'})
    MERGE (b2:Building {title: 'Building C-5', context: 'Electronics faculty building, houses labs and lecture halls'})
    MERGE (b3:Building {title: 'Building A-1', context: 'Central lecture hall building shared across faculties'})
    MERGE (r1:Room {title: 'Room 104 C-3', context: 'Lecture hall with 120 seats in building C-3'})
    MERGE (r2:Room {title: 'Lab 301 C-3', context: 'Computer lab with 30 workstations in building C-3'})
    MERGE (r3:Room {title: 'Room 202 C-5', context: 'Electronics lecture room with 80 seats in building C-5'})
    MERGE (r4:Room {title: 'Aula A-1', context: 'Main auditorium with 400 seats in building A-1'})
    """,

    # ------------------------------------------------------------------
    # Research projects
    # ------------------------------------------------------------------
    """
    MERGE (rp1:Research {title: 'Deep Learning for Medical Image Analysis', context: 'NCN-funded project applying CNNs to MRI segmentation, 2023-2026'})
    MERGE (rp2:Research {title: 'Graph Neural Networks for Knowledge Graphs', context: 'Industry collaboration applying GNNs to enterprise knowledge bases, 2024-2025'})
    MERGE (rp3:Research {title: 'Autonomous Mobile Robot Navigation', context: 'EU-funded Horizon project on SLAM and path planning, 2022-2025'})
    MERGE (rp4:Research {title: 'Post-Quantum Cryptography Implementation', context: 'NCBR project on lattice-based cryptosystems on embedded hardware, 2023-2025'})
    """,

    # ------------------------------------------------------------------
    # Relationships — professors ↔ departments
    # ------------------------------------------------------------------
    """
    MATCH (p1:Professor {title: 'Prof. Jan Kowalski'}), (d1:Department {title: 'Faculty of Computer Science and Management'})
    MERGE (p1)-[:BELONGS_TO]->(d1)

    WITH p1
    MATCH (p2:Professor {title: 'Prof. Anna Wiśniewska'}), (d1:Department {title: 'Faculty of Computer Science and Management'})
    MERGE (p2)-[:BELONGS_TO]->(d1)

    WITH p2
    MATCH (p8:Professor {title: 'Dr. Monika Wojcik'}), (d1:Department {title: 'Faculty of Computer Science and Management'})
    MERGE (p8)-[:BELONGS_TO]->(d1)

    WITH p8
    MATCH (p3:Professor {title: 'Prof. Marek Zielinski'}), (d2:Department {title: 'Faculty of Electronics'})
    MERGE (p3)-[:BELONGS_TO]->(d2)

    WITH p3
    MATCH (p7:Professor {title: 'Dr. Piotr Szymanski'}), (d2:Department {title: 'Faculty of Electronics'})
    MERGE (p7)-[:BELONGS_TO]->(d2)

    WITH p7
    MATCH (p4:Professor {title: 'Prof. Katarzyna Nowak'}), (d3:Department {title: 'Faculty of Mathematics'})
    MERGE (p4)-[:BELONGS_TO]->(d3)

    WITH p4
    MATCH (p5:Professor {title: 'Prof. Tomasz Lewandowski'}), (d4:Department {title: 'Faculty of Mechanical Engineering'})
    MERGE (p5)-[:BELONGS_TO]->(d4)

    WITH p5
    MATCH (p6:Professor {title: 'Prof. Ewa Dabrowska'}), (d5:Department {title: 'Faculty of Civil Engineering'})
    MERGE (p6)-[:BELONGS_TO]->(d5)
    """,

    # ------------------------------------------------------------------
    # Relationships — professors teach courses
    # ------------------------------------------------------------------
    """
    MATCH (p1:Professor {title: 'Prof. Jan Kowalski'}), (c1:Course {title: 'Introduction to Artificial Intelligence'})
    MERGE (p1)-[:TEACHES]->(c1)

    WITH p1
    MATCH (p1x:Professor {title: 'Prof. Jan Kowalski'}), (c9:Course {title: 'Machine Learning'})
    MERGE (p1x)-[:TEACHES]->(c9)

    WITH p1x
    MATCH (p2:Professor {title: 'Prof. Anna Wiśniewska'}), (c2:Course {title: 'Algorithms and Data Structures'})
    MERGE (p2)-[:TEACHES]->(c2)

    WITH p2
    MATCH (p3:Professor {title: 'Prof. Marek Zielinski'}), (c3:Course {title: 'Digital Signal Processing'})
    MERGE (p3)-[:TEACHES]->(c3)

    WITH p3
    MATCH (p4:Professor {title: 'Prof. Katarzyna Nowak'}), (c4:Course {title: 'Mathematical Analysis'})
    MERGE (p4)-[:TEACHES]->(c4)

    WITH p4
    MATCH (p4x:Professor {title: 'Prof. Katarzyna Nowak'}), (c5:Course {title: 'Numerical Methods'})
    MERGE (p4x)-[:TEACHES]->(c5)

    WITH p4x
    MATCH (p5:Professor {title: 'Prof. Tomasz Lewandowski'}), (c6:Course {title: 'Robotics and Automation'})
    MERGE (p5)-[:TEACHES]->(c6)

    WITH p5
    MATCH (p8:Professor {title: 'Dr. Monika Wojcik'}), (c7:Course {title: 'Database Systems'})
    MERGE (p8)-[:TEACHES]->(c7)

    WITH p8
    MATCH (p7:Professor {title: 'Dr. Piotr Szymanski'}), (c8:Course {title: 'Cybersecurity Fundamentals'})
    MERGE (p7)-[:TEACHES]->(c8)

    WITH p7
    MATCH (p2x:Professor {title: 'Prof. Anna Wiśniewska'}), (c10:Course {title: 'Computer Networks'})
    MERGE (p2x)-[:TEACHES]->(c10)
    """,

    # ------------------------------------------------------------------
    # Relationships — courses belong to programs
    # ------------------------------------------------------------------
    """
    MATCH (c1:Course {title: 'Introduction to Artificial Intelligence'}), (pr1:Program {title: 'Computer Science BSc'})
    MERGE (c1)-[:PART_OF]->(pr1)

    WITH c1
    MATCH (c2:Course {title: 'Algorithms and Data Structures'}), (pr1:Program {title: 'Computer Science BSc'})
    MERGE (c2)-[:PART_OF]->(pr1)

    WITH c2
    MATCH (c7:Course {title: 'Database Systems'}), (pr1:Program {title: 'Computer Science BSc'})
    MERGE (c7)-[:PART_OF]->(pr1)

    WITH c7
    MATCH (c10:Course {title: 'Computer Networks'}), (pr1:Program {title: 'Computer Science BSc'})
    MERGE (c10)-[:PART_OF]->(pr1)

    WITH c10
    MATCH (c9:Course {title: 'Machine Learning'}), (pr2:Program {title: 'Computer Science MSc'})
    MERGE (c9)-[:PART_OF]->(pr2)

    WITH c9
    MATCH (c3:Course {title: 'Digital Signal Processing'}), (pr3:Program {title: 'Electronics and Telecommunications BSc'})
    MERGE (c3)-[:PART_OF]->(pr3)

    WITH c3
    MATCH (c8:Course {title: 'Cybersecurity Fundamentals'}), (pr3:Program {title: 'Electronics and Telecommunications BSc'})
    MERGE (c8)-[:PART_OF]->(pr3)

    WITH c8
    MATCH (c6:Course {title: 'Robotics and Automation'}), (pr4:Program {title: 'Robotics MSc'})
    MERGE (c6)-[:PART_OF]->(pr4)

    WITH c6
    MATCH (c4:Course {title: 'Mathematical Analysis'}), (pr5:Program {title: 'Applied Mathematics BSc'})
    MERGE (c4)-[:PART_OF]->(pr5)

    WITH c4
    MATCH (c5:Course {title: 'Numerical Methods'}), (pr5:Program {title: 'Applied Mathematics BSc'})
    MERGE (c5)-[:PART_OF]->(pr5)
    """,

    # ------------------------------------------------------------------
    # Relationships — programs belong to departments
    # ------------------------------------------------------------------
    """
    MATCH (pr1:Program {title: 'Computer Science BSc'}), (d1:Department {title: 'Faculty of Computer Science and Management'})
    MERGE (pr1)-[:OFFERED_BY]->(d1)

    WITH pr1
    MATCH (pr2:Program {title: 'Computer Science MSc'}), (d1:Department {title: 'Faculty of Computer Science and Management'})
    MERGE (pr2)-[:OFFERED_BY]->(d1)

    WITH pr2
    MATCH (pr3:Program {title: 'Electronics and Telecommunications BSc'}), (d2:Department {title: 'Faculty of Electronics'})
    MERGE (pr3)-[:OFFERED_BY]->(d2)

    WITH pr3
    MATCH (pr4:Program {title: 'Robotics MSc'}), (d4:Department {title: 'Faculty of Mechanical Engineering'})
    MERGE (pr4)-[:OFFERED_BY]->(d4)

    WITH pr4
    MATCH (pr5:Program {title: 'Applied Mathematics BSc'}), (d3:Department {title: 'Faculty of Mathematics'})
    MERGE (pr5)-[:OFFERED_BY]->(d3)
    """,

    # ------------------------------------------------------------------
    # Relationships — rooms located in buildings
    # ------------------------------------------------------------------
    """
    MATCH (r1:Room {title: 'Room 104 C-3'}), (b1:Building {title: 'Building C-3'})
    MERGE (r1)-[:LOCATED_IN]->(b1)

    WITH r1
    MATCH (r2:Room {title: 'Lab 301 C-3'}), (b1:Building {title: 'Building C-3'})
    MERGE (r2)-[:LOCATED_IN]->(b1)

    WITH r2
    MATCH (r3:Room {title: 'Room 202 C-5'}), (b2:Building {title: 'Building C-5'})
    MERGE (r3)-[:LOCATED_IN]->(b2)

    WITH r3
    MATCH (r4:Room {title: 'Aula A-1'}), (b3:Building {title: 'Building A-1'})
    MERGE (r4)-[:LOCATED_IN]->(b3)
    """,

    # ------------------------------------------------------------------
    # Relationships — professors supervise research
    # ------------------------------------------------------------------
    """
    MATCH (p1:Professor {title: 'Prof. Jan Kowalski'}), (rp1:Research {title: 'Deep Learning for Medical Image Analysis'})
    MERGE (p1)-[:SUPERVISES]->(rp1)

    WITH p1
    MATCH (p1x:Professor {title: 'Prof. Jan Kowalski'}), (rp2:Research {title: 'Graph Neural Networks for Knowledge Graphs'})
    MERGE (p1x)-[:SUPERVISES]->(rp2)

    WITH p1x
    MATCH (p5:Professor {title: 'Prof. Tomasz Lewandowski'}), (rp3:Research {title: 'Autonomous Mobile Robot Navigation'})
    MERGE (p5)-[:SUPERVISES]->(rp3)

    WITH p5
    MATCH (p7:Professor {title: 'Dr. Piotr Szymanski'}), (rp4:Research {title: 'Post-Quantum Cryptography Implementation'})
    MERGE (p7)-[:SUPERVISES]->(rp4)
    """,

    # ------------------------------------------------------------------
    # Prerequisite relationships between courses
    # ------------------------------------------------------------------
    """
    MATCH (c4:Course {title: 'Mathematical Analysis'}), (c5:Course {title: 'Numerical Methods'})
    MERGE (c4)-[:PREREQUISITE_FOR]->(c5)

    WITH c4
    MATCH (c2:Course {title: 'Algorithms and Data Structures'}), (c1:Course {title: 'Introduction to Artificial Intelligence'})
    MERGE (c2)-[:PREREQUISITE_FOR]->(c1)

    WITH c2
    MATCH (c1x:Course {title: 'Introduction to Artificial Intelligence'}), (c9:Course {title: 'Machine Learning'})
    MERGE (c1x)-[:PREREQUISITE_FOR]->(c9)

    WITH c1x
    MATCH (c7:Course {title: 'Database Systems'}), (c9x:Course {title: 'Machine Learning'})
    MERGE (c7)-[:PREREQUISITE_FOR]->(c9x)
    """,
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def populate(graph: Neo4jGraph) -> None:
    """Execute all synthetic data statements against Neo4j."""
    for i, stmt in enumerate(STATEMENTS, start=1):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            graph.query(stmt)
            print(f"  [OK] block {i}/{len(STATEMENTS)}")
        except Exception as exc:
            print(f"  [ERROR] block {i}/{len(STATEMENTS)}: {exc}", file=sys.stderr)


def main() -> None:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        print("ERROR: set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in your .env", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Neo4j at {uri} as {username} ...")
    graph = Neo4jGraph(url=uri, username=username, password=password)

    print(f"Populating graph with {len(STATEMENTS)} statement blocks ...")
    populate(graph)

    print("Done.")


if __name__ == "__main__":
    main()
