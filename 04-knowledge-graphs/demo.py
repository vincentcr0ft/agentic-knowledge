"""
Module 04: Knowledge Graphs
============================
Demonstrates: Neo4j graph construction, Cypher queries, multi-hop traversal,
and a LangGraph agent that translates natural language to Cypher.

Builds a small company knowledge graph and queries it both directly
and through an LLM-powered natural language interface.

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
"""

import json
from typing import TypedDict
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ─── Connections ──────────────────────────────────────────────────────────
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD THE KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════
# Create a small but rich company graph with multiple entity types
# and relationship types to demonstrate multi-hop traversal.

def build_graph():
    """Create the NovaTech company knowledge graph."""
    with driver.session() as session:
        # Clean slate — remove any previous demo data
        session.run("MATCH (n) DETACH DELETE n")

        # ── People ──
        session.run("""
            CREATE (alice:Person {name: 'Alice Chen', title: 'VP Engineering', department: 'Engineering'})
            CREATE (bob:Person {name: 'Bob Martinez', title: 'Tech Lead', department: 'Engineering'})
            CREATE (carol:Person {name: 'Carol Davis', title: 'Senior Developer', department: 'Engineering'})
            CREATE (dave:Person {name: 'Dave Wilson', title: 'Junior Developer', department: 'Engineering'})
            CREATE (eve:Person {name: 'Eve Thompson', title: 'Product Manager', department: 'Product'})
            CREATE (frank:Person {name: 'Frank Lee', title: 'Data Scientist', department: 'Data'})

            // Management chain
            CREATE (alice)-[:MANAGES {since: '2022'}]->(bob)
            CREATE (bob)-[:MANAGES {since: '2023'}]->(carol)
            CREATE (bob)-[:MANAGES {since: '2024'}]->(dave)

            // Projects
            CREATE (atlas:Project {name: 'Project Atlas', status: 'active', budget: 500000})
            CREATE (beacon:Project {name: 'Project Beacon', status: 'active', budget: 250000})
            CREATE (cipher:Project {name: 'Project Cipher', status: 'completed', budget: 150000})

            // People work on projects
            CREATE (bob)-[:WORKS_ON {role: 'lead'}]->(atlas)
            CREATE (carol)-[:WORKS_ON {role: 'developer'}]->(atlas)
            CREATE (dave)-[:WORKS_ON {role: 'developer'}]->(atlas)
            CREATE (carol)-[:WORKS_ON {role: 'lead'}]->(beacon)
            CREATE (frank)-[:WORKS_ON {role: 'analyst'}]->(beacon)
            CREATE (bob)-[:WORKS_ON {role: 'lead'}]->(cipher)

            // Products
            CREATE (nova7:Product {name: 'Nova 7', category: 'workstation', price: 24999})
            CREATE (novabook:Product {name: 'NovaBook Pro', category: 'laptop', price: 3499})

            // Projects use products
            CREATE (atlas)-[:USES]->(nova7)
            CREATE (beacon)-[:USES]->(novabook)

            // Products created by departments
            CREATE (hw:Department {name: 'Hardware Division'})
            CREATE (sw:Department {name: 'Software Division'})
            CREATE (hw)-[:DESIGNED]->(nova7)
            CREATE (hw)-[:DESIGNED]->(novabook)

            // Eve oversees projects
            CREATE (eve)-[:OVERSEES]->(atlas)
            CREATE (eve)-[:OVERSEES]->(beacon)

            // Alice oversees department
            CREATE (alice)-[:OVERSEES]->(hw)
        """)

    print("  ✓ Knowledge graph created: 6 people, 3 projects, 2 products, 2 departments")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: DIRECT CYPHER QUERIES
# ═══════════════════════════════════════════════════════════════════════════
# Demonstrate different Cypher query patterns.

def run_cypher_demos():
    """Run a series of Cypher queries to demonstrate graph capabilities."""
    queries = [
        {
            "description": "Simple lookup: Find all people",
            "cypher": "MATCH (p:Person) RETURN p.name AS name, p.title AS title ORDER BY p.name",
        },
        {
            "description": "Relationship traversal: Who does Bob manage?",
            "cypher": "MATCH (bob:Person {name: 'Bob Martinez'})-[:MANAGES]->(report) RETURN report.name AS name, report.title AS title",
        },
        {
            "description": "Multi-hop: Who are Alice's indirect reports? (2+ levels)",
            "cypher": "MATCH (alice:Person {name: 'Alice Chen'})-[:MANAGES*2..3]->(report) RETURN report.name AS name, report.title AS title",
        },
        {
            "description": "Path query: Full management chain above Dave",
            "cypher": """MATCH path = (manager:Person)-[:MANAGES*]->(dave:Person {name: 'Dave Wilson'})
                         RETURN [n IN nodes(path) | n.name] AS chain""",
        },
        {
            "description": "Cross-entity: What products are used by projects Bob works on?",
            "cypher": """MATCH (bob:Person {name: 'Bob Martinez'})-[:WORKS_ON]->(project)-[:USES]->(product)
                         RETURN project.name AS project, product.name AS product, product.price AS price""",
        },
        {
            "description": "Aggregation: How many people work on each active project?",
            "cypher": """MATCH (p:Person)-[:WORKS_ON]->(proj:Project {status: 'active'})
                         RETURN proj.name AS project, count(p) AS team_size, proj.budget AS budget
                         ORDER BY team_size DESC""",
        },
        {
            "description": "Complex: Find the management chain for everyone on Project Atlas",
            "cypher": """MATCH (worker:Person)-[:WORKS_ON]->(proj:Project {name: 'Project Atlas'})
                         OPTIONAL MATCH (manager:Person)-[:MANAGES]->(worker)
                         RETURN worker.name AS worker, worker.title AS title, manager.name AS manager""",
        },
    ]

    print(f"\n{'=' * 64}")
    print("  Direct Cypher Queries")
    print("=" * 64)

    with driver.session() as session:
        for q in queries:
            print(f"\n  ▸ {q['description']}")
            print(f"    Cypher: {q['cypher'].strip()[:80]}...")
            result = session.run(q["cypher"])
            records = [dict(r) for r in result]
            for record in records:
                print(f"    → {record}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: LANGGRAPH AGENT — NATURAL LANGUAGE TO CYPHER
# ═══════════════════════════════════════════════════════════════════════════
# A 3-node pipeline:
#   generate_cypher → execute_query → format_answer

# The schema is provided to the LLM so it knows what's available.

GRAPH_SCHEMA = """
Node labels and properties:
  - Person {name: string, title: string, department: string}
  - Project {name: string, status: string ('active'|'completed'), budget: integer}
  - Product {name: string, category: string, price: integer}
  - Department {name: string}

Relationship types:
  - (Person)-[:MANAGES {since: string}]->(Person)
  - (Person)-[:WORKS_ON {role: string}]->(Project)
  - (Person)-[:OVERSEES]->(Project)
  - (Person)-[:OVERSEES]->(Department)
  - (Project)-[:USES]->(Product)
  - (Department)-[:DESIGNED]->(Product)
"""


class NLCypherState(TypedDict):
    question: str              # natural language question
    cypher: str                # generated Cypher query
    query_result: list         # raw query results
    answer: str                # natural language answer
    error: str                 # error message if query fails
    steps: list[str]           # audit trail


CYPHER_GEN_PROMPT = f"""You are a Neo4j Cypher expert. Given a natural language question,
generate a Cypher query to answer it.

Graph schema:
{GRAPH_SCHEMA}

Rules:
- Return ONLY the Cypher query, no explanation
- Use RETURN with meaningful aliases
- Match names EXACTLY as given (properties store proper case like 'Dave Wilson', 'Nova 7')
- Do NOT use toLower() on property values — names are stored in proper case
- Use OPTIONAL MATCH when a relationship might not exist
- Always include ORDER BY when returning lists"""


def generate_cypher(state: NLCypherState) -> dict:
    """Translate natural language to Cypher."""
    messages = [
        SystemMessage(content=CYPHER_GEN_PROMPT),
        HumanMessage(content=state["question"]),
    ]
    result = llm.invoke(messages)

    # Extract Cypher from response (strip markdown code blocks if present)
    cypher = result.content.strip()
    if cypher.startswith("```"):
        # Remove code block markers
        lines = cypher.split("\n")
        cypher = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    return {
        "cypher": cypher,
        "steps": state.get("steps", []) + [f"generated Cypher: {cypher[:60]}..."],
    }


def execute_query(state: NLCypherState) -> dict:
    """Execute the generated Cypher query against Neo4j."""
    try:
        with driver.session() as session:
            result = session.run(state["cypher"])
            records = [dict(r) for r in result]
        return {
            "query_result": records,
            "error": "",
            "steps": state["steps"] + [f"executed query, got {len(records)} results"],
        }
    except Exception as e:
        return {
            "query_result": [],
            "error": str(e),
            "steps": state["steps"] + [f"query error: {str(e)[:60]}"],
        }


ANSWER_PROMPT = """You are a helpful assistant. Given a user's question and the results
from a database query, provide a clear natural language answer.

If there was an error, explain that the query failed and suggest rephrasing.
Be concise and direct."""


def format_answer(state: NLCypherState) -> dict:
    """Format query results into a natural language answer."""
    if state.get("error"):
        context = f"Query error: {state['error']}"
    else:
        context = f"Query results: {json.dumps(state['query_result'], indent=2, default=str)}"

    messages = [
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=f"Question: {state['question']}\n\n{context}"),
    ]
    result = llm.invoke(messages)
    return {
        "answer": result.content,
        "steps": state["steps"] + ["formatted answer"],
    }


def build_nl_cypher_pipeline():
    """Build the NL-to-Cypher LangGraph pipeline."""
    builder = StateGraph(NLCypherState)
    builder.add_node("generate_cypher", generate_cypher)
    builder.add_node("execute_query", execute_query)
    builder.add_node("format_answer", format_answer)
    builder.add_edge(START, "generate_cypher")
    builder.add_edge("generate_cypher", "execute_query")
    builder.add_edge("execute_query", "format_answer")
    builder.add_edge("format_answer", END)
    return builder.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: RUN THE NL-TO-CYPHER AGENT
# ═══════════════════════════════════════════════════════════════════════════

NL_QUESTIONS = [
    "Who does Bob Martinez manage?",
    "What products are used by active projects?",
    "Who is the manager of the person who leads Project Atlas?",
    "How many people work on each project and what's the budget?",
    "What is the full management chain above Dave Wilson?",
    "Which department designed the Nova 7?",
]


def main():
    # Step 1: Build the graph
    print("=" * 64)
    print("  Module 04: Knowledge Graphs")
    print("  Neo4j + LangGraph NL-to-Cypher Agent")
    print("=" * 64)

    print("\n  Building knowledge graph...")
    build_graph()

    # Step 2: Direct Cypher demos
    run_cypher_demos()

    # Step 3: NL-to-Cypher agent
    pipeline = build_nl_cypher_pipeline()

    print(f"\n{'=' * 64}")
    print("  Natural Language → Cypher → Answer Pipeline")
    print("=" * 64)

    for i, question in enumerate(NL_QUESTIONS, 1):
        print(f"\n{'━' * 64}")
        print(f"  Question {i}: {question}")
        print(f"{'━' * 64}")

        config = {"configurable": {"thread_id": f"cypher-{i}"}}
        result = pipeline.invoke(
            {"question": question, "steps": []},
            config,
        )

        print(f"\n  Generated Cypher:\n    {result['cypher']}")
        if result.get("error"):
            print(f"\n  ⚠ Error: {result['error']}")
        else:
            print(f"\n  Raw results: {result['query_result']}")
        print(f"\n  Answer: {result['answer']}")
        print(f"  Steps: {result['steps']}")

    print(f"\n{'=' * 64}")
    print("  Key observations:")
    print("  • Multi-hop queries (manager of lead of project) are NATIVE")
    print("  • The graph schema in the prompt guides Cypher generation")
    print("  • Path queries return the full chain, not just endpoints")
    print("  • Aggregation (count, sum) works naturally in Cypher")
    print("  • Errors are caught and reported, not hallucinated over")
    print("=" * 64)

    driver.close()


if __name__ == "__main__":
    main()
