"""
06 · Event Digital Twin — Query Pipeline
═════════════════════════════════════════

Queryable interface over the completed event graph. Receives natural-language
questions, retrieves relevant subgraphs, and generates answers grounded
exclusively in graph facts — never from the LLM's own knowledge.

Pipeline (LangGraph):

  receive_question  →  retrieve_subgraph  →  generate_answer

Key design (per research):
  - Linearised triples for LLM consumption (Dai et al.)
  - Provenance citations — every claim cites the graph facts that support it
  - KAPING-style fact prepending (Baek et al.)
  - MindMap-style reasoning paths (Wen et al.)

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
  - Graph already populated by ingest.py (+ optionally interview.py)
"""

from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from neo4j import GraphDatabase

from schema import linearise_graph


# ─── Connections ──────────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════

class QueryState(TypedDict):
    question: str                       # user's natural language question
    detected_entities: list[str]        # entities found in the question
    subgraph_triples: str               # relevant portion of graph as triples
    full_graph_triples: str             # full graph for context
    provenance: list[dict]              # source attribution for retrieved facts
    answer: str                         # generated answer
    reasoning_path: str                 # LLM's reasoning through the graph
    steps: list[str]                    # audit trail


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — receive_question (detect entities in question)
# ═══════════════════════════════════════════════════════════════════════════

ENTITY_DETECT_PROMPT = """\
You are analysing a question about a witnessed event (e.g. a traffic collision, \
an assault, a robbery). Identify the key concepts and entities mentioned in \
the question that should be used to search a knowledge graph.

Extract:
- Person references ("the driver", "the suspect", "the witness", "the woman")
- Vehicle references ("the car", "the red car", "the vehicle")
- Location references ("King Street", "the junction")
- Event references ("the collision", "the crash", "the hit-and-run")
- Time references ("2:15 PM", "Tuesday")
- Object references ("the jacket", "the ambulance")

Return a JSON array of search terms. Include both specific names and descriptive \
phrases. Return ONLY the JSON array.

Example: ["the driver", "red car", "collision", "King Street"]"""


def receive_question(state: QueryState) -> dict:
    """Detect entities/concepts in the question for targeted retrieval."""
    question = state["question"]

    messages = [
        SystemMessage(content=ENTITY_DETECT_PROMPT),
        HumanMessage(content=question),
    ]
    result = llm.invoke(messages)

    entities = []
    try:
        content = result.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)
        if isinstance(parsed, list):
            entities = [str(e) for e in parsed]
    except json.JSONDecodeError:
        pass

    if not entities:
        # Fallback: extract nouns/phrases from the question
        entities = [state["question"]]

    print(f"  ▸ Detected entities: {entities}")

    return {
        "detected_entities": entities,
        "steps": state.get("steps", []) + [
            f"receive_question: detected {len(entities)} entities"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — retrieve_subgraph
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_subgraph(state: QueryState) -> dict:
    """Retrieve the relevant portion of the graph for the question.

    Strategy:
    1. Full-text search for entities mentioned in the question
    2. Expand to 2-hop neighbourhood from matched nodes
    3. Collect provenance (source text) for each matched fact
    4. Also provide the full graph as fallback context
    """
    entities = state.get("detected_entities", [])
    matched_triples = []
    provenance_records = []

    with driver.session() as session:
        for entity_term in entities:
            # Search for nodes matching the entity term (case-insensitive
            # substring match across primary descriptor properties)
            cypher = """
                MATCH (n)
                WHERE toLower(coalesce(n.description, n.name_or_description,
                      n.name, n.value, n.summary, ''))
                      CONTAINS toLower($term)
                WITH n LIMIT 5
                OPTIONAL MATCH path = (n)-[r*1..2]-(m)
                WITH n, m, r,
                     [rel IN r | type(rel)] AS rel_types
                RETURN
                    labels(n)[0] AS start_label,
                    coalesce(n.description, n.name_or_description,
                             n.name, n.value, n.summary) AS start_desc,
                    n.source AS start_source,
                    n.source_type AS start_source_type,
                    labels(m)[0] AS end_label,
                    coalesce(m.description, m.name_or_description,
                             m.name, m.value, m.summary) AS end_desc,
                    m.source AS end_source,
                    rel_types
                LIMIT 30
            """
            try:
                records = session.run(cypher, term=entity_term)
                for rec in records:
                    rec_dict = dict(rec)
                    if rec_dict.get("end_desc"):
                        rel_chain = " → ".join(rec_dict.get("rel_types", []))
                        triple = (
                            f"({rec_dict['start_label']}: {rec_dict['start_desc']}) "
                            f"-[{rel_chain}]-> "
                            f"({rec_dict['end_label']}: {rec_dict['end_desc']})"
                        )
                        matched_triples.append(triple)

                        # Collect provenance
                        for source_key in ("start_source", "end_source"):
                            src = rec_dict.get(source_key)
                            src_type = rec_dict.get(
                                source_key.replace("source", "source_type"), ""
                            )
                            if src:
                                provenance_records.append({
                                    "source": src,
                                    "source_type": src_type or "unknown",
                                })
            except Exception as e:
                print(f"    ⚠ Retrieval error for '{entity_term}': {e}")

    # Deduplicate triples
    unique_triples = list(dict.fromkeys(matched_triples))
    subgraph_text = "\n".join(unique_triples) if unique_triples else "(no matches)"

    # Deduplicate provenance
    seen_sources = set()
    unique_prov = []
    for p in provenance_records:
        if p["source"] not in seen_sources:
            seen_sources.add(p["source"])
            unique_prov.append(p)

    # Full graph as fallback
    full_graph = linearise_graph(driver)

    print(f"  ▸ Retrieved {len(unique_triples)} relevant triples "
          f"({len(unique_prov)} provenance records)")

    return {
        "subgraph_triples": subgraph_text,
        "full_graph_triples": full_graph,
        "provenance": unique_prov,
        "steps": state.get("steps", []) + [
            f"retrieve_subgraph: {len(unique_triples)} triples, "
            f"{len(unique_prov)} provenance records"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — generate_answer
# ═══════════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = """\
You are an investigative analyst answering questions about a witnessed event. \
You must answer ONLY from the knowledge graph facts provided below — never \
from your own knowledge.

RELEVANT SUBGRAPH (directly related to the question):
{subgraph}

FULL EVENT GRAPH (for additional context):
{full_graph}

RULES:
1. Ground every claim in specific graph facts — cite them as [FACT: ...]
2. If the graph does not contain enough information to answer, say so clearly
3. Show your REASONING PATH — which graph nodes and relationships you \
traversed to reach your answer
4. Distinguish between facts that are CERTAIN (directly stated) and those \
that are INFERRED (derived from connections)
5. Be precise and investigative in tone
6. If there are gaps or uncertainties in the graph, mention them"""


def generate_answer(state: QueryState) -> dict:
    """Generate a grounded answer with citations and reasoning path.

    Uses KAPING-style fact prepending: relevant subgraph first, then
    full graph as context.  Answer must cite specific graph facts.
    """
    question = state["question"]
    subgraph = state.get("subgraph_triples", "")
    full_graph = state.get("full_graph_triples", "")

    prompt = ANSWER_PROMPT.format(
        subgraph=subgraph,
        full_graph=full_graph,
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=question),
    ]
    result = llm.invoke(messages)
    answer = result.content

    # Extract reasoning path if the LLM included one
    reasoning = ""
    reasoning_match = re.search(
        r"(?:REASONING PATH|Reasoning|Path)[\s:]*(.+?)(?=\n\n|\Z)",
        answer,
        re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    print(f"  ▸ Generated answer ({len(answer)} chars)")

    return {
        "answer": answer,
        "reasoning_path": reasoning,
        "steps": state.get("steps", []) + [
            f"generate_answer: {len(answer)} chars"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_query_graph() -> StateGraph:
    """Build the query pipeline as a LangGraph."""
    builder = StateGraph(QueryState)

    builder.add_node("receive_question", receive_question)
    builder.add_node("retrieve_subgraph", retrieve_subgraph)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "receive_question")
    builder.add_edge("receive_question", "retrieve_subgraph")
    builder.add_edge("retrieve_subgraph", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

_query_graph = None
_query_counter = 0


def ask(question: str) -> str:
    """Ask a question about the event graph. Returns the grounded answer."""
    global _query_graph, _query_counter

    if _query_graph is None:
        _query_graph = build_query_graph()

    _query_counter += 1
    config = {"configurable": {"thread_id": f"query-{_query_counter}"}}

    initial_state: QueryState = {
        "question": question,
        "detected_entities": [],
        "subgraph_triples": "",
        "full_graph_triples": "",
        "provenance": [],
        "answer": "",
        "reasoning_path": "",
        "steps": [],
    }

    result = _query_graph.invoke(initial_state, config)
    return result.get("answer", "No answer generated.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI — interactive query loop + demo questions
# ═══════════════════════════════════════════════════════════════════════════

DEMO_QUESTIONS = [
    "What happened at the junction of King Street and Queen's Road?",
    "Can you describe the suspect and what vehicle they were driving?",
    "What is the timeline of events?",
    "Who called the ambulance and what happened to the cyclist?",
    "In which direction did the suspect flee?",
]


def run_demo():
    """Run the demo questions against the current graph."""
    graph = build_query_graph()

    print(f"{'═' * 70}")
    print(f"  EVENT DIGITAL TWIN — QUERY PHASE")
    print(f"{'═' * 70}")

    # Show current graph state
    triples = linearise_graph(driver)
    print(f"\n  Current graph ({triples.count(chr(10)) + 1} triples):")
    for line in triples.split("\n"):
        print(f"    {line}")
    print()

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        config = {"configurable": {"thread_id": f"demo-{i}"}}

        print(f"{'─' * 70}")
        print(f"  Q{i}: {question}")
        print(f"{'─' * 70}")

        initial_state: QueryState = {
            "question": question,
            "detected_entities": [],
            "subgraph_triples": "",
            "full_graph_triples": "",
            "provenance": [],
            "answer": "",
            "reasoning_path": "",
            "steps": [],
        }

        result = graph.invoke(initial_state, config)

        print(f"\n  Answer:\n")
        for line in result.get("answer", "").split("\n"):
            print(f"    {line}")
        print()

        # Provenance
        prov = result.get("provenance", [])
        if prov:
            print(f"  Sources:")
            for p in prov[:5]:
                print(f"    [{p.get('source_type', '?')}] {p.get('source', '?')}")
            if len(prov) > 5:
                print(f"    … and {len(prov) - 5} more")
        print()


def run_interactive():
    """Run an interactive query session."""
    print(f"{'═' * 70}")
    print(f"  EVENT DIGITAL TWIN — QUERY PHASE (interactive)")
    print(f"{'═' * 70}")
    print(f"  Ask questions about the event. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        print()
        answer = ask(question)
        print(f"\n  Answer:\n")
        for line in answer.split("\n"):
            print(f"    {line}")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_demo()
