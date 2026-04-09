"""
06 · Event Digital Twin — Query Pipeline
═════════════════════════════════════════

Queryable interface over the completed event graph. The user "queries the
witness" — asking natural-language questions about the event. Answers are
grounded exclusively in the knowledge graph (never from the LLM's own
knowledge) and cite specific graph facts, observation provenance, and
confidence levels.

Pipeline (LangGraph):

  receive_question  →  retrieve_subgraph  →  generate_answer

Key design (per research):
  - Linearised triples for LLM consumption (Dai et al.)
  - Provenance citations — every claim cites the graph facts that support it
  - SOSA/PROV awareness — answers reference observation source and confidence
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
    3. Collect provenance via SOSA/PROV layer (Observation nodes,
       DERIVED_FROM edges, confidence levels)
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
                    n.source_type AS start_source_type,
                    n.confidence AS start_confidence,
                    labels(m)[0] AS end_label,
                    coalesce(m.description, m.name_or_description,
                             m.name, m.value, m.summary) AS end_desc,
                    m.source_type AS end_source_type,
                    m.confidence AS end_confidence,
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

                        # Collect provenance from source_type + confidence
                        for prefix in ("start", "end"):
                            src_type = rec_dict.get(f"{prefix}_source_type")
                            confidence = rec_dict.get(f"{prefix}_confidence")
                            desc = rec_dict.get(f"{prefix}_desc", "")
                            if src_type:
                                provenance_records.append({
                                    "entity": desc,
                                    "source_type": src_type,
                                    "confidence": confidence or "unknown",
                                })
            except Exception as e:
                print(f"    ⚠ Retrieval error for '{entity_term}': {e}")

        # ── Pull SOSA/PROV provenance chain ─────────────────────────────
        # Retrieve Observation nodes and their links to provide the LLM
        # with observation-level context (who made it, what was observed)
        prov_cypher = """
            MATCH (obs:Observation)-[:MADE_BY]->(w:Person)
            OPTIONAL MATCH (obs)-[:OBSERVED]->(e:Event)
            RETURN obs.description AS observation,
                   obs.observation_type AS obs_type,
                   obs.confidence AS obs_confidence,
                   w.name_or_description AS witness,
                   w.role AS witness_role,
                   collect(DISTINCT e.description) AS events_observed
        """
        try:
            for rec in session.run(prov_cypher):
                rec_dict = dict(rec)
                if rec_dict.get("observation"):
                    obs_desc = rec_dict["observation"]
                    witness = rec_dict.get("witness", "unknown")
                    events = rec_dict.get("events_observed", [])
                    matched_triples.append(
                        f"(Observation: {obs_desc}) -[MADE_BY]-> "
                        f"(Person: {witness})"
                    )
                    for evt in events:
                        matched_triples.append(
                            f"(Observation: {obs_desc}) -[OBSERVED]-> "
                            f"(Event: {evt})"
                        )
                    provenance_records.append({
                        "entity": obs_desc,
                        "source_type": rec_dict.get("obs_type", "unknown"),
                        "confidence": rec_dict.get("obs_confidence", "unknown"),
                        "witness": witness,
                    })
        except Exception as e:
            print(f"    ⚠ Provenance retrieval error: {e}")

    # Deduplicate triples
    unique_triples = list(dict.fromkeys(matched_triples))
    subgraph_text = "\n".join(unique_triples) if unique_triples else "(no matches)"

    # Deduplicate provenance
    seen_prov = set()
    unique_prov = []
    for p in provenance_records:
        key = f"{p.get('entity', '')}::{p.get('source_type', '')}"
        if key not in seen_prov:
            seen_prov.add(key)
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
You are an investigative analyst. The user is querying a knowledge graph \
built from a witness's account of an event. Your answers must be grounded \
ONLY in the graph facts provided below — never in your own knowledge.

The graph was constructed from a formal witness statement and may have been \
enriched through follow-up interview questions. Facts have provenance \
(source_type: "statement" or "interview_round_N") and confidence levels.

RELEVANT SUBGRAPH (directly related to the question):
{subgraph}

FULL EVENT GRAPH (for additional context):
{full_graph}

PROVENANCE SUMMARY:
{provenance}

RULES:
1. Ground every claim in specific graph facts — cite them as [FACT: ...]
2. If the graph does not contain enough information to answer, say so \
clearly — do NOT speculate
3. Show your REASONING PATH — which graph nodes and relationships you \
traversed to reach your answer
4. Distinguish between:
   - STATED facts (from the original witness statement, confidence: high)
   - FOLLOW-UP facts (from interview answers, confidence: medium)
   - INFERRED connections (derived from graph structure)
5. If there are gaps or uncertainties in the graph, mention them — these \
represent things the witness could not confirm
6. Be precise and investigative in tone
7. Reference the witness's perspective — this is their account of events"""


def generate_answer(state: QueryState) -> dict:
    """Generate a grounded answer with citations and reasoning path.

    Uses KAPING-style fact prepending: relevant subgraph first, then
    full graph as context. Includes SOSA/PROV provenance so the LLM
    can distinguish statement facts from follow-up facts.
    """
    question = state["question"]
    subgraph = state.get("subgraph_triples", "")
    full_graph = state.get("full_graph_triples", "")
    prov = state.get("provenance", [])

    # Format provenance for the prompt
    prov_lines = []
    for p in prov:
        entity = p.get("entity", "?")
        src = p.get("source_type", "unknown")
        conf = p.get("confidence", "unknown")
        witness = p.get("witness", "")
        line = f"  • {entity} — source: {src}, confidence: {conf}"
        if witness:
            line += f", observer: {witness}"
        prov_lines.append(line)
    prov_text = "\n".join(prov_lines) if prov_lines else "(no provenance data)"

    prompt = ANSWER_PROMPT.format(
        subgraph=subgraph,
        full_graph=full_graph,
        provenance=prov_text,
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
# CLI — interactive query session
# ═══════════════════════════════════════════════════════════════════════════


def run_interactive():
    """Run an interactive query session — the user queries the witness."""
    graph = build_query_graph()

    print(f"{'═' * 70}")
    print(f"  EVENT DIGITAL TWIN — QUERY THE WITNESS")
    print(f"{'═' * 70}")

    # Show current graph summary
    triples = linearise_graph(driver)
    triple_count = triples.count("\n") + 1 if triples != "(empty graph)" else 0
    print(f"\n  The knowledge graph contains {triple_count} triples built from")
    print(f"  the witness's statement and any follow-up interview answers.")
    print(f"  Ask anything about the event. Type 'quit' to exit.\n")

    query_num = 0
    while True:
        try:
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q", "done"):
            break

        query_num += 1
        config = {"configurable": {"thread_id": f"interactive-{query_num}"}}

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

        print()
        result = graph.invoke(initial_state, config)

        answer = result.get("answer", "No answer generated.")
        print(f"\n  Witness account:\n")
        for line in answer.split("\n"):
            print(f"    {line}")

        # Show provenance summary
        prov = result.get("provenance", [])
        if prov:
            sources = set()
            for p in prov:
                src = p.get("source_type", "")
                conf = p.get("confidence", "")
                if src:
                    sources.add(f"{src} ({conf})" if conf else src)
            if sources:
                print(f"\n  Sources: {', '.join(sorted(sources))}")

        print()


if __name__ == "__main__":
    run_interactive()
