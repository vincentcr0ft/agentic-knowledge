"""
Module 06 — Agent Run Record
═════════════════════════════
Runs the ingest and query pipelines and prints a human-readable
record of every step: what the agent was given, what it did, why,
and exactly how the state changed.

The interview phase is interactive and is best demonstrated via
  python demo.py <statement_file>
"""

import json

from schema import (
    linearise_graph,
    run_schema_completeness,
    prioritise_gaps,
    NODE_TYPES,
    ONTOLOGY_META,
)
from ingest import build_ingest_graph, IngestState, driver
from query import build_query_graph, QueryState


# ─── Sample statement ────────────────────────────────────────────────────
STATEMENT = (
    "I was walking along King Street at approximately 2:15 PM on Tuesday "
    "when I heard a loud crash. I turned and saw a red car had collided "
    "with a cyclist at the junction of King Street and Queen's Road. The "
    "driver got out — a tall man wearing a dark jacket. He looked at the "
    "cyclist who was on the ground and then got back in his car and drove "
    "off heading north on Queen's Road. Another woman who was nearby "
    "called an ambulance. I stayed with the cyclist until the paramedics "
    "arrived about ten minutes later."
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_val(val, max_len: int = 120) -> str:
    """Format a value for display, truncating if needed."""
    if val is None or val == "" or val == []:
        return "(empty)"
    if isinstance(val, list):
        if not val:
            return "(empty list)"
        if len(val) <= 3:
            return json.dumps(val, ensure_ascii=False)
        return f"[{len(val)} items] first: {json.dumps(val[0], ensure_ascii=False)}"
    if isinstance(val, dict):
        if not val:
            return "(empty dict)"
        keys = list(val.keys())
        key_str = ", ".join(keys[:5])
        suffix = "…" if len(keys) > 5 else ""
        return "{" + key_str + suffix + "}"
    s = str(val)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def _print_state_diff(prev: dict, curr: dict, indent: str = "    "):
    """Print changed fields between two state snapshots."""
    all_keys = sorted(set(list(prev.keys()) + list(curr.keys())))
    changed = False
    for key in all_keys:
        old = prev.get(key)
        new = curr.get(key)
        if old != new:
            changed = True
            print(f"{indent}  {key}:")
            print(f"{indent}    before: {_fmt_val(old)}")
            print(f"{indent}    after:  {_fmt_val(new)}")
    if not changed:
        print(f"{indent}  (no state changes)")


# ═══════════════════════════════════════════════════════════════════════════
# Ingest pipeline narration
# ═══════════════════════════════════════════════════════════════════════════

def _narrate_ingest():
    """Run ingest and narrate each step with full state diffs."""
    from langgraph.checkpoint.memory import MemorySaver

    print(f"\n{'═' * 70}")
    print(f"  AGENT RUN RECORD — INGEST PIPELINE")
    print(f"  Ontology: {ONTOLOGY_META['name']} v{ONTOLOGY_META['version']}")
    print(f"{'═' * 70}")

    # Build the graph with checkpointing so we can inspect state history
    from langgraph.graph import END, START, StateGraph
    from ingest import parse_statement, extract_entities, resolve_entities, load_to_graph

    builder = StateGraph(IngestState)
    builder.add_node("parse_statement", parse_statement)
    builder.add_node("extract_entities", extract_entities)
    builder.add_node("resolve_entities", resolve_entities)
    builder.add_node("load_to_graph", load_to_graph)
    builder.add_edge(START, "parse_statement")
    builder.add_edge("parse_statement", "extract_entities")
    builder.add_edge("extract_entities", "resolve_entities")
    builder.add_edge("resolve_entities", "load_to_graph")
    builder.add_edge("load_to_graph", END)
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "inspect-ingest"}}

    initial_state: IngestState = {
        "raw_statement": STATEMENT,
        "segments": [],
        "extracted": {},
        "resolved": {},
        "load_summary": "",
        "steps": [],
    }

    # ── Print initial state ─────────────────────────────────────────────
    print(f"\n  ── Initial State ──────────────────────────────────────────────")
    print(f"    raw_statement: \"{STATEMENT[:80]}…\"")
    print(f"    segments:      (empty)")
    print(f"    extracted:     (empty)")
    print(f"    resolved:      (empty)")
    print(f"    load_summary:  (empty)")
    print(f"    steps:         (empty)")

    # ── Run and collect history ─────────────────────────────────────────
    result = graph.invoke(initial_state, config)
    history = list(reversed(list(graph.get_state_history(config))))

    # Walk consecutive pairs to show state diffs
    step_names = ["parse_statement", "extract_entities", "resolve_entities", "load_to_graph"]
    step_descs = {
        "parse_statement": (
            "Split the raw statement into sentences for provenance tracking.",
            "Each sentence becomes a source reference so any extracted fact"
            " can be traced back to the sentence it came from."
        ),
        "extract_entities": (
            "Send the full text to the LLM with a schema-guided prompt listing"
            f" all {len(NODE_TYPES)} node types and their required/optional properties.",
            "Schema-guided extraction constrains the LLM to produce only entity"
            " and relationship types defined in the ontology."
        ),
        "resolve_entities": (
            "Ask the LLM to identify co-referent entities (e.g. 'the driver'"
            " and 'he' referring to the same Person) and merge them.",
            "Without resolution, pronouns and descriptions create duplicate"
            " nodes, producing false gaps in the completeness analysis."
        ),
        "load_to_graph": (
            "Load resolved entities and relationships into Neo4j using"
            " parameterised MERGE queries. Every node gets provenance"
            f" properties and ontology_id='{ONTOLOGY_META['id']}'."
            " Then materialise the SOSA/PROV layer (Observation node,"
            " MADE_BY, OBSERVED, DERIVED_FROM relationships).",
            "MERGE ensures idempotency. Provenance lets us trace any graph"
            " fact back to the text and extraction round that created it."
            " SOSA/PROV materialisation completes the observation model."
        ),
    }

    step_idx = 0
    prev_vals = dict(initial_state)
    for cp in history:
        vals = cp.values
        current_steps = vals.get("steps", [])

        # Skip if no new steps were added
        if len(current_steps) <= len(prev_vals.get("steps", [])):
            continue

        if step_idx < len(step_names):
            name = step_names[step_idx]
            action_desc, why = step_descs[name]

            print(f"\n  ── Step {step_idx + 1}: {name} ─────────────────────────────────")
            print(f"  Action:  {action_desc}")
            print(f"  Why:     {why}")
            print(f"\n  State changes:")
            _print_state_diff(prev_vals, vals)

            step_idx += 1

        prev_vals = dict(vals)

    # ── Final state ─────────────────────────────────────────────────────
    triples = linearise_graph(driver)
    triple_count = triples.count("\n") + 1 if triples != "(empty graph)" else 0
    print(f"\n  ── Final Graph: {triple_count} triples ────────────────────────────")
    for line in triples.split("\n"):
        print(f"    {line}")

    gaps = run_schema_completeness(driver)
    gaps = prioritise_gaps(gaps)
    print(f"\n  ── Schema Gaps: {len(gaps)} ───────────────────────────────────────")
    for g in gaps[:8]:
        print(f"    [{g.priority:8s}] {g.gap_description}")
    if len(gaps) > 8:
        print(f"    … and {len(gaps) - 8} more")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Query pipeline narration
# ═══════════════════════════════════════════════════════════════════════════

def _narrate_query():
    """Run a query and narrate each step with full state diffs."""
    question = "What happened at the junction of King Street and Queen's Road?"

    print(f"\n{'═' * 70}")
    print(f"  AGENT RUN RECORD — QUERY PIPELINE")
    print(f"{'═' * 70}")

    graph = build_query_graph()
    config = {"configurable": {"thread_id": "inspect-q1"}}

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

    # ── Print initial state ─────────────────────────────────────────────
    print(f"\n  ── Initial State ──────────────────────────────────────────────")
    print(f"    question:           \"{question}\"")
    print(f"    detected_entities:  (empty)")
    print(f"    subgraph_triples:   (empty)")
    print(f"    provenance:         (empty)")
    print(f"    answer:             (empty)")

    result = graph.invoke(initial_state, config)
    history = list(reversed(list(graph.get_state_history(config))))

    step_names = ["receive_question", "retrieve_subgraph", "generate_answer"]
    step_descs = {
        "receive_question": (
            "Ask the LLM to identify key concepts in the question —"
            " people, vehicles, locations, events, times — and return"
            " them as a JSON array of search terms.",
            "Entity detection drives targeted graph retrieval. Instead"
            " of dumping the whole graph to the LLM, we find the"
            " relevant subgraph first."
        ),
        "retrieve_subgraph": (
            "For each detected entity, searched Neo4j for matching nodes"
            " (case-insensitive substring) and expanded to a 2-hop"
            " neighbourhood. Also retrieved the SOSA/PROV provenance"
            " chain (Observation → MADE_BY → witness, OBSERVED → events).",
            "Targeted retrieval provides focused context. The SOSA/PROV"
            " layer lets the LLM distinguish statement facts from"
            " follow-up facts and cite observation-level provenance."
        ),
        "generate_answer": (
            "Sent the subgraph triples + full graph + provenance summary"
            " to the LLM with a prompt requiring: (1) answer only from"
            " graph facts, (2) cite facts as [FACT: …], (3) show"
            " reasoning path, (4) distinguish stated/follow-up/inferred.",
            "Grounding prevents hallucination. Every claim must trace"
            " back to a graph triple. The provenance summary lets the"
            " LLM report confidence levels."
        ),
    }

    step_idx = 0
    prev_vals = dict(initial_state)
    for cp in history:
        vals = cp.values
        current_steps = vals.get("steps", [])

        if len(current_steps) <= len(prev_vals.get("steps", [])):
            continue

        if step_idx < len(step_names):
            name = step_names[step_idx]
            action_desc, why = step_descs[name]

            print(f"\n  ── Step {step_idx + 1}: {name} ─────────────────────────────────")
            print(f"  Action:  {action_desc}")
            print(f"  Why:     {why}")
            print(f"\n  State changes:")
            _print_state_diff(prev_vals, vals)

            step_idx += 1

        prev_vals = dict(vals)

    # ── Print the answer ────────────────────────────────────────────────
    answer = result.get("answer", "")
    print(f"\n  ── Answer ────────────────────────────────────────────────────")
    for line in answer.split("\n"):
        print(f"    {line}")
    print()


def main():
    print("=" * 70)
    print("  Module 06: Event Digital Twin — Agent Run Records")
    print("  Full state changes at every step of the pipeline.")
    print("=" * 70)

    _narrate_ingest()
    _narrate_query()

    driver.close()


if __name__ == "__main__":
    main()