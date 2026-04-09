"""
Module 06: Event Digital Twin — Full End-to-End Demo
═════════════════════════════════════════════════════

Orchestrates all three phases of the event digital twin:

  Phase 1 · INGEST    — witness statement → knowledge graph
  Phase 2 · INTERVIEW — gap analysis → follow-up questions → graph enrichment
  Phase 3 · QUERY     — grounded Q&A over the completed graph

Demonstrates: event-centric ontology (PROV-O + SOSA + Schema.org Event),
schema-guided extraction, three-level gap analysis, human-in-the-loop
interview via LangGraph interrupt(), and provenance-cited answers.

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
"""

from __future__ import annotations

import sys
from pathlib import Path

from neo4j import GraphDatabase

from schema import linearise_graph, run_schema_completeness, prioritise_gaps
from ingest import ingest_statement
from interview import run_interview
from query import run_demo as run_query_demo, ask, run_interactive


# ─── Connections ──────────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ═══════════════════════════════════════════════════════════════════════════
# Sample statement
# ═══════════════════════════════════════════════════════════════════════════

STATEMENT_PATH = Path(__file__).parent / "statements" / "king_street_collision.txt"

DEFAULT_STATEMENT = (
    "I was walking along King Street at approximately 2:15 PM on Tuesday "
    "when I heard a loud crash. I turned and saw a red car had collided "
    "with a cyclist at the junction of King Street and Queen's Road. The "
    "driver got out — a tall man wearing a dark jacket. He looked at the "
    "cyclist who was on the ground and then got back in his car and drove "
    "off heading north on Queen's Road. Another woman who was nearby "
    "called an ambulance. I stayed with the cyclist until the paramedics "
    "arrived about ten minutes later."
)


def load_statement() -> str:
    """Load the witness statement from file or fallback."""
    if STATEMENT_PATH.exists():
        return STATEMENT_PATH.read_text().strip()
    return DEFAULT_STATEMENT


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Ingest
# ═══════════════════════════════════════════════════════════════════════════

def phase_ingest(statement: str):
    """Run the ingest pipeline: text → extraction → graph."""
    print(f"{'═' * 70}")
    print(f"  PHASE 1: INGEST")
    print(f"  Witness statement → Knowledge graph")
    print(f"{'═' * 70}")
    print(f"\n  Statement ({len(statement)} chars):")
    print(f"  {statement[:120]}…\n")

    result = ingest_statement(statement)

    # Summary
    print(f"\n{'─' * 70}")
    print(f"  Ingest complete.")
    triples = linearise_graph(driver)
    triple_count = triples.count("\n") + 1 if triples != "(empty graph)" else 0
    print(f"  Graph now has {triple_count} triples")

    gaps = run_schema_completeness(driver)
    gaps = prioritise_gaps(gaps)
    print(f"  Schema gaps remaining: {len(gaps)}")
    if gaps:
        for g in gaps[:5]:
            print(f"    [{g.priority:8s}] {g.gap_description}")
        if len(gaps) > 5:
            print(f"    … and {len(gaps) - 5} more")

    print(f"{'─' * 70}\n")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Interview
# ═══════════════════════════════════════════════════════════════════════════

def phase_interview():
    """Run the interview loop: gaps → questions → answers → graph update."""
    print(f"\n{'═' * 70}")
    print(f"  PHASE 2: INTERVIEW")
    print(f"  Gap analysis → Follow-up questions → Graph enrichment")
    print(f"{'═' * 70}")
    print(f"\n  The system will analyse the graph for gaps and ask ")
    print(f"  follow-up questions. Answer as the witness would.")
    print(f"  Type 'done' or 'quit' to end the interview early.\n")

    run_interview(max_rounds=5, thread_id="demo-interview")

    # Post-interview summary
    print(f"\n{'─' * 70}")
    triples = linearise_graph(driver)
    triple_count = triples.count("\n") + 1 if triples != "(empty graph)" else 0
    print(f"  Interview complete. Graph now has {triple_count} triples")

    gaps = run_schema_completeness(driver)
    print(f"  Schema gaps remaining: {len(gaps)}")
    print(f"{'─' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Query
# ═══════════════════════════════════════════════════════════════════════════

def phase_query(interactive: bool = False):
    """Run the query phase: questions → graph retrieval → grounded answers."""
    print(f"\n{'═' * 70}")
    print(f"  PHASE 3: QUERY")
    print(f"  Grounded Q&A over the event graph")
    print(f"{'═' * 70}\n")

    if interactive:
        run_interactive()
    else:
        run_query_demo()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

USAGE = """\
Usage: python demo.py [options]

Options:
  (no args)       Run all three phases end-to-end (ingest → interview → query)
  --ingest-only   Run only Phase 1 (ingest)
  --interview     Run only Phase 2 (interview) — assumes graph is populated
  --query         Run only Phase 3 (query demo) — assumes graph is populated
  --interactive   Run Phase 3 in interactive mode
  --skip-interview  Run Phase 1 + Phase 3, skipping the interview
  --help          Show this help message
"""


def main():
    args = set(sys.argv[1:])

    if "--help" in args or "-h" in args:
        print(USAGE)
        return

    print("=" * 70)
    print("  Module 06: Event Digital Twin")
    print("  Statement → Knowledge Graph → Gap Analysis → Grounded Q&A")
    print("=" * 70)

    statement = load_statement()

    if "--ingest-only" in args:
        phase_ingest(statement)

    elif "--interview" in args:
        phase_interview()

    elif "--query" in args:
        phase_query(interactive=False)

    elif "--interactive" in args:
        phase_query(interactive=True)

    elif "--skip-interview" in args:
        phase_ingest(statement)
        phase_query(interactive=False)

    else:
        # Full pipeline: ingest → interview → query
        phase_ingest(statement)
        phase_interview()
        phase_query(interactive=False)

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Key concepts demonstrated:")
    print("  • Event-centric ontology (PROV-O + SOSA + Schema.org Event)")
    print("  • Schema-guided entity extraction with qwen2.5:7b")
    print("  • Coreference resolution with over-merge safeguard")
    print("  • Three-level gap analysis (schema + narrative + investigative)")
    print("  • Human-in-the-loop interview via LangGraph interrupt()")
    print("  • Provenance tracking — every fact cites its source")
    print("  • Grounded Q&A with [FACT: ...] citations")
    print("  • Linearised triples for LLM graph comprehension (Dai et al.)")
    print("=" * 70)

    driver.close()


if __name__ == "__main__":
    main()
