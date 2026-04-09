"""
Module 06: Event Digital Twin — Full End-to-End Demo
═════════════════════════════════════════════════════

Orchestrates all three phases of the event digital twin:

  Phase 1 · INGEST    — witness statement → knowledge graph
  Phase 2 · INTERVIEW — self-resolution + follow-up questions → graph enrichment
  Phase 3 · QUERY     — grounded Q&A over the completed graph

Usage:
  python demo.py <statement_file>          Full pipeline with a statement file
  python demo.py --interview               Interview only (graph must exist)
  python demo.py --query                   Query demo (graph must exist)
  python demo.py --interactive             Interactive query REPL
  python demo.py <file> --skip-interview   Ingest + query, no interview

The system will:
  1. Ingest the statement and build the initial knowledge graph
  2. Attempt to self-resolve gaps from the existing text
  3. Ask the human only about gaps it cannot resolve itself
  4. Write a plain-text transcript on completion

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
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
# Statement loading
# ═══════════════════════════════════════════════════════════════════════════

def load_statement(file_path: str | None) -> str:
    """Load a witness statement from a file path.

    If no file is provided, prompt the user to type/paste one.
    """
    if file_path:
        p = Path(file_path)
        if not p.exists():
            print(f"  ✗ File not found: {file_path}")
            sys.exit(1)
        text = p.read_text().strip()
        print(f"  ✓ Loaded statement from {p.name} ({len(text)} chars)")
        return text

    # Interactive: prompt user for statement
    print(f"\n{'─' * 70}")
    print("  No statement file provided.")
    print("  Type or paste your witness statement below.")
    print("  Press Enter twice (blank line) when finished.")
    print(f"{'─' * 70}\n")

    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)

    text = " ".join(lines).strip()
    if not text:
        print("  ✗ No statement provided. Exiting.")
        sys.exit(1)

    print(f"\n  ✓ Received statement ({len(text)} chars)")
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Ingest
# ═══════════════════════════════════════════════════════════════════════════

def phase_ingest(statement: str):
    """Run the ingest pipeline: text → extraction → graph."""
    print(f"\n{'═' * 70}")
    print(f"  PHASE 1: INGEST")
    print(f"  Witness statement → Knowledge graph")
    print(f"{'═' * 70}")
    print(f"\n  Statement ({len(statement)} chars):")
    for i, line in enumerate(statement.split(". "), 1):
        print(f"    [{i}] {line.strip()}{'.' if not line.strip().endswith('.') else ''}")
    print()

    result = ingest_statement(statement)

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

def phase_interview(statement: str = "", transcript_path: str | None = None):
    """Run the interview loop: self-resolve → follow-up → graph update."""
    print(f"\n{'═' * 70}")
    print(f"  PHASE 2: INTERVIEW")
    print(f"  Self-resolution → Follow-up questions → Graph enrichment")
    print(f"{'═' * 70}")
    print(f"\n  The system will first try to fill gaps from the existing text.")
    print(f"  For anything it can't resolve, it will ask you directly.")
    print(f"  Type 'done' or 'quit' to end the interview early.\n")

    run_interview(
        max_rounds=5,
        thread_id="demo-interview",
        statement=statement,
        transcript_path=transcript_path,
    )

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
Usage: python demo.py [statement_file] [options]

Arguments:
  statement_file    Path to a .txt file containing the witness statement.
                    If omitted, you will be prompted to type one.

Options:
  --skip-interview  Run Phase 1 + Phase 3, skipping the interview
  --interview       Run only Phase 2 (interview) — assumes graph is populated
  --query           Run only Phase 3 (query demo) — assumes graph is populated
  --interactive     Run Phase 3 in interactive mode
  --transcript FILE Write interview transcript to FILE (default: transcript.txt)
  --help            Show this help message

Examples:
  python demo.py statements/my_statement.txt
  python demo.py                                   # interactive statement entry
  python demo.py statement.txt --skip-interview
  python demo.py --interview --transcript log.txt
"""


def _parse_args():
    """Parse command-line arguments."""
    args = sys.argv[1:]
    flags = set()
    file_path = None
    transcript_path = "transcript.txt"

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--transcript" and i + 1 < len(args):
            transcript_path = args[i + 1]
            i += 2
            continue
        elif arg.startswith("--") or arg.startswith("-"):
            flags.add(arg)
        else:
            file_path = arg
        i += 1

    return file_path, flags, transcript_path


def main():
    file_path, flags, transcript_path = _parse_args()

    if "--help" in flags or "-h" in flags:
        print(USAGE)
        return

    print("=" * 70)
    print("  Module 06: Event Digital Twin")
    print("  Statement → Knowledge Graph → Self-Resolution → Grounded Q&A")
    print("=" * 70)

    if "--interview" in flags:
        phase_interview(transcript_path=transcript_path)

    elif "--query" in flags:
        phase_query(interactive=False)

    elif "--interactive" in flags:
        phase_query(interactive=True)

    else:
        statement = load_statement(file_path)

        if "--skip-interview" in flags:
            phase_ingest(statement)
            phase_query(interactive=False)
        else:
            phase_ingest(statement)
            phase_interview(
                statement=statement,
                transcript_path=transcript_path,
            )
            phase_query(interactive=False)

    print(f"\n{'=' * 70}")
    print("  Key concepts demonstrated:")
    print("  • Event-centric ontology (PROV-O + SOSA + Schema.org Event)")
    print("  • Schema-guided entity extraction with qwen2.5:7b")
    print("  • Self-resolution — fills gaps from existing text before asking")
    print("  • Human-in-the-loop interview via LangGraph interrupt()")
    print("  • Plain-text transcript of all decisions and reasoning")
    print("  • Provenance tracking — every fact cites its source")
    print("  • Grounded Q&A with [FACT: ...] citations")
    print("=" * 70)

    driver.close()


if __name__ == "__main__":
    main()
