"""
Module 08: Digital Twin — Full End-to-End Demo
═══════════════════════════════════════════════

Orchestrates the event digital twin pipeline with pluggable ontology
support and integrated quality assessment.

  Phase 1 · INGEST    — witness statement → knowledge graph
  Phase 2 · INTERVIEW — self-resolution + follow-up → graph enrichment
  Phase 3 · QUALITY   — multi-dimensional graph quality assessment
  Phase 4 · QUERY     — grounded Q&A over the completed graph

Usage:
  python demo.py <statement_file>                          Full pipeline
  python demo.py <file> --ontology sem-event-v1            Use SEM ontology
  python demo.py --query                                   Query only
  python demo.py <file> --skip-interview                   Skip interview
  python demo.py --quality                                 Quality probe only
  python demo.py --hallucination-check <file>              Assess hallucination

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
"""

from __future__ import annotations

import sys
from pathlib import Path

from neo4j import GraphDatabase

from schema import (
    select_ontology,
    get_active_spec,
    linearise_graph,
    run_schema_completeness,
    prioritise_gaps,
    ONTOLOGY_REGISTRY,
)
from ingest import ingest_statement
from interview import run_interview
from query import ask, run_interactive


# ─── Connections ──────────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ═══════════════════════════════════════════════════════════════════════════
# Statement loading
# ═══════════════════════════════════════════════════════════════════════════

def load_statement(file_path: str | None) -> str:
    if file_path:
        p = Path(file_path)
        if not p.exists():
            print(f"  ✗ File not found: {file_path}")
            sys.exit(1)
        text = p.read_text().strip()
        print(f"  ✓ Loaded statement from {p.name} ({len(text)} chars)")
        return text

    print(f"\n{'─' * 70}")
    print("  No statement file provided — type/paste your statement below.")
    print("  Press Enter twice (blank line) to finish.")
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
    spec = get_active_spec()
    print(f"\n{'═' * 70}")
    print(f"  PHASE 1: INGEST  (ontology: {spec.name})")
    print(f"  Witness statement → Knowledge graph")
    print(f"{'═' * 70}\n")

    result = ingest_statement(statement)

    triples = linearise_graph(driver)
    triple_count = triples.count("\n") + 1 if triples != "(empty graph)" else 0
    print(f"\n  Graph: {triple_count} triples")

    gaps = run_schema_completeness(driver)
    gaps = prioritise_gaps(gaps)
    print(f"  Schema gaps: {len(gaps)}")
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
    print(f"\n{'═' * 70}")
    print(f"  PHASE 2: INTERVIEW")
    print(f"  Self-resolution → Follow-up → Graph enrichment")
    print(f"{'═' * 70}\n")

    run_interview(
        max_rounds=5,
        thread_id="demo-interview",
        statement=statement,
        transcript_path=transcript_path,
    )

    triples = linearise_graph(driver)
    triple_count = triples.count("\n") + 1 if triples != "(empty graph)" else 0
    print(f"\n  Interview complete. Graph: {triple_count} triples")
    print(f"{'─' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Quality Assessment
# ═══════════════════════════════════════════════════════════════════════════

def phase_quality(source_text: str | None = None, skip_llm: bool = False):
    """Run quality assessment using the 07-graph-quality module."""
    print(f"\n{'═' * 70}")
    print(f"  PHASE 3: QUALITY ASSESSMENT")
    print(f"{'═' * 70}\n")

    # Import from 07-graph-quality
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07-graph-quality"))

    from cypher_probes import (
        probe_schema_population,
        probe_structural_connectivity,
        probe_consistency,
        probe_source_grounding,
    )
    from quality_core import run_quality_probe

    # Phase 1: structural probes
    spec = get_active_spec()
    labels = list(spec.node_types.keys())
    results = [
        probe_schema_population(driver, labels),
        probe_structural_connectivity(driver),
        probe_consistency(driver),
        probe_source_grounding(driver),
    ]

    for r in results:
        print(f"  {r.dimension:20s}  {r.score:.2f}  ({len(r.violations)} violations)")

    # Phase 2: LLM probes (optional)
    if not skip_llm:
        from llm_probes import probe_coherence, probe_faithfulness

        triples = linearise_graph(driver)
        coherence = probe_coherence(triples)
        results.append(coherence)
        print(f"  {'coherence':20s}  {coherence.score:.2f}  ({len(coherence.violations)} violations)")

        if source_text:
            faithfulness = probe_faithfulness(triples, source_text)
            results.append(faithfulness)
            print(f"  {'faithfulness':20s}  {faithfulness.score:.2f}  "
                  f"({len(faithfulness.violations)} violations)")

    # Phase 3: SHACL (if ontology provides shapes)
    try:
        from shacl_probes import probe_shacl
        from schema import get_shacl_shapes
        shapes_ttl = get_shacl_shapes()
        if shapes_ttl:
            shacl_result = probe_shacl(driver, shapes_ttl=shapes_ttl)
            results.append(shacl_result)
            print(f"  {'shacl':20s}  {shacl_result.score:.2f}  "
                  f"({len(shacl_result.violations)} violations)")
    except Exception as e:
        print(f"  SHACL skipped: {e}")

    report = run_quality_probe(results)
    print(f"\n  Overall score: {report.overall_score:.2f}")

    if report.recommendations:
        print(f"\n  Recommendations:")
        for rec in report.recommendations[:5]:
            print(f"    • {rec}")

    print(f"{'─' * 70}\n")
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Query
# ═══════════════════════════════════════════════════════════════════════════

def phase_query():
    print(f"\n{'═' * 70}")
    print(f"  PHASE 4: QUERY THE WITNESS")
    print(f"  Grounded Q&A over the event graph")
    print(f"{'═' * 70}\n")
    run_interactive()


# ═══════════════════════════════════════════════════════════════════════════
# Hallucination Assessment (stub — to be implemented)
# ═══════════════════════════════════════════════════════════════════════════

def assess_hallucination(source_text: str):
    """Assess hallucination rate of the knowledge graph against source text.

    This compares every graph fact against the original statement to
    identify facts that were hallucinated during extraction.

    TODO: Implement full hallucination rate calculation including:
    - Per-triple faithfulness scoring
    - Aggregate hallucination rate metric
    - Breakdown by entity type and relationship type
    - Comparison across ontologies
    """
    print(f"\n{'═' * 70}")
    print(f"  HALLUCINATION ASSESSMENT")
    print(f"{'═' * 70}\n")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07-graph-quality"))
    from llm_probes import probe_faithfulness

    triples = linearise_graph(driver)
    result = probe_faithfulness(triples, source_text)

    print(f"  Faithfulness score: {result.score:.2f}")
    print(f"  Hallucination rate: {1.0 - result.score:.2%}")

    if result.violations:
        hallucinated = [v for v in result.violations if v.severity == "error"]
        missing = [v for v in result.violations if v.severity == "warning"]
        print(f"\n  Hallucinated facts: {len(hallucinated)}")
        for v in hallucinated[:5]:
            print(f"    ✗ {v.message}")
        print(f"\n  Missing from graph: {len(missing)}")
        for v in missing[:5]:
            print(f"    ⚠ {v.message}")

    print(f"{'─' * 70}\n")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

USAGE = """\
Usage: python demo.py [statement_file] [options]

Arguments:
  statement_file        Path to a .txt file with the witness statement

Options:
  --ontology ID         Select ontology (default: schema-org-event-v1)
                        Available: """ + ", ".join(ONTOLOGY_REGISTRY.keys()) + """
  --skip-interview      Skip Phase 2 (interview)
  --interview           Run only interview (graph must exist)
  --query               Run only query (graph must exist)
  --quality             Run only quality probe
  --hallucination-check Run hallucination assessment against source text
  --skip-llm            Skip LLM probes in quality assessment
  --transcript FILE     Write transcript to FILE (default: transcript.txt)
  --list-ontologies     List available ontologies and exit
  --help                Show this help
"""


def _parse_args():
    args = sys.argv[1:]
    flags = set()
    file_path = None
    transcript_path = "transcript.txt"
    ontology_id = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--transcript" and i + 1 < len(args):
            transcript_path = args[i + 1]
            i += 2
            continue
        elif arg == "--ontology" and i + 1 < len(args):
            ontology_id = args[i + 1]
            i += 2
            continue
        elif arg.startswith("--"):
            flags.add(arg)
        else:
            file_path = arg
        i += 1

    return file_path, flags, transcript_path, ontology_id


def main():
    file_path, flags, transcript_path, ontology_id = _parse_args()

    if "--help" in flags or "-h" in flags:
        print(USAGE)
        return

    if "--list-ontologies" in flags:
        print("Available ontologies:")
        for sid, spec in ONTOLOGY_REGISTRY.items():
            print(f"  {sid:25s}  {spec.name}")
            for layer in spec.layers:
                print(f"    ├ {layer['standard']:20s} — {layer['role']}")
        return

    print("=" * 70)
    print("  Module 08: Digital Twin")
    print("  Statement → KG → Self-Resolution → Quality → Grounded Q&A")
    print("=" * 70)

    # Select ontology
    if ontology_id:
        select_ontology(ontology_id)
    else:
        spec = get_active_spec()
        print(f"  Using default ontology: {spec.name}")

    if "--quality" in flags:
        source_text = None
        if file_path:
            source_text = Path(file_path).read_text().strip()
        phase_quality(source_text=source_text, skip_llm=("--skip-llm" in flags))

    elif "--hallucination-check" in flags:
        if not file_path:
            print("  ✗ --hallucination-check requires a source text file")
            sys.exit(1)
        source_text = Path(file_path).read_text().strip()
        assess_hallucination(source_text)

    elif "--interview" in flags:
        phase_interview(transcript_path=transcript_path)

    elif "--query" in flags:
        phase_query()

    else:
        statement = load_statement(file_path)

        # Phase 1: Ingest
        phase_ingest(statement)

        # Phase 2: Interview (optional)
        if "--skip-interview" not in flags:
            phase_interview(statement=statement, transcript_path=transcript_path)

        # Phase 3: Quality
        phase_quality(source_text=statement, skip_llm=("--skip-llm" in flags))

        # Phase 4: Query
        phase_query()

    print(f"\n{'=' * 70}")
    print("  Key concepts demonstrated:")
    print(f"  • Pluggable ontology ({get_active_spec().name})")
    print("  • Schema-guided extraction with qwen2.5:7b")
    print("  • Self-resolution + human-in-the-loop interview")
    print("  • Multi-dimensional quality assessment")
    print("  • Provenance tracking (SOSA/PROV-O)")
    print("  • Grounded Q&A with [FACT: ...] citations")
    print("  • Hallucination rate assessment (--hallucination-check)")
    print("=" * 70)

    driver.close()


if __name__ == "__main__":
    main()
