"""
07-graph-quality  –  Standalone Graph Quality Assessment

Demonstrates multi-dimensional quality probing against a Neo4j knowledge graph.
This module works independently of any specific ontology or domain — it takes
node labels and triples as input and runs structural, semantic, and constraint
checks.

Usage:
    cd 07-graph-quality
    python demo.py                  # full quality assessment
    python demo.py --phase 1       # structural probes only
    python demo.py --source FILE    # faithfulness check against source text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase

from quality_core import QualityReport, run_quality_probe
from cypher_probes import (
    probe_schema_population,
    probe_structural_connectivity,
    probe_consistency,
    probe_source_grounding,
)
from llm_probes import probe_coherence, probe_faithfulness
from shacl_probes import probe_shacl

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "cabbage123")


# ── Helpers ──────────────────────────────────────────────────────────

def _get_expected_labels(driver) -> list[str]:
    """Discover all labels present in the graph (so we can check population)."""
    with driver.session() as session:
        result = session.run("CALL db.labels() YIELD label RETURN label")
        return [r["label"] for r in result]


def _linearise_graph(driver) -> str:
    """Dump graph as human-readable triples for LLM probes."""
    lines = []
    with driver.session() as session:
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a)[0] AS a_label, "
            "  coalesce(a.description, a.name_or_description, a.value, toString(id(a))) AS a_desc, "
            "  type(r) AS rel, "
            "  labels(b)[0] AS b_label, "
            "  coalesce(b.description, b.name_or_description, b.value, toString(id(b))) AS b_desc"
        )
        for rec in result:
            lines.append(
                f"({rec['a_label']}: {rec['a_desc']}) "
                f"-[{rec['rel']}]-> "
                f"({rec['b_label']}: {rec['b_desc']})"
            )
    return "\n".join(lines) if lines else "(empty graph)"


# ── Phase runners ────────────────────────────────────────────────────

def run_phase_1(driver) -> list:
    """Structural / Cypher-based probes."""
    labels = _get_expected_labels(driver)
    results = [
        probe_schema_population(driver, labels),
        probe_structural_connectivity(driver),
        probe_consistency(driver),
        probe_source_grounding(driver),
    ]
    return results


def run_phase_2(driver, source_text: str | None = None) -> list:
    """LLM-powered probes: coherence + faithfulness."""
    triples = _linearise_graph(driver)
    results = [probe_coherence(triples)]

    if source_text:
        results.append(probe_faithfulness(triples, source_text))
    else:
        print("  ⚠  No source text provided — skipping faithfulness probe")

    return results


def run_phase_3(driver, shapes_ttl: str | None = None) -> list:
    """SHACL constraint validation."""
    result = probe_shacl(driver, shapes_ttl=shapes_ttl)
    return [result]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph Quality Assessment")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run only a specific phase (1=structural, 2=LLM, 3=SHACL)")
    parser.add_argument("--source", type=str,
                        help="Path to source text file for faithfulness check")
    parser.add_argument("--shapes", type=str,
                        help="Path to SHACL shapes .ttl file")
    args = parser.parse_args()

    source_text = None
    if args.source:
        source_text = Path(args.source).read_text()

    shapes_ttl = None
    if args.shapes:
        shapes_ttl = Path(args.shapes).read_text()

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    print("=" * 60)
    print("  GRAPH QUALITY ASSESSMENT")
    print("=" * 60)

    all_results = []

    # Phase 1 — structural
    if args.phase is None or args.phase == 1:
        print("\n── Phase 1: Structural Probes ──")
        p1 = run_phase_1(driver)
        for r in p1:
            print(f"  {r.dimension:20s}  {r.score:.2f}  "
                  f"({len(r.violations)} violations)")
        all_results.extend(p1)

    # Phase 2 — LLM
    if args.phase is None or args.phase == 2:
        print("\n── Phase 2: LLM Probes ──")
        p2 = run_phase_2(driver, source_text)
        for r in p2:
            print(f"  {r.dimension:20s}  {r.score:.2f}  "
                  f"({len(r.violations)} violations)")
        all_results.extend(p2)

    # Phase 3 — SHACL
    if (args.phase is None or args.phase == 3) and shapes_ttl:
        print("\n── Phase 3: SHACL Validation ──")
        p3 = run_phase_3(driver, shapes_ttl)
        for r in p3:
            print(f"  {r.dimension:20s}  {r.score:.2f}  "
                  f"({len(r.violations)} violations)")
        all_results.extend(p3)

    # Build final report
    report = run_quality_probe(all_results)
    print("\n" + report.summary())

    driver.close()


if __name__ == "__main__":
    main()
