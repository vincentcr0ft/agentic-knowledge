"""
07-graph-quality  –  Standalone Graph Quality Assessment

Demonstrates multi-dimensional quality probing against a Neo4j knowledge graph.
This module works independently of any specific ontology or domain.

Usage:
    cd 07-graph-quality
    python demo.py                          # full quality assessment
    python demo.py --phase 1               # structural probes only
    python demo.py --source FILE           # faithfulness check against source text
    python demo.py --shapes shapes.ttl     # include SHACL validation
    python demo.py --calibrate             # run LLM probes 3x for confidence interval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase

from quality_core import (
    QualityReport,
    DimensionResult,
    build_report,
    linearise_graph,
    calibrate_llm_probe,
)
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


# ── Phase runners ────────────────────────────────────────────────────

def run_phase_1(driver) -> list[DimensionResult]:
    """Structural / Cypher-based probes."""
    with driver.session() as session:
        labels = [r["label"] for r in session.run(
            "CALL db.labels() YIELD label RETURN label"
        )]
    return [
        probe_schema_population(driver, labels),
        probe_structural_connectivity(driver),
        probe_consistency(driver),
        probe_source_grounding(driver),
    ]


def run_phase_2(
    driver,
    source_text: str | None = None,
    calibrate: bool = False,
) -> list[DimensionResult]:
    """LLM-powered probes: coherence + faithfulness."""
    triples = linearise_graph(driver)
    results: list[DimensionResult] = []

    if calibrate:
        results.append(calibrate_llm_probe(probe_coherence, triples))
    else:
        results.append(probe_coherence(triples))

    if source_text:
        if calibrate:
            results.append(calibrate_llm_probe(probe_faithfulness, triples, source_text))
        else:
            results.append(probe_faithfulness(triples, source_text))
    else:
        print("  ⚠  No source text provided — skipping faithfulness probe")

    return results


def run_phase_3(driver, shapes_ttl: str | None = None) -> list[DimensionResult]:
    """SHACL constraint validation."""
    return [probe_shacl(driver, shapes_ttl=shapes_ttl)]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph Quality Assessment")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run only a specific phase (1=structural, 2=LLM, 3=SHACL)")
    parser.add_argument("--source", type=str,
                        help="Path to source text file for faithfulness check")
    parser.add_argument("--shapes", type=str,
                        help="Path to SHACL shapes .ttl file")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run LLM probes multiple times for score calibration")
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

    all_results: list[DimensionResult] = []

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
        p2 = run_phase_2(driver, source_text, calibrate=args.calibrate)
        for r in p2:
            cal = r.details.get("calibration", {})
            cal_str = f"  [±{cal['std']:.3f}]" if cal else ""
            print(f"  {r.dimension:20s}  {r.score:.2f}{cal_str}  "
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
    report = build_report(all_results)
    print("\n" + report.summary())

    driver.close()


if __name__ == "__main__":
    main()
