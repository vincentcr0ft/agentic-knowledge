"""
CLI runner for the quality probe.

Usage:
  python -m quality_probe                   Run all probes (requires Neo4j + Ollama)
  python -m quality_probe --skip-llm        Run structural/Cypher probes only
  python -m quality_probe --shacl           Include SHACL validation (requires rdflib + pyshacl)
  python -m quality_probe --deepeval        Use DeepEval G-Eval metrics (Phase 2)
  python -m quality_probe --embeddings      Run PyKEEN embedding probes (Phase 3)
  python -m quality_probe --source FILE     Provide source text for faithfulness scoring
  python -m quality_probe --calibrate       Run LLM probes 3x for score calibration
  python -m quality_probe --json            Output as JSON instead of human-readable

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b (unless --skip-llm)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from neo4j import GraphDatabase

from quality_core import run_quality_probe


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Quality Probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM-based probes (coherence/faithfulness)",
    )
    parser.add_argument(
        "--shacl", action="store_true",
        help="Include SHACL constraint validation (requires rdflib + pyshacl)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to the source statement file for faithfulness scoring",
    )
    parser.add_argument(
        "--deepeval", action="store_true",
        help="Use DeepEval G-Eval metrics for LLM probes (Phase 2). "
             "Falls back to native LLM probes if deepeval is not installed.",
    )
    parser.add_argument(
        "--embeddings", action="store_true",
        help="Run PyKEEN KG embedding probes (Phase 3). "
             "Requires: pip install pykeen",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run LLM probes multiple times and report mean +/- std",
    )
    parser.add_argument(
        "--calibration-runs", type=int, default=3,
        help="Number of calibration runs (default: 3)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--neo4j-uri", type=str, default=NEO4J_URI,
        help=f"Neo4j bolt URI (default: {NEO4J_URI})",
    )
    parser.add_argument(
        "--neo4j-user", type=str, default=NEO4J_USER,
    )
    parser.add_argument(
        "--neo4j-password", type=str, default=NEO4J_PASSWORD,
    )
    args = parser.parse_args()

    # Load source text if provided
    source_text = None
    if args.source:
        source_path = os.path.expanduser(args.source)
        if not os.path.exists(source_path):
            print(f"  ✗ Source file not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        with open(source_path) as f:
            source_text = f.read().strip()
        print(f"  ✓ Loaded source text ({len(source_text)} chars)")

    # Load SHACL shapes
    shapes_ttl = None
    if args.shacl:
        shapes_path = os.path.join(os.path.dirname(__file__), "shapes.ttl")
        if os.path.exists(shapes_path):
            with open(shapes_path) as f:
                shapes_ttl = f.read()
            print(f"  ✓ Loaded SHACL shapes from shapes.ttl")
        else:
            print(f"  ⚠ shapes.ttl not found — SHACL validation will fail", file=sys.stderr)

    # Connect
    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )

    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"  ✗ Cannot connect to Neo4j at {args.neo4j_uri}: {e}", file=sys.stderr)
        sys.exit(1)

    # Run probes
    report = run_quality_probe(
        driver,
        source_text=source_text,
        skip_llm=args.skip_llm,
        use_deepeval=args.deepeval,
        use_embeddings=args.embeddings,
        use_shacl=args.shacl,
        shapes_ttl=shapes_ttl,
        calibrate=args.calibrate,
        calibration_runs=args.calibration_runs,
    )

    driver.close()

    # Output
    if args.json:
        output = {
            "timestamp": report.timestamp,
            "overall_score": round(report.overall_score, 4),
            "scores": {
                "schema": round(report.schema_score, 4),
                "structural": round(report.structural_score, 4),
                "constraint": round(report.constraint_score, 4),
                "consistency": round(report.consistency_score, 4),
                "coherence": round(report.coherence_score, 4),
                "faithfulness": round(report.faithfulness_score, 4),
                "semantic_completeness": round(report.semantic_completeness_score, 4),
                "investigative_readiness": round(report.investigative_readiness_score, 4),
                "link_prediction": round(report.link_prediction_score, 4),
                "triple_plausibility": round(report.triple_plausibility_score, 4),
                "entity_clustering": round(report.entity_clustering_score, 4),
            },
            "violations": [
                {
                    "dimension": v.dimension,
                    "severity": v.severity,
                    "message": v.message,
                    "node_label": v.node_label,
                }
                for v in report.violations
            ],
            "recommendations": report.recommendations,
        }
        # Include calibration data if available
        for dr in report.dimension_results:
            cal = dr.details.get("calibration")
            if cal:
                output.setdefault("calibration", {})[dr.dimension] = cal
        print(json.dumps(output, indent=2))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
