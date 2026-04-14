"""
07 · Graph Quality — Core data structures and orchestration.

Single canonical module shared by all probe files (cypher, llm, shacl,
deepeval, embedding).  Provides:
  - Violation / DimensionResult / QualityReport dataclasses
  - Weight architecture with dynamic normalisation
  - build_report()       — assemble a report from pre-computed results
  - run_quality_probe()  — full pipeline orchestrator
  - linearise_graph()    — graph → text triples for LLM probes
  - calibrate_llm_probe() — run an LLM probe N times, return mean ± std
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Violation:
    """A single quality violation detected by a probe."""
    dimension: str
    severity: str       # "error" | "warning" | "info"
    message: str
    node_label: str | None = None
    node_id: str | None = None


@dataclass
class DimensionResult:
    """Score and violations for a single quality dimension."""
    dimension: str
    score: float                        # 0.0–1.0  (-1.0 = sentinel: skipped)
    violations: list[Violation] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    """Unified output of all quality probes."""
    # Phase 1: Cypher-based
    schema_score: float = 0.0
    structural_score: float = 0.0
    constraint_score: float = 0.0
    consistency_score: float = 0.0
    # Phase 1b native LLM (or Phase 2 DeepEval upgrade)
    coherence_score: float = 0.0
    faithfulness_score: float = 0.0
    # Phase 2: DeepEval extras
    semantic_completeness_score: float = 0.0
    investigative_readiness_score: float = 0.0
    # Phase 3: Embedding-based
    link_prediction_score: float = 0.0
    triple_plausibility_score: float = 0.0
    entity_clustering_score: float = 0.0
    # Aggregate
    overall_score: float = 0.0
    violations: list[Violation] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    dimension_results: list[DimensionResult] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
        """Return a human-readable summary of the quality report."""
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║              KNOWLEDGE GRAPH QUALITY REPORT             ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            f"  Timestamp:  {self.timestamp}",
            f"  Overall:    {self.overall_score:.2f} / 1.00",
            "",
            "  ── Phase 1: Structural Probes ────────────────────────────",
            f"    Schema completeness:     {self.schema_score:.2f}",
            f"    Structural quality:      {self.structural_score:.2f}",
            f"    Constraint conformance:  {self.constraint_score:.2f}",
            f"    Consistency:             {self.consistency_score:.2f}",
            "",
            "  ── Phase 2: Semantic Probes (LLM / DeepEval) ────────────",
            f"    Coherence:               {self.coherence_score:.2f}",
            f"    Faithfulness:            {self.faithfulness_score:.2f}",
            f"    Semantic completeness:   {self.semantic_completeness_score:.2f}",
            f"    Investigative readiness: {self.investigative_readiness_score:.2f}",
            "",
            "  ── Phase 3: Embedding Probes (PyKEEN) ──────────────────",
            f"    Link prediction:         {self.link_prediction_score:.2f}",
            f"    Triple plausibility:     {self.triple_plausibility_score:.2f}",
            f"    Entity clustering:       {self.entity_clustering_score:.2f}",
        ]

        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]
        infos = [v for v in self.violations if v.severity == "info"]

        lines.append("")
        lines.append(f"  ── Violations ({len(self.violations)} total) "
                      "────────────────────────────────")
        lines.append(f"    Errors:   {len(errors)}")
        lines.append(f"    Warnings: {len(warnings)}")
        lines.append(f"    Info:     {len(infos)}")

        if errors:
            lines.append("")
            lines.append("  ── Errors ────────────────────────────────────────────")
            for v in errors:
                lines.append(f"    ✗ [{v.dimension}] {v.message}")

        if warnings:
            lines.append("")
            lines.append("  ── Warnings ──────────────────────────────────────────")
            for v in warnings[:10]:
                lines.append(f"    ⚠ [{v.dimension}] {v.message}")
            if len(warnings) > 10:
                lines.append(f"    ... and {len(warnings) - 10} more")

        if self.recommendations:
            lines.append("")
            lines.append("  ── Recommendations ───────────────────────────────────")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")

        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension weights
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_WEIGHTS = {
    "schema":       0.15,
    "structural":   0.15,
    "constraint":   0.15,
    "consistency":  0.20,
    "coherence":    0.15,
    "faithfulness": 0.20,
}

_PHASE2_WEIGHTS = {
    "semantic_completeness":    0.10,
    "investigative_readiness":  0.10,
}

_PHASE3_WEIGHTS = {
    "link_prediction":      0.08,
    "triple_plausibility":  0.08,
    "entity_clustering":    0.04,
}

DIMENSION_WEIGHTS = _BASE_WEIGHTS


# ═══════════════════════════════════════════════════════════════════════════════
# Graph linearisation (shared by demo.py, llm_probes, deepeval_probes)
# ═══════════════════════════════════════════════════════════════════════════════

def linearise_graph(driver) -> str:
    """Dump graph as human-readable triples for LLM probes."""
    lines = []
    with driver.session() as session:
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a)[0] AS a_label, "
            "  coalesce(a.description, a.name_or_description, a.value, "
            "    a.summary, toString(id(a))) AS a_desc, "
            "  type(r) AS rel, "
            "  labels(b)[0] AS b_label, "
            "  coalesce(b.description, b.name_or_description, b.value, "
            "    b.summary, toString(id(b))) AS b_desc"
        )
        for rec in result:
            lines.append(
                f"({rec['a_label']}: {rec['a_desc']}) "
                f"-[{rec['rel']}]-> "
                f"({rec['b_label']}: {rec['b_desc']})"
            )
    return "\n".join(lines) if lines else "(empty graph)"


# ═══════════════════════════════════════════════════════════════════════════════
# LLM calibration
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_llm_probe(
    probe_fn,
    *args,
    runs: int = 3,
    **kwargs,
) -> DimensionResult:
    """Run an LLM probe multiple times and return the median result with
    calibration metadata (mean, std, min, max).

    Addresses the fundamental non-determinism of LLM-as-judge scoring
    by quantifying the variance.
    """
    results: list[DimensionResult] = []
    for _ in range(runs):
        r = probe_fn(*args, **kwargs)
        if r.score < 0:
            return r
        results.append(r)

    scores = [r.score for r in results]
    median_idx = sorted(range(len(scores)), key=lambda i: scores[i])[len(scores) // 2]
    best = results[median_idx]

    best.details["calibration"] = {
        "runs": runs,
        "mean": round(statistics.mean(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "all_scores": [round(s, 4) for s in scores],
    }
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Overall score computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_overall(report: QualityReport) -> float:
    """Compute the weighted overall score with dynamic normalisation."""
    scores = {
        "schema":       report.schema_score,
        "structural":   report.structural_score,
        "constraint":   report.constraint_score,
        "consistency":  report.consistency_score,
        "coherence":    report.coherence_score,
        "faithfulness": report.faithfulness_score,
    }
    weights = dict(_BASE_WEIGHTS)
    active_dims = {dr.dimension for dr in report.dimension_results}

    for dim, w in _PHASE2_WEIGHTS.items():
        if dim in active_dims:
            scores[dim] = getattr(report, f"{dim}_score")
            weights[dim] = w
    for dim, w in _PHASE3_WEIGHTS.items():
        if dim in active_dims:
            scores[dim] = getattr(report, f"{dim}_score")
            weights[dim] = w

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    return sum(weights[k] * scores[k] for k in weights) / total_weight


# ═══════════════════════════════════════════════════════════════════════════════
# Recommendation generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(report: QualityReport) -> list[str]:
    """Derive actionable recommendations from dimension scores."""
    recs: list[str] = []
    active_dims = {dr.dimension for dr in report.dimension_results}

    if report.schema_score < 0.8:
        recs.append(
            "Schema gaps detected — some expected node types are unpopulated. "
            "Re-run ingestion or add missing entities in the interview phase."
        )
    if report.structural_score < 0.8:
        recs.append(
            "Graph is fragmented — isolated nodes or disconnected components found. "
            "Check for extraction failures or missing relationships."
        )
    if report.constraint_score < 0.8:
        recs.append(
            "Provenance gaps detected — some nodes lack source attribution. "
            "Verify extraction pipeline is tagging all nodes with provenance."
        )
    if report.consistency_score < 0.8:
        recs.append(
            "Consistency issues found — temporal cycles, role conflicts, or "
            "logical contradictions. Review flagged violations."
        )
    if report.coherence_score < 0.6:
        recs.append(
            "Low narrative coherence — the graph does not form a clear story. "
            "Consider additional interview rounds to fill narrative gaps."
        )
    if report.faithfulness_score < 0.6:
        recs.append(
            "Extraction faithfulness concerns — the graph may contain facts "
            "not supported by the source text. Review flagged hallucinations."
        )
    if "semantic_completeness" in active_dims and report.semantic_completeness_score < 0.7:
        recs.append(
            "Semantic completeness is low — key facts from the source text "
            "may not be represented in the graph. Run another extraction pass."
        )
    if "investigative_readiness" in active_dims and report.investigative_readiness_score < 0.7:
        recs.append(
            "Investigative readiness is low — the graph may lack sufficient "
            "detail for case analysis. Target missing who/what/when/where gaps."
        )
    if "link_prediction" in active_dims and report.link_prediction_score < 0.7:
        recs.append(
            "Link prediction found plausible missing relationships — consider "
            "adding suggested links or investigating why they are absent."
        )
    if "triple_plausibility" in active_dims and report.triple_plausibility_score < 0.7:
        recs.append(
            "Some triples score low on plausibility — review flagged facts "
            "for possible extraction errors or hallucinations."
        )
    if "entity_clustering" in active_dims and report.entity_clustering_score < 0.7:
        recs.append(
            "Entity clustering anomalies detected — possible duplicates or "
            "outlier entities. Consider entity resolution / deduplication."
        )
    if report.overall_score < 0.4:
        recs.append(
            "WARNING: Overall quality is low. The graph may be unreliable "
            "for investigative use. Consider re-ingesting the statement."
        )
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# Report builder — from pre-computed DimensionResults
# ═══════════════════════════════════════════════════════════════════════════════

_SCORE_FIELD_MAP = {
    "schema":                   "schema_score",
    "structural":               "structural_score",
    "constraint":               "constraint_score",
    "consistency":              "consistency_score",
    "coherence":                "coherence_score",
    "faithfulness":             "faithfulness_score",
    "semantic_completeness":    "semantic_completeness_score",
    "investigative_readiness":  "investigative_readiness_score",
    "link_prediction":          "link_prediction_score",
    "triple_plausibility":      "triple_plausibility_score",
    "entity_clustering":        "entity_clustering_score",
}


def build_report(results: list[DimensionResult]) -> QualityReport:
    """Assemble a QualityReport from a list of DimensionResult objects."""
    report = QualityReport(timestamp=datetime.now(timezone.utc).isoformat())
    for dr in results:
        if dr.score < 0:
            continue
        field_name = _SCORE_FIELD_MAP.get(dr.dimension)
        if field_name:
            setattr(report, field_name, dr.score)
        report.violations.extend(dr.violations)
        report.dimension_results.append(dr)

    report.overall_score = compute_overall(report)
    report.recommendations = generate_recommendations(report)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_quality_probe(
    driver,
    source_text: str | None = None,
    *,
    skip_llm: bool = False,
    use_deepeval: bool = False,
    use_embeddings: bool = False,
    use_shacl: bool = False,
    shapes_ttl: str | None = None,
    calibrate: bool = False,
    calibration_runs: int = 3,
) -> QualityReport:
    """Run all quality probes and return a unified report.

    Args:
        driver:             Neo4j driver instance.
        source_text:        Original source text (for faithfulness scoring).
        skip_llm:           Skip all LLM-based probes.
        use_deepeval:       Use DeepEval G-Eval metrics (Phase 2).
        use_embeddings:     Run PyKEEN embedding probes (Phase 3).
        use_shacl:          Run SHACL constraint validation.
        shapes_ttl:         SHACL shapes as TTL string.
        calibrate:          Run LLM probes multiple times for calibration.
        calibration_runs:   Number of calibration runs (default 3).
    """
    from cypher_probes import (
        probe_schema_population,
        probe_structural_connectivity,
        probe_consistency,
        probe_source_grounding,
    )

    all_results: list[DimensionResult] = []

    # Phase 1: Cypher-based probes
    with driver.session() as session:
        labels = [r["label"] for r in session.run(
            "CALL db.labels() YIELD label RETURN label"
        )]

    all_results.append(probe_schema_population(driver, labels))
    all_results.append(probe_structural_connectivity(driver))
    all_results.append(probe_consistency(driver))
    all_results.append(probe_source_grounding(driver))

    # SHACL validation
    if use_shacl:
        from shacl_probes import probe_shacl
        all_results.append(probe_shacl(driver, shapes_ttl=shapes_ttl))

    # Phase 2: LLM probes
    if not skip_llm:
        triples = linearise_graph(driver)

        if use_deepeval:
            all_results.extend(
                _run_deepeval_probes(triples, source_text,
                                    calibrate=calibrate, runs=calibration_runs)
            )
        else:
            all_results.extend(
                _run_native_llm_probes(triples, source_text,
                                      calibrate=calibrate, runs=calibration_runs)
            )

    # Phase 3: Embedding probes
    if use_embeddings:
        all_results.extend(_run_embedding_probes(driver))

    return build_report(all_results)


def _run_native_llm_probes(
    triples: str,
    source_text: str | None,
    *,
    calibrate: bool = False,
    runs: int = 3,
) -> list[DimensionResult]:
    from llm_probes import probe_coherence, probe_faithfulness

    results: list[DimensionResult] = []
    if calibrate:
        results.append(calibrate_llm_probe(probe_coherence, triples, runs=runs))
    else:
        results.append(probe_coherence(triples))

    if source_text:
        if calibrate:
            results.append(calibrate_llm_probe(probe_faithfulness, triples, source_text, runs=runs))
        else:
            results.append(probe_faithfulness(triples, source_text))
    return results


def _run_deepeval_probes(
    triples: str,
    source_text: str | None,
    *,
    calibrate: bool = False,
    runs: int = 3,
) -> list[DimensionResult]:
    from deepeval_probes import (
        probe_coherence_deepeval,
        probe_faithfulness_deepeval,
        probe_semantic_completeness,
        probe_investigative_readiness,
    )
    from llm_probes import probe_coherence, probe_faithfulness

    results: list[DimensionResult] = []

    coherence_result = probe_coherence_deepeval(triples)
    if coherence_result.score < 0:
        coherence_result = (
            calibrate_llm_probe(probe_coherence, triples, runs=runs)
            if calibrate else probe_coherence(triples)
        )
    results.append(coherence_result)

    if source_text:
        faith_result = probe_faithfulness_deepeval(triples, source_text)
        if faith_result.score < 0:
            faith_result = (
                calibrate_llm_probe(probe_faithfulness, triples, source_text, runs=runs)
                if calibrate else probe_faithfulness(triples, source_text)
            )
        results.append(faith_result)

    if source_text:
        sc_result = probe_semantic_completeness(triples, source_text)
        if sc_result.score >= 0:
            results.append(sc_result)

    ir_result = probe_investigative_readiness(triples)
    if ir_result.score >= 0:
        results.append(ir_result)

    return results


def _run_embedding_probes(driver) -> list[DimensionResult]:
    from embedding_probes import (
        probe_link_prediction,
        probe_triple_plausibility,
        probe_entity_clusters,
    )
    results: list[DimensionResult] = []
    for probe_fn in (probe_link_prediction, probe_triple_plausibility, probe_entity_clusters):
        r = probe_fn(driver)
        if r.score >= 0:
            results.append(r)
    return results
