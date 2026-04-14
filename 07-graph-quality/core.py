"""
Core data structures and orchestration for quality probing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Violation:
    """A single quality violation detected by a probe."""
    dimension: str          # "schema" | "structural" | "constraint" | "consistency" | "coherence" | "faithfulness"
    severity: str           # "error" | "warning" | "info"
    message: str
    node_label: str | None = None
    node_id: str | None = None


@dataclass
class DimensionResult:
    """Score and violations for a single quality dimension."""
    dimension: str
    score: float                        # 0.0 – 1.0
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
    # Phase 1 native LLM (or Phase 2 DeepEval upgrade)
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
            for v in warnings:
                lines.append(f"    ⚠ [{v.dimension}] {v.message}")

        if self.recommendations:
            lines.append("")
            lines.append("  ── Recommendations ───────────────────────────────────")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")

        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension weights for overall score
# ═══════════════════════════════════════════════════════════════════════════════

# Base weights for Phase 1 dimensions (always active)
_BASE_WEIGHTS = {
    "schema":       0.15,
    "structural":   0.15,
    "constraint":   0.15,
    "consistency":  0.20,
    "coherence":    0.15,
    "faithfulness": 0.20,
}

# Phase 2 DeepEval weights (added when those probes run)
_PHASE2_WEIGHTS = {
    "semantic_completeness":    0.10,
    "investigative_readiness":  0.10,
}

# Phase 3 embedding weights (added when those probes run)
_PHASE3_WEIGHTS = {
    "link_prediction":      0.08,
    "triple_plausibility":  0.08,
    "entity_clustering":    0.04,
}

# Legacy alias for backward compatibility
DIMENSION_WEIGHTS = _BASE_WEIGHTS


def _compute_overall(report: QualityReport) -> float:
    """Compute the weighted overall score.

    Dynamically adjusts weights based on which phases were actually run.
    Phases that were skipped (score still at default 0.0 with no
    dimension_result) are excluded and weights are renormalised.
    """
    scores = {
        "schema":       report.schema_score,
        "structural":   report.structural_score,
        "constraint":   report.constraint_score,
        "consistency":  report.consistency_score,
        "coherence":    report.coherence_score,
        "faithfulness": report.faithfulness_score,
    }
    weights = dict(_BASE_WEIGHTS)

    # Include Phase 2 dimensions if they were actually probed
    active_dims = {dr.dimension for dr in report.dimension_results}

    if "semantic_completeness" in active_dims:
        scores["semantic_completeness"] = report.semantic_completeness_score
        weights["semantic_completeness"] = _PHASE2_WEIGHTS["semantic_completeness"]
    if "investigative_readiness" in active_dims:
        scores["investigative_readiness"] = report.investigative_readiness_score
        weights["investigative_readiness"] = _PHASE2_WEIGHTS["investigative_readiness"]

    if "link_prediction" in active_dims:
        scores["link_prediction"] = report.link_prediction_score
        weights["link_prediction"] = _PHASE3_WEIGHTS["link_prediction"]
    if "triple_plausibility" in active_dims:
        scores["triple_plausibility"] = report.triple_plausibility_score
        weights["triple_plausibility"] = _PHASE3_WEIGHTS["triple_plausibility"]
    if "entity_clustering" in active_dims:
        scores["entity_clustering"] = report.entity_clustering_score
        weights["entity_clustering"] = _PHASE3_WEIGHTS["entity_clustering"]

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    return sum(weights[k] * scores[k] for k in weights) / total_weight


def run_quality_probe(
    driver,
    source_text: str | None = None,
    *,
    skip_llm: bool = False,
    use_deepeval: bool = False,
    use_embeddings: bool = False,
) -> QualityReport:
    """Run all quality probes and return a unified report.

    Args:
        driver:         Neo4j driver instance.
        source_text:    Original witness statement text.  Required for
                        faithfulness scoring; optional otherwise.
        skip_llm:       If True, skip all LLM-based probes.
        use_deepeval:   If True, use DeepEval G-Eval metrics (Phase 2).
                        Falls back to native LLM probes if DeepEval
                        is not installed.
        use_embeddings: If True, run PyKEEN embedding probes (Phase 3).
                        Requires: pip install pykeen

    Returns:
        A populated QualityReport.
    """
    from quality_probe.cypher_probes import (
        probe_schema_population,
        probe_structural_connectivity,
        probe_consistency,
        probe_source_grounding,
    )
    from quality_probe.llm_probes import (
        probe_coherence,
        probe_faithfulness,
    )

    report = QualityReport(timestamp=datetime.now(timezone.utc).isoformat())

    # ── Phase 1: Cypher-based probes (zero extra deps) ──────────────────
    schema_result = probe_schema_population(driver)
    report.schema_score = schema_result.score
    report.violations.extend(schema_result.violations)
    report.dimension_results.append(schema_result)

    structural_result = probe_structural_connectivity(driver)
    report.structural_score = structural_result.score
    report.violations.extend(structural_result.violations)
    report.dimension_results.append(structural_result)

    consistency_result = probe_consistency(driver)
    report.consistency_score = consistency_result.score
    report.violations.extend(consistency_result.violations)
    report.dimension_results.append(consistency_result)

    grounding_result = probe_source_grounding(driver)
    report.constraint_score = grounding_result.score
    report.violations.extend(grounding_result.violations)
    report.dimension_results.append(grounding_result)

    # ── Phase 2: LLM / DeepEval probes ─────────────────────────────────
    if skip_llm:
        report.coherence_score = 1.0
        report.faithfulness_score = 1.0
    elif use_deepeval:
        _run_deepeval_probes(driver, source_text, report)
    else:
        coherence_result = probe_coherence(driver)
        report.coherence_score = coherence_result.score
        report.violations.extend(coherence_result.violations)
        report.dimension_results.append(coherence_result)

        if source_text:
            faithfulness_result = probe_faithfulness(driver, source_text)
            report.faithfulness_score = faithfulness_result.score
            report.violations.extend(faithfulness_result.violations)
            report.dimension_results.append(faithfulness_result)
        else:
            report.faithfulness_score = 1.0

    # ── Phase 3: Embedding probes ──────────────────────────────────────
    if use_embeddings:
        _run_embedding_probes(driver, report)

    # ── Compute overall ────────────────────────────────────────────────
    report.overall_score = _compute_overall(report)

    # ── Generate recommendations ────────────────────────────────────────
    report.recommendations = _generate_recommendations(report)

    return report


def _run_deepeval_probes(
    driver, source_text: str | None, report: QualityReport
) -> None:
    """Run Phase 2 DeepEval probes, falling back to native LLM probes."""
    from quality_probe.deepeval_probes import (
        probe_coherence_deepeval,
        probe_faithfulness_deepeval,
        probe_semantic_completeness,
        probe_investigative_readiness,
    )
    from quality_probe.llm_probes import probe_coherence, probe_faithfulness

    # Coherence: try DeepEval, fall back to native
    coherence_result = probe_coherence_deepeval(driver)
    if coherence_result.score < 0:  # sentinel: DeepEval not installed
        coherence_result = probe_coherence(driver)
    report.coherence_score = coherence_result.score
    report.violations.extend(coherence_result.violations)
    report.dimension_results.append(coherence_result)

    # Faithfulness: try DeepEval, fall back to native
    if source_text:
        faith_result = probe_faithfulness_deepeval(driver, source_text)
        if faith_result.score < 0:
            faith_result = probe_faithfulness(driver, source_text)
        report.faithfulness_score = faith_result.score
        report.violations.extend(faith_result.violations)
        report.dimension_results.append(faith_result)
    else:
        report.faithfulness_score = 1.0

    # Semantic completeness (DeepEval only, no native fallback)
    if source_text:
        sc_result = probe_semantic_completeness(driver, source_text)
        if sc_result.score >= 0:
            report.semantic_completeness_score = sc_result.score
            report.violations.extend(sc_result.violations)
            report.dimension_results.append(sc_result)

    # Investigative readiness (DeepEval only)
    ir_result = probe_investigative_readiness(driver)
    if ir_result.score >= 0:
        report.investigative_readiness_score = ir_result.score
        report.violations.extend(ir_result.violations)
        report.dimension_results.append(ir_result)


def _run_embedding_probes(driver, report: QualityReport) -> None:
    """Run Phase 3 PyKEEN embedding probes."""
    from quality_probe.embedding_probes import (
        probe_link_prediction,
        probe_triple_plausibility,
        probe_entity_clusters,
    )

    lp_result = probe_link_prediction(driver)
    if lp_result.score >= 0:
        report.link_prediction_score = lp_result.score
        report.violations.extend(lp_result.violations)
        report.dimension_results.append(lp_result)

    tp_result = probe_triple_plausibility(driver)
    if tp_result.score >= 0:
        report.triple_plausibility_score = tp_result.score
        report.violations.extend(tp_result.violations)
        report.dimension_results.append(tp_result)

    ec_result = probe_entity_clusters(driver)
    if ec_result.score >= 0:
        report.entity_clustering_score = ec_result.score
        report.violations.extend(ec_result.violations)
        report.dimension_results.append(ec_result)


def _generate_recommendations(report: QualityReport) -> list[str]:
    """Derive actionable recommendations from dimension scores."""
    recs = []
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

    # Phase 2 recommendations
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

    # Phase 3 recommendations
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
