"""
07 · Graph Quality — Multi-dimensional KG Quality Assessment
═════════════════════════════════════════════════════════════

Core data structures for quality reporting.  Shared by all probe
modules (cypher, llm, shacl, embedding).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


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
    # Phase 2: LLM / DeepEval
    coherence_score: float = 0.0
    faithfulness_score: float = 0.0
    # Aggregate
    overall_score: float = 0.0
    violations: list[Violation] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    dimension_results: list[DimensionResult] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
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
            "  ── Phase 2: Semantic Probes (LLM) ───────────────────────",
            f"    Coherence:               {self.coherence_score:.2f}",
            f"    Faithfulness:            {self.faithfulness_score:.2f}",
        ]

        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]

        lines.append("")
        lines.append(f"  ── Violations ({len(self.violations)} total) ──────────────")
        lines.append(f"    Errors:   {len(errors)}")
        lines.append(f"    Warnings: {len(warnings)}")

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

DIMENSION_WEIGHTS = {
    "schema":       0.15,
    "structural":   0.15,
    "constraint":   0.15,
    "consistency":  0.20,
    "coherence":    0.15,
    "faithfulness": 0.20,
}


def compute_overall(report: QualityReport) -> float:
    """Compute weighted overall score from dimension scores."""
    scores = {
        "schema":       report.schema_score,
        "structural":   report.structural_score,
        "constraint":   report.constraint_score,
        "consistency":  report.consistency_score,
        "coherence":    report.coherence_score,
        "faithfulness": report.faithfulness_score,
    }
    total_weight = sum(DIMENSION_WEIGHTS.values())
    if total_weight == 0:
        return 0.0
    return sum(
        DIMENSION_WEIGHTS[k] * scores[k] for k in DIMENSION_WEIGHTS
    ) / total_weight


def generate_recommendations(report: QualityReport) -> list[str]:
    """Generate actionable recommendations from the report."""
    recs = []
    if report.schema_score < 0.8:
        recs.append("Schema population is low — check that all expected node types are being extracted")
    if report.structural_score < 0.8:
        recs.append("Graph has connectivity issues — look for isolated nodes or disconnected components")
    if report.consistency_score < 0.8:
        recs.append("Consistency violations found — check for temporal cycles or duplicate entities")
    if report.constraint_score < 0.8:
        recs.append("Source provenance is incomplete — ensure every node has a source property")
    if report.coherence_score < 0.7:
        recs.append("Narrative coherence is low — review event ordering and participant consistency")
    if report.faithfulness_score < 0.7:
        recs.append("Faithfulness issues detected — the graph may contain hallucinated or missing facts")
    return recs
