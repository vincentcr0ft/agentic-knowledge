"""
08 · Digital Twin — Extended Quality Probes
═══════════════════════════════════════════

Additional quality assessment probes beyond what Chapter 07 provides:
  - Population completeness (domain rules)
  - Temporal consistency
  - Cross-source consistency
  - Narrative reconstruction test
  - Composite quality score

Prerequisites:
  - Neo4j running with populated event graph
  - Ollama running with qwen2.5:7b (for narrative probe)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from schema import linearise_graph, get_active_spec

# Import quality structures from ch07
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07-graph-quality"))
try:
    from quality_core import ProbeResult, Violation
except ImportError:
    @dataclass
    class Violation:
        dimension: str
        message: str
        severity: str = "warning"

    @dataclass
    class ProbeResult:
        dimension: str
        score: float
        violations: list = field(default_factory=list)
        details: dict = field(default_factory=dict)


llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════
# Probe: Population Completeness
# ═══════════════════════════════════════════════════════════════════════════

def probe_population_completeness(driver) -> ProbeResult:
    """Check domain-specific population rules.

    For a collision event, we expect:
    - At least 2 participants (e.g. driver + cyclist)
    - At least 1 location
    - At least 1 time reference
    - At least 1 event node
    """
    violations = []
    checks = 0
    passed = 0

    with driver.session() as session:
        # Check: at least 1 Event
        checks += 1
        result = session.run("MATCH (e:Event) RETURN count(e) AS cnt")
        rec = result.single()
        event_count = rec["cnt"] if rec else 0
        if event_count == 0:
            violations.append(Violation(
                dimension="population_completeness",
                message="No Event nodes in graph",
                severity="error",
            ))
        else:
            passed += 1

        # Check: at least 2 Person nodes
        checks += 1
        result = session.run("MATCH (p:Person) RETURN count(p) AS cnt")
        rec = result.single()
        person_count = rec["cnt"] if rec else 0
        if person_count < 2:
            violations.append(Violation(
                dimension="population_completeness",
                message=f"Only {person_count} Person nodes (expected ≥2 for collision)",
                severity="warning",
            ))
        else:
            passed += 1

        # Check: at least 1 Location
        checks += 1
        result = session.run(
            "MATCH (l) WHERE l:Location OR l:Place OR l:SpatialRegion "
            "RETURN count(l) AS cnt"
        )
        rec = result.single()
        loc_count = rec["cnt"] if rec else 0
        if loc_count == 0:
            violations.append(Violation(
                dimension="population_completeness",
                message="No Location/Place nodes in graph",
                severity="warning",
            ))
        else:
            passed += 1

        # Check: at least 1 Time
        checks += 1
        result = session.run(
            "MATCH (t) WHERE t:Time OR t:TemporalRegion "
            "RETURN count(t) AS cnt"
        )
        rec = result.single()
        time_count = rec["cnt"] if rec else 0
        if time_count == 0:
            violations.append(Violation(
                dimension="population_completeness",
                message="No Time nodes in graph",
                severity="warning",
            ))
        else:
            passed += 1

        # Check: Events linked to participants
        checks += 1
        result = session.run(
            "MATCH (e:Event) "
            "WHERE NOT EXISTS { MATCH (e)-[:HAS_PARTICIPANT|:INVOLVED_IN|:HAS_ACTOR]->(:Person) } "
            "AND NOT EXISTS { MATCH (:Person)-[:INVOLVED_IN|:PARTICIPATES_IN]->(e) } "
            "RETURN count(e) AS cnt"
        )
        rec = result.single()
        orphan_events = rec["cnt"] if rec else 0
        if orphan_events > 0:
            violations.append(Violation(
                dimension="population_completeness",
                message=f"{orphan_events} Events without any linked participants",
                severity="warning",
            ))
        else:
            passed += 1

        # Check: source provenance
        checks += 1
        result = session.run(
            "MATCH (n) WHERE n.source IS NULL AND NOT n:GraphVersion "
            "AND NOT n:Observation "
            "RETURN count(n) AS cnt"
        )
        rec = result.single()
        no_source = rec["cnt"] if rec else 0
        if no_source > 0:
            violations.append(Violation(
                dimension="population_completeness",
                message=f"{no_source} nodes without source provenance",
                severity="warning",
            ))
        else:
            passed += 1

    score = passed / checks if checks > 0 else 0.0
    return ProbeResult(
        dimension="population_completeness",
        score=score,
        violations=violations,
        details={"checks": checks, "passed": passed},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Probe: Temporal Consistency
# ═══════════════════════════════════════════════════════════════════════════

def probe_temporal_consistency(driver) -> ProbeResult:
    """Check for temporal inconsistencies in the graph."""
    from temporal import check_consistency

    issues = check_consistency(driver)
    violations = [
        Violation(
            dimension="temporal_consistency",
            message=issue,
            severity="warning" if "without timestamp" in issue else "error",
        )
        for issue in issues
    ]

    # Score: penalise per issue, but floor at 0
    score = max(0.0, 1.0 - len(issues) * 0.1)
    return ProbeResult(
        dimension="temporal_consistency",
        score=score,
        violations=violations,
        details={"issues": len(issues)},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Probe: Cross-Source Consistency
# ═══════════════════════════════════════════════════════════════════════════

def probe_cross_source_consistency(driver) -> ProbeResult:
    """Assess consistency across different sources."""
    violations = []
    details: dict[str, Any] = {}

    with driver.session() as session:
        # Count sources
        result = session.run(
            "MATCH (n) WHERE n.source IS NOT NULL "
            "RETURN DISTINCT n.source AS source, count(n) AS cnt"
        )
        sources = {rec["source"]: rec["cnt"] for rec in result}
        details["sources"] = sources
        details["source_count"] = len(sources)

        # Count contradictions
        result = session.run(
            "MATCH ()-[r:CONTRADICTS]->() RETURN count(r) AS cnt"
        )
        rec = result.single()
        contradiction_count = rec["cnt"] if rec else 0
        details["contradictions"] = contradiction_count

        if contradiction_count > 0:
            violations.append(Violation(
                dimension="cross_source_consistency",
                message=f"{contradiction_count} cross-source contradictions detected",
                severity="warning",
            ))

        # Count corroborations
        result = session.run(
            "MATCH ()-[r:CORROBORATED_BY]->() RETURN count(r) AS cnt"
        )
        rec = result.single()
        corroboration_count = rec["cnt"] if rec else 0
        details["corroborations"] = corroboration_count

        # Count unresolved POSSIBLY_SAME_AS
        result = session.run(
            "MATCH ()-[r:POSSIBLY_SAME_AS]->() RETURN count(r) AS cnt"
        )
        rec = result.single()
        possibly_same = rec["cnt"] if rec else 0
        details["possibly_same_as"] = possibly_same

    # Score: corroborations boost, contradictions penalise
    if len(sources) <= 1:
        score = 1.0  # single source → no cross-source issues possible
    else:
        base = 0.7
        boost = min(0.3, corroboration_count * 0.05)
        penalty = min(0.5, contradiction_count * 0.1)
        score = max(0.0, min(1.0, base + boost - penalty))

    return ProbeResult(
        dimension="cross_source_consistency",
        score=score,
        violations=violations,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Probe: Narrative Reconstruction
# ═══════════════════════════════════════════════════════════════════════════

NARRATIVE_PROMPT = """\
You are a quality assessor for a knowledge graph built from witness statements \
about an incident. Given ONLY the graph triples below, reconstruct what happened \
as a coherent narrative.

GRAPH TRIPLES:
{triples}

Then score the graph on a scale of 1–5:
1 = Incoherent, major gaps, cannot understand what happened
2 = Fragmented, significant gaps, partial understanding
3 = Understandable but with notable gaps
4 = Clear narrative with minor gaps
5 = Complete, coherent narrative

Return EXACTLY this format:
NARRATIVE: <your reconstruction>
SCORE: <1-5>
GAPS: <list any missing information>
"""


def probe_narrative_reconstruction(driver) -> ProbeResult:
    """Use LLM-as-judge to assess whether the graph tells a coherent story."""
    triples = linearise_graph(driver)

    if triples == "(empty graph)":
        return ProbeResult(
            dimension="narrative_reconstruction",
            score=0.0,
            violations=[Violation(
                dimension="narrative_reconstruction",
                message="Empty graph — no narrative possible",
                severity="error",
            )],
        )

    result = llm.invoke([
        SystemMessage(content=NARRATIVE_PROMPT.format(triples=triples)),
        HumanMessage(content="Reconstruct the narrative and score."),
    ])

    content = result.content
    score = 3  # default
    narrative = ""
    gaps = ""

    import re
    score_match = re.search(r"SCORE:\s*(\d)", content)
    if score_match:
        score = int(score_match.group(1))

    narrative_match = re.search(r"NARRATIVE:\s*(.+?)(?=\nSCORE:|\nGAPS:|\Z)",
                                content, re.DOTALL)
    if narrative_match:
        narrative = narrative_match.group(1).strip()

    gaps_match = re.search(r"GAPS:\s*(.+?)(?=\Z)", content, re.DOTALL)
    if gaps_match:
        gaps = gaps_match.group(1).strip()

    violations = []
    if score < 3:
        violations.append(Violation(
            dimension="narrative_reconstruction",
            message=f"Low narrative coherence score: {score}/5",
            severity="error",
        ))
    if gaps:
        violations.append(Violation(
            dimension="narrative_reconstruction",
            message=f"Identified gaps: {gaps[:200]}",
            severity="warning",
        ))

    normalised_score = score / 5.0
    return ProbeResult(
        dimension="narrative_reconstruction",
        score=normalised_score,
        violations=violations,
        details={"raw_score": score, "narrative": narrative[:500], "gaps": gaps},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Composite Quality Score
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QualityReport:
    """Comprehensive quality report across all dimensions."""
    overall_score: float
    dimensions: dict[str, float]
    violations: list
    recommendations: list[str]


def run_extended_quality(driver, source_text: str | None = None) -> QualityReport:
    """Run all extended quality probes and produce a composite report."""
    results = []

    print("  ▸ Population completeness...")
    results.append(probe_population_completeness(driver))

    print("  ▸ Temporal consistency...")
    results.append(probe_temporal_consistency(driver))

    print("  ▸ Cross-source consistency...")
    results.append(probe_cross_source_consistency(driver))

    print("  ▸ Narrative reconstruction...")
    results.append(probe_narrative_reconstruction(driver))

    # Aggregate
    dimensions = {r.dimension: r.score for r in results}
    all_violations = []
    for r in results:
        all_violations.extend(r.violations)

    overall = sum(dimensions.values()) / len(dimensions) if dimensions else 0.0

    # Generate recommendations
    recs = []
    for r in results:
        if r.score < 0.5:
            recs.append(f"Critical: {r.dimension} scored {r.score:.2f} — needs attention")
        elif r.score < 0.7:
            recs.append(f"Improve: {r.dimension} scored {r.score:.2f}")

    for v in all_violations:
        if v.severity == "error":
            recs.append(f"Fix: {v.message}")

    return QualityReport(
        overall_score=overall,
        dimensions=dimensions,
        violations=all_violations,
        recommendations=recs[:10],
    )
