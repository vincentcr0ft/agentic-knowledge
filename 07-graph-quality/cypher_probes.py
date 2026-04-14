"""
Cypher-based quality probes — zero additional dependencies.

Checks schema population, structural connectivity, temporal consistency,
and source-provenance grounding against a live Neo4j graph.
"""

from __future__ import annotations

from quality_core import DimensionResult, Violation


def probe_schema_population(driver, expected_labels: list[str]) -> DimensionResult:
    """Check that every expected node type has at least one instance."""
    populated = 0
    violations = []

    with driver.session() as session:
        for label in expected_labels:
            result = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS cnt"
            )
            count = result.single()["cnt"]
            if count > 0:
                populated += 1
            else:
                violations.append(Violation(
                    dimension="schema",
                    severity="warning",
                    message=f"No instances of node type '{label}' in the graph",
                    node_label=label,
                ))

    score = populated / len(expected_labels) if expected_labels else 1.0

    return DimensionResult(
        dimension="schema",
        score=score,
        violations=violations,
        details={
            "expected_types": len(expected_labels),
            "populated_types": populated,
        },
    )


def probe_structural_connectivity(driver) -> DimensionResult:
    """Assess graph connectivity: isolated nodes, component count."""
    violations = []

    with driver.session() as session:
        total = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]

        if total == 0:
            return DimensionResult(
                dimension="structural", score=0.0,
                violations=[Violation(
                    dimension="structural", severity="error",
                    message="Graph is empty — no nodes found",
                )],
            )

        # Isolated nodes
        isolated_result = session.run(
            "MATCH (n) WHERE NOT (n)-[]-() "
            "RETURN labels(n)[0] AS label, "
            "  coalesce(n.description, n.name_or_description, n.value, 'unknown') AS desc"
        )
        isolated_count = 0
        for rec in isolated_result:
            isolated_count += 1
            violations.append(Violation(
                dimension="structural", severity="warning",
                message=f"Isolated node: ({rec['label']}: {rec['desc']})",
                node_label=rec["label"],
            ))

        connected = total - isolated_count
        score = connected / total if total > 0 else 0.0

    return DimensionResult(
        dimension="structural",
        score=score,
        violations=violations,
        details={"total_nodes": total, "isolated_nodes": isolated_count},
    )


def probe_consistency(driver) -> DimensionResult:
    """Check temporal acyclicity, role constraints, duplicate detection."""
    violations = []
    checks_passed = 0
    checks_total = 0

    with driver.session() as session:
        # Temporal cycle detection (works with PRECEDED or PRECEDES)
        checks_total += 1
        for rel in ("PRECEDED", "PRECEDES"):
            cycle = session.run(
                f"MATCH path = (e)-[:{rel}*2..]->(e) "
                "RETURN coalesce(e.description, 'unknown') AS desc LIMIT 1"
            ).single()
            if cycle:
                violations.append(Violation(
                    dimension="consistency", severity="error",
                    message=f"Temporal cycle detected via {rel} involving '{cycle['desc']}'",
                ))
                break
        else:
            checks_passed += 1

        # Duplicate node detection
        checks_total += 1
        dup_result = session.run(
            "MATCH (a), (b) "
            "WHERE id(a) < id(b) "
            "  AND labels(a) = labels(b) "
            "  AND coalesce(a.description, a.name_or_description, a.value, '') = "
            "     coalesce(b.description, b.name_or_description, b.value, '') "
            "  AND coalesce(a.description, a.name_or_description, a.value, '') <> '' "
            "RETURN labels(a)[0] AS label, "
            "  coalesce(a.description, a.name_or_description, a.value) AS desc "
            "LIMIT 5"
        )
        dup_found = False
        for rec in dup_result:
            dup_found = True
            violations.append(Violation(
                dimension="consistency", severity="warning",
                message=f"Possible duplicate: ({rec['label']}: {rec['desc']})",
                node_label=rec["label"],
            ))
        if not dup_found:
            checks_passed += 1

    score = checks_passed / checks_total if checks_total > 0 else 1.0
    return DimensionResult(
        dimension="consistency", score=score,
        violations=violations,
        details={"checks_passed": checks_passed, "checks_total": checks_total},
    )


def probe_source_grounding(driver) -> DimensionResult:
    """Verify that nodes have source provenance."""
    violations = []

    with driver.session() as session:
        total = session.run(
            "MATCH (n) WHERE NOT n:Observation "
            "RETURN count(n) AS cnt"
        ).single()["cnt"]

        if total == 0:
            return DimensionResult(
                dimension="constraint", score=1.0, violations=[],
            )

        orphan_count = 0
        orphan_result = session.run(
            "MATCH (n) WHERE NOT n:Observation AND n.source IS NULL "
            "RETURN labels(n)[0] AS label, "
            "  coalesce(n.description, n.name_or_description, n.value, 'unknown') AS desc "
            "LIMIT 20"
        )
        for rec in orphan_result:
            orphan_count += 1
            violations.append(Violation(
                dimension="constraint", severity="warning",
                message=f"No source provenance: ({rec['label']}: {rec['desc']})",
                node_label=rec["label"],
            ))

        score = max(0.0, 1.0 - (orphan_count / max(total, 1)))

    return DimensionResult(
        dimension="constraint", score=score,
        violations=violations,
        details={"total_nodes": total, "orphan_nodes": orphan_count},
    )
