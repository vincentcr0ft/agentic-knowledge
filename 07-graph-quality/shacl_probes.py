"""
SHACL validation probe — validates graph RDF export against shapes.

Works with both static shapes files and dynamically generated shapes
from an OntologySpec (see 06-ontologies/).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rdflib import Graph as RDFGraph

from quality_core import DimensionResult, Violation

try:
    from pyshacl import validate as shacl_validate
except ImportError:
    shacl_validate = None


def _neo4j_to_rdf(driver, *, ontology_id: str | None = None) -> RDFGraph:
    """Export Neo4j nodes/relationships to a minimal RDF graph for SHACL."""
    from rdflib import Namespace, Literal, URIRef, RDF

    EX = Namespace("http://example.org/kg/")
    g = RDFGraph()
    g.bind("ex", EX)

    with driver.session() as session:
        # Nodes
        node_query = "MATCH (n) RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props"
        for rec in session.run(node_query):
            if ontology_id and rec["props"].get("ontology_id") != ontology_id:
                continue
            uri = EX[f"node_{rec['nid']}"]
            for label in rec["labels"]:
                g.add((uri, RDF.type, EX[label]))
            for k, v in rec["props"].items():
                if v is not None:
                    g.add((uri, EX[k], Literal(str(v))))

        # Relationships
        rel_query = (
            "MATCH (a)-[r]->(b) "
            "RETURN id(a) AS aid, id(b) AS bid, type(r) AS rtype, properties(r) AS props"
        )
        for rec in session.run(rel_query):
            a_uri = EX[f"node_{rec['aid']}"]
            b_uri = EX[f"node_{rec['bid']}"]
            g.add((a_uri, EX[rec["rtype"]], b_uri))

    return g


def probe_shacl(
    driver,
    shapes_ttl: str | None = None,
    shapes_path: str | Path | None = None,
    ontology_id: str | None = None,
) -> DimensionResult:
    """Run SHACL validation against the graph.

    Provide shapes as a TTL string (shapes_ttl) or a file path (shapes_path).
    """
    if shacl_validate is None:
        return DimensionResult(
            dimension="constraint",
            score=0.0,
            violations=[Violation(
                dimension="constraint", severity="error",
                message="pyshacl not installed — cannot run SHACL probes",
            )],
        )

    data_graph = _neo4j_to_rdf(driver, ontology_id=ontology_id)

    if not shapes_ttl and shapes_path:
        shapes_ttl = Path(shapes_path).read_text()

    if not shapes_ttl:
        return DimensionResult(
            dimension="constraint", score=0.0,
            violations=[Violation(
                dimension="constraint", severity="error",
                message="No SHACL shapes provided",
            )],
        )

    with tempfile.NamedTemporaryFile(suffix=".ttl", mode="w", delete=False) as f:
        f.write(shapes_ttl)
        shapes_file = f.name

    conforms, results_graph, results_text = shacl_validate(
        data_graph, shacl_graph=shapes_file,
    )

    violations = []
    if not conforms:
        for line in results_text.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("Validation"):
                violations.append(Violation(
                    dimension="constraint", severity="warning",
                    message=line[:300],
                ))

    total_checks = max(len(data_graph), 1)
    violation_penalty = min(len(violations) / total_checks, 1.0)
    score = max(0.0, 1.0 - violation_penalty)

    return DimensionResult(
        dimension="constraint", score=score,
        violations=violations,
        details={
            "conforms": conforms,
            "violation_count": len(violations),
            "data_triples": len(data_graph),
        },
    )
