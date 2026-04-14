"""Dump graph state and current ontology configuration."""

from __future__ import annotations

from neo4j import GraphDatabase
from schema import get_active_spec, linearise_graph, ONTOLOGY_REGISTRY

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "cabbage123")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    spec = get_active_spec()

    print("Digital Twin State")
    print("=" * 40)
    print(f"Active ontology: {spec.name} ({spec.id})")
    print(f"Available ontologies: {len(ONTOLOGY_REGISTRY)}")
    for sid in ONTOLOGY_REGISTRY:
        marker = " ← active" if sid == spec.id else ""
        print(f"  {sid}{marker}")

    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        labels = session.run("CALL db.labels() YIELD label RETURN label").data()
        rel_types = session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        ).data()

    print(f"\nGraph:")
    print(f"  Nodes:          {node_count}")
    print(f"  Relationships:  {rel_count}")
    print(f"  Node labels:    {[r['label'] for r in labels]}")
    print(f"  Rel types:      {[r['relationshipType'] for r in rel_types]}")

    if node_count > 0:
        triples = linearise_graph(driver)
        print(f"\nTriples:")
        for line in triples.split("\n")[:20]:
            print(f"  {line}")
        total = triples.count("\n") + 1
        if total > 20:
            print(f"  … ({total - 20} more)")

    driver.close()


if __name__ == "__main__":
    main()
