"""Dump graph quality state for inspection."""

from __future__ import annotations

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "cabbage123")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with driver.session() as session:
        labels = session.run("CALL db.labels() YIELD label RETURN label").data()
        rel_types = session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        ).data()
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

    driver.close()

    print("Graph State")
    print("=" * 40)
    print(f"Nodes:          {node_count}")
    print(f"Relationships:  {rel_count}")
    print(f"\nNode Labels:    {[r['label'] for r in labels]}")
    print(f"Rel Types:      {[r['relationshipType'] for r in rel_types]}")


if __name__ == "__main__":
    main()
