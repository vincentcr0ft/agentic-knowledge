"""
08 · Digital Twin — What-If Simulation
══════════════════════════════════════

Enables counterfactual reasoning over the event graph by supporting
scenario branching, evidence removal, hypothesis testing, and
scenario comparison.

This is what makes the system a true "digital twin" — the ability to
ask "what if?" questions and see how the graph would change.

Prerequisites:
  - Neo4j running with populated event graph
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase

from schema import linearise_graph, get_active_spec


# ═══════════════════════════════════════════════════════════════════════════
# Scenario management
# ═══════════════════════════════════════════════════════════════════════════

class Scenario:
    """A named scenario with a list of operations applied to it."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.operations: list[dict] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "operations": self.operations,
            "created_at": self.created_at,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot / Restore
# ═══════════════════════════════════════════════════════════════════════════

def snapshot_graph(driver) -> dict[str, Any]:
    """Take a full snapshot of the current graph state.

    Returns nodes and relationships as serialisable dicts.
    """
    nodes = []
    rels = []

    with driver.session() as session:
        # Snapshot nodes
        result = session.run(
            "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, "
            "properties(n) AS props"
        )
        for rec in result:
            nodes.append({
                "id": rec["id"],
                "labels": list(rec["labels"]),
                "props": dict(rec["props"]),
            })

        # Snapshot relationships
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN id(r) AS id, type(r) AS type, "
            "id(a) AS start_id, id(b) AS end_id, "
            "properties(r) AS props"
        )
        for rec in result:
            rels.append({
                "id": rec["id"],
                "type": rec["type"],
                "start_id": rec["start_id"],
                "end_id": rec["end_id"],
                "props": dict(rec["props"]),
            })

    return {
        "nodes": nodes,
        "relationships": rels,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "rel_count": len(rels),
    }


def restore_snapshot(driver, snapshot: dict[str, Any]) -> None:
    """Restore a graph from a snapshot, wiping the current state."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        # Restore nodes
        old_to_new = {}
        for node in snapshot["nodes"]:
            labels = ":".join(node["labels"])
            if not labels:
                labels = "Node"
            result = session.run(
                f"CREATE (n:{labels} $props) RETURN id(n) AS new_id",
                props=node["props"],
            )
            rec = result.single()
            if rec:
                old_to_new[node["id"]] = rec["new_id"]

        # Restore relationships
        for rel in snapshot["relationships"]:
            new_start = old_to_new.get(rel["start_id"])
            new_end = old_to_new.get(rel["end_id"])
            if new_start is None or new_end is None:
                continue
            rel_type = rel["type"]
            session.run(
                f"MATCH (a) WHERE id(a) = $start_id "
                f"MATCH (b) WHERE id(b) = $end_id "
                f"CREATE (a)-[r:{rel_type} $props]->(b)",
                start_id=new_start, end_id=new_end, props=rel.get("props", {}),
            )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario operations
# ═══════════════════════════════════════════════════════════════════════════

def remove_source(driver, source_id: str) -> dict[str, int]:
    """Remove all nodes and relationships from a specific source.

    Returns counts of removed nodes and relationships.
    """
    with driver.session() as session:
        # Count before
        result = session.run(
            "MATCH (n) WHERE n.source = $sid "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS rels",
            sid=source_id,
        )
        rec = result.single()
        node_count = rec["nodes"] if rec else 0
        rel_count = rec["rels"] if rec else 0

        # Remove
        session.run(
            "MATCH (n) WHERE n.source = $sid DETACH DELETE n",
            sid=source_id,
        )

        # Also remove the Observation for this source
        session.run(
            "MATCH (obs:Observation) WHERE obs.description CONTAINS $sid "
            "DETACH DELETE obs",
            sid=source_id,
        )

    return {"removed_nodes": node_count, "removed_rels": rel_count}


def modify_entity(driver, entity_desc: str, property_name: str, new_value: str) -> bool:
    """Modify a property on an entity for hypothesis testing."""
    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE coalesce(n.description, n.name_or_description, "
            "n.name, n.value, n.summary) = $desc "
            f"SET n.{property_name} = $val, n.modified_scenario = true "
            "RETURN count(n) AS cnt",
            desc=entity_desc, val=new_value,
        )
        rec = result.single()
        return rec["cnt"] > 0 if rec else False


def compare_scenarios(
    snapshot_a: dict[str, Any],
    snapshot_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two graph snapshots and report differences.

    Returns a dict with:
    - only_in_a: nodes/rels only in snapshot A
    - only_in_b: nodes/rels only in snapshot B
    - common: nodes/rels in both
    - changed: nodes with different properties
    """
    def _node_key(node):
        labels = tuple(sorted(node["labels"]))
        desc = (node["props"].get("description")
                or node["props"].get("name_or_description")
                or node["props"].get("name")
                or node["props"].get("value")
                or node["props"].get("summary")
                or "")
        return (labels, desc)

    nodes_a = {_node_key(n): n for n in snapshot_a["nodes"]}
    nodes_b = {_node_key(n): n for n in snapshot_b["nodes"]}

    keys_a = set(nodes_a.keys())
    keys_b = set(nodes_b.keys())

    only_a = [nodes_a[k] for k in keys_a - keys_b]
    only_b = [nodes_b[k] for k in keys_b - keys_a]
    common_keys = keys_a & keys_b

    changed = []
    for key in common_keys:
        props_a = {k: v for k, v in nodes_a[key]["props"].items()
                   if k not in ("extracted_at",)}
        props_b = {k: v for k, v in nodes_b[key]["props"].items()
                   if k not in ("extracted_at",)}
        if props_a != props_b:
            changed.append({
                "entity": key,
                "props_a": props_a,
                "props_b": props_b,
                "diff": {k: (props_a.get(k), props_b.get(k))
                         for k in set(props_a) | set(props_b)
                         if props_a.get(k) != props_b.get(k)},
            })

    return {
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "common": len(common_keys),
        "changed": len(changed),
        "details": {
            "nodes_only_a": [{"labels": n["labels"],
                              "desc": _node_key(n)[1]} for n in only_a],
            "nodes_only_b": [{"labels": n["labels"],
                              "desc": _node_key(n)[1]} for n in only_b],
            "property_changes": changed[:10],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# High-level what-if API
# ═══════════════════════════════════════════════════════════════════════════

def run_what_if(
    driver,
    scenario_name: str,
    operation: str,
    **kwargs,
) -> dict[str, Any]:
    """Run a what-if scenario and return the comparison.

    Operations:
      - "remove_source": kwargs must include source_id
      - "modify_entity": kwargs must include entity_desc, property_name, new_value

    Returns a comparison dict between before and after states.
    """
    # Take baseline snapshot
    baseline = snapshot_graph(driver)
    print(f"  ▸ Baseline: {baseline['node_count']} nodes, {baseline['rel_count']} rels")

    # Apply operation
    scenario = Scenario(scenario_name)

    if operation == "remove_source":
        source_id = kwargs["source_id"]
        result = remove_source(driver, source_id)
        scenario.operations.append({"op": "remove_source", "source_id": source_id, **result})
        print(f"  ▸ Removed source '{source_id}': "
              f"{result['removed_nodes']} nodes, {result['removed_rels']} rels")

    elif operation == "modify_entity":
        entity_desc = kwargs["entity_desc"]
        prop = kwargs["property_name"]
        val = kwargs["new_value"]
        success = modify_entity(driver, entity_desc, prop, val)
        scenario.operations.append({
            "op": "modify_entity", "entity": entity_desc,
            "property": prop, "new_value": val, "success": success,
        })
        print(f"  ▸ Modified '{entity_desc}'.{prop} = '{val}' (success={success})")

    else:
        print(f"  ✗ Unknown operation: {operation}")
        return {}

    # Take post-operation snapshot
    after = snapshot_graph(driver)
    print(f"  ▸ After: {after['node_count']} nodes, {after['rel_count']} rels")

    # Compare
    comparison = compare_scenarios(baseline, after)
    print(f"  ▸ Diff: "
          f"-{comparison['only_in_a']} nodes, "
          f"+{comparison['only_in_b']} nodes, "
          f"~{comparison['changed']} changed")

    # Restore baseline
    restore_snapshot(driver, baseline)
    print(f"  ✓ Baseline restored")

    return {
        "scenario": scenario.to_dict(),
        "comparison": comparison,
        "baseline_nodes": baseline["node_count"],
        "after_nodes": after["node_count"],
    }
