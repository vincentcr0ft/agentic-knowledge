"""
08 · Digital Twin — Export & Serialisation
═════════════════════════════════════════

Exports the Neo4j event graph to portable formats:
  - RDF/Turtle (linked data interoperability)
  - JSON-LD (Schema.org compatible)
  - Cypher dump (snapshot/restore)
  - DOT/Graphviz (static visualisation)

Prerequisites:
  - Neo4j running with populated event graph
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schema import get_active_spec, get_ontology_id


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_all_graph_data(driver) -> dict[str, Any]:
    """Fetch all nodes and relationships from Neo4j."""
    nodes = []
    rels = []

    with driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE NOT n:GraphVersion "
            "RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props"
        )
        for rec in result:
            nodes.append({
                "id": rec["id"],
                "labels": list(rec["labels"]),
                "props": dict(rec["props"]),
            })

        result = session.run(
            "MATCH (a)-[r]->(b) WHERE NOT a:GraphVersion AND NOT b:GraphVersion "
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

    return {"nodes": nodes, "relationships": rels}


def _node_uri(node_id: int) -> str:
    """Generate a URI for a node."""
    return f"urn:event-twin:node:{node_id}"


def _sanitise_for_turtle(value: str) -> str:
    """Escape a string for Turtle format."""
    return (value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r"))


# ═══════════════════════════════════════════════════════════════════════════
# RDF/Turtle export
# ═══════════════════════════════════════════════════════════════════════════

def export_turtle(driver, output_path: str | Path | None = None) -> str:
    """Export the graph as RDF Turtle.

    Uses PROV-O, Schema.org, and custom event-twin vocabulary.
    """
    data = _get_all_graph_data(driver)
    ontology_id = get_ontology_id()

    lines = [
        "# Event Digital Twin — RDF/Turtle Export",
        f"# Generated: {datetime.now(timezone.utc).isoformat()}",
        f"# Ontology: {ontology_id}",
        "",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix schema: <http://schema.org/> .",
        "@prefix sosa: <http://www.w3.org/ns/sosa/> .",
        "@prefix evt: <urn:event-twin:vocab:> .",
        "@prefix node: <urn:event-twin:node:> .",
        "",
    ]

    # Map Neo4j labels to RDF classes
    label_to_class = {
        "Event": "schema:Event",
        "Person": "schema:Person",
        "Vehicle": "evt:Vehicle",
        "Location": "schema:Place",
        "Place": "schema:Place",
        "Time": "schema:DateTime",
        "Object": "evt:PhysicalObject",
        "Observation": "sosa:Observation",
        "PhysicalDescription": "evt:PhysicalDescription",
        "Role": "evt:Role",
    }

    # Nodes
    for node in data["nodes"]:
        uri = f"node:{node['id']}"
        label = node["labels"][0] if node["labels"] else "Thing"
        rdf_class = label_to_class.get(label, f"evt:{label}")

        lines.append(f"{uri}")
        lines.append(f"    a {rdf_class} ;")

        props = node["props"]
        prop_lines = []
        for key, value in sorted(props.items()):
            if value is None:
                continue
            safe_val = _sanitise_for_turtle(str(value))
            if key == "confidence":
                prop_lines.append(f'    evt:{key} "{safe_val}"^^xsd:float')
            elif key == "extracted_at":
                prop_lines.append(f'    prov:generatedAtTime "{safe_val}"^^xsd:dateTime')
            elif key == "source":
                prop_lines.append(f'    prov:wasAttributedTo "{safe_val}"')
            elif key in ("description", "name_or_description", "name"):
                prop_lines.append(f'    rdfs:label "{safe_val}"')
            else:
                prop_lines.append(f'    evt:{key} "{safe_val}"')

        if prop_lines:
            lines.append(" ;\n".join(prop_lines) + " .")
        else:
            lines[-1] = lines[-1].rstrip(" ;") + " ."
        lines.append("")

    # Relationships
    rel_to_predicate = {
        "AT_TIME": "schema:startDate",
        "OCCURRED_AT_TIME": "schema:startDate",
        "HAS_TIME": "schema:startDate",
        "AT_LOCATION": "schema:location",
        "OCCURRED_AT": "schema:location",
        "HAS_PARTICIPANT": "schema:participant",
        "INVOLVED_IN": "evt:involvedIn",
        "PRECEDED_BY": "evt:precededBy",
        "OBSERVED": "sosa:observedProperty",
        "MADE_BY": "sosa:madeBySensor",
        "DERIVED_FROM": "prov:wasDerivedFrom",
        "POSSIBLY_SAME_AS": "evt:possiblySameAs",
        "CORROBORATED_BY": "evt:corroboratedBy",
        "CONTRADICTS": "evt:contradicts",
    }

    for rel in data["relationships"]:
        s = f"node:{rel['start_id']}"
        o = f"node:{rel['end_id']}"
        p = rel_to_predicate.get(rel["type"], f"evt:{rel['type']}")
        lines.append(f"{s} {p} {o} .")

    turtle_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(turtle_text)
        print(f"  ✓ Exported Turtle to {output_path} ({len(data['nodes'])} nodes, "
              f"{len(data['relationships'])} rels)")

    return turtle_text


# ═══════════════════════════════════════════════════════════════════════════
# JSON-LD export
# ═══════════════════════════════════════════════════════════════════════════

def export_jsonld(driver, output_path: str | Path | None = None) -> dict:
    """Export the graph as Schema.org-compatible JSON-LD."""
    data = _get_all_graph_data(driver)
    ontology_id = get_ontology_id()

    context = {
        "@context": {
            "schema": "http://schema.org/",
            "prov": "http://www.w3.org/ns/prov#",
            "sosa": "http://www.w3.org/ns/sosa/",
            "evt": "urn:event-twin:vocab:",
        }
    }

    # Build node index
    node_index = {}
    for node in data["nodes"]:
        label = node["labels"][0] if node["labels"] else "Thing"
        desc = (node["props"].get("description")
                or node["props"].get("name_or_description")
                or node["props"].get("name")
                or node["props"].get("value")
                or node["props"].get("summary")
                or f"node_{node['id']}")

        entry = {
            "@id": _node_uri(node["id"]),
            "@type": label,
            "name": desc,
        }

        for key, value in node["props"].items():
            if key in ("description", "name_or_description", "name",
                       "value", "summary"):
                continue
            if value is not None:
                entry[key] = value

        node_index[node["id"]] = entry

    # Add relationships
    for rel in data["relationships"]:
        start = node_index.get(rel["start_id"])
        end = node_index.get(rel["end_id"])
        if start and end:
            rel_key = rel["type"].lower()
            existing = start.get(rel_key)
            end_ref = {"@id": end["@id"]}
            if existing:
                if isinstance(existing, list):
                    existing.append(end_ref)
                else:
                    start[rel_key] = [existing, end_ref]
            else:
                start[rel_key] = end_ref

    doc = {
        **context,
        "@graph": list(node_index.values()),
        "evt:ontologyId": ontology_id,
        "prov:generatedAtTime": datetime.now(timezone.utc).isoformat(),
    }

    if output_path:
        Path(output_path).write_text(json.dumps(doc, indent=2, default=str))
        print(f"  ✓ Exported JSON-LD to {output_path}")

    return doc


# ═══════════════════════════════════════════════════════════════════════════
# Cypher dump (snapshot)
# ═══════════════════════════════════════════════════════════════════════════

def export_cypher(driver, output_path: str | Path | None = None) -> str:
    """Export the graph as a Cypher script for snapshot/restore."""
    data = _get_all_graph_data(driver)

    lines = [
        "// Event Digital Twin — Cypher Snapshot",
        f"// Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "// Clear existing graph",
        "MATCH (n) DETACH DELETE n;",
        "",
        "// Create nodes",
    ]

    for node in data["nodes"]:
        labels = ":".join(node["labels"]) if node["labels"] else "Node"
        props_str = json.dumps(node["props"], default=str)
        lines.append(f"CREATE (:{labels} {props_str});")

    lines.append("")
    lines.append("// Create relationships")

    for rel in data["relationships"]:
        start_node = next((n for n in data["nodes"] if n["id"] == rel["start_id"]), None)
        end_node = next((n for n in data["nodes"] if n["id"] == rel["end_id"]), None)

        if not start_node or not end_node:
            continue

        def _match_clause(node, var):
            label = node["labels"][0] if node["labels"] else "Node"
            desc = (node["props"].get("description")
                    or node["props"].get("name_or_description")
                    or node["props"].get("name")
                    or node["props"].get("value")
                    or node["props"].get("summary")
                    or "")
            safe_desc = desc.replace("'", "\\'")
            return (f"MATCH ({var}:{label}) WHERE "
                    f"coalesce({var}.description, {var}.name_or_description, "
                    f"{var}.name, {var}.value, {var}.summary) = '{safe_desc}'")

        match_a = _match_clause(start_node, "a")
        match_b = _match_clause(end_node, "b")
        rel_type = rel["type"]
        props = rel.get("props", {})
        props_str = f" {json.dumps(props, default=str)}" if props else ""
        lines.append(f"{match_a} {match_b} CREATE (a)-[:{rel_type}{props_str}]->(b);")

    cypher_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(cypher_text)
        print(f"  ✓ Exported Cypher to {output_path}")

    return cypher_text


# ═══════════════════════════════════════════════════════════════════════════
# DOT/Graphviz export
# ═══════════════════════════════════════════════════════════════════════════

LABEL_COLORS = {
    "Event": "#e74c3c",
    "Person": "#3498db",
    "Vehicle": "#f39c12",
    "Location": "#2ecc71",
    "Place": "#2ecc71",
    "Time": "#9b59b6",
    "Object": "#95a5a6",
    "Observation": "#1abc9c",
    "PhysicalDescription": "#e67e22",
    "Role": "#34495e",
}


def export_dot(driver, output_path: str | Path | None = None) -> str:
    """Export the graph as a Graphviz DOT file."""
    data = _get_all_graph_data(driver)

    lines = [
        "digraph EventDigitalTwin {",
        '    rankdir=LR;',
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '    edge [fontname="Helvetica", fontsize=10];',
        "",
    ]

    for node in data["nodes"]:
        label_type = node["labels"][0] if node["labels"] else "Node"
        desc = (node["props"].get("description")
                or node["props"].get("name_or_description")
                or node["props"].get("name")
                or node["props"].get("value")
                or node["props"].get("summary")
                or f"node_{node['id']}")

        # Truncate long descriptions
        if len(desc) > 40:
            desc = desc[:37] + "..."

        color = LABEL_COLORS.get(label_type, "#bdc3c7")
        safe_desc = desc.replace('"', '\\"')
        source = node["props"].get("source", "")
        conf = node["props"].get("confidence", "")

        node_label = f"{label_type}\\n{safe_desc}"
        if source:
            node_label += f"\\n[{source}]"
        if conf:
            node_label += f" ({conf})"

        lines.append(
            f'    n{node["id"]} [label="{node_label}", '
            f'fillcolor="{color}", fontcolor="white"];'
        )

    lines.append("")

    for rel in data["relationships"]:
        rel_type = rel["type"]
        style = ""
        if rel_type == "CONTRADICTS":
            style = ', color="red", style="dashed"'
        elif rel_type == "POSSIBLY_SAME_AS":
            style = ', color="blue", style="dotted"'
        elif rel_type == "CORROBORATED_BY":
            style = ', color="green"'
        elif rel_type == "PRECEDED_BY":
            style = ', color="purple"'

        lines.append(
            f'    n{rel["start_id"]} -> n{rel["end_id"]} '
            f'[label="{rel_type}"{style}];'
        )

    lines.append("}")

    dot_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(dot_text)
        print(f"  ✓ Exported DOT to {output_path}")

    return dot_text


# ═══════════════════════════════════════════════════════════════════════════
# HTML visualisation (pyvis-style, but without the dependency)
# ═══════════════════════════════════════════════════════════════════════════

def export_html(driver, output_path: str | Path = "graph.html") -> str:
    """Export the graph as an interactive HTML visualisation.

    Uses vis.js directly (no pyvis dependency required).
    """
    data = _get_all_graph_data(driver)

    # Build vis.js nodes and edges
    vis_nodes = []
    for node in data["nodes"]:
        label_type = node["labels"][0] if node["labels"] else "Node"
        desc = (node["props"].get("description")
                or node["props"].get("name_or_description")
                or node["props"].get("name")
                or node["props"].get("value")
                or node["props"].get("summary")
                or f"node_{node['id']}")
        color = LABEL_COLORS.get(label_type, "#bdc3c7")
        source = node["props"].get("source", "unknown")
        conf = node["props"].get("confidence", "?")
        vis_nodes.append({
            "id": node["id"],
            "label": f"{label_type}\n{desc[:30]}",
            "title": f"<b>{label_type}</b><br>{desc}<br>Source: {source}<br>Confidence: {conf}",
            "color": color,
            "font": {"color": "white"},
        })

    vis_edges = []
    for rel in data["relationships"]:
        color = "#888"
        dashes = False
        if rel["type"] == "CONTRADICTS":
            color = "#e74c3c"
            dashes = True
        elif rel["type"] == "POSSIBLY_SAME_AS":
            color = "#3498db"
            dashes = True
        elif rel["type"] == "CORROBORATED_BY":
            color = "#2ecc71"
        elif rel["type"] == "PRECEDED_BY":
            color = "#9b59b6"
        vis_edges.append({
            "from": rel["start_id"],
            "to": rel["end_id"],
            "label": rel["type"],
            "color": color,
            "dashes": dashes,
            "arrows": "to",
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Event Digital Twin — Graph Visualisation</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Helvetica, Arial, sans-serif; background: #1a1a2e; color: #eee; }}
        #graph {{ width: 100%; height: 85vh; border: 1px solid #333; }}
        #header {{ padding: 10px 20px; background: #16213e; }}
        #header h1 {{ margin: 0; font-size: 18px; }}
        #header p {{ margin: 4px 0 0 0; font-size: 12px; color: #888; }}
        #legend {{ padding: 5px 20px; display: flex; gap: 15px; flex-wrap: wrap; font-size: 11px; }}
        .legend-item {{ display: flex; align-items: center; gap: 4px; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Event Digital Twin — Knowledge Graph</h1>
        <p>{len(vis_nodes)} nodes, {len(vis_edges)} relationships | Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
    </div>
    <div id="legend">
        {"".join(f'<span class="legend-item"><span class="legend-dot" style="background:{c}"></span>{l}</span>' for l, c in LABEL_COLORS.items())}
    </div>
    <div id="graph"></div>
    <script>
        var nodes = new vis.DataSet({json.dumps(vis_nodes, default=str)});
        var edges = new vis.DataSet({json.dumps(vis_edges, default=str)});
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            physics: {{ solver: 'forceAtlas2Based', stabilization: {{ iterations: 100 }} }},
            interaction: {{ hover: true, tooltipDelay: 100 }},
            edges: {{ font: {{ size: 9, color: '#aaa' }}, smooth: {{ type: 'curvedCW' }} }},
            nodes: {{ shape: 'box', margin: 8, font: {{ size: 11 }} }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"  ✓ Exported interactive HTML to {output_path}")
    return html
