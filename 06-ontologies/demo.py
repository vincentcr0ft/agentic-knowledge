"""
06 · Ontology Comparison — Demo
═══════════════════════════════

Demonstrates why ontology choice matters for knowledge-graph construction
in agentic AI pipelines.  The SAME source text is processed through three
different ontologies, and the resulting extraction prompts, graph
structures, and completeness rules are compared side-by-side.

Ontologies compared:
  1. Schema.org Event  — lightweight, high LLM extractability
  2. SEM (Simple Event Model) — first-class roles, sub-event decomposition
  3. BFO / CCO         — ISO-standard upper ontology, formal rigour

Prerequisites:
  - Ollama running with qwen2.5:7b
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase

from ontology_spec import OntologySpec
from schema_org_event import SCHEMA_ORG_EVENT
from sem_event import SEM_EVENT
from bfo_cco_event import BFO_CCO_EVENT


# ─── Connections ──────────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ─── Test scenario ────────────────────────────────────────────────────────

SAMPLE_TEXT = """\
At 3:15 PM on Wednesday 5 March, Dr Sarah Chen was walking along the 
north side of Riverside Avenue when she heard a loud crash.  She turned 
and saw a white tanker truck had collided with a green hatchback at the 
junction of Riverside Avenue and Bridge Lane.  The truck driver — a 
heavy-set man in a hi-vis jacket — climbed out, looked at the damage, 
then got back in and drove off heading east.  A young woman from the 
hatchback appeared injured and was sitting on the kerb.  Dr Chen ran 
over to help.  Another bystander — an elderly man with a walking stick 
— called 999.  Paramedics arrived approximately ten minutes later from 
City Hospital.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json(content: str) -> dict | None:
    """Try to parse JSON from LLM output, tolerating markdown fences."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return None


def extract_with_ontology(spec: OntologySpec, text: str) -> dict:
    """Run LLM extraction using the given ontology spec."""
    prompt = spec.build_extraction_prompt()
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text),
    ]
    result = llm.invoke(messages)
    extracted = _parse_json(result.content)

    if extracted is None:
        messages.append(HumanMessage(
            content="Your response was not valid JSON. Return ONLY the JSON object."
        ))
        result = llm.invoke(messages)
        extracted = _parse_json(result.content)

    return extracted or {"entities": [], "relationships": []}


def print_section(title: str) -> None:
    """Print a section header."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_extraction_summary(name: str, data: dict) -> None:
    """Print a compact summary of extraction results."""
    entities = data.get("entities", [])
    rels = data.get("relationships", [])

    # Count by label
    label_counts: dict[str, int] = {}
    for e in entities:
        lbl = e.get("label", "?")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    rel_counts: dict[str, int] = {}
    for r in rels:
        rt = r.get("rel_type", "?")
        rel_counts[rt] = rel_counts.get(rt, 0) + 1

    print(f"\n  ┌─ {name}")
    print(f"  │  Entities: {len(entities)}")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  │    {lbl}: {cnt}")
    print(f"  │  Relationships: {len(rels)}")
    for rt, cnt in sorted(rel_counts.items()):
        print(f"  │    {rt}: {cnt}")
    print(f"  └─")


# ═══════════════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ontologies = [
        ("Schema.org Event", SCHEMA_ORG_EVENT),
        ("SEM (Simple Event Model)", SEM_EVENT),
        ("BFO / CCO", BFO_CCO_EVENT),
    ]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         06 · ONTOLOGY COMPARISON DEMO                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    print("\nSource text:")
    for line in SAMPLE_TEXT.strip().split("\n"):
        print(f"  │ {line.strip()}")

    # ── Phase 1: Compare ontology specs ─────────────────────────────────

    print_section("Phase 1 — Ontology Specifications")

    for name, spec in ontologies:
        print(f"\n  {name}")
        print(f"    ID:       {spec.id}")
        print(f"    Nodes:    {', '.join(spec.node_types.keys())}")

        extractable_rels = [
            r for r in spec.relationship_types
            if r not in spec.system_managed_rels
        ]
        print(f"    Rels:     {', '.join(extractable_rels)}")
        print(f"    Rules:    {len(spec.completeness_rules)} completeness rules")

    # ── Phase 2: Extract with each ontology ─────────────────────────────

    print_section("Phase 2 — LLM Extraction (same text, different ontology)")

    results: dict[str, dict] = {}
    for name, spec in ontologies:
        print(f"\n  ▸ Extracting with {name}...")
        extracted = extract_with_ontology(spec, SAMPLE_TEXT)
        results[name] = extracted
        print_extraction_summary(name, extracted)

    # ── Phase 3: Compare key differences ────────────────────────────────

    print_section("Phase 3 — Structural Comparison")

    print("\n  What each ontology captures differently:\n")

    # Role modelling
    print("  ROLE MODELLING:")
    schema_roles = []
    for e in results.get("Schema.org Event", {}).get("entities", []):
        if e.get("label") == "Person" and "role" in e.get("properties", {}):
            schema_roles.append(
                f"{e['properties'].get('name_or_description', '?')}"
                f" → {e['properties']['role']}"
            )
    if schema_roles:
        print(f"    Schema.org: roles as Person properties: {'; '.join(schema_roles)}")
    else:
        print("    Schema.org: (no role data extracted)")

    sem_roles = [
        e for e in results.get("SEM (Simple Event Model)", {}).get("entities", [])
        if e.get("label") == "Role"
    ]
    if sem_roles:
        role_types = [r.get("properties", {}).get("role_type", "?") for r in sem_roles]
        print(f"    SEM:        roles as first-class nodes: {', '.join(role_types)}")
    else:
        print("    SEM:        (no Role nodes extracted)")

    bfo_roles = [
        e for e in results.get("BFO / CCO", {}).get("entities", [])
        if e.get("label") == "AgentRole"
    ]
    if bfo_roles:
        role_types = [r.get("properties", {}).get("role_type", "?") for r in bfo_roles]
        print(f"    BFO/CCO:    roles as bfo:Role nodes: {', '.join(role_types)}")
    else:
        print("    BFO/CCO:    (no AgentRole nodes extracted)")

    # Event decomposition
    print("\n  EVENT DECOMPOSITION:")
    for name, _ in ontologies:
        rels = results.get(name, {}).get("relationships", [])
        sub_event_rels = [
            r for r in rels
            if r.get("rel_type") in (
                "HAS_SUB_EVENT", "HAS_PART", "PRECEDED", "PRECEDES", "CAUSED"
            )
        ]
        print(f"    {name[:15]:15s}: {len(sub_event_rels)} structural event links")

    # ── Phase 4: SHACL shapes comparison ────────────────────────────────

    print_section("Phase 4 — Generated SHACL Shapes (excerpt)")

    for name, spec in ontologies:
        shapes = spec.build_shacl_shapes()
        # Show first shape only
        first_shape = shapes.split("\n\n")[1] if "\n\n" in shapes else shapes[:300]
        print(f"\n  {name}:")
        for line in first_shape.strip().split("\n")[:8]:
            print(f"    {line}")
        print(f"    ...")

    # ── Phase 5: Completeness rules comparison ──────────────────────────

    print_section("Phase 5 — Completeness Rules")

    for name, spec in ontologies:
        print(f"\n  {name}:")
        for rule in spec.completeness_rules:
            print(f"    [{rule.priority:8s}] {rule.description}")

    # ── Summary ─────────────────────────────────────────────────────────

    print_section("Summary")

    print("""
  Schema.org Event:
    ✓ Most LLM-friendly — simple labels, high extraction accuracy
    ✗ Shallow role model — can't represent same person in different roles
    ✗ No sub-event decomposition
    Best for: rapid prototyping, lightweight applications

  SEM (Simple Event Model):
    ✓ First-class roles via Role nodes
    ✓ Sub-event decomposition via HAS_SUB_EVENT
    ✓ Designed for narrative/historical event modelling
    ✗ Less well-known, fewer tools
    Best for: multi-source event analysis, media/historical events

  BFO / CCO:
    ✓ ISO standard (21838-2:2021), adopted by DOD/IC
    ✓ Formal process ontology with mereological decomposition
    ✓ Information Content Entities for provenance
    ✗ Heavier — more nodes, harder for LLMs to extract cleanly
    ✗ Requires more training/prompt engineering
    Best for: forensic/intelligence applications, formal reasoning
""")


if __name__ == "__main__":
    main()
