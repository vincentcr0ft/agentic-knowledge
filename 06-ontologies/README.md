# 06 · Ontology Comparison for Agentic AI Pipelines

## Purpose

When building knowledge graphs from unstructured text, the **choice of ontology** fundamentally shapes what gets extracted, how it's structured, and what questions you can answer. This module introduces a **pluggable ontology framework** and compares three ontologies on the same source text.

## Ontologies Compared

| Ontology | Standard | Strengths | Weaknesses |
|----------|----------|-----------|------------|
| **Schema.org Event** | W3C community | Simple labels, high LLM extractability | Shallow roles, no sub-events |
| **SEM** | VU Amsterdam | First-class roles, sub-event decomposition | Less tooling |
| **BFO / CCO** | ISO 21838 + DOD/IC | Formal rigour, mereological reasoning | Heavier, harder for LLMs |

## Key Concepts

### The OntologySpec Abstraction

Every ontology is expressed as an `OntologySpec` — a single Python data structure that drives:

- **Extraction prompts** — the LLM is told which node types, relationship types, and properties to extract
- **Graph loading** — MERGE statements are generated from the spec
- **Completeness rules** — Cypher gap-detection queries per ontology
- **SHACL shapes** — constraint validation shapes generated from the spec
- **Neo4j constraints** — uniqueness constraints for entity resolution

```python
from ontology_spec import OntologySpec
from schema_org_event import SCHEMA_ORG_EVENT

# Generate an extraction prompt for this ontology
prompt = SCHEMA_ORG_EVENT.build_extraction_prompt()

# Generate SHACL shapes
shapes_ttl = SCHEMA_ORG_EVENT.build_shacl_shapes()
```

### Role Modelling — The Critical Difference

**Schema.org**: `Person.role = "witness"` — static property, one role per person.

**SEM**: `Actor -[HAS_ROLE]-> Role {role_type: "witness"}` — first-class, the same actor can have multiple roles across different events.

**BFO/CCO**: `Agent -[BEARS_ROLE]-> AgentRole -[REALIZED_IN]-> Process` — formal ontological role with realization semantics.

### Event Decomposition

**Schema.org**: flat `PRECEDED` / `CAUSED` relations between events.

**SEM**: `sem:hasSubEvent` — events can nest (a collision *contains* impact + exit + fleeing).

**BFO/CCO**: `bfo:has_part` — mereological decomposition with formal part-whole semantics.

## Running the Demo

```bash
cd 06-ontologies
python demo.py
```

The demo extracts entities from a sample incident report using all three ontologies and compares the resulting graph structures.

## Prerequisites

- Ollama running with `qwen2.5:7b`
- Neo4j on `bolt://localhost:7687` (neo4j / cabbage123)
- Python packages: `langchain-ollama`, `neo4j`

## Files

| File | Purpose |
|------|---------|
| `ontology_spec.py` | Abstract `OntologySpec` dataclass — the pluggable interface |
| `schema_org_event.py` | Schema.org Event ontology instance |
| `sem_event.py` | SEM (Simple Event Model) ontology instance |
| `bfo_cco_event.py` | BFO / CCO ontology instance |
| `demo.py` | Side-by-side comparison demo |
| `inspect_state.py` | Quick dump of all registered ontology specs |

## Research References

- van Hage et al. (2011) — SEM: Design and use of the Simple Event Model
- Smith et al. (2015) — Building Ontologies with Basic Formal Ontology (MIT Press)
- DOD/IC Memorandum (2024) — BFO+CCO adopted as baseline standard
- Dai et al. (2024) — Linearised triples for LLM reasoning
- Zhu et al. (2023) — LLMs for KG reasoning vs extraction
