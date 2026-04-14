# Implementation Deep-Dive: The Pluggable Ontology Framework

## Architecture Overview

The `06-ontologies` module implements a **spec-driven pipeline** where a single data structure — `OntologySpec` — controls every stage of processing: extraction prompts, graph loading, completeness validation, SHACL shape generation, and Neo4j constraint creation. Swapping the spec changes the entire pipeline's behaviour without touching any pipeline code.

```
Source Text ──→ OntologySpec.build_extraction_prompt() ──→ LLM ──→ JSON
                                                                    │
     OntologySpec.completeness_rules ←── Neo4j ←── MERGE statements ←┘
                    │
                    ▼
            Gap Detection Report
```

## Core Data Structures

### `OntologySpec` ([ontology_spec.py](ontology_spec.py))

The central dataclass contains:

| Field | Type | Purpose |
|-------|------|---------|
| `node_types` | `dict[str, NodeSpec]` | All node labels, their required/optional properties, merge keys |
| `relationship_types` | `dict[str, RelSpec]` | All edge types with source/target labels |
| `completeness_rules` | `list[CompletenessRule]` | Cypher queries that detect structural gaps |
| `system_managed_rels` | `tuple[str, ...]` | Rels excluded from LLM extraction (generated programmatically) |
| `system_managed_nodes` | `tuple[str, ...]` | Nodes excluded from LLM extraction |
| `event_types` / `person_roles` | `tuple[str, ...]` | Controlled vocabularies injected into extraction prompts |
| `provenance_props` | `dict[str, str]` | Standard provenance metadata fields |

### Key Methods

- **`build_extraction_prompt()`** — Generates a complete LLM instruction containing target entity types with required/optional properties, relationship types with source/target constraints, controlled vocabularies, and a JSON output schema. System-managed types are automatically excluded.

- **`build_shacl_shapes()`** — Generates SHACL Turtle from the spec. Each `NodeSpec` becomes a `sh:NodeShape` with `sh:property` constraints for each required property. This enables RDF-world validation tooling.

- **`get_constraint_cypher()`** — Generates `CREATE CONSTRAINT IF NOT EXISTS` statements for Neo4j, using each node's `merge_key` for uniqueness enforcement.

## The Three Ontology Instances

### Schema.org Event ([schema_org_event.py](schema_org_event.py))

- **8 node types**: Event, Person, Vehicle, Location, Time, Object, PhysicalDescription, Observation
- **13 relationship types** (3 system-managed: OBSERVED, MADE_BY, DERIVED_FROM)
- **5 completeness rules** (event needs time, location, participant; suspect needs description; vehicle needs identifiers)
- **Role model**: `Person.role` property — flat, single-valued
- **SOSA/PROV integration**: Observation nodes model the witness-as-sensor pattern

### SEM ([sem_event.py](sem_event.py))

- **7 node types**: Event, Actor, Place, Time, Role, Object, PhysicalDescription
- **10 relationship types** (all LLM-extractable)
- **5 completeness rules** (similar to Schema.org, plus actor-needs-role)
- **Role model**: `Role` as first-class node, linked via `HAS_ROLE` — multi-valued
- **Sub-event support**: `HAS_SUB_EVENT` enables hierarchical event decomposition

### BFO / CCO ([bfo_cco_event.py](bfo_cco_event.py))

- **9 node types**: Process, Act, Agent, AgentRole, MaterialEntity, Site, TemporalInterval, InformationContentEntity, DescriptiveICE
- **14 relationship types** (all LLM-extractable)
- **5 completeness rules** (process needs temporal/spatial/agent; agent needs role; suspect needs description)
- **Role model**: `AgentRole` node with `BEARS_ROLE` + `REALIZED_IN` — three-place relation
- **Information entities**: `InformationContentEntity` for provenance; `IS_ABOUT` and `CREATED_BY` for statement tracking

## Demo Pipeline ([demo.py](demo.py))

The demo executes five phases:

1. **Spec comparison** — Prints node types, relationship types, and rule counts for each ontology
2. **LLM extraction** — Sends the same sample text (a traffic collision witness statement) through each ontology's extraction prompt using `qwen2.5:7b` via Ollama
3. **Structural comparison** — Compares role modelling (flat vs. reified vs. BFO roles) and event decomposition (structural link counts) across results
4. **SHACL shape comparison** — Generates and displays SHACL shapes per ontology
5. **Completeness rules comparison** — Lists all gap-detection rules per ontology

### JSON Extraction and Error Handling

The `_parse_json()` function handles common LLM output issues:
- Strips markdown code fences
- Attempts direct JSON parse
- Falls back to extracting the first `{...}` block
- On failure, sends a retry message asking for JSON-only output

## Current Limitations

1. **No actual graph loading** — The demo extracts and compares but doesn't MERGE into Neo4j. The `08-digital-twin` module handles actual graph construction.

2. **No cross-ontology mapping** — Each ontology runs independently. There's no mechanism to align or merge results across ontologies.

3. **No extraction metrics** — The demo doesn't measure precision, recall, or F1 against a gold-standard annotation. Comparison is qualitative only.

4. **Single sample text** — Only one witness statement is used. Different text types (formal reports, transcripts, social media) may shift the ontology trade-offs.

5. **No confidence calibration** — Extracted entities don't carry extraction-confidence scores that downstream consumers could use for filtering.

6. **SHACL shapes are basic** — Only `sh:minCount` + `sh:datatype xsd:string` constraints are generated. No cardinality limits, value ranges, or relationship shape constraints.

7. **No OWL export** — The ontology specs live as Python dataclasses. There's no export to OWL/RDF for use in standard ontology tooling (Protégé, reasoners).

8. **Prompt engineering is static** — The extraction prompt is template-generated. No few-shot examples, no chain-of-thought decomposition, no ontology-specific extraction strategies.

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Ollama + `qwen2.5:7b` | Local, private text extraction |
| Graph DB | Neo4j (bolt://localhost:7687) | Property graph storage |
| LLM framework | LangChain (langchain-ollama) | Message handling, model abstraction |
| Validation | Custom Cypher rules | Gap detection |

## How to Extend

**Adding a new ontology:**
1. Create a new file (e.g., `dolce_event.py`)
2. Define an `OntologySpec` instance with node types, relationship types, and completeness rules
3. Import it in `demo.py` and add it to the `ontologies` list
4. Run — the entire pipeline adapts automatically

**Adding new completeness rules:**
Add `CompletenessRule` instances to any ontology's `completeness_rules` list. Each rule needs a Cypher query that returns `entity_desc` and `label` columns for entities violating the constraint.

---

*This module is part of the Agentic AI training series. See [README.md](README.md) for setup instructions.*
