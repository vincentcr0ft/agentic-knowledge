# Event Digital Twin — Implementation Walkthrough

## Overview

This guide walks through the complete extended implementation: architecture, every module, every command, and how the pieces connect.

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           demo.py                                │
│                     (pipeline orchestrator)                       │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  ingest   │──▶│  fusion   │──▶│ temporal  │──▶│ interview│     │
│  │  .py      │   │  .py      │   │  .py      │   │  .py     │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       │                                              │           │
│       ▼                                              ▼           │
│  ┌──────────┐                                  ┌──────────┐     │
│  │ schema.py │                                  │ query.py  │     │
│  │ (ontology)│                                  │ (Q&A REPL)│     │
│  └──────────┘                                  └──────────┘     │
│                                                      │           │
│                          ┌───────────────────────────┤           │
│                          ▼           ▼               ▼           │
│                    ┌──────────┐ ┌──────────┐  ┌──────────┐      │
│                    │ export.py│ │simulation│  │quality   │      │
│                    │          │ │  .py     │  │ _ext.py  │      │
│                    └──────────┘ └──────────┘  └──────────┘      │
└──────────────────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
    ┌──────────┐        ┌──────────┐
    │  Neo4j   │        │  Ollama  │
    │ bolt:7687│        │ :11434   │
    └──────────┘        └──────────┘
```

## 2. Prerequisites

```bash
# Start the infrastructure
cd 08-event-digital-twin
docker-compose up -d          # Neo4j on bolt://localhost:7687

# Start Ollama with the model
ollama pull qwen2.5:7b
ollama serve                  # http://localhost:11434
```

Environment:
- **Neo4j**: bolt://localhost:7687 — credentials: `neo4j` / `cabbage123`
- **Ollama**: http://localhost:11434 — model: `qwen2.5:7b`
- **Python packages**: `neo4j`, `langchain-ollama`, `langchain-core`, `langgraph`, `rdflib` (for RDF export)

## 3. Running the Pipeline

### Basic (single source, same as original)

```bash
python demo.py statements/king_street_collision.txt
```

### Multi-source (extended)

```bash
python demo.py statements/king_street_collision.txt \
               statements/queen_road_witness.txt \
               statements/cctv_log.txt \
               statements/paramedic_report.txt
```

### With clear (fresh start)

```bash
python demo.py --clear statements/king_street_collision.txt \
                        statements/queen_road_witness.txt
```

The `--clear` flag deletes the entire graph before the first source is ingested. Without it, each run is additive.

## 4. Module-by-Module

---

### 4.1 `schema.py` — Ontology Definition

Defines the knowledge graph ontology using constructs from Chapter 06:

- **Node types**: `Event`, `Person`, `Vehicle`, `Location`, `Time`, `Observation`
- **Relationship types**: `INVOLVED_IN`, `OCCURRED_AT`, `AT_TIME`, `OBSERVED`, `DROVE`, `HIT`, `CALLED`
- **Pluggable**: The ontology ID (`schema-org-event-v1`) is stamped onto every extracted node

Key function:

```python
def get_ontology() -> OntologySpec
```

Returns the ontology specification that constrains the LLM extraction prompt.

---

### 4.2 `ingest.py` — Source Ingestion Pipeline

A LangGraph state machine that transforms text into graph nodes.

**Pipeline steps:**

```
segment → extract → resolve_coreferences → load_to_graph
```

**State definition:**

```python
class IngestState(TypedDict):
    text: str
    segments: list[str]
    raw_extractions: list[dict]
    resolved: list[dict]
    source_id: str          # ← NEW: identifies the source
    source_type: str        # ← NEW: "statement" | "cctv" | "medical"
```

**Key changes from original:**

1. **No destructive delete** — `MATCH (n) DETACH DELETE n` removed
2. **Source tagging** — every node gets `source` and `source_type` properties
3. **Graph versioning** — `GraphVersion` node created with timestamp and source list
4. **Numerical confidence** — `confidence: 0.8` (float) instead of `"high"` (string)
5. **Additive MERGE** — uses Cypher MERGE to avoid duplicates within a source

**Public API:**

```python
def ingest_statement(text: str,
                     source_id: str = "statement_1",
                     source_type: str = "statement",
                     clear: bool = False) -> dict
```

---

### 4.3 `fusion.py` — Multi-Source Entity Resolution

Runs after all sources are ingested. Compares entities across sources:

**Pipeline:**

1. **Get sources** — query graph for distinct `source` values
2. **Get entities by source** — collect all nodes grouped by source
3. **Resolve** — for each pair of sources, find candidate entity matches
4. **Merge** — create `POSSIBLY_SAME_AS` relationships with confidence
5. **Corroborate** — create `CORROBORATED_BY` relationships between matching observations
6. **Contradict** — create `CONTRADICTS` relationships for conflicting values

**Entity resolution approach:**

```python
def _resolve_across_sources(entities_a, entities_b, driver, llm):
    """
    For each pair of entities with the same label:
    1. Compare attributes (name, description, type)
    2. If attributes alone are ambiguous, ask the LLM
    3. Return match decisions with confidence scores
    """
```

**Confidence updates:**

```python
# Corroboration: boost confidence toward 1.0
new_confidence = old_confidence + (1 - old_confidence) * 0.3

# Contradiction: decay confidence
new_confidence = old_confidence * 0.7
```

**Public API:**

```python
def run_fusion(driver, llm) -> dict
# Returns: {"sources": [...], "merges": N, "corroborations": N, "contradictions": N}
```

---

### 4.4 `temporal.py` — Temporal Reasoning

Extracts temporal information and reasons about event ordering.

**Capabilities:**

1. **Time parsing** — handles multiple formats:
   - `14:13:45` → 853.75 minutes
   - `2:15 PM` → 855.0 minutes
   - `quarter past two` → 855.0 minutes
   - `about ten minutes later` → contextually resolved

2. **Timeline construction:**

```python
def build_timeline(driver) -> list[dict]:
    """
    Query graph for all time-related nodes and relationships.
    Parse all time values.
    Sort by parsed time.
    Materialise PRECEDED_BY relationships.
    Return ordered timeline.
    """
```

3. **Consistency checking:**

```python
def check_consistency(driver) -> dict:
    """
    Check for:
    - Temporal cycles (A before B before A)
    - Contradicting timestamps (same event, different times from different sources)
    - Missing timestamps (events with no temporal anchor)
    Returns: {"consistent": bool, "issues": [...]}
    """
```

4. **ASCII timeline output:**

```
14:13:45 │ Impact detected (CCTV)
14:14:02 │ Driver exits vehicle (CCTV)
14:14:22 │ Driver departs on foot (CCTV)
~14:15   │ Collision observed (Witness 1)
~14:15   │ Collision observed (Witness 2)
14:24:38 │ Ambulance arrives (CCTV)
```

---

### 4.5 `interview.py` — Gap Analysis Loop

Unchanged from original. Analyses the graph for missing information and poses questions.

**Pipeline:**

```
analyse_graph → generate_questions → (user answers) → incorporate_answers
```

The interview loop identifies:
- Events with no participants
- People with no descriptions
- Locations with no spatial context
- Missing temporal anchors
- Unconnected subgraphs

---

### 4.6 `query.py` — Interactive Q&A with Provenance

Extended with source-aware commands.

**Slash commands:**

| Command | Description |
|---------|-------------|
| `/sources` | List all ingested sources with entity counts |
| `/from <source_id>` | Show all entities from a specific source |
| `/contradictions` | List all CONTRADICTS relationships |
| `/timeline` | Show the event timeline (ASCII) |
| `/export <format>` | Export graph (turtle, jsonld, cypher, dot, html) |
| `/whatif remove <source_id>` | Simulate removing a source — show what would change |

**Regular queries** are answered by:
1. Extracting entities from the question
2. Querying Neo4j for matching nodes and their neighbourhood
3. Passing the graph context + question to the LLM
4. Generating a grounded answer with provenance citations

---

### 4.7 `simulation.py` — What-If Scenario Engine

Supports counterfactual reasoning by snapshot/modify/diff/restore:

```python
# Public API
def run_what_if(driver, operation: str, **kwargs) -> dict:
    """
    1. Snapshot the current graph state
    2. Apply the operation (remove_source, modify_entity)
    3. Compare before/after
    4. Restore the baseline
    Returns: diff of what changed
    """
```

**Operations:**

```python
# Remove a source and see what's lost
run_what_if(driver, "remove_source", source_id="queen_road_witness")

# Modify an entity and check consistency
run_what_if(driver, "modify_entity",
            label="Person", name="James Chen",
            property="age", new_value=28)
```

---

### 4.8 `export.py` — Graph Export

Five export formats, all invocable from the REPL via `/export <format>`:

```python
export_turtle(driver, output_path)   # RDF/Turtle with PROV-O, Schema.org, SOSA
export_jsonld(driver, output_path)   # JSON-LD (Schema.org compatible)
export_cypher(driver, output_path)   # Cypher CREATE statements for snapshot
export_dot(driver, output_path)      # Graphviz DOT with entity-type colours
export_html(driver, output_path)     # Self-contained vis.js interactive page
```

The HTML export uses vis.js loaded from CDN — no Python dependencies beyond Neo4j.

---

### 4.9 `quality_ext.py` — Extended Quality Probes

Builds on the Chapter 07 quality framework with event-digital-twin-specific probes:

| Probe | What it Checks |
|-------|---------------|
| `probe_population_completeness` | At least 1 Event, 2+ Persons, 1+ Location, 1+ Time; events linked to participants; source provenance |
| `probe_temporal_consistency` | Delegates to `temporal.check_consistency()` |
| `probe_cross_source_consistency` | Counts CONTRADICTS, CORROBORATED_BY, POSSIBLY_SAME_AS relationships |
| `probe_narrative_reconstruction` | LLM-as-judge: given full graph, reconstruct a narrative and rate it 1–5 |

```python
def run_extended_quality(driver, llm) -> QualityReport:
    """Returns composite quality score and per-probe results."""
```

---

## 5. The Test Corpus

Four sources of increasing formality:

| File | Type | Perspective | Precision |
|------|------|------------|-----------|
| `king_street_collision.txt` | Witness statement | Pedestrian on King Street | Natural time ("quarter past two"), subjective descriptions |
| `queen_road_witness.txt` | Witness statement | Person at bus stop on Queen's Road | Different vantage point, partial plate |
| `cctv_log.txt` | CCTV log | Traffic camera KS-QR-001 | Precise timestamps (14:13:45), objective measurements (38 mph) |
| `paramedic_report.txt` | Medical report | Ambulance crew | Patient identity (James Chen, age 34), clinical observations |

These sources intentionally overlap and conflict:
- **Witness 1** says "about quarter past two" — CCTV says 14:13:45
- **Witness 1** describes "dark jacket" — Witness 2 says "dark coat"
- **CCTV** gives partial plate KV68 — Witness 2 gives partial plate KV
- **Paramedic** identifies the victim by name — witnesses describe appearance only

---

## 6. Pipeline Execution Flow

```
1. Parse arguments → list of file paths + optional --clear flag

2. For each file:
   a. Read the file content
   b. Infer source type from content/filename
   c. Call ingest_statement(text, source_id, source_type, clear=first_file_and_clear)

3. If more than one source was ingested:
   a. Run fusion (cross-source entity resolution)
   b. Print fusion summary (merges, corroborations, contradictions)

4. Build temporal timeline
   a. Parse all time values in the graph
   b. Materialise PRECEDED_BY relationships
   c. Check temporal consistency

5. Run interview loop
   a. Analyse graph for gaps
   b. Generate questions
   c. Incorporate user answers (or skip)

6. Run quality probes
   a. Population completeness
   b. Temporal consistency
   c. Cross-source consistency
   d. Narrative reconstruction

7. Start interactive query REPL
   a. Show source summary
   b. Accept natural language questions or /commands
```

---

## 7. Graph Schema (Neo4j)

After a full multi-source ingestion, the graph contains:

**Node labels:**
- `Event` — what happened (collision, departure, arrival)
- `Person` — participants (witnesses, victim, driver, paramedic)
- `Vehicle` — vehicles involved (red hatchback, bicycle, ambulance)
- `Location` — places (King Street, Queen's Road junction)
- `Time` — temporal anchors ("14:13:45", "quarter past two")
- `Observation` — provenance records linking facts to sources
- `GraphVersion` — version tracking nodes

**Relationship types (original):**
- `INVOLVED_IN`, `OCCURRED_AT`, `AT_TIME`, `OBSERVED`, `DROVE`, `HIT`, `CALLED`

**Relationship types (new):**
- `POSSIBLY_SAME_AS` — cross-source entity match (confidence ≥ 0.7)
- `CORROBORATED_BY` — same fact reported by multiple sources
- `CONTRADICTS` — conflicting values between sources
- `PRECEDED_BY` — temporal ordering between events

**Node properties (new):**
- `source` — source identifier
- `source_type` — "statement", "cctv", "medical"
- `confidence` — 0.0 to 1.0
- `extracted_at` — ISO timestamp
- `ontology_id` — which ontology was used for extraction

---

## 8. Extending Further

See [EXTENSIONS.md](EXTENSIONS.md) for research-level extensions:
- Full Bayesian Network integration (pgmpy)
- SHACL shape validation for automatic constraint checking
- Geospatial reasoning with GeoSPARQL
- Integration with real forensic ontologies (UCO/CASE)
- Agent-based multi-stakeholder simulation
