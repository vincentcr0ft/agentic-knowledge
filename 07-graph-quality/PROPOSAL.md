# Knowledge Graph Quality Probing — Tool Investigation & Proposal

This document extends **§3 Knowledge Graph Quality Assessment** of [EXTENSIONS.md](EXTENSIONS.md) with a concrete tool investigation and implementation proposal. The goal is to move beyond the current schema-completeness checks in `schema.py` to a multi-dimensional quality probing framework that can be run after every ingest/interview round.

---

## 1. Problem Statement

`run_schema_completeness()` currently checks whether required properties and relationships exist. This covers one facet of KG quality — **property completeness** — but leaves four critical dimensions unaddressed:

| Dimension | Current Coverage | Gap |
|-----------|-----------------|-----|
| Schema completeness | Partial — checks props, not node-type population | Missing population coverage (e.g. "does the graph have ≥ 1 Location?") |
| Consistency | None | No temporal acyclicity checks, no role-constraint validation |
| Accuracy | Minimal — `confidence` tag only | No re-extraction verification, no source-grounding audit |
| Coherence | None | No narrative-reconstruction test, no community-level summary scoring |
| Timeliness | None | No staleness detection |

---

## 2. Tool Landscape — Investigation Results

### 2.1 Structural & Schema Validation

#### pySHACL (`pyshacl` v0.31.0)
- **What it does**: W3C SHACL (Shapes Constraint Language) validator for RDF graphs. Validates data graphs against declarative shape constraints: cardinality, value types, patterns, closed shapes, SPARQL-based constraints.
- **Fit for our system**: The current schema is in Python dataclasses, not RDF. However, `rdflib` can export the Neo4j graph to RDF, and SHACL shapes can encode every constraint currently in `NODE_TYPES` and `RELATIONSHIP_TYPES` — plus new ones like cardinality (e.g. "every Event must have exactly 1 OCCURRED_AT_TIME") and closed-shape validation (no unexpected properties).
- **Key capability**: Returns a formal `ValidationReport` with per-node violation details, severity levels (Info/Warning/Violation), and machine-readable results.
- **Integration path**: Export graph → RDF via `rdflib` → validate against SHACL shapes → parse report → feed violations into gap prioritisation.
- **License**: Apache 2.0
- **Verdict**: **Recommended for formal schema/constraint validation.** Replaces hand-written Cypher completeness queries with a standards-based approach that can grow with the ontology.

#### kglab (`kglab` v1.0.1)
- **What it does**: Abstraction layer for KG construction and analysis. Wraps `rdflib`, NetworkX, pySHACL, and RAPIDS into a unified API. Includes a `Measure` class that computes edge count, node count, and basic graph statistics.
- **Fit for our system**: Useful as a convenience wrapper — combines RDF export, SHACL validation, and NetworkX graph metrics in one API. The `Measure` class provides quick structural health checks (isolated nodes, edge density, degree distribution).
- **Key capability**: `kglab.Measure().measure_graph(kg)` gives node/edge counts; integrates pySHACL for constraint validation; can convert to NetworkX for centrality/connectivity analysis.
- **Integration path**: Build `kglab.KnowledgeGraph` from Neo4j export → run `Measure` → run SHACL validation → compute NetworkX metrics.
- **License**: MIT
- **Verdict**: **Recommended as the orchestration layer** that ties together RDF, SHACL, and graph-metric tools.

### 2.2 Graph-Structural Quality (Neo4j Native)

#### Neo4j Graph Data Science (GDS) Library
- **What it does**: In-database graph algorithms: centrality (PageRank, betweenness), community detection (Louvain, Label Propagation), similarity (node similarity, Jaccard), path finding, and DAG algorithms.
- **Fit for our system**: Already available since the system runs Neo4j. Key quality-relevant algorithms:
  - **Weakly Connected Components (WCC)**: Detect disconnected subgraphs — a quality signal (orphan nodes, fragmented narratives).
  - **Degree centrality**: Find nodes with suspiciously low connectivity (potential extraction failures) or suspiciously high connectivity (potential over-linking).
  - **Louvain community detection**: Identify natural clusters — each community should represent a coherent sub-narrative. Communities that span unrelated events suggest entity resolution errors.
  - **Topological link prediction**: Predict missing relationships based on graph structure — a completeness signal.
- **Integration path**: Cypher calls via the existing Neo4j driver; no additional dependencies.
- **License**: Neo4j Community Edition (GPLv3) or Enterprise (commercial).
- **Verdict**: **Recommended for in-database structural quality metrics.** Zero additional dependencies; runs where the data lives.

#### APOC (Awesome Procedures on Cypher)
- **What it does**: Utility library for Neo4j — includes graph refactoring, data import/export, and metadata inspection procedures.
- **Fit for our system**: `apoc.meta.stats()` provides node/relationship type counts; `apoc.meta.schema()` returns the actual schema as a map; `apoc.path.expandConfig()` enables targeted reachability checks. Useful for schema-drift detection (comparing expected vs. actual schema).
- **Integration path**: Already available if APOC is installed in the Neo4j instance.
- **Verdict**: **Recommended for quick schema-drift detection and metadata inspection.**

### 2.3 LLM-Based Quality Evaluation

#### DeepEval (`deepeval` v3.9.6)
- **What it does**: LLM evaluation framework with metrics like G-Eval (LLM-as-judge for arbitrary criteria), answer relevancy, hallucination detection, and faithfulness scoring. Supports custom metrics and pytest integration.
- **Fit for our system**: The **coherence** and **accuracy** dimensions require semantic judgment that structural tools cannot provide. DeepEval's G-Eval metric can:
  - Score narrative coherence: given the linearised graph, can an LLM produce a coherent summary? (1–5 score)
  - Score extraction faithfulness: given source text and extracted triples, does the extraction faithfully represent the source? (hallucination detection)
  - Score completeness from a semantic perspective: given the source text, are there important facts not captured in the graph?
- **Key capability**: `GEval(criteria="...", evaluation_params=[...])` creates custom evaluation metrics with explanations. Can use local LLMs via Ollama.
- **Integration path**: Linearise graph → create `LLMTestCase(input=source_text, actual_output=linearised_graph)` → run faithfulness/coherence metrics.
- **License**: Apache 2.0
- **Verdict**: **Recommended for coherence and extraction-faithfulness scoring.** The only tool that can assess semantic quality dimensions.

#### Ragas (`ragas` v0.4.3)
- **What it does**: RAG evaluation framework with metrics for answer relevancy, faithfulness, and context recall/precision.
- **Fit for our system**: Overlaps with DeepEval but specialised for RAG pipelines. Less applicable to pure KG quality probing since its metrics assume a retrieval-generation-answer flow.
- **Verdict**: **Not recommended for KG quality probing** — better suited for evaluating the query phase (§3 of the demo pipeline). Could be used later to evaluate the `query.py` GraphRAG answers.

### 2.4 Knowledge Graph Embedding & Link Prediction

#### AmpliGraph (`ampligraph` v2.1.0)
- **What it does**: Knowledge graph embedding library for link prediction, triple classification, and entity clustering. Trains embedding models (TransE, ComplEx, DistMult, HolE) on KG triples and predicts missing links.
- **Fit for our system**: Link prediction can estimate **population completeness** — if the model predicts a relationship that doesn't exist, it may represent a gap. Triple classification can flag low-plausibility triples.
- **Key capability**: `evaluate_performance()` computes MRR, Hits@K for link prediction; `discover_facts()` predicts missing triples.
- **Integration path**: Export triples from Neo4j → train embedding model → predict missing links → compare predictions against known schema expectations.
- **License**: Apache 2.0
- **Verdict**: **Conditionally recommended.** Powerful for larger graphs but may be overkill for single-statement graphs with ~20–50 triples. More valuable in the multi-statement fusion scenario (EXTENSIONS.md §5). Last released Feb 2024 — verify active maintenance.

### 2.5 RDF Foundation

#### rdflib (`rdflib` v7.6.0)
- **What it does**: Pure Python library for working with RDF — parsers/serializers for Turtle, JSON-LD, N-Triples, etc. SPARQL 1.1 implementation. Foundation for pySHACL and kglab.
- **Fit for our system**: Required bridge between Neo4j (property graph) and RDF-based validation tools. The schema already defines `NAMESPACES` for PROV-O, SOSA/SSN, and Schema.org — `rdflib` can materialise these.
- **Integration path**: Export Neo4j nodes/relationships to `rdflib.Graph` → use as input for pySHACL and kglab.
- **License**: BSD 3-Clause
- **Verdict**: **Required dependency** for the SHACL validation path.

---

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Quality Probe Pipeline                           │
│                                                                     │
│  Neo4j Graph ──┬──► Neo4j GDS Algorithms ──► Structural Report     │
│                │      • WCC (connectivity)                          │
│                │      • Degree centrality                           │
│                │      • Louvain communities                         │
│                │                                                    │
│                ├──► APOC meta.schema() ──► Schema Drift Report      │
│                │      • Expected vs actual node/rel types           │
│                │      • Missing node types                          │
│                │                                                    │
│                ├──► rdflib Export ──► pySHACL Validation             │
│                │      • Cardinality constraints                     │
│                │      • Value-type constraints                      │
│                │      • Closed-shape validation                     │
│                │      • Custom SPARQL constraints                   │
│                │             │                                      │
│                │             ▼                                      │
│                │      SHACL ValidationReport ──► Constraint Report  │
│                │                                                    │
│                ├──► Linearise Graph ──► DeepEval                    │
│                │      • LLM coherence scoring (G-Eval)              │
│                │      • Extraction faithfulness (hallucination)      │
│                │      • Semantic completeness                       │
│                │             │                                      │
│                │             ▼                                      │
│                │      LLM Quality Scores ──► Semantic Report        │
│                │                                                    │
│                └──► Cypher Consistency Checks ──► Consistency Report │
│                       • Temporal acyclicity (PRECEDED chains)       │
│                       • Role exclusivity constraints                │
│                       • Source-grounding audit (orphan triples)      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               QualityReport (unified output)                 │   │
│  │  • structural_score    (0.0–1.0)                             │   │
│  │  • schema_score        (0.0–1.0)                             │   │
│  │  • constraint_score    (0.0–1.0)                             │   │
│  │  • consistency_score   (0.0–1.0)                             │   │
│  │  • coherence_score     (0.0–1.0)                             │   │
│  │  • faithfulness_score  (0.0–1.0)                             │   │
│  │  • overall_score       (weighted average)                    │   │
│  │  • violations: list[Violation]                               │   │
│  │  • recommendations: list[str]                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Recommended Tool Stack

| Layer | Tool | Purpose | New Dependency? |
|-------|------|---------|-----------------|
| **Structural** | Neo4j GDS | WCC, degree centrality, community detection | No (Neo4j plugin) |
| **Schema drift** | APOC | `meta.schema()` vs expected schema comparison | No (Neo4j plugin) |
| **Constraint validation** | pySHACL + rdflib | SHACL shape validation after RDF export | Yes: `pyshacl`, `rdflib` |
| **Orchestration** | kglab | Wraps rdflib + pySHACL + NetworkX | Yes: `kglab` |
| **Coherence/faithfulness** | DeepEval | LLM-as-judge scoring (via Ollama) | Yes: `deepeval` |
| **Consistency** | Custom Cypher | Temporal acyclicity, role constraints, provenance audit | No |
| **Link prediction** | AmpliGraph | Missing-triple prediction (Phase 2 — multi-statement) | Deferred |

### Minimal viable stack (Phase 1)

For a first implementation, only **two new dependencies** are needed:

```
pip install rdflib pyshacl
```

Everything else uses the existing Neo4j driver and Ollama LLM. DeepEval can be added in Phase 2 once the structural/constraint probes are working.

---

## 5. Implementation Plan

### Phase 1: Structural & Constraint Probes

**New file: `quality.py`**

```python
@dataclass
class Violation:
    dimension: str          # "schema" | "consistency" | "structural" | ...
    severity: str           # "error" | "warning" | "info"
    message: str
    node_label: str | None
    node_id: str | None

@dataclass
class QualityReport:
    structural_score: float
    schema_score: float
    constraint_score: float
    consistency_score: float
    coherence_score: float
    faithfulness_score: float
    overall_score: float
    violations: list[Violation]
    recommendations: list[str]
    timestamp: str
```

#### 5.1 Schema Population Completeness (Cypher)
Check that every expected node type from `NODE_TYPES` has at least one instance:

```cypher
// For each label in NODE_TYPES, count instances
MATCH (n:{label}) RETURN count(n) AS count
```

Compute: `schema_score = (# populated types) / (# expected types)`

#### 5.2 Structural Connectivity (Neo4j GDS)
```cypher
// Weakly Connected Components
CALL gds.wcc.stream('event-graph')
YIELD nodeId, componentId
RETURN componentId, count(*) AS size
ORDER BY size DESC
```

- **1 component** → fully connected → score 1.0
- **>1 components** → fragmented → score = (largest component size) / (total nodes)
- Flag isolated nodes (degree 0) as warnings

#### 5.3 Temporal Consistency (Custom Cypher)
```cypher
// Detect cycles in PRECEDED chains
MATCH path = (e1:Event)-[:PRECEDED*]->(e1)
RETURN path LIMIT 1
```

```cypher
// Check monotonic time ordering along PRECEDED chains
MATCH (e1:Event)-[:PRECEDED]->(e2:Event),
      (e1)-[:OCCURRED_AT_TIME]->(t1:Time),
      (e2)-[:OCCURRED_AT_TIME]->(t2:Time)
WHERE t1.value > t2.value
RETURN e1.description, t1.value, e2.description, t2.value
```

#### 5.4 Source Grounding Audit (Custom Cypher)
```cypher
// Find nodes without source provenance
MATCH (n)
WHERE n.source IS NULL
  AND NOT n:Observation
RETURN labels(n)[0] AS type, n.description AS desc
```

Score: `accuracy_score = 1.0 - (orphan_count / total_count)`

#### 5.5 SHACL Constraint Validation (Phase 1b)

Define SHACL shapes for the domain, for example:

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix evt: <http://example.org/event-twin#> .
@prefix schema: <https://schema.org/> .

evt:EventShape a sh:NodeShape ;
    sh:targetClass evt:Event ;
    sh:property [
        sh:path evt:description ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
    ] ;
    sh:property [
        sh:path evt:occurred_at ;
        sh:minCount 1 ;
        sh:message "Every Event must have at least one OCCURRED_AT relationship" ;
    ] ;
    sh:property [
        sh:path evt:occurred_at_time ;
        sh:minCount 1 ;
        sh:message "Every Event must have a time anchor" ;
    ] .

evt:PersonShape a sh:NodeShape ;
    sh:targetClass evt:Person ;
    sh:property [
        sh:path evt:name_or_description ;
        sh:minCount 1 ;
    ] ;
    sh:property [
        sh:path evt:role ;
        sh:minCount 1 ;
        sh:in ("witness" "suspect" "victim" "bystander" "driver" "passenger") ;
    ] .
```

Validate via:
```python
from pyshacl import validate
conforms, report_graph, report_text = validate(data_graph, shacl_graph=shapes)
```

### Phase 2: Semantic Quality Probes (DeepEval)

#### 5.6 Coherence Scoring

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

coherence_metric = GEval(
    name="NarrativeCoherence",
    criteria=(
        "Given only the knowledge graph linearisation (the 'actual_output'), "
        "evaluate whether it represents a coherent, internally consistent "
        "narrative of a witnessed event. Score 1-5."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

test_case = LLMTestCase(
    input="Evaluate knowledge graph coherence",
    actual_output=linearise_graph(driver),  # existing function in schema.py
)
coherence_metric.measure(test_case)
```

#### 5.7 Extraction Faithfulness

```python
faithfulness_metric = GEval(
    name="ExtractionFaithfulness",
    criteria=(
        "Compare the source witness statement (input) against the extracted "
        "knowledge graph (actual_output). Are there facts in the graph that "
        "are NOT supported by the source text? Score 0 for hallucinated, "
        "1 for fully faithful."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.8,
)

test_case = LLMTestCase(
    input=original_statement_text,
    actual_output=linearise_graph(driver),
)
faithfulness_metric.measure(test_case)
```

### Phase 3: Predictive Quality (AmpliGraph — deferred to multi-statement)

Train KG embeddings to predict missing links and flag low-plausibility triples. Only valuable once graphs grow beyond single-statement size (~50+ triples).

---

## 6. Integration with Existing Pipeline

```
Current pipeline:
  extract → validate → load → gap_analysis → interview → query

Extended pipeline:
  extract → validate → load → QUALITY_PROBE → gap_analysis → interview → QUALITY_PROBE → query
```

The quality probe runs **after ingestion** and **after each interview round**, producing a `QualityReport` that:

1. Feeds improved recommendations into `prioritise_gaps()` — structural violations become high-priority gaps
2. Provides a quality dashboard in the query phase — users can ask "what is the quality of this graph?"
3. Gates the pipeline — if `overall_score < 0.4`, the system should warn that the graph is unreliable

---

## 7. Scoring Framework

| Dimension | Weight | Measurement Tool |
|-----------|--------|-----------------|
| Schema completeness | 0.15 | Cypher population check |
| Property completeness | 0.15 | Existing `run_schema_completeness()` |
| Structural connectivity | 0.15 | Neo4j GDS (WCC) |
| Constraint conformance | 0.15 | pySHACL validation |
| Temporal consistency | 0.10 | Custom Cypher |
| Source grounding | 0.10 | Custom Cypher provenance audit |
| Coherence | 0.10 | DeepEval G-Eval |
| Faithfulness | 0.10 | DeepEval G-Eval |

$$\text{overall\_score} = \sum_{i} w_i \cdot s_i$$

Where $w_i$ is the weight and $s_i \in [0, 1]$ is the normalised score for each dimension.

---

## 8. New Dependencies Summary

| Package | Version | Purpose | Phase |
|---------|---------|---------|-------|
| `rdflib` | ≥ 7.0 | RDF export from Neo4j property graph | 1b |
| `pyshacl` | ≥ 0.31 | SHACL constraint validation | 1b |
| `kglab` | ≥ 1.0 | Orchestration: wraps rdflib + pySHACL + NetworkX | 1b |
| `deepeval` | ≥ 3.9 | LLM-as-judge coherence/faithfulness | 2 |
| `ampligraph` | ≥ 2.1 | KG embeddings + link prediction | 3 (deferred) |

Phase 1 core (Cypher-only probes) requires **zero new dependencies**.

---

## 9. Research Basis

This proposal builds on the research already documented in [EXTENSIONS.md](EXTENSIONS.md) §3:

- **Xue & Zou (2022)** — five intrinsic KG quality dimensions → our six-dimension scoring framework
- **Issa et al. (2021)** — four completeness sub-dimensions → schema, property, population, interlinking checks
- **Zhang & Xiao (2024)** — requirements-driven assessment → domain-specific population rules (collision needs ≥ 2 participants)
- **Huaman (2022)** — GQM methodology → our Goal (reliable witness KG) → Questions (is it complete? consistent? coherent?) → Metrics (the scoring framework above)

Additional tool-specific references:
- **SHACL W3C Recommendation (2017)** — formal constraint language for RDF graphs
- **Prud'hommeaux et al. (2014)** — Shape Expressions as an alternative to SHACL for more concise constraints
- **Neo4j GDS documentation** — graph algorithms for structural quality metrics
