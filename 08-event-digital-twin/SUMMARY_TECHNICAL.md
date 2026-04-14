# Event Digital Twins — Technical State of the Art

## 1. Digital Twin Maturity Model

The literature identifies four maturity levels for digital representations (Kritzinger et al., 2018; Wright & Davidson, 2020):

| Level | Name | Data Flow | Characteristics |
|-------|------|-----------|----------------|
| 1 | **Digital Model** | Manual | Static representation, no live data connection |
| 2 | **Digital Shadow** | Physical → Digital (one-way) | Automated data ingestion, but no feedback to physical system |
| 3 | **Digital Twin** | Bidirectional | Two-way synchronisation between physical and digital; simulation and what-if capability |
| 4 | **Digital Twin Aggregate** | Many instances → aggregate | Fleet-level analysis across multiple twin instances |

### Where this implementation sits

The initial Chapter 08 implementation was a **Digital Shadow** — data flowed from statement to graph, but with no state evolution, no bidirectional feedback, and the graph was destructively overwritten on each run.

The extended implementation achieves **Digital Twin** status:

- **Additive ingestion**: each source enriches the graph rather than replacing it
- **Bidirectional feedback**: the interview loop analyses the graph for gaps and feeds new information back
- **What-if simulation**: scenario branching allows exploration of hypothetical modifications
- **State versioning**: `GraphVersion` nodes track the evolution of the graph over time
- **Multi-source fusion**: cross-source entity resolution, corroboration, and contradiction detection

## 2. Knowledge Graph Construction from Unstructured Text

### 2.1 Extraction Architecture

The pipeline follows the standard KG construction pattern validated in recent literature:

```
Source Document
    │
    ▼
┌──────────────────┐
│ Segment (sentence│    Split text into segments for context
│ splitting)       │    management
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Schema-guided    │    Ontology-derived prompt constrains
│ LLM extraction   │    extraction to valid entity types
└──────────────────┘    and relationship types
    │
    ▼
┌──────────────────┐
│ Coreference      │    Within-document entity resolution:
│ resolution       │    "the driver" = "he" = "a tall man"
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Cross-document   │    Across-source entity resolution:
│ entity resolution│    "tall man in dark jacket" (W1) =
└──────────────────┘    "suspect, about 6 foot" (W2)
    │
    ▼
┌──────────────────┐
│ Graph loading    │    MERGE into Neo4j with source
│ (additive)       │    tagging and confidence scores
└──────────────────┘
```

### 2.2 Related Work

**Ramadan et al. (2025)** — *"Reconstructing Judicial Digital Forensic Evidence Graphs from Legal Documents Using LLMs"* (IEEE COMPSAC 2025)
- Uses the KRYSTAL ontology for structuring evidence graphs
- LLM-driven extraction from legal documents — architecture directly parallels this system
- Validates that LLM-driven KG construction from legal text is a published, peer-reviewed approach

**Pandey, Brantingham & Uchida (2020)** — *"Building Knowledge Graphs of Homicide Investigation Chronologies"* (IEEE BigData 2020)
- Builds KGs from real homicide case chronologies with event-centric nodes
- Validates the event-centric design in our schema

**Spyropoulos et al. (2023)** — *"Interoperability-Enhanced Knowledge Management in Law Enforcement"* (Information, 14(11), 607)
- Full forensic ontology in OWL with DL reasoning
- Establishes the upper bound of what formal ontology can achieve in this domain
- Uses Protégé + Pellet for automated reasoning

### 2.3 Extraction Quality

Current SOTA for KG extraction from unstructured text:

| Technique | Status in this System |
|-----------|----------------------|
| Schema-guided prompting | ✓ Implemented — ontology drives the extraction prompt |
| Structured output validation | ✓ Improved — JSON parsing with retry and format enforcement |
| Within-document coreference | ✓ Implemented — LLM-based coreference resolution |
| Cross-document entity resolution | ✓ **New** — LLM-assisted with attribute/role matching |
| Iterative refinement | ✓ Implemented — interview loop fills gaps |
| Confidence scoring | ✓ **New** — numerical 0.0–1.0 with Bayesian updates |
| Hallucination detection | ✓ Implemented — faithfulness probes compare graph to source |

## 3. Provenance and Trust

### 3.1 W3C Standards

The system implements a layered provenance model:

| Layer | Standard | Implementation |
|-------|----------|---------------|
| **Provenance** | W3C PROV-O | `Observation` nodes track who said what, when |
| **Observation** | W3C SOSA/SSN | Witness-as-sensor model: statements are observations |
| **Event structure** | Schema.org / SEM / BFO-CCO | Pluggable ontology for event modelling |

### 3.2 PROV-O Coverage

| PROV-O Concept | Implementation |
|---------------|---------------|
| `prov:Entity` | Graph nodes with `source` and `source_type` properties |
| `prov:Activity` | `Observation` nodes with extraction timestamps |
| `prov:Agent` | `Person` nodes with witness/source roles |
| `prov:wasDerivedFrom` | `DERIVED_FROM` relationships from Events to Observations |
| `prov:wasGeneratedBy` | `extracted_at` timestamp + `GraphVersion` nodes |
| `prov:wasAttributedTo` | `source` property on every node |

### 3.3 Multi-Source Provenance

Every entity in the unified graph carries:

```
(Event {description: "collision"})
  .source = "king_street_collision"     ← which source
  .source_type = "statement"            ← what kind of source
  .confidence = 0.93                    ← numerical confidence
  .extracted_at = "2026-04-14T..."      ← when extracted
  .ontology_id = "schema-org-event-v1"  ← which ontology
```

Cross-source relationships:

```
(Entity A)-[:POSSIBLY_SAME_AS {confidence: 0.9}]->(Entity B)
(Observation A)-[:CORROBORATED_BY]->(Observation B)
(Entity A)-[:CONTRADICTS {field: "time", value_a: "14:15", value_b: "14:13"}]->(Entity B)
```

## 4. Probabilistic Reasoning

### 4.1 Confidence Model

The system uses a numerical confidence model with Bayesian-inspired updates:

| Event | Confidence Change |
|-------|-------------------|
| Initial extraction | 0.8 (single source baseline) |
| Corroborated by second source | +30% of remaining distance to 1.0 |
| Contradicted by another source | ×0.7 decay |
| Resolved from source text (self-resolution) | maintained at extraction level |
| Inferred by interview | 0.6–0.7 (lower than direct extraction) |

### 4.2 Research Context

**Fenton, Neil & Yet (2020)** — *"Analyzing the Simonshaven Case Using Bayesian Networks"*
- Models a real murder case with Bayesian networks encoding witness testimony, motive, opportunity
- Demonstrates moving from "what facts do we have?" to "what do these facts imply?"

**Van Leeuwen et al. (2024)** — *"Building a Stronger Case"* (HHAI 2024)
- Scenario-based Bayesian networks combining evidence, legal elements, and competing scenarios
- Directly evaluates witness testimony within the probabilistic framework

**Xu & Vinci (2024)** — Chain Event Graphs for forensic evidence
- Tree-structured Bayesian models for asymmetric event progressions
- Natural fit for branching witness narratives

The current implementation uses a simplified confidence propagation model. A full Bayesian Network extension (using `pgmpy`) is outlined in EXTENSIONS.md for future work.

## 5. Quality Assessment

### 5.1 Quality Dimensions

The KG quality literature (Xue & Zou, 2022) identifies five intrinsic quality dimensions. The system covers:

| Dimension | Approach | Probe |
|-----------|----------|-------|
| **Completeness** | Schema rules + population rules + domain expectations | `probe_schema_population` + `probe_population_completeness` |
| **Accuracy** | LLM faithfulness scoring (graph vs. source text) | `probe_faithfulness` |
| **Consistency** | Temporal cycle detection, cross-source contradiction counting | `probe_temporal_consistency` + `probe_cross_source_consistency` |
| **Coherence** | LLM-as-judge narrative reconstruction | `probe_narrative_reconstruction` |
| **Timeliness** | `extracted_at` timestamps + `GraphVersion` nodes | Tracked but not scored |

### 5.2 Cross-Source Quality

The multi-source fusion introduces new quality signals:

- **Corroboration rate**: what fraction of facts are reported by ≥2 sources?
- **Contradiction density**: how many cross-source conflicts per entity?
- **Coverage overlap**: how much do different sources cover the same events vs. adding new information?

## 6. Temporal Reasoning

### 6.1 Approach

The system extracts temporal information from multiple representations:

1. **Explicit timestamps**: "14:13:45", "2:15 PM"
2. **Relative temporal markers**: "then", "about ten minutes later"
3. **Temporal relationships**: `AT_TIME`, `OCCURRED_AT_TIME`, `PRECEDED_BY`

A timeline is constructed by:
1. Parsing all time values to minutes-since-midnight
2. Sorting events by parsed time
3. Materialising `PRECEDED_BY` relationships between consecutive events
4. Checking for temporal contradictions (same event, different reported times)

### 6.2 Allen's Interval Algebra

Allen (1983) defines 13 temporal relations between events. The current system uses:

| Relation | Implementation |
|----------|---------------|
| `BEFORE` / `AFTER` | Via `PRECEDED_BY` chain |
| `MEETS` | Adjacent events with no gap |
| `DURING` | Not yet implemented |
| `OVERLAPS` | Not yet implemented |
| `EQUALS` | Same-event detection via entity resolution |

Full Allen's algebra implementation is a research extension requiring interval-based temporal representation rather than point-based.

## 7. Multi-Source Fusion Architecture

```
Statement A        CCTV Log         Paramedic Report
    │                  │                   │
    ▼                  ▼                   ▼
┌─────────┐      ┌─────────┐        ┌─────────┐
│ Ingest  │      │ Ingest  │        │ Ingest  │
│ (additive)     │ (additive)       │ (additive)
└─────────┘      └─────────┘        └─────────┘
    │                  │                   │
    └──────────┬───────┘───────────────────┘
               ▼
    ┌────────────────────┐
    │ Cross-Source        │   LLM-assisted entity matching
    │ Entity Resolution   │   with attribute/role fallback
    └────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 POSSIBLY   CORROBO-   CONTRA-
 _SAME_AS   RATED_BY   DICTS
    │          │          │
    └──────────┼──────────┘
               ▼
    ┌────────────────────┐
    │ Confidence Update   │   Bayesian-style boost/decay
    └────────────────────┘
               │
               ▼
        Unified Event Graph
        (with per-source provenance)
```

### 7.1 Entity Resolution Strategies

| Strategy | When Used |
|----------|-----------|
| **Attribute matching** | Compare properties (height, clothing colour, vehicle type) |
| **Role matching** | Same role relative to same event → likely same entity |
| **LLM-assisted** | Present candidate pairs with context from both sources |
| **Confidence threshold** | Only create `POSSIBLY_SAME_AS` at ≥0.7 confidence |

### 7.2 Contradiction Handling

Contradictions are **never silently resolved**. Both claims are preserved with full provenance:

```
(Time {value: "14:15", source: "witness_1"})
(Time {value: "14:13:45", source: "cctv_log"})
(Time "14:15")-[:CONTRADICTS {field: "timestamp"}]->(Time "14:13:45")
```

The system flags contradictions but leaves resolution to the human investigator. The confidence model reduces confidence on contradicted facts.

## 8. What-If Simulation

The what-if capability is what elevates the system from a "digital shadow" to a true "digital twin." It supports:

| Operation | Description |
|-----------|-------------|
| **Remove source** | "What if Witness 2 is unreliable?" → remove all their contributions |
| **Modify entity** | "What if the driver was female?" → change a property and check consistency |
| **Scenario comparison** | Diff two graph states: what's lost, what's gained, what changes |
| **Snapshot/restore** | Full graph state capture and restoration for safe experimentation |

The implementation:
1. Takes a baseline snapshot of the entire graph
2. Applies the hypothetical operation
3. Compares before/after states
4. Restores the baseline

This ensures the evidence graph is never corrupted by speculative exploration.

## 9. Export and Interoperability

| Format | Standard | Use Case |
|--------|----------|----------|
| **RDF/Turtle** | W3C Semantic Web | Linked data interoperability, SPARQL querying |
| **JSON-LD** | Schema.org + PROV-O | Web-compatible structured data exchange |
| **Cypher dump** | Neo4j | Snapshot/restore, version control |
| **DOT/Graphviz** | Graphviz | Static graph diagrams for reports |
| **Interactive HTML** | vis.js | Browser-based interactive exploration |

The RDF export maps:
- Neo4j labels → RDF classes (Schema.org, SOSA, PROV-O vocabularies)
- Neo4j relationships → RDF predicates
- Properties → RDF literals with appropriate XSD types

## 10. References

### Digital Twin Foundations
- Grieves, M. & Vickers, J. (2017). *Digital Twin: Mitigating Unpredictable, Undesirable Emergent Behavior in Complex Systems.* Springer.
- Kritzinger, W. et al. (2018). Digital Twin in manufacturing: A categorical literature review. *IFAC-PapersOnLine*, 51(11), 1016–1022.
- Wright, L. & Davidson, S. (2020). How to tell the difference between a model and a digital twin. *Advanced Modeling and Simulation in Engineering Sciences*, 7(1), 13.
- US DoD Digital Engineering Strategy (2018). Digital Engineering Working Group.

### Forensic Knowledge Graphs
- Spyropoulos, A.Z. et al. (2023). Interoperability-Enhanced Knowledge Management in Law Enforcement. *Information*, 14(11), 607.
- Pandey, R. et al. (2020). Building Knowledge Graphs of Homicide Investigation Chronologies. *IEEE BigData 2020*.
- Ramadan, O. et al. (2025). Reconstructing Judicial Digital Forensic Evidence Graphs from Legal Documents Using LLMs. *IEEE COMPSAC 2025*.

### Bayesian Evidence Reasoning
- Fenton, N., Neil, M. & Yet, B. (2020). Analyzing the Simonshaven Case Using Bayesian Networks. *Topics in Cognitive Science*.
- Van Leeuwen, L. et al. (2024). Building a Stronger Case. *HHAI 2024*.
- Xu, X. & Vinci, G. (2024). Forensic Science and How Statistics Can Help It. *WIREs Computational Statistics*.

### Knowledge Graph Quality
- Xue, B. & Zou, L. (2022). Knowledge Graph Quality Management: A Comprehensive Survey. *IEEE TKDE*.
- Issa, S. et al. (2021). Knowledge Graph Completeness: A Systematic Literature Review. *IEEE Access*.

### Temporal Reasoning
- Allen, J.F. (1983). Maintaining knowledge about temporal intervals. *Communications of the ACM*, 26(11), 832–843.
