# Chapter 08 — Event Digital Twin: Extension Plan

## Audit of Current State

### What exists

| File | Purpose | Status |
|------|---------|--------|
| `demo.py` | 4-phase orchestrator (ingest → interview → quality → query) | Works but single-statement only |
| `ingest.py` | LangGraph: text → extraction → coreference → Neo4j | **Destructive**: `MATCH (n) DETACH DELETE n` before every load |
| `interview.py` | 3-level gap analysis → self-resolution → human-in-the-loop | Functional but gaps remain unaddressed at scale |
| `query.py` | Grounded Q&A with provenance citations | Functional |
| `schema.py` | Bridge to 06-ontologies, gap analysis, provenance materialisation | Functional |
| `EXTENSIONS.md` | Research literature review + 5 extension tracks | Research only, nothing implemented |

### Critical problems

1. **The graph is wiped on every ingest.** `ingest.py` line 295: `session.run("MATCH (n) DETACH DELETE n")`. This makes multi-source fusion impossible and contradicts the README's claim of building a "provably complete graph from those sources."

2. **Single source only.** The pipeline processes exactly one `.txt` file. There is no mechanism for combining statements, CCTV logs, forensic reports, or any second source.

3. **No temporal reasoning.** Events have no ordering beyond what the LLM happens to extract. There is no timeline, no Allen's interval algebra, no temporal constraint propagation.

4. **No contradiction detection.** When two sources disagree, there is no mechanism to detect, record, or resolve the conflict.

5. **No confidence model.** Confidence is a string (`"high"`, `"medium"`, `"low"`) with no propagation, aggregation, or update logic.

6. **Hallucination assessment is partially stubbed.** `demo.py`'s `assess_hallucination()` delegates to `llm_probes.probe_faithfulness` but the "full per-triple scoring and cross-ontology comparison" described in the README are marked TODO.

7. **No graph export.** The graph lives only in Neo4j. No RDF/JSON-LD serialisation, no snapshot/restore, no portable output.

8. **No visualisation.** No way to see the graph structure, timeline, or provenance chain visually.

9. **Not a digital twin.** The system is a one-shot extraction pipeline. A digital twin requires continuous synchronisation with its physical counterpart, state evolution over time, and the ability to simulate what-if scenarios. None of these are present.

---

## Part 1 — Corporate Deep Dive (Non-Technical)

> *What is a digital twin, why does it matter, and what does this chapter demonstrate?*

### SUMMARY_CORPORATE.md — Proposed Structure

#### 1.1 What is a digital twin?

A digital twin is a live digital replica of a real-world system that stays continuously synchronised with that system through data feeds. The concept was formalised by Michael Grieves (University of Michigan, 2002) and named by NASA engineer John Vickers (2010). It consists of three parts:

- **The physical system** — the thing being modelled (a jet engine, a building, *an incident under investigation*)
- **The digital representation** — a structured model that mirrors the physical system's state
- **The communication channel** ("digital thread") — the data flow that keeps the two synchronised

The key distinction from a simulation or a static model: a digital twin is *live*. It reflects the current state of its physical counterpart and evolves as new data arrives.

#### 1.2 What is an *event* digital twin?

Traditional digital twins model physical objects (turbines, aircraft, buildings). An event digital twin models *something that happened* — a collision, a crime, a disaster. The "physical system" is the event itself, reconstructed from evidence.

| Traditional DT | Event DT |
|---------------|----------|
| Sensors on a turbine | Witness statements, CCTV, forensic reports |
| Real-time telemetry | Batch evidence ingestion + interview |
| Predict mechanical failure | Reconstruct what happened, detect gaps |
| One physical object | One event, many perspectives |

The core challenge is different: instead of continuous sensor data, an event DT must combine *incomplete, contradictory, human-generated accounts* into a single coherent model.

#### 1.3 Why knowledge graphs?

An event does not have a single "correct" representation the way a turbine's RPM has a single value. An event involves:
- Multiple actors with different roles
- Temporal sequences that may conflict across accounts
- Spatial relationships ("at the junction of King Street and Queen's Road")
- Provenance chains (who said what, and how reliable are they?)

A knowledge graph captures all of these naturally as nodes and edges, with properties for confidence, provenance, and temporal ordering. It also enables formal reasoning: "if A preceded B, and B preceded C, then A preceded C."

#### 1.4 The investigation metaphor

The chapter models the pipeline an investigator follows:

1. **Collect** — take the first witness statement
2. **Extract** — identify who, what, when, where from each account
3. **Corroborate** — compare accounts, flag agreements and disagreements
4. **Fill gaps** — interview witnesses about missing information
5. **Assess quality** — how complete and consistent is the reconstruction?
6. **Query** — answer specific investigative questions grounded in evidence

#### 1.5 Business value

| Capability | Value |
|-----------|-------|
| Multi-source fusion | Single source of truth from many accounts |
| Provenance tracking | Every fact traceable to its source |
| Gap analysis | Systematic identification of what's still unknown |
| Contradiction detection | Automatic flagging of conflicting accounts |
| Quality scoring | Objective measure of reconstruction completeness |
| Grounded Q&A | Answers cite evidence, not LLM knowledge |

#### 1.6 Where industry is going

- **Azure Digital Twins** (Microsoft): graph-based modelling with DTDL (Digital Twins Definition Language), live execution environments, IoT integration
- **IBM Maximo**: asset lifecycle management with digital twin technology
- **NVIDIA Omniverse**: physics-based simulation of digital twin environments
- The digital twin market is projected to grow from $24.5B (2025) to $259.3B (2032) (Fortune Business Insights)
- The US DoD defines a digital twin as "an integrated multiphysics, multiscale, probabilistic simulation of an as-built system, enabled by a Digital Thread"

The event digital twin concept extends this paradigm from physical assets to investigative and forensic domains.

---

## Part 2 — Technical Deep Dive (State of the Art)

> *What does the current research say, and how does this implementation compare?*

### SUMMARY_TECHNICAL.md — Proposed Structure

#### 2.1 Digital twin architecture taxonomy

The literature identifies four maturity levels (Kritzinger et al., 2018):

| Level | Name | Data flow | This chapter |
|-------|------|-----------|-------------|
| 1 | Digital Model | Manual | ✗ |
| 2 | Digital Shadow | Physical → Digital (one-way) | **Current state** — ingest extracts, but nothing flows back |
| 3 | Digital Twin | Bidirectional | **Target** — interview loop feeds back, query informs investigation |
| 4 | Digital Twin Aggregate | Many instances → aggregate insights | **Future** — multi-case pattern analysis |

The current implementation is a **Digital Shadow** at best: data flows from statement to graph, but there is no continuous synchronisation or bidirectional feedback with the physical world of the investigation.

#### 2.2 Knowledge graph construction from unstructured text

The pipeline follows the standard KG construction pattern validated in recent literature:

- **Ramadan et al. (2025)**: LLM-driven extraction from legal documents into the KRYSTAL ontology — directly mirrors this chapter's approach
- **Pandey et al. (2020)**: KGs of homicide investigation chronologies with event-centric nodes — validates the schema design
- **Spyropoulos et al. (2023)**: Full forensic ontology in OWL with DL reasoning — establishes the upper bound of what formal ontology can achieve

Current SOTA augments LLM extraction with:
1. **Schema-guided prompting** (implemented ✓)
2. **Structured output validation** (partially: JSON parsing with retry, but no Pydantic/schema enforcement)
3. **Coreference resolution** (implemented ✓, but within single documents only)
4. **Cross-document entity resolution** (not implemented ✗)
5. **Iterative refinement** (implemented ✓ via interview loop)

#### 2.3 Provenance and trust

The W3C PROV-O ontology provides the standard for provenance tracking. The current implementation uses a simplified version:

| PROV-O concept | Current implementation | Gap |
|---------------|----------------------|-----|
| `prov:Entity` | Graph nodes with `source` property | ✓ |
| `prov:Activity` | `Observation` nodes | ✓ |
| `prov:Agent` | `Person` nodes with witness role | ✓ |
| `prov:wasDerivedFrom` | Not implemented | ✗ — no derivation chains between graph revisions |
| `prov:wasInformedBy` | Not implemented | ✗ — no activity-to-activity dependency |
| `prov:wasGeneratedBy` | Partial — `extracted_at` timestamp | ✗ — no link to the extraction activity itself |

#### 2.4 Probabilistic reasoning over evidence

Three approaches from the literature:

1. **Bayesian Networks** (Fenton et al., 2020; Van Leeuwen et al., 2024): model competing hypotheses with posterior probabilities. The Simonshaven murder case study demonstrates encoding witness testimony, motive, opportunity, and physical evidence as BN nodes.

2. **Chain Event Graphs** (Xu & Vinci, 2024): tree-structured Bayesian models that represent asymmetric event progressions — natural fit for branching witness narratives.

3. **Perpetrator Knowledge Analysis** (Jellema, 2024): Bayesian framework to detect whether a witness's testimony contains information only someone present at the scene could know.

None of these are implemented. The current confidence model is categorical strings with no mathematical basis.

#### 2.5 Quality assessment

The current quality assessment imports from Chapter 07 and covers:
- Schema population (Cypher probes)
- Structural connectivity
- Consistency (basic)
- Source grounding
- LLM coherence and faithfulness
- SHACL validation

The KG quality literature (Xue & Zou, 2022) identifies five dimensions:

| Dimension | Covered? | Current approach |
|-----------|----------|-----------------|
| **Completeness** | Partial | Schema completeness rules, not population or interlinking completeness |
| **Accuracy** | Partial | Faithfulness probe (LLM-based) |
| **Consistency** | Partial | Basic Cypher checks, no temporal logic validation |
| **Timeliness** | No | `extracted_at` timestamp exists but isn't used for staleness detection |
| **Accessibility** | No | No export formats, no API |

#### 2.6 Multi-source fusion (SOTA)

Cross-document entity resolution and evidence fusion are active research areas:

- **Attribute-based matching**: compare properties (height, clothing, vehicle colour) across statements
- **Role-based matching**: same role relative to same event suggests same entity
- **LLM-assisted matching**: prompt the LLM with candidate pairs and context from both sources
- **Confidence-weighted linking**: `POSSIBLY_SAME_AS` relationships with confidence scores rather than forced merges

The current system cannot do any of this because the graph is wiped before each ingest.

#### 2.7 Temporal reasoning

Allen's interval algebra defines 13 temporal relations between events (before, after, during, overlaps, meets, starts, finishes, equals, and their inverses). These enable:

- Timeline construction from partial temporal information
- Consistency checking (A cannot be both before and after B)
- Temporal constraint propagation (if A before B and B during C, then A before C)

The current system stores Time nodes with string values but performs no temporal reasoning.

#### 2.8 What makes this a "twin" vs. a "model"

To merit the term "digital twin," the system needs:

| Property | Digital model | Digital shadow | Digital twin |
|----------|-------------|----------------|-------------|
| Static data load | ✓ | ✓ | ✓ |
| Live data feed | ✗ | ✓ | ✓ |
| Bidirectional sync | ✗ | ✗ | ✓ |
| What-if simulation | ✗ | ✗ | ✓ |
| State versioning | ✗ | ✗ | ✓ |

**Minimum viable twin** requires: (1) additive ingestion (no wipe), (2) state versioning so you can ask "what did the graph look like after statement 1 vs. after statement 3?", (3) what-if queries ("if we remove Witness 2's testimony, what changes?"), (4) continuous refinement as new evidence arrives.

---

## Part 3 — Practical Implementation Plan

> *What exactly do we build, in what order, and how?*

### Phase 0: Fix the Foundations (Prerequisite)

**Problem**: The graph is destroyed on every ingest. Nothing else works until this is fixed.

| Task | File | Change |
|------|------|--------|
| 0.1 Remove `MATCH (n) DETACH DELETE n` | `ingest.py` | Replace with additive MERGE logic (already partially there for individual nodes) |
| 0.2 Add source tagging | `ingest.py` | Every node/relationship gets `source_id` property linking to the specific source document |
| 0.3 Add graph versioning | `schema.py` | `GraphVersion` node tracking: version number, timestamp, source file, ontology used |
| 0.4 Add `--clear` flag | `demo.py` | Explicit flag to wipe the graph, rather than wiping by default |
| 0.5 Create test fixtures | `statements/` | Add 2-3 additional witness statements for the same incident, from different perspectives |

### Phase 1: Multi-Source Fusion

**Goal**: Ingest multiple statements into the same graph with cross-statement entity resolution.

| Task | File | What |
|------|------|------|
| 1.1 Source-aware ingestion | `ingest.py` | Accept `source_id` parameter, tag all nodes, skip `DETACH DELETE` |
| 1.2 Cross-statement entity resolution | `fusion.py` (new) | LLM-assisted entity matching across statements + attribute/role matching fallback |
| 1.3 Corroboration detection | `fusion.py` | When two witnesses report the same fact → `CORROBORATED_BY` relationship, confidence boost |
| 1.4 Contradiction detection | `fusion.py` | When two witnesses disagree → `CONTRADICTS` relationship, both facts preserved with provenance |
| 1.5 Fusion orchestration | `demo.py` | Accept multiple statement files: `python demo.py statements/*.txt` |
| 1.6 Provenance-filtered queries | `query.py` | "What did Witness 2 say?" / "Which facts are corroborated?" |

**New schema additions**:
```python
"CORROBORATED_BY": RelDef(...)  # Observation ↔ Observation
"CONTRADICTS":     RelDef(...)  # Observation ↔ Observation  
"POSSIBLY_SAME_AS": RelDef(...) # Entity ↔ Entity (cross-statement)
```

**New file: `statements/queen_road_witness.txt`** — Second witness with overlapping but slightly different account.
**New file: `statements/cctv_log.txt`** — Simulated CCTV evidence log with timestamps.

### Phase 2: Temporal Reasoning

**Goal**: Construct and validate event timelines.

| Task | File | What |
|------|------|------|
| 2.1 Temporal extraction | `ingest.py` | Parse times into ISO 8601; extract relative temporal markers ("before", "after", "then") |
| 2.2 Allen's relations | `temporal.py` (new) | Encode `BEFORE`, `AFTER`, `DURING`, `OVERLAPS`, `MEETS` as relationships between Event nodes |
| 2.3 Timeline construction | `temporal.py` | Build ordered timeline from temporal constraints; detect and flag inconsistencies |
| 2.4 Temporal consistency probe | `quality_ext.py` (new) | Check for cycles in PRECEDED_BY chains; validate timestamp monotonicity |
| 2.5 Timeline visualisation | `visualise.py` (new) | ASCII or HTML timeline of events with confidence bands |

### Phase 3: Confidence & Uncertainty Model

**Goal**: Replace string-based confidence with numerical scores that propagate and update.

| Task | File | What |
|------|------|------|
| 3.1 Numerical confidence | `schema.py` | Replace `"high"/"medium"/"low"` with `float` confidence scores `[0.0, 1.0]` |
| 3.2 Source reliability weighting | `ingest.py` | Weight extraction confidence by source type: direct witness > hearsay > CCTV timestamps (high precision) |
| 3.3 Corroboration update | `fusion.py` | When fact is corroborated, boost confidence via Bayesian update |
| 3.4 Contradiction handling | `fusion.py` | When facts contradict, flag both, assign split posteriors |
| 3.5 Confidence propagation | `schema.py` | Inferred facts inherit minimum confidence of their supporting facts |
| 3.6 Uncertainty in queries | `query.py` | Answers include confidence levels; low-confidence facts flagged |

### Phase 4: Graph Export & Serialisation

**Goal**: Make the graph portable and interoperable.

| Task | File | What |
|------|------|------|
| 4.1 RDF/Turtle export | `export.py` (new) | Export Neo4j graph to RDF using `rdflib`, mapping to PROV-O + domain ontology |
| 4.2 JSON-LD export | `export.py` | Schema.org-compatible JSON-LD output |
| 4.3 Graph snapshot/restore | `export.py` | Cypher dump for graph versioning; restore to a previous version |
| 4.4 Graphviz/DOT export | `export.py` | Static graph visualisation via Graphviz |
| 4.5 Interactive visualisation | `visualise.py` (new) | `pyvis` or `neovis.js`-based interactive graph exploration in browser |

### Phase 5: What-If Simulation (True Twin Capability)

**Goal**: Enable counterfactual reasoning over the event graph.

| Task | File | What |
|------|------|------|
| 5.1 Scenario branching | `simulation.py` (new) | Clone the graph state, apply hypothetical modifications |
| 5.2 Evidence removal | `simulation.py` | "What if Witness 2 is unreliable?" → remove their contributions, recompute confidences |
| 5.3 Hypothesis testing | `simulation.py` | "What if the driver was a woman?" → modify entity, check consistency with remaining evidence |
| 5.4 Scenario comparison | `simulation.py` | Compare two scenario branches: what differs? what is invariant across scenarios? |
| 5.5 Scenario queries | `query.py` | Answer questions within a specific scenario context |

### Phase 6: Enhanced Quality & Completeness

**Goal**: Move from "did we populate the schema?" to "is the reconstruction faithful and complete?"

| Task | File | What |
|------|------|------|
| 6.1 Population completeness | `quality_ext.py` | Domain rules: a collision needs ≥2 participants, ≥1 location, ≥1 time |
| 6.2 Interlinking completeness | `quality_ext.py` | Every Person should be linked to ≥1 Event; every Event should have temporal context |
| 6.3 Full hallucination assessment | `demo.py` | Per-triple faithfulness scoring, aggregate metrics, breakdown by type |
| 6.4 Cross-source consistency | `quality_ext.py` | Quality score incorporates corroboration rate and contradiction resolution |
| 6.5 Narrative reconstruction test | `quality_ext.py` | LLM-as-judge: given only the graph, reconstruct the narrative; compare to sources |
| 6.6 Completeness score | `quality_ext.py` | Single 0.0–1.0 score combining all quality dimensions |

---

## Missing Tools & Dependencies

| Tool | Purpose | Currently used? |
|------|---------|----------------|
| `rdflib` | RDF/Turtle/JSON-LD export | No |
| `pyvis` | Interactive graph visualisation | No |
| `networkx` | Graph analytics (centrality, communities, path analysis) | No |
| `pgmpy` or `pomegranate` | Bayesian network inference for confidence scoring | No |
| `pydantic` | Structured output validation for LLM extraction | No — currently raw JSON parsing with regex fallback |
| `graphviz` | Static graph visualisation | No |
| `faster-whisper` | Speech-to-text for audio witness statements | No (future: Phase 7) |
| `pyannote.audio` | Speaker diarisation for multi-speaker recordings | No (future: Phase 7) |
| `langdetect` | Language detection for multilingual statements | No (future: Phase 7) |
| `rdflib-shacl` or `pyshacl` | SHACL validation directly in Python (vs current ch07 approach) | No |

### Structured Output Validation

The current extraction pipeline parses raw JSON from the LLM with regex fallback. This is fragile. Pydantic models for extraction output would:
- Enforce that extracted entities have required fields
- Validate relationship types against the ontology
- Catch type errors (e.g. a "time" field with non-temporal content)
- Provide clear error messages when extraction fails

```python
class ExtractedEntity(BaseModel):
    id: str
    label: Literal["Event", "Person", "Vehicle", "Location", "Time", ...]
    properties: dict[str, str | list[str]]

class ExtractedRelationship(BaseModel):
    from_id: str
    rel_type: str
    to_id: str

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
```

---

## Implementation Priority

```
Phase 0 ──► Phase 1 ──► Phase 2
  │              │           │
  │              ▼           ▼
  │         Phase 3     Phase 6
  │              │
  ▼              ▼
Phase 4     Phase 5
```

- **Phase 0** is prerequisite — nothing works without additive ingestion
- **Phase 1** (multi-source) and **Phase 2** (temporal) are the two highest-value extensions
- **Phase 3** (confidence) builds on Phase 1's corroboration/contradiction data
- **Phase 4** (export) is independent and can be done anytime
- **Phase 5** (simulation) requires Phases 0+1+3
- **Phase 6** (quality) benefits from all prior phases

---

## Proposed File Structure (Post-Extension)

```
08-event-digital-twin/
  demo.py                    # Orchestrator (extended for multi-source, scenarios)
  ingest.py                  # Additive ingestion with source tagging
  interview.py               # Interview loop (unchanged core, confidence-aware)
  query.py                   # Extended with scenario and provenance filters
  schema.py                  # Extended with new relationship types, confidence model
  fusion.py                  # NEW: cross-statement entity resolution, corroboration, contradiction
  temporal.py                # NEW: Allen's interval algebra, timeline construction
  simulation.py              # NEW: scenario branching, what-if queries
  export.py                  # NEW: RDF, JSON-LD, Cypher snapshot, Graphviz
  visualise.py               # NEW: interactive graph + timeline visualisation
  quality_ext.py             # NEW: extended quality probes (temporal, cross-source, population)
  inspect_state.py           # Existing
  SUMMARY_CORPORATE.md       # NEW: non-technical deep dive
  SUMMARY_TECHNICAL.md       # NEW: SOTA review
  SUMMARY_IMPLEMENTATION.md  # NEW: practical walkthrough
  EXTENSIONS.md              # Existing research notes (reference)
  PLAN.md                    # This document
  README.md                  # Updated to reflect new capabilities
  statements/
    king_street_collision.txt       # Existing: Witness 1
    queen_road_witness.txt          # NEW: Witness 2 (overlapping account)
    cctv_log.txt                    # NEW: Simulated CCTV evidence
    paramedic_report.txt            # NEW: Arriving paramedic's account
  transcript.txt
```

---

## Summary Deliverables

| Document | Audience | Content |
|----------|----------|---------|
| `SUMMARY_CORPORATE.md` | Non-technical stakeholders | What digital twins are, why event reconstruction matters, business value, industry context |
| `SUMMARY_TECHNICAL.md` | Technical audience, not implementing | SOTA review, architecture comparison, research positioning, maturity model |
| `SUMMARY_IMPLEMENTATION.md` | Developers following the course | Step-by-step walkthrough of the implemented solution, code explanations, how to run each phase |

---

## What "Done" Looks Like

A completed Chapter 08 should demonstrate:

1. **Ingest 3+ sources** into the same graph without data loss
2. **Resolve entities** across sources (the "tall man in a dark jacket" = "the suspect")
3. **Detect corroboration** (both witnesses saw a red car → confidence boost)
4. **Detect contradiction** (Witness 1 says 2:15 PM, CCTV says 2:22 PM → both preserved, flagged)
5. **Build a timeline** of events with temporal ordering and consistency checking
6. **Score confidence** numerically across the entire graph
7. **Assess quality** across all five dimensions (completeness, accuracy, consistency, timeliness, accessibility)
8. **Answer questions** with provenance citations and confidence levels
9. **Export** the graph to RDF/JSON-LD for interoperability
10. **Simulate** what-if scenarios ("what if we exclude Witness 2?")
11. **Visualise** the graph and timeline interactively

Running `python demo.py statements/*.txt` should execute the full pipeline end-to-end and produce a quality report, a visualisation, and an interactive query session over the fused, validated, temporally-ordered event graph.
