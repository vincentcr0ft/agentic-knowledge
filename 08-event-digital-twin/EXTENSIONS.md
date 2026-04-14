# Event Digital Twin — Extensions & Future Research

This document outlines five extension tracks for the event digital twin system. Each section covers the research basis, the connection to the current architecture, and practical implementation directions.

---

## 1. Alternative & Complementary Ontologies

### Current State

The system composites three W3C/community ontologies into a single witness-statement schema:

| Layer | Standard | Role |
|-------|----------|------|
| Provenance | PROV-O (W3C) | Who said what, when, derived from where |
| Observation | SOSA/SSN (W3C) | Witness-as-sensor, statement-as-observation |
| Event structure | Schema.org Event | Event-centric modelling with temporal/spatial properties |

### Research Basis

**Spyropoulos et al. (2023)** — *"Interoperability-Enhanced Knowledge Management in Law Enforcement: An Integrated Data-Driven Forensic Ontological Approach to Crime Scene Analysis"* (Information, 14(11), 607)

- Develops a full forensic ontology in OWL with classes for Crime Scene, Evidence (physical/digital), Witness, Suspect, and Investigative Actions
- Uses description logic (DL) for automated reasoning — e.g. linking fingerprints across scenes to infer a common perpetrator
- Includes a detailed class/subclass hierarchy and object/data property definitions
- Built and tested in Protégé with the Pellet reasoning engine
- **Relevance**: provides an independently developed forensic class hierarchy that can serve as a benchmark comparison for the current schema

**Pandey, Brantingham & Uchida (2020)** — *"Building Knowledge Graphs of Homicide Investigation Chronologies"* (IEEE BigData 2020, cited 11×)

- Builds KGs of real homicide case chronologies with suspect, witness, and victim nodes
- Ontology is event-chronology-focused: events are temporally ordered and linked to participants
- **Relevance**: validates the event-centric design choice in `schema.py` and provides a tested node/edge vocabulary for investigative chronologies

**Müller et al. (2022)** — *"Knowledge Engineering and Ontology for Crime Investigation"* (AIAI 2022, cited 8×)

- Defines event types as special crime types, actions in crime preparation/execution, and witness observations
- Represents facts as RDF triples in a knowledge graph
- **Relevance**: intermediate-level ontology between our lightweight Python schema and the full OWL approach of Spyropoulos et al.

**Ramadan et al. (2025)** — *"Reconstructing Judicial Digital Forensic Evidence Graphs from Legal Documents Using Large Language Models"* (IEEE COMPSAC 2025)

- Uses the KRYSTAL ontology as a standard for structuring evidence graphs
- Employs LLMs for extraction from legal documents — architecture closely mirrors the current ingest pipeline
- **Relevance**: demonstrates that LLM-driven KG construction from legal text is a validated approach, and identifies the KRYSTAL ontology as an alternative schema standard

### Candidate Ontology Extensions

| Ontology | What it adds | Integration point |
|----------|-------------|-------------------|
| **LKIF** (Legal Knowledge Interchange Format) | Legal standing of statements, admissibility, procedural context | New node types: `LegalDocument`, `ProceduralStep` |
| **UCO/CASE** (Unified Cyber Ontology) | Forensic evidence chain-of-custody, tamper detection | Strengthens PROV-O layer with formal custody chains |
| **OWL-Time** | Interval-based temporal reasoning, Allen's relations (during, overlaps, before) | Replaces or extends the current `Time` node with richer temporal semantics |
| **DOLCE+DnS Ultralite** | Foundational ontology for events, objects, and descriptions | Upper ontology layer linking domain-specific nodes to formal categories |

### Implementation Direction

1. Export the current Python schema to OWL/RDF using `rdflib`
2. Import into Protégé alongside the Spyropoulos forensic ontology
3. Run DL reasoner to identify class alignment and gaps
4. Extend `schema.py` with any validated additions (new node types, relationship types, or property constraints)

---

## 2. Bayesian Networks for Evidence Reasoning

### Current State

Gap analysis is deterministic: `run_schema_completeness()` checks which required properties and relationships are missing and returns a flat list of gaps. Prioritisation uses a simple rule-based ranking. There is no probabilistic reasoning about the quality or reliability of extracted facts.

### Research Basis

**Fenton, Neil & Yet (2020)** — *"Analyzing the Simonshaven Case Using Bayesian Networks"* (Topics in Cognitive Science, cited 30×)

- Models a real Dutch murder case with Bayesian networks (BNs)
- Encodes witness testimony, motive, opportunity, and physical evidence as nodes with conditional probability tables
- Demonstrates how to combine witness statements with forensic evidence probabilistically
- **Relevance**: provides a worked example of exactly the kind of reasoning the digital twin should support — moving from "what facts do we have?" to "what do these facts imply?"

**Van Leeuwen, Verbrugge et al. (2024)** — *"Building a Stronger Case: Combining Evidence and Law in Scenario-Based Bayesian Networks"* (HHAI 2024, Utrecht, cited 3×)

- Builds BN models that combine three aspects: evidence items, legal elements of the charge, and competing scenarios
- Directly evaluates witness testimony within the probabilistic framework
- **Relevance**: demonstrates scenario-based reasoning — the graph maintains multiple hypotheses (e.g. "suspect drove the car" vs "suspect was a passenger") with posterior probabilities

**Xu & Vinci (2024)** — *"Forensic Science and How Statistics Can Help It: Evidence, Likelihood Ratios, and Graphical Models"* (WIREs Computational Statistics, cited 3×)

- Surveys object-oriented Bayesian networks (OOBNs) and chain event graphs (CEGs) for forensic evidence
- CEGs are tree-structured Bayesian models that represent asymmetric event progressions — branching witness narratives where different paths have different evidence structures
- **Relevance**: CEGs map naturally onto witness statements where the narrative branches ("I saw him turn left" vs "she said he went straight")

**Jellema (2024)** — *"Perpetrator Knowledge: A Bayesian Account"* (Law, Probability and Risk, cited 1×)

- Formalises when witness testimony contains "perpetrator knowledge" — information that only someone present at the scene could know
- Uses Bayesian networks to distinguish genuine perpetrator knowledge from coincidence
- **Relevance**: directly applicable to the digital twin's witness interview phase — the system could flag statements that contain perpetrator knowledge signals

**Wang et al. (2020)** — *"A Knowledge-Based Reasoning Model for Crime Reconstruction and Investigation"* (Expert Systems with Applications, cited 10×)

- Proposes a case-type-based Bayesian model for crime reconstruction
- Combines knowledge graphs with Bayesian inference for evidence chain reasoning
- **Relevance**: shows how to bridge from a KG (our current approach) to a BN (the proposed extension)

### How a Bayesian Layer Would Work

```
Current pipeline:
  extract → validate → load → gap_analysis (deterministic) → interview

Extended pipeline:
  extract → validate → load → confidence_scoring (BN) → gap_analysis (probabilistic) → interview
```

| Capability | Current | With Bayesian extension |
|-----------|---------|----------------------|
| **Triple confidence** | Single `confidence` property (high/medium/low) | Posterior probability conditioned on source reliability, corroboration, internal consistency |
| **Gap prioritisation** | Rule-based ranking | Information-theoretic: ask the question that maximally reduces overall uncertainty |
| **Contradiction detection** | None | Conflicting triples detected via posterior collapse (P(A) and P(¬A) both supported) |
| **Multi-hypothesis** | Single graph | Multiple scenario subgraphs with comparative likelihoods |
| **Interview strategy** | Ask about all gaps | Ask about gaps where expected information gain is highest |

### Implementation Direction

1. Add a `pgmpy` or `pomegranate` dependency for BN inference
2. After graph loading, construct a BN where:
   - Nodes = extracted entities and relationships
   - Edges = evidential support relationships
   - CPTs = initialised from extraction confidence and source type
3. Compute posterior probabilities for each triple
4. Replace the gap prioritisation in `interview.py` with an information-gain ranking
5. Expose scenario comparison in the query phase

---

## 3. Knowledge Graph Quality Assessment

### Current State

Quality is assessed via `run_schema_completeness()` — a set of Cypher queries that check for missing required properties and relationships. This covers schema completeness but not accuracy, consistency, or coherence.

### Research Basis

**Xue & Zou (2022)** — *"Knowledge Graph Quality Management: A Comprehensive Survey"* (IEEE TKDE, cited 203×)

- Defines five intrinsic quality dimensions: accuracy, consistency, completeness, timeliness, and accessibility
- Provides specific metrics and measurement methods for each dimension
- **Relevance**: provides the canonical framework for KG quality assessment — the current system covers completeness; the other four dimensions are unaddressed

**Issa et al. (2021)** — *"Knowledge Graph Completeness: A Systematic Literature Review"* (IEEE Access, cited 126×)

- Categorises completeness into four sub-dimensions:
  - **Schema completeness**: are all relevant classes and properties defined?
  - **Property completeness**: for existing nodes, are all properties populated?
  - **Population completeness**: are all expected instances present?
  - **Interlinking completeness**: are all expected relationships instantiated?
- **Relevance**: the current `SCHEMA_COMPLETENESS_RULES` cover property and interlinking completeness but not schema or population completeness

**Zhang & Xiao (2024)** — *"How to Implement a Knowledge Graph Completeness Assessment with the Guidance of User Requirements"* (IEEE JSEE, cited 8×)

- Requirements-driven approach: define what "complete" means for your domain, then measure against it
- **Relevance**: the witness statement domain has clear expectations (e.g. a collision event should have ≥ 2 participants, a time, a location) that can be formalised as population completeness rules

**Huaman (2022)** — *"Steps to Knowledge Graphs Quality Assessment"* (arXiv:2208.07779, cited 15×)

- Uses the GQM (Goal-Question-Metric) approach to systematically derive quality metrics from quality goals
- **Relevance**: provides a methodology for designing the quality assessment framework rather than just listing metrics

### Proposed Quality Metrics

| Dimension | Metric | Measurement |
|-----------|--------|-------------|
| **Schema completeness** | Coverage of expected node types | % of `NODE_TYPES` that have ≥ 1 instance in the graph |
| **Property completeness** | Required properties populated | % of required props populated per node (already partially implemented) |
| **Population completeness** | Domain expectations met | Domain rules: e.g. a collision needs ≥ 2 participants, ≥ 1 location, ≥ 1 time |
| **Consistency** | Temporal logic | Check that PRECEDED chains are acyclic and times are monotonic |
| **Consistency** | Role constraints | A Person cannot be both witness and suspect for the same Event (domain rule) |
| **Accuracy** | Source grounding | Every triple must trace to a `source` provenance property; orphan triples flagged |
| **Accuracy** | Extraction verification | Re-extract from source text and compare — measure agreement rate |
| **Coherence** | Narrative reconstruction | LLM-as-judge: given only the graph, can a coherent narrative be generated? Score 1–5 |
| **Coherence** | Community coverage | After GraphRAG-style community detection, each community should have a coherent summary |
| **Timeliness** | Extraction staleness | Time between statement and extraction; flag if graph was built from outdated source |

### Implementation Direction

1. Extend `schema.py` with a `QualityReport` dataclass that captures all dimensions
2. Implement consistency checks as Cypher queries (temporal acyclicity, role constraints)
3. Implement accuracy checks by re-running extraction on source sentences and comparing
4. Add an LLM-based coherence scorer in `query.py`
5. Generate a quality report after each ingest/interview round

---

## 4. Speech-to-Text & Translation Layers

### Motivation

Witness statements are often given verbally and may be in a language other than English. The current system assumes clean English text input. Two preprocessing layers would significantly broaden applicability:

1. **Speech-to-text (STT)**: accept audio recordings of witness statements
2. **Translation**: accept statements in any language and translate to English before extraction

### Speech-to-Text Layer

#### Architecture

```
Audio input (.wav/.mp3/.m4a)
    │
    ▼
┌──────────────────┐
│  Speech-to-Text  │   Whisper (OpenAI) / whisper.cpp / faster-whisper
│  with diarisation │   + pyannote.audio for speaker diarisation
└──────────────────┘
    │
    ▼
Diarised transcript
    │
    ▼
┌──────────────────┐
│  Speaker label   │   Map speakers to roles: witness, interviewer, etc.
│  resolution      │
└──────────────────┘
    │
    ▼
Structured text segments (per-speaker)
    │
    ▼
Existing ingest pipeline
```

#### Key Components

| Component | Tool | Purpose |
|-----------|------|---------|
| **STT engine** | `faster-whisper` (CTranslate2 backend) | Transcription — runs locally, supports 99 languages |
| **Speaker diarisation** | `pyannote.audio` | Identifies who is speaking when — critical for multi-speaker recordings |
| **Speaker labelling** | LLM prompt | Given the diarised transcript, assign roles (witness, interviewer, other) based on content |
| **Timestamp alignment** | Whisper word-level timestamps | Align provenance to specific audio timestamps, not just text segments |

#### Provenance Extension

The `PROVENANCE_PROPS` in `schema.py` would gain:

```python
"audio_source":     "Filename of the source audio recording",
"audio_timestamp":  "Start timestamp in source audio (seconds)",
"speaker_id":       "Diarisation speaker label",
"transcription_confidence": "STT model confidence score",
```

#### Considerations

- **Accuracy**: Whisper large-v3 achieves ~5% WER on English; noisy field recordings will be worse. The system should expose transcription confidence so downstream extraction can weight accordingly
- **Diarisation errors**: speaker misattribution could cause provenance errors — a review step before ingestion is advisable
- **Real-time vs batch**: batch mode (upload a file) is simplest; real-time transcription during interview would require streaming STT

### Translation Layer

#### Architecture

```
Foreign-language text or transcript
    │
    ▼
┌──────────────────┐
│  Language         │   langdetect / fasttext language ID
│  detection        │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Translation      │   Helsinki-NLP/OPUS-MT (local) or LLM-based
└──────────────────┘
    │
    ▼
English text + original text (preserved for provenance)
    │
    ▼
Existing ingest pipeline
```

#### Key Components

| Component | Tool | Purpose |
|-----------|------|---------|
| **Language detection** | `langdetect` or `fasttext` | Auto-detect source language |
| **Translation** | Helsinki-NLP OPUS-MT models via `transformers` | Local, offline translation for ~100 language pairs |
| **Translation (alt)** | LLM-based (Ollama with multilingual model) | Higher quality for complex/legal text; slower |
| **Terminology alignment** | Domain glossary | Ensure legal/forensic terms are translated consistently (e.g. Dutch "getuige" → "witness", not "testifier") |

#### Provenance Extension

```python
"original_language":    "ISO 639-1 code of the source language",
"original_text":        "Untranslated source text (preserved verbatim)",
"translation_method":   "Model used for translation",
"translation_confidence": "Confidence or quality estimate",
```

#### Considerations

- **Legal terminology**: generic translation models may mistranslate domain-specific terms. A legal glossary lookup or post-translation term alignment step is recommended
- **Preserving nuance**: some witness descriptions are culturally or linguistically specific. The original text must always be preserved alongside the translation
- **Chain**: STT and translation compose naturally — a Dutch audio recording would flow through `faster-whisper` (with Dutch model) → `langdetect` → `OPUS-MT nl→en` → ingest pipeline

---

## 5. Multi-Statement Graph Fusion

### Motivation

Real investigations involve multiple witnesses to the same event. The current system processes a single statement into a single graph. Multi-statement fusion combines statements from different witnesses into a unified event graph, enabling cross-witness corroboration and contradiction detection.

### Architecture

```
Statement A (Witness 1)     Statement B (Witness 2)     Statement C (Witness 3)
    │                            │                            │
    ▼                            ▼                            ▼
┌──────────┐              ┌──────────┐                ┌──────────┐
│ Ingest A │              │ Ingest B │                │ Ingest C │
└──────────┘              └──────────┘                └──────────┘
    │                            │                            │
    ▼                            ▼                            ▼
Subgraph A                 Subgraph B                  Subgraph C
    │                            │                            │
    └────────────┬───────────────┘────────────────────────────┘
                 ▼
    ┌────────────────────────┐
    │  Entity Resolution     │   Cross-statement coreference
    │  (across statements)   │   "the tall man" in A = "suspect" in B?
    └────────────────────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │  Corroboration &       │   Matching facts increase confidence
    │  Contradiction         │   Conflicting facts flagged
    │  Detection             │
    └────────────────────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │  Temporal Alignment    │   Align event timelines across statements
    └────────────────────────┘
                 │
                 ▼
         Unified Event Graph
         (with per-statement provenance)
```

### Key Challenges

#### Cross-Statement Entity Resolution

The hardest problem. Within a single statement, coreference resolution handles "the driver" → "he" → "a tall man". Across statements, the same real-world person may be described differently by different witnesses:

| Witness 1 | Witness 2 | Witness 3 |
|-----------|-----------|-----------|
| "a tall man in a red jacket" | "the suspect, about 6 foot" | "some bloke near the car" |

Resolution strategies:
1. **Attribute matching**: compare PhysicalDescription properties across statements
2. **Role matching**: same role relative to the same event suggests same entity
3. **LLM-assisted**: prompt the LLM with candidate entity pairs and context from both statements
4. **Confidence-weighted**: don't force a merge — create a `POSSIBLY_SAME_AS` relationship with a confidence score

#### Corroboration & Contradiction

When two witnesses describe the same fact, this increases confidence. When they disagree, this must be flagged, not silently resolved.

```
Witness 1: (Event:collision)-[:OCCURRED_AT_TIME]->(Time{value:"14:30"})
Witness 2: (Event:collision)-[:OCCURRED_AT_TIME]->(Time{value:"14:45"})

Result: both triples kept, linked to their respective Observation nodes.
        A CONTRADICTS relationship is created between the two Time nodes.
        The Bayesian layer (§2) can assign posteriors.
```

#### Temporal Alignment

Different witnesses may describe events in different orders or with different temporal anchors. Temporal alignment:
1. Identify shared reference events (the collision itself, the arrival of emergency services)
2. Align per-witness timelines to these anchors
3. Construct a unified timeline with uncertainty intervals where accounts diverge

### Provenance Model

Every node and relationship in the unified graph carries per-statement provenance:

```
(Event {description: "collision"})-[:DERIVED_FROM]->(Observation {source_type: "statement_witness_1"})
(Event {description: "collision"})-[:DERIVED_FROM]->(Observation {source_type: "statement_witness_2"})
```

This means:
- Any fact can be traced to the specific witness(es) who reported it
- Facts reported by only one witness are distinguishable from corroborated facts
- The query phase can filter by witness or show the provenance chain for any claim

### Graph Schema Additions

```python
# New relationship types for multi-statement fusion

"POSSIBLY_SAME_AS": RelDef(
    "POSSIBLY_SAME_AS", "Person", "Person",
    "Cross-statement entity resolution candidate — same real-world person?",
    "owl:sameAs (tentative)",
),
"CORROBORATES": RelDef(
    "CORROBORATES", "Observation", "Observation",
    "Two observations from different witnesses that support the same fact",
    "evt:corroborates",
),
"CONTRADICTS": RelDef(
    "CONTRADICTS", "Observation", "Observation",
    "Two observations from different witnesses that conflict on a fact",
    "evt:contradicts",
),
```

### Implementation Direction

1. Modify `ingest.py` to accept a statement identifier and tag all nodes with the source statement
2. After each statement is ingested, run cross-statement entity resolution (LLM-assisted with attribute matching fallback)
3. Run corroboration/contradiction detection as a post-merge Cypher query
4. Extend `interview.py` to generate questions that target cross-witness discrepancies
5. Extend `query.py` to support provenance-filtered queries ("what did Witness 2 say about the vehicle?")

---

## Extension Dependency Map

```
                    ┌─────────────────────┐
                    │  Speech-to-Text (4) │
                    └──────────┬──────────┘
                               │ audio → text
                    ┌──────────▼──────────┐
                    │  Translation (4)    │
                    └──────────┬──────────┘
                               │ foreign text → English
                    ┌──────────▼──────────┐
                    │  Multi-Statement    │
                    │  Fusion (5)         │◄─── feeds into ───┐
                    └──────────┬──────────┘                   │
                               │ unified graph                │
              ┌────────────────┼────────────────┐             │
              ▼                ▼                ▼             │
    ┌─────────────────┐ ┌───────────┐ ┌──────────────┐       │
    │ Alt Ontologies  │ │ Bayesian  │ │ KG Quality   │       │
    │ (1)             │ │ Layer (2) │ │ Assessment(3)│───────┘
    └─────────────────┘ └───────────┘ └──────────────┘
                              │
                              ▼
                    Probabilistic gap analysis
                    & interview prioritisation
```

Extensions 1–3 are independent of each other and can be pursued in parallel. Extension 4 (STT + translation) is a preprocessing layer that feeds into the existing pipeline. Extension 5 (multi-statement fusion) benefits from all other extensions — the Bayesian layer enables probabilistic corroboration scoring, quality assessment catches fusion errors, and alternative ontologies may better capture cross-witness relationships.

---

## References

### Ontologies & Crime Investigation KGs
- Spyropoulos, A.Z. et al. (2023). Interoperability-Enhanced Knowledge Management in Law Enforcement. *Information*, 14(11), 607.
- Pandey, R. et al. (2020). Building Knowledge Graphs of Homicide Investigation Chronologies. *IEEE BigData 2020*.
- Müller, W. et al. (2022). Knowledge Engineering and Ontology for Crime Investigation. *AIAI 2022*.
- Ramadan, O. et al. (2025). Reconstructing Judicial Digital Forensic Evidence Graphs from Legal Documents Using LLMs. *IEEE COMPSAC 2025*.
- Mazepa, S. et al. (2022). Relationships Knowledge Graphs Construction between Evidence Based on Crime Reports. *IEEE 2022*.

### Bayesian Networks & Evidence Reasoning
- Fenton, N., Neil, M. & Yet, B. (2020). Analyzing the Simonshaven Case Using Bayesian Networks. *Topics in Cognitive Science*.
- Van Leeuwen, L., Verbrugge, R. et al. (2024). Building a Stronger Case: Combining Evidence and Law in Scenario-Based Bayesian Networks. *HHAI 2024*.
- Xu, X. & Vinci, G. (2024). Forensic Science and How Statistics Can Help It. *WIREs Computational Statistics*.
- Jellema, H. (2024). Perpetrator Knowledge: A Bayesian Account. *Law, Probability and Risk*.
- Wang, L. et al. (2020). A Knowledge-Based Reasoning Model for Crime Reconstruction and Investigation. *Expert Systems with Applications*.

### Knowledge Graph Quality Assessment
- Xue, B. & Zou, L. (2022). Knowledge Graph Quality Management: A Comprehensive Survey. *IEEE TKDE*.
- Issa, S. et al. (2021). Knowledge Graph Completeness: A Systematic Literature Review. *IEEE Access*.
- Zhang, Y. & Xiao, G. (2024). How to Implement a KG Completeness Assessment with User Requirements. *IEEE JSEE*.
- Huaman, E. (2022). Steps to Knowledge Graphs Quality Assessment. *arXiv:2208.07779*.

### Forensic Ontology Standards
- Sikos, L.F. (2021). AI in Digital Forensics: Ontology Engineering for Cybercrime Investigations. *WIREs Forensic Science*.
- Xu, W. & Xu, D. (2022). Visualizing and Reasoning about Presentable Digital Forensic Evidence with Knowledge Graphs. *IEEE ARES 2022*.
