# Quality Probe — Testing & Scoring Guide

A complete reference for the 11-dimension quality assessment system used to
evaluate knowledge graphs produced by the event digital twin pipeline.

---

## Table of Contents

1. [Setup & Prerequisites](#setup--prerequisites)
2. [Running the Tests](#running-the-tests)
3. [Phase 1 — Structural Probes](#phase-1--structural-probes-cypher-based)
   - [Schema Completeness](#1-schema-completeness)
   - [Structural Quality](#2-structural-quality)
   - [Consistency](#3-consistency)
   - [Source Grounding / Constraint Conformance](#4-source-grounding--constraint-conformance)
4. [Phase 2 — Semantic Probes](#phase-2--semantic-probes-llm--deepeval)
   - [Coherence](#5-coherence)
   - [Faithfulness](#6-faithfulness)
   - [Semantic Completeness](#7-semantic-completeness)
   - [Investigative Readiness](#8-investigative-readiness)
5. [Phase 3 — Embedding Probes](#phase-3--embedding-probes-pykeen)
   - [Link Prediction](#9-link-prediction)
   - [Triple Plausibility](#10-triple-plausibility)
   - [Entity Clustering](#11-entity-clustering)
6. [Overall Score Calculation](#overall-score-calculation)
7. [How to Improve Your Scores](#how-to-improve-your-scores)

---

## Setup & Prerequisites

### Core Requirements (Phase 1 — always needed)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python    | ≥ 3.10  | Runtime (3.12 recommended) |
| Neo4j     | ≥ 5.x   | Knowledge graph database |
| neo4j (pip) | ≥ 5.x | Python Neo4j driver |

Phase 1 probes run Cypher queries directly — no additional dependencies.

### LLM Requirements (Phases 1 native LLM + 2 fallback)

| Component | Version | Purpose |
|-----------|---------|---------|
| Ollama    | any     | Local LLM serving |
| qwen2.5:7b | —     | LLM model for scoring |
| langchain-ollama | ≥ 1.0 | LLM integration |

### Optional: Phase 2 — DeepEval

```bash
pip install deepeval
```

Adds structured G-Eval metrics. Falls back to native LLM probes if not
installed.

### Optional: Phase 2b — SHACL validation

```bash
pip install rdflib pyshacl
```

Validates the graph against formal SHACL shape constraints.

### Optional: Phase 3 — Embedding Probes

```bash
pip install pykeen
```

Adds KG embedding-based analysis (link prediction, triple plausibility,
entity clustering). Requires PyTorch (installed automatically with pykeen).

---

## Running the Tests

All commands assume you are in the `06-event-digital-twin/` directory.

### Quick structural check (no LLM needed)

```bash
python -m quality_probe --skip-llm
```

Runs only Phase 1 Cypher-based probes. Fast, no Ollama required.

### Standard run with LLM scoring

```bash
python -m quality_probe --source transcript.txt
```

Runs Phase 1 + native LLM probes (coherence and faithfulness).
The `--source` flag enables faithfulness scoring against the original text.

### Full run — all three phases

```bash
python -m quality_probe --deepeval --embeddings --source transcript.txt
```

Runs Phase 1 (structural) + Phase 2 (DeepEval G-Eval) + Phase 3 (PyKEEN
embeddings). This is the most comprehensive assessment.

### With SHACL validation

```bash
python -m quality_probe --shacl --source transcript.txt
```

Adds formal SHACL shape validation against `shapes.ttl`.

### JSON output (for programmatic use)

```bash
python -m quality_probe --deepeval --embeddings --source transcript.txt --json 2>/dev/null
```

Outputs a structured JSON report. Redirect stderr (`2>/dev/null`) to suppress
Neo4j deprecation warnings.

### With LLM score calibration

```bash
python -m quality_probe --calibrate --source transcript.txt
```

Runs each LLM probe 3 times and reports the median score plus mean ± std.
This addresses the fundamental non-determinism of LLM-as-judge scoring.

### CLI Flag Summary

| Flag | Effect |
|------|--------|
| `--skip-llm` | Skip all LLM-based probes (fast structural-only check) |
| `--shacl` | Include SHACL shape validation |
| `--deepeval` | Use DeepEval G-Eval for LLM metrics (falls back to native if unavailable) |
| `--embeddings` | Run PyKEEN KG embedding probes (RotatE model) |
| `--calibrate` | Run LLM probes multiple times for score calibration |
| `--calibration-runs N` | Number of calibration runs (default: 3) |
| `--source FILE` | Provide source text file for faithfulness/completeness scoring |
| `--json` | Machine-readable JSON output |
| `--neo4j-uri` | Neo4j connection URI (default: `bolt://localhost:7687`) |
| `--neo4j-user` | Neo4j username (default: `neo4j`) |
| `--neo4j-password` | Neo4j password (default: `cabbage123`) |

---

## Phase 1 — Structural Probes (Cypher-Based)

These probes query the Neo4j graph directly. Zero additional dependencies
beyond the neo4j driver. They check whether the graph is well-formed,
connected, consistent, and properly attributed.

### 1. Schema Completeness

**What it tests:**
Whether every expected node type from the ontology is populated in the graph.
The ontology defines 8 node types: Event, Person, Vehicle, Location, Time,
Object, PhysicalDescription, and Observation.

**How it works:**
1. For each node type in the ontology, counts how many instances exist
2. Checks domain-specific rules:
   - Incident events must have ≥ 2 participants
   - At least one witness (Person with role `witness`) must exist
   - At least one Event node must exist
3. Flags missing types and domain violations

**How the score is calculated:**

```
base_score = populated_types / total_expected_types
penalty    = number_of_domain_violations × 0.05
score      = max(0.0, base_score − penalty)
```

**Example:** If 6 of 8 node types are populated and there is 1 domain
violation: score = 6/8 − 0.05 = 0.70

**Violation severities:**
- Missing node type → ⚠ Warning
- Domain rule failure → ⚠ Warning

**How to score higher:**
- Ensure the extraction pipeline creates all 8 node types
- If Object or PhysicalDescription nodes are relevant to the statement, make
  sure the LLM extracts them
- Every incident Event should reference at least 2 participants (via
  `PARTICIPATED_IN` or `WITNESSED` relationships)
- At least one Person should have `role: "witness"`

---

### 2. Structural Quality

**What it tests:**
Whether the graph is well-connected — no isolated nodes, ideally a single
connected component, and no extreme hub nodes.

**How it works:**
1. Finds isolated nodes (nodes with no relationships at all)
2. Counts connected components (disjoint subgraphs)
3. Identifies high-degree hub nodes (degree > 15)

**How the score is calculated:**

```
connectivity_ratio = (total_nodes − isolated_nodes) / total_nodes
component_penalty  = max(0, (number_of_components − 1) × 0.1)
score              = clamp(connectivity_ratio − component_penalty, 0.0, 1.0)
```

**Example:** 15 nodes, 1 isolated, 2 components:
connectivity = 14/15 = 0.93, penalty = 0.1, score = 0.83

**Violation severities:**
- Isolated node → ⚠ Warning
- Multiple components → ⚠ Warning
- High-degree node (> 15 connections) → ℹ Info

**How to score higher:**
- Connect every node to at least one other node via a relationship
- Time and Location nodes are common offenders — make sure Events link to
  them via `OCCURRED_AT` and `OCCURRED_AT_TIME`
- If the graph has disconnected components, use the interview phase to elicit
  connections between them

---

### 3. Consistency

**What it tests:**
Whether the graph is logically consistent — events happen in order,
roles don't conflict, and no duplicate nodes exist.

**How it works — four sub-checks (each pass/fail; score = passed / 4):**

| Sub-check | What it detects | Severity |
|-----------|----------------|----------|
| **Temporal cycles** | An event that precedes itself (directly or transitively via `PRECEDED` chains) | Error |
| **Time monotonicity** | Events connected by `PRECEDED` where the earlier event has a later timestamp | Error |
| **Role exclusivity** | A suspect who is both `WITNESSED` and `PARTICIPATED_IN` the same event | Warning |
| **Duplicate nodes** | Two nodes of the same type with identical descriptions | Warning |

**How the score is calculated:**

```
score = number_of_checks_passed / 4
```

**Example:** All 4 sub-checks pass → score = 1.00.
One duplicate pair found → score = 0.75

**How to score higher:**
- Ensure `PRECEDED` chains match temporal order
- Do not create cycles in event sequences
- Deduplicate entities during extraction — if the same person appears twice
  with slightly different descriptions, merge them
- Make sure a suspect is not simultaneously a witness to the same event

---

### 4. Source Grounding / Constraint Conformance

**What it tests:**
Whether nodes have proper provenance — a `source` property tracing them back
to the original statement text. This is critical for investigative use: every
fact in the graph should be attributable to a specific source.

**How it works:**
1. Finds all non-Observation nodes missing the `source` property
2. Separately counts nodes missing a `confidence` rating

**How the score is calculated:**

```
orphan_count = nodes without source property (excluding Observations)
score        = 1.0 − (orphan_count / total_nodes)
```

**Example:** 15 nodes, 1 missing `source` → score = 14/15 = 0.93

**Violation severities:**
- Missing source → ⚠ Warning
- Missing confidence → ℹ Info

**How to score higher:**
- Ensure the ingestion pipeline stamps every node with a `source` property
  indicating which part of the statement it came from
- Add `confidence` ratings (0.0–1.0) to reflect extraction certainty
- The Observation node itself is exempt (it *is* the source)

---

## Phase 2 — Semantic Probes (LLM / DeepEval)

These probes use a language model to evaluate the *meaning* of the graph —
not just its structure. By default they use the native qwen2.5:7b model via
Ollama. With `--deepeval`, they use the DeepEval G-Eval framework for more
structured, research-backed evaluations.

### 5. Coherence

**What it tests:**
Whether the graph tells a coherent, internally consistent narrative. Can you
read the graph and reconstruct a clear story of what happened?

**How it works:**

The graph is linearised into text triples (e.g., `Person:the witness →
WITNESSED → Event:a loud crash`) and submitted to the LLM with instructions
to evaluate:

1. Do events form a logical sequence?
2. Are participants consistently described?
3. Do spatial movements make physical sense?
4. Are causal links plausible?
5. Can you reconstruct a clear story?

**Native LLM mode:** The LLM returns a JSON object with a `coherence_score`
(0.0–1.0) and a list of specific issues (temporal, spatial, participant,
causal, or narrative).

**DeepEval mode (`--deepeval`):** Uses G-Eval structured evaluation with the
`NarrativeCoherence` metric. The score is LLM-assigned on a 0–10 scale,
normalised to 0.0–1.0.

**Score range:**
- 1.0 = perfectly coherent; the graph reads as a clear, unambiguous narrative
- 0.6 = threshold below which a warning is raised
- 0.0 = completely incoherent; contradictory or nonsensical

**How to score higher:**
- Ensure events are connected via `PRECEDED` in the correct order
- Each person should be referenced consistently (same name/description)
- Locations should make physical sense (don't have someone at two locations
  during the same event)
- Causal chains should be complete — don't skip intermediate events

---

### 6. Faithfulness

**What it tests:**
Whether the extracted graph faithfully represents what the source text
actually says, without hallucinated, distorted, or omitted facts.

> **Requires:** `--source FILE` to provide the original statement text

**How it works:**

The graph is linearised and compared against the original source text. The
LLM evaluates three categories:

| Category | Severity | Meaning |
|----------|----------|---------|
| **Hallucinations** | Error | Facts in the graph not stated in the source |
| **Distortions** | Error/Warning | Facts changed from what the source says |
| **Omissions** | Warning/Info | Source facts missing from the graph |

**Native LLM mode:** Returns a JSON object with `faithfulness_score` and
categorised issues.

**DeepEval mode:** Uses the `ExtractionFaithfulness` G-Eval metric. Threshold
is 0.8 (stricter than coherence).

**Score range:**
- 1.0 = every graph fact is supported by the source, nothing invented
- 0.8 = DeepEval threshold; below this triggers a warning
- 0.5 = significant concerns; below this triggers an error
- 0.0 = entirely hallucinated

**How to score higher:**
- Do *not* infer facts beyond what the source explicitly states
- If the source says "a red car", don't extract "a red Toyota"
- Include all key facts from the source — omissions lower the score
- If using interview-derived facts, attribute them clearly

---

### 7. Semantic Completeness

**What it tests:**
Whether the graph captures *all important facts* from the source text. This
is the complement of faithfulness — faithfulness checks for falsehoods,
completeness checks for gaps.

> **Requires:** `--deepeval` and `--source FILE`

**How it works:**

Uses the DeepEval `SemanticCompleteness` G-Eval metric to compare source text
against the linearised graph. The LLM evaluates:

1. Are all people mentioned in the source represented?
2. Are all events and actions captured?
3. Are all times, locations, and objects included?
4. Are physical descriptions and vehicle details preserved?
5. Are relationships between entities correctly represented?

**Score range:**
- 1.0 = every fact in the source is captured in the graph
- 0.7 = threshold; below raises a warning
- 0.4 = below raises an error
- 0.0 = the graph captures nothing from the source

**How to score higher:**
- Extract every person, place, time, and object mentioned in the statement
- Don't skip physical descriptions (height, clothing, hair colour)
- Capture vehicle details (colour, make, model, registration)
- Include all events, even minor ones (arriving, leaving, looking)
- Run the interview phase to fill gaps the initial extraction missed

---

### 8. Investigative Readiness

**What it tests:**
Whether the graph is detailed enough to be useful as an investigative tool
for police case analysis.

> **Requires:** `--deepeval`

**How it works:**

The graph is evaluated from an investigator's perspective using the
`InvestigativeReadiness` G-Eval metric. No source text is needed — this
assesses the graph on its own merits:

1. Can you reconstruct the sequence of events?
2. Can you identify suspects and witnesses?
3. Is there a timeline with specific times/dates?
4. Are locations identifiable?
5. Is there enough physical evidence detail to corroborate accounts?
6. Are there obvious gaps needing follow-up?

**Score range:**
- 1.0 = fully investigation-ready
- 0.5 = threshold; below raises a warning
- 0.0 = useless for investigation

**How to score higher:**
- Include specific times, not vague ones ("14:15" not "afternoon")
- Include full addresses or precise location descriptions
- Capture physical descriptions of all people involved
- Make sure suspect descriptions are detailed enough to act on
- Include vehicle registration numbers if mentioned
- Don't skip environmental conditions (weather, lighting)

---

## Phase 3 — Embedding Probes (PyKEEN)

These probes train a knowledge graph embedding model (RotatE) on the graph
and use the learned vector representations to detect structural anomalies.
They are most valuable for larger graphs (50+ triples) and multi-statement
fusion scenarios.

> **Requires:** `--embeddings` and `pip install pykeen`

### How the embedding model works

The system exports all Neo4j relationships as (subject, predicate, object)
triples and trains a **RotatE** model (Sun et al., ICLR 2019):

- **Embedding dimension:** 50
- **Training epochs:** 100
- **Batch size:** min(64, number_of_triples)
- **Random seed:** 42 (reproducible results)

RotatE models each relation as a rotation in complex space, which handles
asymmetric relations (PRECEDED, WITNESSED) better than earlier models like
ComplEx. After training, the model can predict missing links, score existing
triples, and compare entity similarity.

---

### 9. Link Prediction

**What it tests:**
Whether the graph has missing relationships. The model predicts links that
*should* exist based on the overall graph structure but are absent.

**Minimum graph size:** 10 triples (skipped with score 1.0 if smaller)

**How it works:**
1. For a sample of (head, relation) pairs from existing triples, the model
   scores every possible tail entity
2. Top-5 predicted tails per pair are collected
3. Predictions that already exist in the graph are filtered out
4. The top 10 novel predictions are reported as info-level violations

**How the score is calculated:**

```
missing_ratio = predicted_missing_links / total_triples
score         = max(0.0, 1.0 − missing_ratio)
```

A high number of predicted missing links (relative to graph size) means the
graph may have structural gaps.

**Score range:**
- 1.0 = the model finds no plausible missing links
- 0.7 = threshold for recommendations
- 0.0 = more predicted missing links than existing triples

**What the violations mean:**
Each info-level violation shows a predicted missing link, e.g.:
```
Predicted missing link: (Person:the witness)-[WITNESSED]->(Event:the driver drove off)
```
This means the model thinks this relationship should exist based on the graph
patterns. It's a *suggestion*, not an error.

**How to score higher:**
- Close structural gaps by adding missing relationships
- If two entities should be connected, make sure the extraction creates a
  relationship between them
- A lower score may simply reflect a genuinely incomplete witness account —
  use the interview phase to fill gaps

---

### 10. Triple Plausibility

**What it tests:**
Whether existing triples are "plausible" in the embedding space. Triples that
score low may be extraction errors or hallucinations, because they don't fit
the patterns learned from the rest of the graph.

**Minimum graph size:** 10 triples (skipped with score 1.0 if smaller)

**How it works:**
1. The model scores every existing triple using `score_hrt()`
2. Scores are normalised to 0–1 using sigmoid
3. Triples below the threshold (default: 0.3) are flagged

**How the score is calculated:**

```
score = triples_above_threshold / total_triples
```

**Score range:**
- 1.0 = every triple scores above the plausibility threshold
- 0.7 = threshold for recommendations
- 0.0 = no triples are plausible

**What the violations mean:**
Each warning shows a low-scoring triple:
```
Low-plausibility triple (0.058): (Observation:Original witness statement)-[OBSERVED]->(Event:the driver got out...)
```
This triple contradicts the patterns the model learned from the rest of the
graph. Possible causes:
- Extraction error (wrong relationship type)
- Rare but valid relationship pattern
- Hallucinated fact

**How to score higher:**
- Review flagged low-scoring triples and correct extraction errors
- Ensure relationship types are used consistently (e.g., use `WITNESSED` for
  a witness observing an event, `OBSERVED` for an Observation)
- With larger graphs, the model learns better patterns and scores improve
- Note: some low-scoring triples may be correct but unusual — they are
  suggestions for review, not definitive errors

---

### 11. Entity Clustering

**What it tests:**
Whether entities of the same type cluster together in the embedding space.
Outliers may be incorrectly typed, and very close embeddings may indicate
duplicate entities that should be merged.

**Minimum graph size:** 15 triples (skipped with score 1.0 if smaller)

**How it works:**
1. Entity embeddings are extracted from the trained model
2. Entities are grouped by their Neo4j label (Person, Event, etc.)
3. For each group with ≥ 2 entities:
   - **Outlier detection:** Distance to group centroid > mean + 2σ
   - **Duplicate detection:** Distance between two entities of the same type < 0.1

**How the score is calculated:**

```
outlier_count = number of outlier entities
score         = clamp(1.0 − (outlier_count × 0.15), 0.0, 1.0)
```

Each outlier reduces the score by 0.15.

**Score range:**
- 1.0 = entities cluster cleanly by type, no outliers or duplicates
- 0.7 = threshold for recommendations
- 0.0 = heavy outlier contamination

**What the violations mean:**

| Type | Severity | Meaning |
|------|----------|---------|
| Outlier | ⚠ Warning | Entity is far from others of its type — may be mislabelled or incorrectly linked |
| Duplicate | ℹ Info | Two entities of the same type have near-identical embeddings — may refer to the same real-world entity |

**How to score higher:**
- Run entity resolution/deduplication before quality assessment
- Make sure nodes are correctly labelled (a Location shouldn't be labelled
  as a Person)
- If two Person nodes refer to the same person (e.g., "the driver" and
  "a tall man in a dark jacket"), merge them into one node

---

## Overall Score Calculation

The overall score is a weighted average of all active dimension scores.
Weights are **dynamically normalised** based on which phases were actually run.

### Base Weights (always active)

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Schema completeness | 0.15 | Important but not critical |
| Structural quality | 0.15 | Important but not critical |
| Constraint conformance | 0.15 | Provenance matters for investigative use |
| Consistency | 0.20 | Logical consistency is fundamental |
| Coherence | 0.15 | Narrative quality matters |
| Faithfulness | 0.20 | Most critical — hallucinations are dangerous |

### Phase 2 Weights (added when `--deepeval` is used)

| Dimension | Weight |
|-----------|--------|
| Semantic completeness | 0.10 |
| Investigative readiness | 0.10 |

### Phase 3 Weights (added when `--embeddings` is used)

| Dimension | Weight |
|-----------|--------|
| Link prediction | 0.08 |
| Triple plausibility | 0.08 |
| Entity clustering | 0.04 |

### How normalisation works

When all three phases run, the raw weight total is
0.15+0.15+0.15+0.20+0.15+0.20 + 0.10+0.10 + 0.08+0.08+0.04 = **1.40**.
Each weight is divided by 1.40 to produce the effective weight.

When only Phase 1 runs, the total is 1.00 and weights are used as-is.

```
overall_score = Σ(weight_i × score_i) / Σ(weight_i)
```

Dimensions that were not run (no entry in `dimension_results`) are excluded
entirely — they don't drag the score down.

---

## How to Improve Your Scores

### Quick wins (biggest score improvements)

| Issue | Fix | Affected Dimensions |
|-------|-----|---------------------|
| Missing node types (Object, PhysicalDescription) | Add extraction prompts for these types | Schema (+0.1–0.2) |
| Isolated nodes | Connect Time/Location nodes to Events | Structural (+0.1) |
| Missing `source` property | Stamp all nodes during ingestion | Constraint (+0.1) |
| Duplicate entities | Merge identical nodes | Consistency (+0.25) |

### Pipeline improvements

1. **Better extraction prompts:** If schema completeness is low, refine the
   LLM prompt in `ingest.py` to explicitly request all 8 node types.

2. **Run the interview phase:** The interview probes the LLM to identify
   implicit facts and fill gaps. This directly improves semantic completeness
   and investigative readiness.

3. **Entity resolution:** After ingestion, deduplicate entities that refer to
   the same real-world object. This improves consistency and entity clustering.

4. **Provenance tagging:** Ensure every node carries a `source` property linking
   it to the specific passage in the source text. This improves constraint
   conformance and faithfulness.

5. **Multi-statement fusion:** When combining multiple witness statements, run
   the quality probe after each addition to track how scores change. Entity
   clustering becomes increasingly valuable as the graph grows.

### Score interpretation guide

| Overall Score | Interpretation |
|---------------|----------------|
| 0.90–1.00 | Excellent — graph is investigation-ready |
| 0.75–0.89 | Good — minor gaps, usable with caveats |
| 0.60–0.74 | Fair — notable issues, review violations before use |
| 0.40–0.59 | Poor — significant quality problems, re-run pipeline |
| < 0.40 | Critical — graph is unreliable, re-ingest from scratch |
