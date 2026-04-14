# Knowledge Graph Quality Assessment — Corporate Deep Dive

> **Audience:** Executive stakeholders, product owners, programme managers.
> No code, no math — only outcomes, risks, and value.

---

## What This Is

When we extract structured knowledge from unstructured text (witness
statements, reports, intelligence feeds), we build a **knowledge graph** — a
network of people, places, events, vehicles, and their relationships. This
module answers a single question:

> **"How much can we trust this graph?"**

It is an automated quality auditor that scores a knowledge graph from 0 to
100% across multiple trust dimensions, flags specific problems, and tells you
what to fix.

---

## Why This Matters

| Risk without quality assessment | Business impact |
|--------------------------------|-----------------|
| Hallucinated facts make it into the graph | Investigators act on information that was never in the source material |
| Missing facts are invisible | Lines of enquiry are overlooked — gaps aren't even surfaced |
| Duplicate entities create confusion | The same person appears twice under different names; analysis gives wrong counts |
| Disconnected graph fragments | Entire sub-narratives become invisible to downstream queries |
| No provenance trail | When challenged in court or review, you cannot trace a fact back to its source |

The module provides **continuous, reproducible, auditable quality scoring**
that can be run after every ingest cycle, after every interview round, and
before any graph is used for operational decisions.

---

## What It Measures — Three Layers of Trust

### Layer 1: Is the graph well-formed? (Structural)

These are the checks you would do by hand if you inspected the database:

| Check | Plain-English meaning |
|-------|----------------------|
| **Schema completeness** | Does the graph contain all the entity types we expect? If a statement mentions a vehicle but the graph has no Vehicle node, something went wrong. |
| **Connectivity** | Is every piece of the graph reachable from every other piece? Isolated, orphaned nodes are lost information. |
| **Consistency** | Do events happen in a logically possible order? Are there duplicate entities that should be merged? |
| **Provenance** | Can every fact in the graph be traced back to a specific line in the original source document? |

These checks are fast, deterministic, and require no AI — they are direct
database queries.

### Layer 2: Does the graph make sense? (Semantic)

These checks use a language model to read the graph and judge its quality the
way a human analyst would:

| Check | Plain-English meaning |
|-------|----------------------|
| **Coherence** | If you read all the facts in the graph as a story, does it make sense? Are there contradictions, logical impossibilities, or gaps in the narrative? |
| **Faithfulness** | Does the graph say what the source actually said? This is the hallucination detector — it catches facts the AI invented that aren't in the original text. |
| **Semantic completeness** | The inverse of faithfulness: did the graph capture everything important from the source, or did it miss key facts? |
| **Investigative readiness** | From a practitioner's perspective: is this graph detailed enough to actually use for case analysis? Does it have specific times, identifiable locations, physical descriptions? |

### Layer 3: What's hidden in the patterns? (Embedding)

These checks train a mathematical model of the entire graph's structure and
use it to detect anomalies a human would struggle to find:

| Check | Plain-English meaning |
|-------|----------------------|
| **Link prediction** | Based on the overall structure of the graph, are there relationships that *should* exist but don't? These are gap predictions. |
| **Triple plausibility** | Are there relationships that look structurally unusual compared to the rest of the graph? These may be extraction errors. |
| **Entity clustering** | Do similar things group together? If one "Person" node clusters far from all other Person nodes, it may be mislabelled. If two Person nodes are extremely similar, they may be duplicates. |

---

## The Scoring System

Each dimension produces a **0.0 – 1.0 score**. These are combined into an
overall score using a weighted average:

| Overall Score | Interpretation | Recommended action |
|---------------|----------------|--------------------|
| 0.90 – 1.00 | **Excellent** | Graph is ready for operational use |
| 0.75 – 0.89 | **Good** | Minor gaps; usable with noted caveats |
| 0.60 – 0.74 | **Fair** | Notable issues; review violations before use |
| 0.40 – 0.59 | **Poor** | Significant problems; re-run the pipeline |
| Below 0.40 | **Critical** | Graph is unreliable; re-ingest from scratch |

**Faithfulness** (0.20 weight) and **consistency** (0.20 weight) carry the
most weight because hallucinated facts and logical contradictions are the
most dangerous failure modes.

---

## How It Fits Into the Pipeline

```
Source text  →  AI extraction  →  Knowledge graph  →  QUALITY PROBE  →  Decision
                                                           │
                                                    ┌──────┴──────┐
                                                    │ Score ≥ 0.75 │──→ Use graph
                                                    │ Score < 0.75 │──→ Fix & re-run
                                                    └─────────────┘
```

The quality probe runs **after** extraction and **before** the graph is used
for analysis, served via an API, or presented to an analyst. It acts as a
quality gate.

---

## State-of-the-Art Position

This module aligns with current best practice in KG quality assessment:

| Capability | Industry standard | Our implementation |
|-----------|-------------------|-------------------|
| Structural validation | SHACL (W3C standard) | SHACL shapes in Turtle format, validated via pySHACL |
| Schema conformance | Database constraint checks | Direct Cypher queries against Neo4j |
| LLM-based evaluation | G-Eval (NeurIPS 2023 paper) | DeepEval framework with custom G-Eval metrics |
| KG embeddings | ComplEx, TransE, RotatE | PyKEEN with RotatE model (handles asymmetric relations) |
| Multi-dimensional scoring | ISO 25012 data quality dimensions | 11 dimensions across 3 phases |
| Graceful degradation | — | Optional dependencies; phases run independently |
| LLM calibration | Verbalized confidence (Tian et al. 2023) | Multi-run scoring with mean ± std reporting |

What's **ahead of curve**: The combination of all three layers (structural +
semantic + embedding) in a single automated pipeline is not commonly seen in
production systems. Most KG quality tools use one approach.

What's **not yet covered** (potential future work): GNN-based quality
detection and cross-source consistency scoring (when fusing multiple witness
statements).

---

## Key Takeaways for Decision-Makers

1. **Quality is measurable.** Every graph gets a score you can track over time.
2. **Quality is actionable.** Each probe produces specific violations with
   recommendations — not just a number, but a fix list.
3. **Quality is layered.** Fast structural checks catch the obvious issues.
   Semantic checks catch the subtle ones. Embedding checks find hidden
   patterns.
4. **The system degrades gracefully.** If the LLM or embedding tools are
   unavailable, structural checks still run. You always get *some* score.
5. **Faithfulness is paramount.** The system is tuned to treat hallucinated
   facts as the most serious failure mode — because in investigative contexts,
   a false fact is worse than a missing fact.
