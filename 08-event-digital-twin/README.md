# 08 — Digital Twin: Event Reconstruction from Witness Statements

End-to-end pipeline for building and querying a knowledge graph from witness
statements, with pluggable ontology support and integrated quality assessment.

## Architecture

```
 06-ontologies/          07-graph-quality/         08-digital-twin/
 ┌─────────────┐         ┌──────────────┐          ┌──────────────┐
 │OntologySpec │◄────────│ QualityReport │◄─────────│   demo.py    │
 │schema_org   │         │ cypher_probes │          │  ingest.py   │
 │sem_event    │         │ llm_probes    │          │ interview.py │
 │bfo_cco      │         │ shacl_probes  │          │  query.py    │
 └─────────────┘         └──────────────┘          │  schema.py   │
                                                    └──────────────┘
```

- **schema.py** — Bridge to 06-ontologies. Select any registered ontology.
- **ingest.py** — LangGraph pipeline: text → extraction → coreference → Neo4j.
- **interview.py** — Self-resolution + human-in-the-loop gap filling.
- **query.py** — Grounded Q&A with provenance citations.
- **demo.py** — Orchestrates all four phases.

## Usage

```bash
# Full pipeline with default ontology (Schema.org Event)
python demo.py statements/king_street_collision.txt

# Use a different ontology
python demo.py statements/king_street_collision.txt --ontology sem-event-v1
python demo.py statements/king_street_collision.txt --ontology bfo-cco-event-v1

# List available ontologies
python demo.py --list-ontologies

# Individual phases
python demo.py --interview                     # interview only (graph must exist)
python demo.py --query                         # query only
python demo.py --quality                       # quality probe only
python demo.py --quality statement.txt         # quality + faithfulness check

# Hallucination assessment
python demo.py --hallucination-check statements/king_street_collision.txt

# Skip interview or LLM probes
python demo.py statement.txt --skip-interview
python demo.py --quality --skip-llm
```

## Ontology Options

| ID | Name | Best For |
|----|------|----------|
| `schema-org-event-v1` | Schema.org Event | General event modelling, web-compatible |
| `sem-event-v1` | SEM Event Model | Role-centric events, multi-actor scenarios |
| `bfo-cco-event-v1` | BFO/CCO Event | Forensic/intelligence-grade, ISO-standard |

## Pipeline Phases

### Phase 1: Ingest
Statement → parse → schema-guided LLM extraction → coreference resolution → Neo4j

### Phase 2: Interview
3-level gap analysis → self-resolution → targeted follow-up questions → graph enrichment

### Phase 3: Quality Assessment
Structural probes → LLM coherence/faithfulness → SHACL validation → quality report

### Phase 4: Query
Entity detection → subgraph retrieval → grounded answer generation with [FACT: ...] citations

## Hallucination Assessment

The `--hallucination-check` flag runs a faithfulness analysis comparing every
graph fact against the original source text. This produces:
- A faithfulness score (0.0 – 1.0)
- A hallucination rate (inverse of faithfulness)
- Specific hallucinated facts identified
- Facts missing from the graph

Full per-triple scoring and cross-ontology comparison are planned for future
implementation.
