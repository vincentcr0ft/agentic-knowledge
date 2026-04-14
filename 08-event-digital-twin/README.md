# 08 — Event Digital Twin: Multi-Source Event Reconstruction

End-to-end pipeline for building a **provably complete knowledge graph** from
multiple sources (witness statements, CCTV logs, medical reports), with
cross-source fusion, temporal reasoning, what-if simulation, and interactive
provenance-aware querying.

## Architecture

```
 06-ontologies/    07-graph-quality/    08-event-digital-twin/
 ┌─────────────┐   ┌──────────────┐    ┌────────────────────────┐
 │OntologySpec │◄──│ QualityReport│◄───│  demo.py (orchestrator)│
 │schema_org   │   │ cypher_probes│    │  ingest.py    fusion.py│
 │sem_event    │   │ llm_probes   │    │  temporal.py  query.py │
 │bfo_cco      │   │ shacl_probes │    │  interview.py export.py│
 └─────────────┘   └──────────────┘    │  simulation.py         │
                                        │  quality_ext.py        │
                                        │  schema.py             │
                                        └────────────────────────┘
```

### Modules

| Module | Purpose |
|--------|---------|
| **schema.py** | Bridge to 06-ontologies. Select any registered ontology |
| **ingest.py** | LangGraph pipeline: text → extraction → coreference → Neo4j (additive, source-tagged) |
| **fusion.py** | Cross-source entity resolution, corroboration, contradiction detection |
| **temporal.py** | Time parsing, timeline construction, Allen's interval algebra consistency |
| **interview.py** | Self-resolution + human-in-the-loop gap filling |
| **query.py** | Grounded Q&A with provenance citations + slash commands |
| **simulation.py** | What-if scenario engine (snapshot/modify/diff/restore) |
| **export.py** | Graph export: RDF/Turtle, JSON-LD, Cypher, DOT, interactive HTML |
| **quality_ext.py** | Extended quality probes: completeness, temporal, cross-source, narrative |
| **demo.py** | Orchestrates the full pipeline across all modules |

## Quick Start

```bash
# Start infrastructure
docker-compose up -d
ollama pull qwen2.5:7b && ollama serve

# Single source (original behaviour)
python demo.py statements/king_street_collision.txt

# Multi-source fusion (extended)
python demo.py statements/king_street_collision.txt \
               statements/queen_road_witness.txt \
               statements/cctv_log.txt \
               statements/paramedic_report.txt

# Fresh start (clear graph first)
python demo.py --clear statements/king_street_collision.txt \
                        statements/queen_road_witness.txt
```

## All Options

```bash
# Ontology selection
python demo.py statements/*.txt --ontology sem-event-v1
python demo.py --list-ontologies

# Individual phases
python demo.py --interview                     # interview only
python demo.py --query                         # query REPL only
python demo.py --quality                       # quality probes only

# Hallucination assessment
python demo.py --hallucination-check statements/king_street_collision.txt

# Skip options
python demo.py statements/*.txt --skip-interview
python demo.py --quality --skip-llm
```

## Test Corpus

Four sources with intentional overlaps and conflicts:

| File | Type | Perspective |
|------|------|------------|
| `king_street_collision.txt` | Witness statement | Pedestrian on King Street |
| `queen_road_witness.txt` | Witness statement | Person at bus stop, Queen's Road |
| `cctv_log.txt` | CCTV traffic log | Camera KS-QR-001, precise timestamps |
| `paramedic_report.txt` | Medical report | Ambulance crew, patient identification |

## Pipeline Phases

### Phase 1: Ingest (per source)
Source text → segment → schema-guided LLM extraction → coreference resolution → additive MERGE into Neo4j with source tagging and confidence scores

### Phase 2: Fusion (if multiple sources)
Cross-source entity resolution → `POSSIBLY_SAME_AS` / `CORROBORATED_BY` / `CONTRADICTS` relationships → Bayesian confidence updates

### Phase 3: Temporal Reasoning
Parse time values → build ordered timeline → materialise `PRECEDED_BY` relationships → check for temporal contradictions

### Phase 4: Interview
3-level gap analysis → self-resolution → targeted follow-up questions → graph enrichment

### Phase 5: Quality Assessment
Population completeness → temporal consistency → cross-source consistency → narrative reconstruction (LLM-as-judge)

### Phase 6: Interactive Query
Source summary → natural language questions with provenance citations → slash commands

## Query REPL Commands

| Command | Description |
|---------|-------------|
| `/sources` | List all ingested sources with entity counts |
| `/from <source_id>` | Show all entities from a specific source |
| `/contradictions` | List all cross-source contradictions |
| `/timeline` | Display ordered event timeline |
| `/export <format>` | Export graph (turtle, jsonld, cypher, dot, html) |
| `/whatif remove <source_id>` | Simulate removing a source |

## Documentation

| Document | Audience |
|----------|----------|
| [SUMMARY_CORPORATE.md](SUMMARY_CORPORATE.md) | Non-technical deep dive into concepts and business value |
| [SUMMARY_TECHNICAL.md](SUMMARY_TECHNICAL.md) | State-of-the-art review with academic references |
| [SUMMARY_IMPLEMENTATION.md](SUMMARY_IMPLEMENTATION.md) | Practical walkthrough of every module and command |
| [EXTENSIONS.md](EXTENSIONS.md) | Research-level extensions and future work |
| [PLAN.md](PLAN.md) | Original audit and implementation plan |

## Ontology Options

| ID | Name | Best For |
|----|------|----------|
| `schema-org-event-v1` | Schema.org Event | General event modelling, web-compatible |
| `sem-event-v1` | SEM Event Model | Role-centric events, multi-actor scenarios |
| `bfo-cco-event-v1` | BFO/CCO Event | Forensic/intelligence-grade, ISO-standard |
