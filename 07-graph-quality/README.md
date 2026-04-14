# 07 — Graph Quality Assessment

Standalone module for multi-dimensional quality assessment of knowledge graphs
stored in Neo4j. Works with **any** knowledge graph — not tied to a particular
ontology or domain.

## Quality Dimensions

### Phase 1 — Structural (Cypher-based, zero extra deps)
| Probe | What it checks |
|-------|---------------|
| Schema population | Every expected node type has at least one instance |
| Structural connectivity | No isolated nodes, graph is connected |
| Consistency | No temporal cycles, no duplicates |
| Source grounding | Nodes carry provenance information |

### Phase 2 — Semantic (LLM-powered)
| Probe | What it checks |
|-------|---------------|
| Coherence | Triples form a non-contradictory narrative |
| Faithfulness | Graph content is supported by the source text |
| Semantic completeness | All important source facts are captured (DeepEval) |
| Investigative readiness | Graph is detailed enough for case analysis (DeepEval) |

### Phase 3 — Constraint (SHACL)
| Probe | What it checks |
|-------|---------------|
| SHACL validation | Graph conforms to declared shape constraints |

### Phase 4 — Embedding (PyKEEN)
| Probe | What it checks |
|-------|---------------|
| Link prediction | Missing relationships predicted by RotatE embeddings |
| Triple plausibility | Existing triples scored for structural consistency |
| Entity clustering | Outlier entities and potential duplicates |

## Scoring

Each dimension returns a 0.0 – 1.0 score. The overall score is a weighted
average with dynamic weight normalisation (only active phases contribute).

**LLM calibration** is available via `--calibrate`: runs LLM probes N times
and reports mean ± std to quantify scoring variance.

## Usage

```bash
# Full assessment (Phase 1 + 2)
python demo.py

# Structural probes only
python demo.py --phase 1

# With faithfulness check against source text
python demo.py --source ../08-event-digital-twin/transcript.txt

# With SHACL shapes
python demo.py --shapes shapes.ttl

# With LLM score calibration (3 runs, reports std)
python demo.py --calibrate

# Via the CLI module (supports all phases)
python -m quality_probe --deepeval --embeddings --source ../08-event-digital-twin/transcript.txt

# JSON output
python -m quality_probe --json --skip-llm

# Inspect current graph state
python inspect_state.py
```

## Files

| File | Purpose |
|------|---------|
| `quality_core.py` | Core data structures, weight system, orchestration, linearisation, calibration |
| `cypher_probes.py` | Phase 1 — Cypher-based structural probes |
| `llm_probes.py` | Phase 2 — Native LLM coherence and faithfulness probes |
| `deepeval_probes.py` | Phase 2 — DeepEval G-Eval metrics (optional upgrade) |
| `embedding_probes.py` | Phase 4 — PyKEEN RotatE embedding probes |
| `shacl_probes.py` | Phase 3 — SHACL constraint validation |
| `shapes.ttl` | SHACL shape definitions for the event domain |
| `demo.py` | Standalone CLI demo |
| `__main__.py` | Full CLI runner (`python -m quality_probe`) |
| `inspect_state.py` | Quick graph state dump |
| `test_quality_framework.py` | Offline unit tests (no Neo4j/Ollama required) |
| `TESTING_GUIDE.md` | Exhaustive testing & scoring reference |
| `PROPOSAL.md` | Original tool investigation & architecture proposal |
| `SUMMARY_CORPORATE.md` | Non-technical executive deep dive |
| `SUMMARY_TECHNICAL.md` | Theoretical derivation & state-of-art analysis |
| `SUMMARY_IMPLEMENTATION.md` | Practical implementation walkthrough |

## Dependencies

| Feature | Required packages | Fallback |
|---------|------------------|----------|
| Phase 1 (structural) | `neo4j` | — (always runs) |
| Native LLM probes | `langchain-ollama` + Ollama | Skipped with `--skip-llm` |
| DeepEval G-Eval | `deepeval` | Falls back to native LLM probes |
| SHACL validation | `rdflib`, `pyshacl` | Skipped if not installed |
| Embedding probes | `pykeen` (installs PyTorch) | Skipped if not installed |

## Integration with 06-ontologies

When used alongside the ontology comparison module, you can generate SHACL
shapes from any `OntologySpec` and pass them in:

```python
import sys; sys.path.insert(0, "../06-ontologies")
from ontology_spec import ONTOLOGY_REGISTRY

spec = ONTOLOGY_REGISTRY["schema-org-event-v1"]
shapes_ttl = spec.build_shacl_shapes()

# Then pass to shacl_probes.probe_shacl(driver, shapes_ttl=shapes_ttl)
```
