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

### Phase 3 — Constraint (SHACL)
| Probe | What it checks |
|-------|---------------|
| SHACL validation | Graph conforms to declared shape constraints |

## Scoring

Each dimension returns a 0.0 – 1.0 score. The overall score is a weighted
average with dynamic weight normalisation (only active phases contribute).

## Usage

```bash
# Full assessment (Phase 1 + 2)
python demo.py

# Structural probes only
python demo.py --phase 1

# With faithfulness check against source text
python demo.py --source ../06-event-digital-twin/transcript.txt

# With SHACL shapes
python demo.py --shapes shapes.ttl

# Inspect current graph state
python inspect_state.py
```

## Files

| File | Purpose |
|------|---------|
| `quality_core.py` | Core data structures: `Violation`, `DimensionResult`, `QualityReport` |
| `cypher_probes.py` | Cypher-based structural probes |
| `llm_probes.py` | LLM coherence and faithfulness probes |
| `shacl_probes.py` | SHACL constraint validation |
| `demo.py` | CLI demo running all phases |
| `inspect_state.py` | Quick graph state dump |

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
