# Knowledge Graph Quality Assessment — Implementation Walkthrough

> **Audience:** Developers who need to understand, run, extend, or debug
> the code.  This document maps concepts to files + functions.

---

## 1. Repository Layout

```
07-graph-quality/
├── __init__.py              # Package declaration; re-exports public API
├── __main__.py              # CLI entry point (python -m quality_probe)
├── quality_core.py          # Single canonical core: dataclasses, weights, orchestration, calibration
├── cypher_probes.py         # Phase 1 — 4 Cypher-based structural probes
├── llm_probes.py            # Phase 2 — Native LLM coherence + faithfulness (Ollama/qwen2.5:7b)
├── deepeval_probes.py       # Phase 2 — DeepEval G-Eval metrics (4 probes, optional)
├── embedding_probes.py      # Phase 3 — PyKEEN RotatE embedding probes (3 probes, optional)
├── shacl_probes.py          # SHACL constraint validation via pySHACL + rdflib
├── demo.py                  # Standalone demo script (alternative to __main__.py)
├── inspect_state.py         # Quick graph state dump (node/rel counts, labels)
├── shapes.ttl               # SHACL shape definitions for the event domain
├── test_quality_framework.py # Offline unit tests (no Neo4j/Ollama needed)
├── docker-compose.yml       # Neo4j container configuration
├── README.md                # End-user overview
├── PROPOSAL.md              # Original tool investigation & architecture proposal
├── TESTING_GUIDE.md         # Exhaustive scoring & testing reference
├── SUMMARY_CORPORATE.md     # Non-technical deep dive
├── SUMMARY_TECHNICAL.md     # Theoretical derivation
└── SUMMARY_IMPLEMENTATION.md # This file
```

### Module architecture

`quality_core.py` is the **single canonical module** that all other files
import from.  It provides:

| Symbol | Purpose |
|--------|---------|
| `Violation` | Single quality issue |
| `DimensionResult` | Score + violations for one dimension |
| `QualityReport` | Aggregate 11-dimension report with summary output |
| `linearise_graph(driver)` | Graph → text triples for LLM probes |
| `calibrate_llm_probe(fn)` | Run an LLM probe N times, return median + stats |
| `build_report(results)` | Assemble a report from pre-computed `DimensionResult` objects |
| `run_quality_probe(driver)` | Full pipeline orchestrator |
| `compute_overall(report)` | Weighted score with dynamic normalisation |
| `generate_recommendations(report)` | Threshold-driven actionable suggestions |

All probe files (`cypher_probes.py`, `llm_probes.py`, etc.) import only
`Violation` and `DimensionResult` from `quality_core`. No circular imports,
no `sys.path` hacks.

---

## 2. How to Run It

### Prerequisites

```bash
# Required
pip install neo4j

# For LLM probes
# Ollama must be running with the qwen2.5:7b model
ollama pull qwen2.5:7b

# Optional — SHACL validation
pip install rdflib pyshacl

# Optional — DeepEval G-Eval (Phase 2)
pip install deepeval

# Optional — KG embeddings (Phase 3)
pip install pykeen
```

### Running via the CLI module

```bash
cd 07-graph-quality

# Quick structural check — no LLM needed
python -m quality_probe --skip-llm

# Standard run: structural + LLM coherence + faithfulness
python -m quality_probe --source ../08-event-digital-twin/transcript.txt

# Full run: all 11 dimensions
python -m quality_probe --deepeval --embeddings --source ../08-event-digital-twin/transcript.txt

# With SHACL validation
python -m quality_probe --shacl --source ../08-event-digital-twin/transcript.txt

# With LLM calibration (3 runs, reports mean ± std)
python -m quality_probe --calibrate --source ../08-event-digital-twin/transcript.txt

# JSON output
python -m quality_probe --json --deepeval --embeddings --source ../08-event-digital-twin/transcript.txt 2>/dev/null
```

### Running via the demo script

```bash
cd 07-graph-quality

python demo.py                              # Full assessment (Phase 1 + 2)
python demo.py --phase 1                    # Structural only
python demo.py --source ../08-event-digital-twin/transcript.txt  # With faithfulness
python demo.py --shapes shapes.ttl          # With SHACL shapes
python demo.py --calibrate                  # LLM calibration
```

### Inspecting the graph state

```bash
python inspect_state.py
```

---

## 3. Data Structures

### `Violation`

```python
@dataclass
class Violation:
    dimension: str          # "schema" | "structural" | "constraint" | ...
    severity: str           # "error" | "warning" | "info"
    message: str
    node_label: str | None
    node_id: str | None
```

### `DimensionResult`

```python
@dataclass
class DimensionResult:
    dimension: str          # Name matching the weight key
    score: float            # 0.0–1.0  (-1.0 = sentinel: probe skipped)
    violations: list[Violation]
    details: dict           # Probe-specific metadata (inc. calibration data)
```

A score of -1.0 signals "this probe was skipped; fall back to an alternative."

### `QualityReport`

11 named score fields, plus:
- `overall_score` — weighted average
- `violations` — all violations from all probes
- `recommendations` — actionable fix suggestions
- `dimension_results` — per-dimension raw results
- `summary()` — formatted terminal output

---

## 4. Probe-by-Probe Walkthrough

### Phase 1: Cypher Probes (`cypher_probes.py`)

All take a Neo4j `driver` and return `DimensionResult`. Pure Cypher — no
external dependencies.

| Function | Dimension | Score formula |
|----------|-----------|---------------|
| `probe_schema_population(driver, labels)` | schema | populated / total labels |
| `probe_structural_connectivity(driver)` | structural | (total − isolated) / total |
| `probe_consistency(driver)` | consistency | checks_passed / 2 |
| `probe_source_grounding(driver)` | constraint | 1 − (orphan / total) |

### Phase 2: Native LLM Probes (`llm_probes.py`)

Use `langchain_ollama.ChatOllama` with `qwen2.5:7b` at temperature 0.
Both accept **linearised triples** (a string), not a driver.

| Function | Input | Score |
|----------|-------|-------|
| `probe_coherence(triples)` | Linearised graph text | LLM score / 10 |
| `probe_faithfulness(triples, source_text)` | Triples + source (truncated 3000 chars) | LLM score / 10 |

### Phase 2: DeepEval Probes (`deepeval_probes.py`)

All accept **linearised triples** (not a driver) for consistency with native
probes. Return -1.0 sentinel if DeepEval is not installed.

| Function | G-Eval metric | Threshold |
|----------|--------------|-----------|
| `probe_coherence_deepeval(triples)` | NarrativeCoherence | 0.6 |
| `probe_faithfulness_deepeval(triples, source_text)` | ExtractionFaithfulness | 0.8 |
| `probe_semantic_completeness(triples, source_text)` | SemanticCompleteness | 0.7 |
| `probe_investigative_readiness(triples)` | InvestigativeReadiness | 0.5 |

### Phase 3: Embedding Probes (`embedding_probes.py`)

Train a **RotatE** model (Sun et al., ICLR 2019) via PyKEEN. RotatE handles
asymmetric relations (PRECEDED, WITNESSED) better than ComplEx via rotation
in complex space.

| Function | Min triples | Score formula |
|----------|------------|---------------|
| `probe_link_prediction(driver)` | 10 | 1 − (predicted_missing / total) |
| `probe_triple_plausibility(driver)` | 10 | above_threshold / total |
| `probe_entity_clusters(driver)` | 15 | 1 − 0.15 × outlier_count |

Training: RotatE, dim=50, 100 epochs, batch=min(64, |triples|), seed=42.

### SHACL Validation (`shacl_probes.py`)

| Function | Input | Score |
|----------|-------|-------|
| `probe_shacl(driver, shapes_ttl)` | Neo4j → RDF export + SHACL shapes | 1 − (violations / triples) |

---

## 5. Orchestration Flow

### Via `__main__.py` / `run_quality_probe()`

```
parse CLI args → connect Neo4j
    │
    ├── Phase 1: 4 × cypher_probes → DimensionResult[]
    │
    ├── SHACL (if --shacl): probe_shacl()
    │
    ├── Phase 2 (unless --skip-llm):
    │   ├── linearise_graph(driver) → triples string
    │   ├── If --deepeval: deepeval probes (fallback to native on -1.0)
    │   └── Else: native LLM probes
    │   └── If --calibrate: wrap in calibrate_llm_probe()
    │
    ├── Phase 3 (if --embeddings): 3 × embedding_probes
    │
    └── build_report(all_results) → QualityReport
        ├── compute_overall()  (dynamic weight normalisation)
        └── generate_recommendations()
```

### Via `demo.py`

Simpler: individual phase runners → `build_report(all_results)`.
Supports Phase 1, Phase 2 (native LLM), and Phase 3 (SHACL only).

---

## 6. LLM Calibration

`calibrate_llm_probe()` addresses LLM-as-judge non-determinism:

```python
result = calibrate_llm_probe(probe_coherence, triples, runs=3)
# result.details["calibration"] == {
#   "runs": 3, "mean": 0.75, "std": 0.05,
#   "min": 0.70, "max": 0.80, "all_scores": [0.70, 0.75, 0.80]
# }
```

- Runs the probe function `N` times
- Returns the **median** result (not mean — robust to outliers)
- Injects calibration metadata into `details["calibration"]`
- If the probe returns -1.0 (skipped), returns immediately without re-running

---

## 7. Weight Architecture

Three tiers with dynamic normalisation:

| Tier | Dimensions | Raw weight sum |
|------|-----------|---------------|
| Phase 1 | schema (0.15), structural (0.15), constraint (0.15), consistency (0.20), coherence (0.15), faithfulness (0.20) | 1.00 |
| Phase 2 | semantic completeness (0.10), investigative readiness (0.10) | 0.20 |
| Phase 3 | link prediction (0.08), triple plausibility (0.08), entity clustering (0.04) | 0.20 |

When all phases run: total = 1.40, each weight normalised by dividing by 1.40.
When only Phase 1 runs: weights already sum to 1.00.

---

## 8. SHACL Shapes (`shapes.ttl`)

7 node shapes + 1 provenance shape:

| Shape | Key constraints |
|-------|----------------|
| EventShape | description, type (enum), OCCURRED_AT (Location), OCCURRED_AT_TIME (Time) |
| PersonShape | name_or_description, role (enum) |
| VehicleShape | description |
| LocationShape | description |
| TimeShape | value |
| ObservationShape | description, observation_type, MADE_BY (Person) |
| PhysicalDescriptionShape | summary |
| ProvenanceShape | source (Warning), confidence (Info) — all content nodes |

---

## 9. Extending the System

### Adding a new probe

1. Write `probe_my_dim(driver) -> DimensionResult` in a probe file
2. Add `"my_dim": 0.10` to the appropriate weight dict in `quality_core.py`
3. Add `my_dim_score: float = 0.0` to `QualityReport`
4. Add `"my_dim": "my_dim_score"` to `_SCORE_FIELD_MAP`
5. Wire into the orchestration in `run_quality_probe()`

### Swapping the embedding model

In `embedding_probes.py`, change `DEFAULT_MODEL`:

```python
DEFAULT_MODEL = "ComplEx"  # or "TransE", "DistMult", "TuckER"
```

### Using a different LLM

In `llm_probes.py`, change `_MODEL`. In `deepeval_probes.py`, change the
model name in `OllamaModel.__init__()`.

---

## 10. Dependency Matrix

| Feature | Required packages | Fallback |
|---------|------------------|----------|
| Phase 1 (structural) | `neo4j` | — (always runs) |
| Native LLM probes | `langchain-ollama` + Ollama | Skipped with `--skip-llm` |
| DeepEval G-Eval | `deepeval` | Falls back to native LLM probes |
| SHACL validation | `rdflib`, `pyshacl` | Skipped; reports "pyshacl not installed" |
| Embedding probes | `pykeen` (installs PyTorch) | Skipped; reports "pykeen not installed" |
| LLM calibration | (built-in `statistics`) | — (always available) |

Minimum viable installation: `pip install neo4j`.
