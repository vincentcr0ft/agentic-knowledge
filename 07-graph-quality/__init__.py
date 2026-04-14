"""
Quality Probe — Multi-dimensional knowledge graph quality assessment.

Probes the quality of knowledge graphs produced by the event digital twin
pipeline across multiple dimensions:

  Phase 1 — Structural (Cypher-based, zero extra deps)
    1. Schema completeness  — are all expected node types populated?
    2. Structural quality   — is the graph connected? degree distribution healthy?
    3. Constraint conformance — do nodes/relationships satisfy SHACL shape constraints?
    4. Consistency          — temporal acyclicity, role constraints, provenance audit

  Phase 2 — Semantic (LLM / DeepEval)
    5. Coherence            — does the graph tell a coherent narrative?
    6. Faithfulness         — does the graph faithfully represent the source text?
    7. Semantic completeness — are all key facts captured from the source?
    8. Investigative readiness — is the graph detailed enough for case analysis?

  Phase 3 — Embedding (PyKEEN)
    9. Link prediction      — can the embedding model discover missing links?
   10. Triple plausibility  — do existing triples score high in the embedding space?
   11. Entity clustering    — are entities well-separated with no hidden duplicates?
"""

from quality_probe.core import QualityReport, Violation, run_quality_probe
