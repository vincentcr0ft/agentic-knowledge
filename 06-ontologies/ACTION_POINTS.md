# Action Points: Improving the Ontology Comparison Module

Based on the research and implementation review, the following improvements are recommended, ordered by impact and feasibility.

---

## High Priority

### AP-1: Add Quantitative Extraction Metrics

**Problem**: The demo compares outputs qualitatively (entity/rel counts) but doesn't measure extraction quality against a ground truth.

**Action**: Create a gold-standard JSON annotation for the sample text under each ontology. After extraction, compute precision, recall, and F1 per entity type and relationship type. Display these alongside the structural comparison.

**Why**: Without metrics, the claim that "Schema.org has higher extractability" remains anecdotal. Quantitative evidence strengthens the pedagogical value and makes ontology selection data-driven.

---

### AP-2: Add Few-Shot Examples to Extraction Prompts

**Problem**: The extraction prompt is purely template-based (zero-shot). BFO/CCO extraction quality suffers because the LLM has no examples of the expected output pattern for complex constructs like `BEARS_ROLE → REALIZED_IN` chains.

**Action**: Extend `OntologySpec` with an optional `few_shot_examples` field. Include 1–2 example input/output pairs in each spec, injected into `build_extraction_prompt()`.

**Why**: Few-shot prompting is the single most effective technique for improving structured extraction accuracy, particularly for ontologies with non-obvious patterns. Research consistently shows 10–30% improvement in extraction F1 with even one example.

---

### AP-3: Add a Fourth Ontology — DOLCE or a Custom Hybrid

**Problem**: The comparison covers three ontologies but omits DOLCE (Descriptive Ontology for Linguistic and Cognitive Engineering), which represents a fourth distinct philosophical position — cognitive/linguistic grounding rather than BFO's ontological realism.

**Action**: Implement a `DOLCE_EVENT` OntologySpec or, alternatively, a custom "best-of-breed" hybrid ontology that combines SEM's role model with BFO's process decomposition with Schema.org's simplicity where appropriate. This demonstrates the pluggable architecture's value.

**Why**: Showing four ontologies — and especially a custom hybrid — makes the "pluggable" claim concrete and demonstrates that practitioners can tailor ontologies to their specific domain without starting from scratch.

---

## Medium Priority

### AP-4: Integrate Actual Graph Loading with Visual Comparison

**Problem**: The demo extracts but doesn't load into Neo4j. The user can't visualise or query the resulting graphs.

**Action**: Add a `--load` flag to `demo.py` that MERGEs extracted entities into Neo4j with ontology-prefixed labels (e.g., `Schema_Event`, `SEM_Event`, `BFO_Process`). Include Cypher queries that return side-by-side comparison views.

**Why**: Seeing the graph structures in Neo4j Browser makes the structural differences tangible and memorable — far more impactful than text-based comparison.

---

### AP-5: Enrich SHACL Shape Generation

**Problem**: Generated SHACL shapes only enforce `sh:minCount 1` and `sh:datatype xsd:string` for required properties. No relationship constraints, cardinality limits, or value-range restrictions.

**Action**: Extend `build_shacl_shapes()` to generate:
- `sh:class` constraints for relationship targets
- `sh:maxCount` where appropriate (e.g., a Person has at most one name)
- `sh:pattern` for formatted values (dates, coordinates)
- `sh:in` for controlled vocabulary properties (event types, roles)
- Relationship path constraints via `sh:path` with property paths

**Why**: The current SHACL output is too simple to demonstrate the real power of ontology-driven validation. Richer shapes also serve as better documentation of the ontology's structural expectations.

---

### AP-6: Add Multiple Source Text Scenarios

**Problem**: Only one witness statement is used. The ontology trade-offs may differ for different text types.

**Action**: Add 2–3 additional sample texts:
- A formal police/incident report (structured language)
- An interview transcript (conversational, with hedging and uncertainty)
- A social media post or informal account (noisy, abbreviated)

Run all ontologies against all text types and compare how extraction quality varies.

**Why**: Demonstrates that ontology suitability is not absolute but depends on the source material characteristics. This is a key practical insight for practitioners.

---

### AP-7: Add OWL/RDF Export

**Problem**: Ontology specs exist only as Python dataclasses. They can't be loaded into standard ontology tools (Protégé, OWL reasoners, SHACL validators).

**Action**: Add an `export_owl()` method to `OntologySpec` that generates an OWL ontology in Turtle or RDF/XML format. Include `rdfs:subClassOf` hierarchies, `owl:ObjectProperty` declarations, and domain/range constraints.

**Why**: Bridges the gap between the Python implementation and the wider semantic web ecosystem. Enables users to view the ontology in Protégé, run OWL reasoners for consistency checking, and share specs with ontology engineers who don't use Python.

---

## Lower Priority / Future Work

### AP-8: Cross-Ontology Alignment Demonstration

**Action**: After extracting with all three ontologies, demonstrate automated alignment — mapping Schema.org `Person` to SEM `Actor` to BFO `Agent`, merging equivalent entities, and comparing what each ontology captured that the others missed.

**Why**: In practice, organisations often need to merge data extracted under different schemas. Demonstrating this addresses a real-world integration challenge.

---

### AP-9: Confidence Scoring on Extracted Entities

**Action**: Extend the extraction prompt to request a confidence score (high/medium/low) for each entity and relationship. Propagate these scores to the graph and use them in completeness rule evaluation (e.g., low-confidence entities might exempt from strict validation).

**Why**: LLM extraction is inherently uncertain. Making uncertainty explicit enables downstream filtering and prioritisation.

---

### AP-10: Benchmark Against Larger LLMs

**Action**: Run the same extraction pipeline with multiple LLMs (e.g., qwen2.5:7b, llama3:8b, mistral, a larger 70B model) and compare how ontology complexity interacts with model capability. Smaller models may lose more accuracy on BFO/CCO than larger ones.

**Why**: Provides practical guidance on the LLM-vs-ontology trade-off: "If you use a 7B model, stick to Schema.org or SEM. If you have access to a 70B model, BFO/CCO becomes viable."

---

## Summary Table

| ID | Action | Priority | Effort | Impact |
|----|--------|----------|--------|--------|
| AP-1 | Quantitative extraction metrics | High | Medium | High |
| AP-2 | Few-shot examples in prompts | High | Low | High |
| AP-3 | Add fourth ontology (DOLCE/hybrid) | High | Medium | Medium |
| AP-4 | Graph loading + visualisation | Medium | Medium | High |
| AP-5 | Enrich SHACL shapes | Medium | Medium | Medium |
| AP-6 | Multiple source text scenarios | Medium | Low | Medium |
| AP-7 | OWL/RDF export | Medium | Medium | Medium |
| AP-8 | Cross-ontology alignment | Low | High | Medium |
| AP-9 | Confidence scoring | Low | Low | Low |
| AP-10 | Multi-LLM benchmark | Low | Medium | Medium |
