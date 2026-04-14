# Ontological Foundations for Event Knowledge Graphs: A Technical Survey

## 1. Introduction

The construction of knowledge graphs (KGs) from unstructured text has become a central task in AI pipelines, yet the **ontological commitment** — the choice of what categories of entities exist and how they relate — remains an under-examined design variable. This survey examines three ontological frameworks through the lens of their suitability for LLM-driven event extraction and graph construction, situating them within the broader landscape of formal ontology, semantic web standards, and contemporary AI research.

## 2. Ontological Landscape

### 2.1 What Constitutes an Ontology

Following Gruber's (1993) foundational definition, an ontology is "a formal, explicit specification of a shared conceptualisation." More precisely, Feilmayr and Wöß (2016) refine this as a specification "characterised by high semantic expressiveness required for increased complexity." In information science, ontologies occupy a spectrum from lightweight vocabularies (taxonomies, thesauri) through to axiomatically rich formal theories.

The key dimensions along which ontologies vary are:

- **Ontological commitment** — what entities and relations are posited as existing
- **Expressivity** — what can be stated (e.g., mereological decomposition, role reification)
- **Formality** — whether the ontology has a model-theoretic semantics (e.g., OWL DL)
- **Extractability** — how readily an LLM can populate instances given the schema

### 2.2 Upper Ontologies and Their Descendants

A top-level (or upper) ontology provides domain-independent categories. The principal contenders include **BFO** (ISO/IEC 21838-2:2021), **DOLCE** (Descriptive Ontology for Linguistic and Cognitive Engineering), **SUMO** (Suggested Upper Merged Ontology), and **Cyc**. BFO distinguishes itself by its commitment to **ontological realism** — entities in the ontology are intended to correspond to entities in reality, not merely to concepts in a model (Smith & Ceusters, 2010).

BFO's core bifurcation divides entities into:
- **Continuants** — entities that persist through time (objects, qualities, roles, spatial regions)
- **Occurrents** — entities that unfold through time (processes, temporal regions)

This division has profound consequences for event modelling: an "event" in BFO is a `bfo:Process`, and participant roles are `bfo:Role` instances that *inhere* in agents and are *realized in* processes — a significantly richer representation than flat event-property models.

### 2.3 Mid-Level and Domain Ontologies

The **Common Core Ontologies (CCO)** extend BFO with eleven mid-level modules covering agents, events, information entities, geospatial regions, and temporal intervals. Following the February 2024 DOD/IC memorandum, BFO+CCO now constitute the baseline standard for US defence and intelligence ontology work (Gambini, 2024). The CCO's `InformationContentEntity` class is particularly relevant for provenance modelling — statements, claims, and observations can be first-class objects with their own creation histories.

## 3. The Three Ontologies Under Examination

### 3.1 Schema.org Event

Schema.org is a collaborative, community-developed vocabulary primarily designed for web markup. Its `Event` type is intentionally shallow:

- **Participants** are modelled as properties on the Event (e.g., `schema:attendee`, `schema:performer`) or as `Person` nodes with a role string property
- **Temporal extent** is captured via `schema:startDate` / `schema:endDate`
- **Spatial location** via `schema:location`
- **No sub-event decomposition** — events are atomic or linked via ad-hoc relations

**Ontological analysis**: Schema.org makes minimal ontological commitments. Roles are *property values* rather than first-class entities, meaning the same person cannot hold different roles across different events without duplication. There is no formal distinction between continuants and occurrents — everything is a `schema:Thing`. This simplicity is simultaneously its strength (high LLM extractability) and its limitation (expressional poverty for complex narratives).

### 3.2 SEM (Simple Event Model)

The Simple Event Model (van Hage et al., 2011) was designed at the VU Amsterdam Semantic Web group specifically for event-centric information integration. Its key innovations:

- **First-class roles**: `sem:Role` is a constraint class that qualifies participation. An `Actor`'s engagement in an `Event` via `sem:hasActor` can be constrained by a `sem:Role` with a `sem:roleType`, enabling the same actor to play different roles in different events.
- **Sub-event decomposition**: `sem:hasSubEvent` allows hierarchical nesting (a collision *contains* impact, exit, fleeing as sub-events).
- **Temporal uncertainty**: The `sem:Time` class supports uncertain intervals via `sem:hasEarliestBeginTimeStamp` / `sem:hasLatestEndTimeStamp`.
- **Authority and viewpoint modelling**: `sem:View` and `sem:accordingTo` enable propositional attitudes — the same event can be characterised differently by different authorities.

**Ontological analysis**: SEM occupies a middle ground. It introduces reified roles (a significant expressivity gain over Schema.org) and event decomposition, while remaining lightweight enough for practical deployment. However, SEM lacks a formal upper-ontological grounding — its categories are pragmatic rather than philosophically motivated. It does not distinguish between continuants and occurrents at the meta-level.

### 3.3 BFO / CCO

Basic Formal Ontology provides the most rigorous framework:

- **Process ontology**: Events are `bfo:Process` instances — occurrents that have temporal parts. A collision is a process that `bfo:has_part` sub-processes (impact, flight, intervention).
- **Role realism**: Roles are `bfo:Role` instances — *realizable entities* that *inhere in* agents (`bfo:bearer_of`) and are *realized in* processes (`bfo:realized_in`). This three-place relation (agent → role → process) is maximally expressive.
- **Mereological decomposition**: `bfo:has_part` provides formally defined part-whole relations with transitivity and antisymmetry guarantees.
- **Information Content Entities**: CCO's `InformationContentEntity` class enables first-class modelling of statements, claims, and their provenance chains.

**Ontological analysis**: BFO/CCO provides the most complete formal semantics but at substantial complexity cost. The number of node and relationship types is significantly larger, and LLMs must navigate a deeper conceptual hierarchy. Empirically, extraction accuracy degrades with ontological complexity unless significant prompt engineering or few-shot examples are provided.

## 4. Key Theoretical Tensions

### 4.1 Extractability vs. Expressivity

There is a fundamental tension between what an ontology can *represent* and what an LLM can reliably *extract*. Dai et al. (2024) demonstrate that linearised triple formats improve LLM reasoning over structured data, while Zhu et al. (2023) show that LLMs perform better on KG reasoning than on KG *extraction* — precisely the scenario where ontological complexity becomes a burden.

Our empirical observation across the three ontologies confirms this: Schema.org consistently yields the highest extraction accuracy for simple scenarios, while BFO/CCO captures the most structural information when extraction succeeds.

### 4.2 Role Modelling: The Critical Differentiator

The handling of participant roles is the most significant structural difference between the ontologies:

| Level | Pattern | Example |
|-------|---------|---------|
| **L0** (Schema.org) | `Person.role = "witness"` | Static property; one role per entity |
| **L1** (SEM) | `Actor ←[HAS_ROLE]→ Role {type: "witness"}` | Reified; multi-role per event |
| **L2** (BFO) | `Agent ←[BEARS_ROLE]→ Role ←[REALIZED_IN]→ Process` | Reified + realisation semantics |

For investigative scenarios where the same individual is a witness to one event and a suspect in another, L0 is structurally inadequate, L1 is sufficient, and L2 provides additional formal guarantees about role identity and lifecycle.

### 4.3 The Validation Problem

SHACL (Shapes Constraint Language, W3C Recommendation 2017) provides the standard mechanism for validating RDF graphs against structural constraints. Our implementation generates SHACL shapes directly from the ontology specification — each `NodeSpec` produces a `sh:NodeShape` with `sh:property` constraints for required properties. This enables automated quality assurance: the graph is validated against the ontology's structural expectations, and violations are surfaced for human review.

However, SHACL validation in the property-graph world (Neo4j / Cypher) requires translation — our implementation uses Cypher-based "completeness rules" as a property-graph analogue of SHACL shapes. This is a pragmatic compromise that sacrifices some of SHACL's formal expressivity.

## 5. State of the Art: LLMs Meet Ontologies (2024–2026)

Several recent developments shape the landscape:

1. **GraphRAG** (Edge et al., 2024): Entity knowledge graphs are used for query-focused summarisation, with community structure providing hierarchical context. This approach validates the utility of ontology-structured extraction for downstream reasoning.

2. **Ontology-guided extraction**: Recent work demonstrates that providing explicit ontological schemas in LLM prompts significantly improves extraction precision and recall, particularly for domain-specific entities and relations. The prompt-as-schema pattern — where the ontology specification *is* the extraction prompt — is the approach our implementation follows.

3. **Knowledge graph evolution**: Jiang et al. (2023, revised 2025) survey the evolution from static KGs through dynamic and temporal KGs to event KGs, identifying the integration of LLMs with structured knowledge as the most promising research direction.

4. **Neurosymbolic approaches**: Combining LLM extraction with formal ontological reasoning (constraint checking, inference, consistency verification) represents an emerging paradigm where the strengths of statistical and symbolic AI are complementary.

## 6. Open Research Questions

1. **Adaptive ontology selection**: Can the extraction pipeline dynamically choose the optimal ontology depth based on text complexity and downstream task requirements?
2. **Ontology-aware fine-tuning**: To what extent does fine-tuning LLMs on ontology-structured data improve extraction fidelity for complex ontologies like BFO/CCO?
3. **Cross-ontology alignment**: How can knowledge graphs extracted under different ontologies be systematically merged while preserving semantic fidelity?
4. **Formal verification of LLM-extracted graphs**: Can SHACL validation be extended with probabilistic semantics to handle the inherent uncertainty of LLM extraction?

## 7. Conclusion

The choice of ontology for event knowledge graph construction is not merely a schema design decision — it is a commitment to a particular set of metaphysical assumptions about the structure of events, participants, and their relations. Schema.org optimises for breadth and simplicity; SEM for narrative richness; BFO/CCO for formal rigour and institutional interoperability. The optimal choice depends on the intended reasoning tasks, the complexity of the source material, the required quality assurance level, and the downstream integration requirements.

---

## References

- Arp, R., Smith, B., & Spear, A. D. (2015). *Building Ontologies with Basic Formal Ontology*. MIT Press.
- Dai, Y., et al. (2024). Linearised triples for LLM reasoning over structured data. *arXiv preprint*.
- Edge, D., et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv:2404.16130*.
- Feilmayr, C., & Wöß, W. (2016). An analysis of ontologies and their success factors for application to business. *Data & Knowledge Engineering*, 101, 1–23.
- Gambini, B. (2024). DOD, Intelligence Community adopt resource developed by UB ontologists. *University at Buffalo News*.
- Gruber, T. R. (1993). Toward Principles for the Design of Ontologies Used for Knowledge Sharing. *International Journal of Human-Computer Studies*, 43(5–6), 907–928.
- ISO/IEC 21838-2:2021. Information technology — Top-level ontologies (TLO) — Part 2: Basic Formal Ontology (BFO).
- Jensen, M., et al. (2024). The Common Core Ontologies. *Formal Ontology in Information Systems*, IOS Press, pp. 47–58.
- Jiang, X., et al. (2023). On the Evolution of Knowledge Graphs: A Survey and Perspective. *arXiv:2310.04835*.
- Otte, N., Beverley, J., & Ruttenberg, A. (2022). BFO: Basic Formal Ontology. *Applied Ontology*, 17(1), 17–43.
- Smith, B., & Ceusters, W. (2010). Ontological Realism as a Methodology for Coordinated Evolution of Scientific Ontologies. *Applied Ontology*, 5(3–4), 139–188.
- van Hage, W. R., et al. (2011). Design and use of the Simple Event Model (SEM). *Web Semantics*, 9(2), 128–136.
- W3C (2017). Shapes Constraint Language (SHACL). W3C Recommendation 20 July 2017.
- Zhu, Y., et al. (2023). LLMs for KG reasoning vs extraction. *arXiv preprint*.
