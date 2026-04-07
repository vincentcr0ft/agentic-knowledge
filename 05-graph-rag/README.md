# Module 05: Graph RAG — Combining Graphs and Retrieval

## The Convergence

Modules 03 and 04 each solved half of the problem:

- **RAG** (Module 03) retrieves relevant text based on semantic similarity — excellent for "what does the policy say about refunds?" but blind to relationships between entities
- **Knowledge Graphs** (Module 04) traverse explicit relationships — excellent for "who manages the person who leads Project Atlas?" but requires a pre-built graph

**Graph RAG** combines both: use an LLM to *extract* entities and relationships from unstructured text, load them into a graph, and then use *both* vector similarity and graph traversal to answer questions. The result is a system that can reason over relationships it was never explicitly told about, discovered from raw documents.

---

## The Graph RAG Pipeline

### Phase 1: Knowledge Graph Construction (Offline)

This happens once per document corpus, not at query time.

```
Documents → Chunk → Extract entities & relationships (LLM) → Resolve duplicates → Load into graph
```

#### Entity Extraction
An LLM reads each chunk and identifies entities (people, organisations, products, concepts) and relationships between them. The key challenge is **structured extraction** — the LLM must output entities in a consistent, parseable format.

```
Input: "Dr. Sarah Chen leads the AI Research team at NovaTech. Her team 
        developed the prediction engine used in Project Atlas."

Extracted:
  Entities:  (Dr. Sarah Chen, Person), (AI Research, Team), 
             (NovaTech, Organisation), (prediction engine, Product),
             (Project Atlas, Project)
  Relations: (Dr. Sarah Chen)-[LEADS]->(AI Research)
             (AI Research)-[PART_OF]->(NovaTech)
             (AI Research)-[DEVELOPED]->(prediction engine)
             (Project Atlas)-[USES]->(prediction engine)
```

#### Entity Resolution
Different chunks may refer to the same entity differently: "Dr. Chen", "Sarah Chen", "Chen". **Entity resolution** merges these into a single node. This can be done with:
- Exact matching with normalisation (lowercase, strip titles)
- Fuzzy matching (Levenshtein distance, Jaro-Winkler similarity)
- LLM-based resolution ("Are 'Dr. Sarah Chen' and 'Sarah' in this context the same person?")

#### Schema-Guided Extraction
Providing the LLM with a target schema improves extraction quality dramatically. Instead of "extract all entities", you say "extract entities matching these labels: Person, Team, Project, Product, with relationships: LEADS, WORKS_ON, DEVELOPED, USES." This constrains the LLM's output and produces a consistent, queryable graph.

### Phase 2: Hybrid Retrieval (Query Time)

At query time, Graph RAG uses a two-pronged retrieval strategy:

```
Question → [Vector Search] → relevant text chunks
         → [Entity Detection] → [Graph Traversal] → related entities & paths
         → [Merge Context] → Generate Answer
```

1. **Vector search**: finds chunks semantically similar to the question (same as Module 03)
2. **Entity detection**: identifies entities mentioned in the question
3. **Graph traversal**: starting from detected entities, traverses the knowledge graph to find related entities, paths, and context
4. **Context merging**: combines text chunks with graph-derived facts into a single, rich context for generation

The graph doesn't replace vector search — it *augments* it. Vector search finds relevant text; graph traversal finds relevant *structure*.

---

## Why Graph RAG Outperforms Pure RAG

### Multi-Hop Questions
**Pure RAG**: "Who leads the team that developed the prediction engine used in Project Atlas?"
→ Retrieves chunks mentioning "prediction engine" and "Project Atlas" but may not find the chunk about team leadership. The connection chain is lost.

**Graph RAG**: Detects "Project Atlas" → traverses USES → finds "prediction engine" → traverses DEVELOPED_BY → finds "AI Research" → traverses LEADS → finds "Dr. Sarah Chen". Every hop is explicit.

### Cross-Document Reasoning
When information about related entities is spread across different documents, pure RAG retrieves each piece independently with no connection between them. Graph RAG links them through the knowledge graph, enabling reasoning across document boundaries.

### Entity-Centric Queries
"Tell me everything about Project Atlas" with pure RAG retrieves the top-K most similar chunks. With Graph RAG, you also get every entity connected to Project Atlas — its team, its products, its dependencies — even if those connections are described in different documents that wouldn't rank highly in vector similarity.

### Contradictions and Authority
When two chunks contain contradictory information, the graph can encode provenance: which document, which date, which author. This gives the system a basis for resolving conflicts rather than randomly picking one chunk over another.

---

## Entity Extraction: The Hard Part

Extraction quality determines everything downstream. Poor extraction → sparse graph → no advantage over plain RAG.

### Common Failure Modes

1. **Over-extraction**: the LLM creates an entity for every noun ("the meeting", "the issue", "the plan"). These generic nodes pollute the graph with meaningless connections
2. **Under-extraction**: the LLM misses implicit entities. "We launched last quarter" has an implicit company and timing that may be important
3. **Inconsistent typing**: the same entity gets different labels in different chunks ("Alice" as Person vs Employee vs TeamMember)
4. **Hallucinated relationships**: the LLM infers a relationship that isn't stated in the text. "Alice works at NovaTech" and "Bob works at NovaTech" does NOT imply Alice and Bob work together

### Mitigation Strategies

- **Constrained schemas**: provide explicit label and relationship type lists
- **Few-shot examples**: include 2-3 extraction examples in the prompt
- **Validation**: check extracted entities and relationships against the source text
- **Post-processing**: merge near-duplicate entities, remove orphan nodes

---

## Graph RAG in LangGraph

The full pipeline as a LangGraph agent:

```
START → extract_entities → detect_question_entities → parallel:
            ├─ vector_retrieve
            └─ graph_traverse
        → merge_context → generate → END
```

Key design decisions:
- **Extraction is a preprocessing step**, not a query-time operation. You build the graph once and query it many times
- **Entity detection at query time** identifies which graph nodes to start traversal from
- **Parallel retrieval** (vector + graph) combines both sources of information
- **Context merging** deduplicates and formats the combined context for generation

---

## Key Concepts Summary

| Concept | What It Does |
|---|---|
| **Entity extraction** | LLM identifies entities and relationships from unstructured text |
| **Entity resolution** | Merges duplicate references to the same real-world entity |
| **Schema-guided extraction** | Constrains LLM output to match a predefined graph schema |
| **Hybrid retrieval** | Combines vector similarity search with graph traversal |
| **Graph traversal** | Follows relationships from detected entities to find structured context |
| **Context merging** | Combines text chunks and graph paths into generation context |
| **Cross-document reasoning** | Links information across document boundaries via the graph |
