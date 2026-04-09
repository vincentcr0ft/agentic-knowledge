# Event Digital Twin: Research & Implementation Plan

## The Idea

An agent that constructs a "digital twin" of an event from narrative text (e.g. a police witness statement). The system:

1. **Ingests** unstructured narrative text
2. **Extracts** entities, relationships, and temporal/spatial facts into a knowledge graph
3. **Analyses** the graph for gaps against a domain schema
4. **Generates** follow-up questions targeting those gaps
5. **Incorporates** answers, updating the graph
6. **Repeats** until the graph reaches a completeness threshold
7. **Serves** as a queryable knowledge base for subsequent investigation

---

## Research Foundations

### 1. LLM-Driven Knowledge Graph Construction

**Zhu et al. (2023)** — *"LLMs for Knowledge Graph Construction and Reasoning"* (arXiv:2305.13168)

Key findings relevant to this POC:
- GPT-4-class models are **better at KG reasoning than KG construction** — they excel when given a graph and asked to infer, but are less reliable at raw extraction from text
- The paper proposes **AutoKG**, a multi-agent approach where different LLM agents handle different extraction tasks (entity identification, relation classification, triple validation)
- **Implication for us**: Don't trust a single extraction pass. Use a multi-step pipeline: extract → validate → resolve. Our Module 05 already demonstrates this pattern

**Dong (2023)** — *"Generations of Knowledge Graphs: The Crazy Ideas and the Business Impact"* (arXiv:2308.14217, PVLDB 2023)

- Describes three generations of KGs, culminating in **"dual neural KGs"** — the integration of structured KGs with LLMs where each compensates for the other's weaknesses
- KGs provide **grounding and explainability**; LLMs provide **flexibility and natural language understanding**
- **Implication for us**: The digital twin should be a dual system — the graph is the source of truth, the LLM is the interface. The LLM never answers from its own knowledge; it always grounds in the graph

### 2. Graph RAG and Community Detection

**Edge et al. (2024)** — *"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"* (arXiv:2404.16130)

Microsoft's GraphRAG paper, directly relevant:
- Builds an entity knowledge graph from source documents using LLM extraction
- Pre-generates **community summaries** — clusters of closely related entities get a narrative summary
- At query time, each community summary generates a partial response; these are then synthesised
- **Implication for us**: After building the event graph, we should generate community summaries (e.g. "the vehicle incident", "the suspect description", "the timeline"). These summaries help the gap analysis node identify what's missing at a thematic level, not just node-by-node

### 3. LLM Comprehension of Graph Structure

**Dai et al. (2024)** — *"Large Language Models Can Better Understand Knowledge Graphs Than We Thought"* (arXiv:2402.11541)

Critical finding:
- **Linearized triples** (e.g. `(John, witnessed, collision)`) are **more effective than fluent natural language text** for helping LLMs understand KG information
- Different LLMs have different preferences for triple organisation format
- Larger models are **more susceptible to noisy/incomplete subgraphs** — noise degrades their performance more than it does smaller models
- **Implication for us**: When feeding the current graph state to the LLM for gap analysis, use linearized triple format, not prose. And with qwen2.5:7b (a smaller model), noise tolerance may actually be reasonable

### 4. Knowledge-Augmented Prompting

**Baek et al. (2023)** — *"KAPING: Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering"* (arXiv:2306.04136)

- Retrieves relevant facts from a KG based on semantic similarity to the question, prepends them to the prompt
- Zero-shot (no fine-tuning needed) — outperforms baselines by up to 48%
- **Implication for us**: For the query phase, retrieve relevant subgraphs via entity detection + traversal, inject as linearized triples into the prompt. This is essentially what our Module 04 NL-to-Cypher pipeline does, but with KAPING's insight that prepending facts works better than asking the LLM to generate Cypher

### 5. Graph-of-Thoughts Reasoning

**Wen et al. (2023)** — *"MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models"* (arXiv:2308.09729)

- LLMs can build **internal reasoning graphs** when prompted with KG structure
- The "mind map" reveals the LLM's reasoning pathway — which entities it connected and how
- Showed significant improvements in medical QA by combining KG facts with LLM reasoning
- **Implication for us**: When answering queries against the completed graph, we can ask the LLM to show its reasoning path through the graph, making answers explainable and auditable

### 6. Event Knowledge Graphs

**Jiang et al. (2023)** — *"On the Evolution of Knowledge Graphs: A Survey and Perspective"* (arXiv:2310.04835)

Particularly relevant section on **Event KGs**:
- Event KGs represent events as first-class nodes with temporal/spatial attributes, participants, and causal links
- Unlike entity-centric KGs (Person → WORKS_AT → Company), event KGs centre on **what happened**: (Event: collision) → HAS_PARTICIPANT → (Person: witness), HAS_LOCATION → (Location: High Street), HAS_TIME → (Time: 14:30)
- **Implication for us**: The witness statement domain is fundamentally an EVENT graph, not an entity graph. The schema should centre on events, with entities as participants

---

## Domain Schema: Witness Statement Ontology

The gap analysis engine needs a **target schema** — a model of what a "complete" event description looks like. Based on standard investigative frameworks (5W1H: Who, What, When, Where, Why, How):

### Node Types

| Label | Required Properties | Description |
|-------|-------------------|-------------|
| **Event** | description, type (incident/observation/action) | Central node — something that happened |
| **Person** | name or description, role (witness/suspect/victim/bystander) | Human participant |
| **Vehicle** | description, colour, make, model, registration | Vehicle involved |
| **Location** | description, type (street/building/area) | Where something happened |
| **Time** | value, precision (exact/approximate/relative) | When something happened |
| **Object** | description, type (weapon/clothing/item) | Physical object mentioned |
| **PhysicalDescription** | height, build, hair, clothing, distinguishing_features | Appearance of a person |

### Relationship Types

| Type | From → To | Description |
|------|-----------|-------------|
| **PARTICIPATED_IN** | Person → Event | Someone was involved |
| **WITNESSED** | Person → Event | Someone observed |
| **OCCURRED_AT** | Event → Location | Where it happened |
| **OCCURRED_AT_TIME** | Event → Time | When it happened |
| **USED** | Person → Object/Vehicle | Someone used something |
| **DESCRIBED_AS** | Person → PhysicalDescription | Appearance |
| **CAUSED** | Event → Event | Causal chain |
| **PRECEDED** | Event → Event | Temporal ordering |
| **LOCATED_NEAR** | Location → Location | Spatial relation |
| **OWNED_BY** | Vehicle/Object → Person | Ownership |

### Completeness Rules

These define what "gaps" look like — conditions that should be true for a complete graph:

1. Every **Event** must have at least one OCCURRED_AT_TIME and one OCCURRED_AT location
2. Every **Event** must have at least one participant (PARTICIPATED_IN or WITNESSED)
3. Every **Person** with role=suspect must have a DESCRIBED_AS → PhysicalDescription
4. Every **Vehicle** mentioned in relation to an event should have as many identifying properties as possible (colour, make, registration)
5. Causal/temporal ordering between events should be established (CAUSED or PRECEDED)
6. The **witness** (statement author) should have WITNESSED relationships to all events they describe
7. Time values should be as precise as possible (exact > approximate > relative)
8. Location descriptions should be specific enough to be locatable

---

## Architecture

### LangGraph Agent Design

```
                    ┌─────────────────────────────────┐
                    │         INGEST PHASE             │
                    │                                  │
  statement ──►     │  parse_statement                 │
                    │       │                          │
                    │       ▼                          │
                    │  extract_entities                │
                    │       │                          │
                    │       ▼                          │
                    │  resolve_entities                │
                    │       │                          │
                    │       ▼                          │
                    │  load_to_graph                   │
                    │       │                          │
                    └───────┼──────────────────────────┘
                            ▼
                    ┌─────────────────────────────────┐
                    │       INTERVIEW PHASE            │
                    │                                  │
                    │  analyse_gaps ◄─────────────┐    │
                    │       │                     │    │
                    │       ▼                     │    │
                    │  (gaps found?)              │    │
                    │    yes │  no──► done         │    │
                    │       ▼                     │    │
                    │  generate_questions         │    │
                    │       │                     │    │
                    │       ▼                     │    │
                    │  [INTERRUPT: human answers] │    │
                    │       │                     │    │
                    │       ▼                     │    │
                    │  extract_from_answers       │    │
                    │       │                     │    │
                    │       ▼                     │    │
                    │  update_graph ──────────────┘    │
                    │                                  │
                    └───────┼──────────────────────────┘
                            ▼
                    ┌─────────────────────────────────┐
                    │        QUERY PHASE               │
                    │                                  │
                    │  receive_question                │
                    │       │                          │
                    │       ▼                          │
                    │  retrieve_subgraph               │
                    │       │                          │
                    │       ▼                          │
                    │  generate_answer                 │
                    │       │                          │
                    │       ▼                          │
                    │  [return answer with evidence]   │
                    │                                  │
                    └─────────────────────────────────┘
```

### Phase 1: Ingest

| Node | Purpose |
|------|---------|
| `parse_statement` | Clean and segment the raw statement text into paragraphs/sentences |
| `extract_entities` | Schema-guided extraction — LLM identifies entities and relationships matching the ontology. Uses the extraction pattern from Module 05 but with the witness statement schema |
| `resolve_entities` | Merge duplicates ("the man", "he", "the suspect" → same Person node). Uses coreference resolution via LLM |
| `load_to_graph` | Create nodes and relationships in Neo4j. Tag each fact with its source (which sentence of the statement) |

### Phase 2: Interview (the novel part)

| Node | Purpose |
|------|---------|
| `analyse_gaps` | Query the graph against completeness rules. For each rule violation, record a gap: `{type: "missing_time", entity: "Event:collision", rule: "Events must have timestamps"}` |
| `generate_questions` | LLM generates natural language questions targeting the specific gaps. Prioritises high-value gaps (events without times/locations before suspects without hair colour) |
| `[INTERRUPT]` | Agent pauses. Questions are presented to the user. User answers in natural language |
| `extract_from_answers` | LLM extracts new entities/relationships from the user's answers, using the same schema-guided extraction |
| `update_graph` | Merge new facts into the graph. Re-run entity resolution. Return to `analyse_gaps` |

**Loop termination**: The interview phase ends when either (a) no gaps remain, (b) a maximum number of interview rounds is reached, or (c) the user indicates they have no more information.

### Phase 3: Query

| Node | Purpose |
|------|---------|
| `receive_question` | Accept a natural language question about the event |
| `retrieve_subgraph` | Detect entities in the question → traverse the graph → collect relevant nodes, relationships, and paths. Also retrieve the source sentences that each fact was extracted from |
| `generate_answer` | LLM answers grounded in the retrieved subgraph. Must cite which graph facts support the answer. Uses linearized triples (per Dai et al. finding) |

---

## Gap Analysis Strategy

This is the core innovation. The gap analyser operates at three levels:

### Level 1: Schema Completeness
Cypher queries that check structural rules:

```cypher
// Events without timestamps
MATCH (e:Event) WHERE NOT (e)-[:OCCURRED_AT_TIME]->(:Time) RETURN e

// Persons without descriptions
MATCH (p:Person {role: 'suspect'}) WHERE NOT (p)-[:DESCRIBED_AS]->(:PhysicalDescription) RETURN p

// Events without locations
MATCH (e:Event) WHERE NOT (e)-[:OCCURRED_AT]->(:Location) RETURN e

// Orphan entities (no relationships)
MATCH (n) WHERE NOT (n)-[]-() RETURN n
```

### Level 2: Narrative Coherence
LLM-driven checks against the graph as a whole:

- **Temporal consistency**: Are events ordered logically? Are there gaps in the timeline?
- **Spatial consistency**: Does the movement of people/vehicles between locations make sense?
- **Participant consistency**: Is the same person described differently in different events?
- **Causal plausibility**: Do the CAUSED relationships make sense?

### Level 3: Investigative Completeness
Domain-specific questions an investigator would want answered:

- Can each person be identified or described well enough to find them?
- Is the sequence of events clear enough to reconstruct what happened?
- Are there implied participants who aren't explicitly mentioned?
- Are there environmental conditions (weather, lighting, visibility) that matter?

---

## Question Generation Strategy

Questions should be:
- **Specific**: "What colour was the car?" not "Can you tell me more?"
- **Prioritised**: Missing event timestamps and locations before suspect shoe colour
- **Non-leading**: "What did the person look like?" not "Was the person tall?"
- **Grouped**: Related gaps bundled into 2-3 questions per round, not 20 at once
- **Context-aware**: Reference what IS known: "You mentioned a man running. Can you describe him?"

Priority ordering:
1. **Critical**: Event timestamps, locations, participant count
2. **High**: Suspect/vehicle descriptions, direction of travel, sequence of events
3. **Medium**: Bystander details, environmental conditions, distances/durations
4. **Low**: Peripheral details that don't affect core event reconstruction

---

## Implementation Plan

### Module 06: `06-event-digital-twin/`

**Files**:

| File | Purpose |
|------|---------|
| `README.md` | Concepts: event ontology, gap analysis, conversational KG construction |
| `schema.py` | Domain ontology definition — node types, relationship types, completeness rules, all as data structures |
| `ingest.py` | Ingest pipeline — extraction, resolution, graph loading |
| `interview.py` | Interview loop — gap analysis, question generation, answer extraction |
| `query.py` | Query interface — subgraph retrieval, grounded answer generation |
| `demo.py` | Full end-to-end demo orchestrating all three phases |
| `statements/` | Sample witness statement texts for testing |

### State Schema

```python
class DigitalTwinState(TypedDict):
    # Ingest
    raw_statement: str
    segments: list[str]           # sentences/paragraphs
    extracted_entities: list       # raw extraction results
    resolved_entities: dict        # after resolution

    # Interview
    gaps: list[dict]              # current gap analysis results
    questions: list[str]          # generated questions for user
    user_answers: list[str]       # answers from human
    interview_round: int          # current round number
    max_rounds: int               # termination limit

    # Query
    question: str                 # user query
    subgraph: list[dict]          # retrieved graph context
    answer: str                   # generated answer
    evidence: list[str]           # source citations

    # Meta
    phase: str                    # ingest | interview | query
    steps: list[str]              # audit trail
```

### Build Sequence

1. **schema.py** — Define the ontology as Python data structures. Completeness rules as callable functions that return gap descriptions
2. **ingest.py** — Build the extraction pipeline (reuse patterns from Module 05, adapted to the witness schema). Test with a sample statement
3. **interview.py** — Build the gap analysis → question generation → answer extraction loop. Test the loop with simulated answers first, then with real human-in-the-loop (LangGraph interrupt)
4. **query.py** — Build the query pipeline (reuse patterns from Module 04). Add provenance tracking so answers cite specific graph facts
5. **demo.py** — Wire everything together into a single LangGraph that transitions between phases
6. **Test end-to-end** with a realistic sample statement

### Sample Test Statement

```
I was walking along King Street at approximately 2:15 PM on Tuesday when I heard
a loud crash. I turned and saw a red car had collided with a cyclist at the
junction of King Street and Queen's Road. The driver got out — a tall man wearing
a dark jacket. He looked at the cyclist who was on the ground and then got back
in his car and drove off heading north on Queen's Road. Another woman who was
nearby called an ambulance. I stayed with the cyclist until the paramedics arrived
about ten minutes later.
```

Expected extraction: 5+ events, 4+ persons, 2+ locations, 3+ times, 1 vehicle.

Expected gap analysis would identify:
- No registration for the vehicle
- No make/model for the vehicle ("red car" — colour but no make)
- Approximate time only ("approximately 2:15 PM")
- No date precision ("Tuesday" — which Tuesday?)
- No physical description of the cyclist
- No description of the woman who called the ambulance
- No physical description detail for the driver beyond "tall, dark jacket"
- The witness's own identity/position isn't captured
- Direction of travel for the cyclist before the collision
- Lighting/weather conditions not mentioned

---

## Technical Constraints

- **Sovereign stack**: Same as existing modules — ChatOllama with qwen2.5:7b, Neo4j at bolt://localhost:7687
- **No external APIs**: Everything runs locally
- **Human-in-the-loop**: Uses LangGraph's interrupt mechanism for the interview phase
- **Provenance**: Every fact in the graph carries a `source` property linking back to the original text or interview round that produced it
- **Idempotent re-ingestion**: Running the same statement through ingest twice should not duplicate the graph

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| qwen2.5:7b produces poor entity extraction | High | Schema-guided extraction with few-shot examples. Validation node checks extracted JSON. Fallback to simpler extraction if structured output fails |
| Coreference resolution is unreliable | Medium | Use conservative merging — only merge when LLM confidence is high. Allow manual correction in interview phase |
| Gap analysis produces too many/too few gaps | Medium | Prioritisation system. Cap questions per round at 3-5. Allow user to skip |
| Interview loop doesn't terminate | Low | Hard cap at max_rounds (default 5). User can always say "I don't know" |
| Neo4j schema drift across interview rounds | Low | Use MERGE not CREATE. Entity resolution runs after every update |
| Generated questions are leading/biased | Medium | Prompt engineering: explicit "do not suggest answers" instruction. Review generated questions before presenting |

---

## References

- Zhu et al. — *LLMs for KG Construction and Reasoning* (arXiv:2305.13168, 2023)
- Edge et al. — *From Local to Global: A Graph RAG Approach* (arXiv:2404.16130, 2024)
- Dai et al. — *LLMs Can Better Understand KGs Than We Thought* (arXiv:2402.11541, 2024)
- Wen et al. — *MindMap: KG Prompting Sparks Graph of Thoughts* (arXiv:2308.09729, 2023)
- Dong — *Generations of Knowledge Graphs* (arXiv:2308.14217, PVLDB 2023)
- Baek et al. — *KAPING: Knowledge-Augmented LM Prompting for Zero-Shot KGQA* (arXiv:2306.04136, 2023)
- Jiang et al. — *On the Evolution of Knowledge Graphs* (arXiv:2310.04835, 2023)
- Luo et al. — *ChatKBQA: Generate-then-Retrieve for KBQA* (arXiv:2310.08975, ACL 2024)
