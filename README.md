# Agentic AI and Knowledge Graphs

A nine-module learning project exploring agentic AI systems, from fundamentals through to event digital twins and cloud deployment. Each module contains a concepts guide (README.md) and a runnable demonstration (demo.py).

## Project Structure

| Module | Topic | Key Concepts |
|--------|-------|-------------|
| [01-agentic-fundamentals](01-agentic-fundamentals/) | Agentic AI Fundamentals | State, graphs, tools, conditional routing, checkpointing |
| [02-prompt-engineering](02-prompt-engineering/) | Prompt Engineering for Agents | System prompts, decomposition, structured output, temperature |
| [03-rag](03-rag/) | Retrieval-Augmented Generation | Chunking, embedding, vector search, self-corrective retrieval |
| [04-knowledge-graphs](04-knowledge-graphs/) | Knowledge Graphs | Neo4j, Cypher, multi-hop traversal, NL-to-Cypher |
| [05-graph-rag](05-graph-rag/) | Graph RAG | Entity extraction, KG construction, hybrid retrieval |
| [06-ontologies](06-ontologies/) | Ontology Comparison | Pluggable ontologies, Schema.org vs SEM vs BFO/CCO, SHACL shapes |
| [07-graph-quality](07-graph-quality/) | Graph Quality Assessment | Multi-dimensional probes, Cypher/LLM/SHACL validation, scoring |
| [08-event-digital-twin](08-event-digital-twin/) | Event Digital Twin | Single & multi-source witness graphs, interview, fine-tuning with quality feedback |
| [09-deployment](09-deployment/) | Deployment & Model Serving | Containers, cloud, model swapping, LoRA adapter serving |

## Getting Started

Each module includes a `docker-compose.yml` to start the required services:

```bash
cd <module-folder>
docker compose up -d          # start Ollama (+ Neo4j for modules 04-08)
../langgraph-test/langgraph-env/bin/python demo.py
```

**Stack**: LangGraph 0.6.11 · Ollama (qwen2.5:7b) · Neo4j Community (modules 04-08)

---

## Concepts Overview

This document covers the foundational ideas behind agentic AI systems and knowledge graphs — what they are, how they work, and how they fit together. Each section corresponds to one of the modules above.

---

## What makes AI "agentic"

A conventional LLM interaction is stateless: you send a prompt, you get a response, it's over. An **agentic** system is different in three ways:

1. **It maintains state across steps.** The system carries structured data (not just a message log) from one reasoning step to the next. That state might include what has been collected, what has been validated, what decisions have been made, and what remains to be done.

2. **It makes decisions about its own control flow.** Rather than following a fixed sequence, an agent inspects its current state and decides which step to take next. It might call a tool, ask a human for input, retry a failed validation, or skip ahead — based on conditions, not a script.

3. **It uses tools.** An agent can reach outside the LLM to query a database, call an API, read a file, run a calculation, or write to a system of record. The LLM decides *when* to use a tool and *what arguments* to pass, then incorporates the result into its next reasoning step.

The combination of persistent state, conditional branching, and tool use is what separates an agent from a chatbot. A chatbot generates text. An agent *does things*.

### The graph metaphor

Most agentic frameworks model the workflow as a **directed graph**. Each node is a processing step (call the LLM, run a tool, validate data, ask a human). Each edge defines what can happen next. Some edges are unconditional ("after extraction, always validate"). Others are **conditional** — a routing function inspects the current state and decides which node to activate.

This is powerful because it lets you express complex logic — loops, branches, fallbacks, parallel paths — in a structure the framework can checkpoint, resume, and inspect. The graph is the program.

### State: the backbone of agentic systems

In a simple chatbot, state is a list of messages. In an agentic system, state is a **structured object with multiple typed fields**: the user's query, extracted entities, validation flags, collected form data, the current step in a multi-turn flow, error counts, and so on.

This matters because conditional edges route based on state. If `email` is empty, route to the collection node. If `validation_errors` is non-empty, route to the retry node. If `approval_status` is "rejected", route to the escalation node. The richer your state schema, the more sophisticated your control flow.

State also enables **memory**. By persisting state to a database (checkpointing), an agent can survive process restarts, resume interrupted conversations, and even "time-travel" back to earlier states for debugging. Every invocation gets a thread ID, and the checkpointer stores a snapshot after each node execution.

---

## Prompt engineering in agentic settings

Prompting an agent is fundamentally different from prompting a standalone LLM. You're not crafting a single prompt — you're designing a **system of prompts** that work together across multiple steps, each with a different purpose.

### System prompts as behavioural contracts

Each node in an agent graph typically has its own system prompt. A classification node might say: "You are a query classifier. Respond with exactly one of: greeting, technical, general." A response node might say: "You are a helpful assistant. Answer concisely in 2-3 sentences." A validation node might say: "Check whether the following extracted fields are complete and consistent. Return a JSON object with `is_valid` and `errors`."

The key insight is that **narrow, single-purpose prompts dramatically outperform broad, do-everything prompts**. By splitting responsibilities across nodes, each prompt can be specific and constrained, which reduces hallucination and improves reliability.

### Structured output and tool calling

Agents need LLMs to produce structured data, not free-form text. Two mechanisms achieve this:

- **Tool/function calling**: The LLM is told about available tools (with typed parameter schemas) and can choose to invoke one. The framework intercepts the tool call, executes it, and feeds the result back. This is how agents interact with external systems.

- **Structured output**: The LLM is instructed to respond in a specific format (typically JSON matching a schema). This is how agents extract typed data from unstructured input — pulling names, dates, categories, and confidence scores out of natural language.

The quality of tool calling and structured output varies significantly between models. Some models reliably produce valid JSON and well-formed function calls; others frequently produce malformed output that breaks the pipeline. This is a practical constraint when choosing which LLM to use in an agentic setting.

### Conversation flow control

In a multi-turn agent, conversation flow is **not** controlled by the LLM. The graph controls it. The LLM is a component that processes individual steps — it classifies, extracts, generates, validates — but the routing logic that decides what happens next is deterministic code.

This separation is deliberate. LLMs are unreliable at managing complex multi-step processes on their own. They lose track of what's been collected, skip steps, repeat themselves, or hallucinate completion. By putting the control flow in the graph and the intelligence in the nodes, you get the best of both: the LLM's language understanding combined with the graph's structural guarantees.

### Human-in-the-loop

Agents often need to pause and wait for human input. An interview flow collects answers one at a time. An approval workflow waits for a manager's sign-off. A tool-calling agent might need a human to confirm before executing a destructive action.

The mechanism is an **interrupt**: the graph pauses mid-execution, serialises its state, and waits. When the human responds, the graph resumes from exactly where it stopped with the new input incorporated into state. This requires checkpointing — without persistent state, there's nothing to resume from.

Four common human-in-the-loop patterns:
- **Approve/reject** — the agent proposes an action; a human approves or blocks it
- **Edit state** — a human corrects or enriches the agent's work before it continues
- **Review tool calls** — a human inspects what the agent wants to do before execution
- **Multi-turn conversation** — the agent collects information across multiple exchanges

---

## Retrieval-Augmented Generation (RAG)

RAG addresses a fundamental limitation of LLMs: they only know what was in their training data, and they can't tell you when they don't know something. RAG gives the LLM access to external knowledge at query time.

### How RAG works

The process has two phases:

**Indexing (offline):** Documents are split into chunks, each chunk is converted into a vector embedding (a high-dimensional numerical representation that captures semantic meaning), and the embeddings are stored in a vector database alongside the original text.

**Retrieval + Generation (at query time):**
1. The user's query is converted into an embedding using the same model
2. The vector database finds the chunks whose embeddings are most similar to the query embedding
3. The retrieved chunks are inserted into the LLM's prompt as context
4. The LLM generates a response grounded in the retrieved information

The result is an LLM that can answer questions about documents it was never trained on, with the ability to cite sources.

### Why RAG matters for agents

In an agentic setting, RAG becomes a **tool**. The agent decides when it needs external knowledge, formulates a search query, retrieves relevant documents, and incorporates the results into its reasoning. This is more powerful than naive RAG because the agent can:

- Reformulate queries if initial retrieval returns poor results
- Combine information from multiple retrieval calls
- Decide whether retrieved context is sufficient or whether it needs to search again
- Route to different knowledge sources depending on the question type

### Limitations of pure vector RAG

Vector similarity search finds text that is *semantically similar* to the query, but similarity isn't always what you need. If you ask "What companies has Alice worked for?", vector search might return chunks mentioning Alice, and chunks mentioning companies, but it can't reliably *connect* them. It doesn't understand relationships.

This is where knowledge graphs enter the picture.

---

## Knowledge graphs

A **knowledge graph** stores information as entities (nodes) and relationships (edges) rather than as flat text or rows in a table. The node `Alice` connects via an edge `WORKS_AT` to the node `Acme Corp`, with properties like `since: 2019` and `role: "engineer"`. Another edge might connect `Acme Corp` to `Python` via `USES`.

### Why graphs beat tables and documents for certain questions

Consider the question: "Which technologies are used by companies where Alice has worked?" In a relational database, this requires joining multiple tables with knowledge of the schema. In a document store, it requires hoping the answer appears verbatim in some paragraph. In a knowledge graph, it's a traversal: `Alice → WORKS_AT → Company → USES → Technology`. The structure of the data *is* the query path.

Graphs excel when:
- Relationships between entities matter as much as the entities themselves
- Questions require traversing connections across multiple hops
- The schema is heterogeneous or evolving (new entity and relationship types appear over time)
- You need to explain *why* an answer is correct (the path through the graph is the explanation)

### Cypher: querying graphs

Graph databases like Neo4j use **Cypher**, a declarative query language designed for pattern matching on graphs. A query looks like:

```
MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:USES]->(t:Technology)
WHERE p.name = "Alice"
RETURN t.name
```

This reads naturally: "Find a person named Alice, follow WORKS_AT edges to companies, follow USES edges to technologies, return the technology names." The pattern-matching syntax makes multi-hop relationship queries intuitive.

### Combining knowledge graphs with LLMs

There are two primary integration patterns:

**Natural language to Cypher:** The LLM translates a user's question into a Cypher query, executes it against the graph database, and uses the results to formulate an answer. This requires the LLM to understand the graph's schema (what node labels and relationship types exist) and produce syntactically valid Cypher. The graph schema is typically injected into the system prompt.

**Graph-enhanced RAG (GraphRAG):** Instead of (or in addition to) vector similarity search, the retrieval step traverses the knowledge graph to find relevant entities and their relationships. This produces structured context ("Alice works at Acme Corp since 2019; Acme Corp uses Python and Kubernetes") rather than raw text chunks, which the LLM can reason about more reliably.

GraphRAG solves the key weakness of pure vector RAG: it can answer relationship questions that require connecting multiple entities, because the connections are explicit in the graph rather than implicit in text.

---

## Learning knowledge graphs from data

A knowledge graph is only useful if it contains relevant, accurate knowledge. Manually constructing a graph is tedious and doesn't scale. **Automated knowledge graph construction** uses LLMs to extract entities and relationships from unstructured text.

### Entity and relationship extraction

The core task is: given a document, identify the entities (people, organisations, technologies, locations, events) and the relationships between them (works at, acquired, founded, located in, depends on). An LLM does this by reading the text and producing structured output — typically a list of `(subject, predicate, object)` triples.

For example, from the sentence "Alice joined Acme Corp in 2019 as a senior engineer", the LLM extracts:
- Entity: `Alice` (Person)
- Entity: `Acme Corp` (Organisation)
- Relationship: `Alice → JOINED → Acme Corp` with properties `{year: 2019, role: "senior engineer"}`

### Schema-guided extraction

Better results come from **constraining the extraction** with a predefined schema: a list of allowed entity types and relationship types. Rather than letting the LLM invent arbitrary labels (which leads to inconsistency — "works at", "employed by", "works for" all meaning the same thing), you specify that the only valid entity types are `Person`, `Organisation`, `Technology` and the only valid relationships are `WORKS_AT`, `DEVELOPED`, `USES`. The LLM maps the text to your schema.

### Entity resolution

A persistent challenge is **entity resolution** — recognising that "Acme Corp", "Acme Corporation", "ACME", and "the company" all refer to the same entity. Without resolution, the knowledge graph fractures into disconnected nodes that should be one. LLMs can help here too, by comparing candidate entities and judging whether they're the same, but it remains an imperfect process.

### Embeddings and the role of vector representations

Embedding models convert text into dense numerical vectors where semantically similar texts are close together in the vector space. In the knowledge graph context, embeddings serve two purposes:

1. **Node embeddings**: Each entity's description or associated text is embedded, enabling similarity search over the graph's content ("find entities similar to X")
2. **Retrieval**: When a user asks a question, their query is embedded and compared against node embeddings to find relevant starting points for graph traversal

This bridges the gap between unstructured natural-language queries and structured graph traversal.

### Iterative refinement

Knowledge graph construction isn't a one-shot process. The typical workflow is:

1. Extract entities and relationships from a corpus using an LLM
2. Load the triples into the graph database
3. Inspect the results — look for inconsistencies, missing connections, incorrect relationships
4. Refine the extraction prompts, adjust the schema, add entity resolution rules
5. Re-extract and merge

The graph grows and improves over time as more documents are processed and the extraction pipeline is refined. This is fundamentally different from training an LLM — you can see exactly what the system knows, inspect individual facts, and correct errors directly.

---

## Putting it all together: agentic RAG over knowledge graphs

The most powerful architecture combines all these concepts:

1. **A knowledge graph** stores structured facts extracted from documents
2. **An embedding layer** enables semantic search over graph entities
3. **An agent** orchestrates the interaction — it receives a user query, decides whether to search the graph, formulates Cypher queries or vector searches, retrieves relevant subgraphs, and generates a grounded response
4. **State management** tracks the conversation, remembers what's been retrieved, and enables multi-turn refinement
5. **Human-in-the-loop** allows a human to correct, guide, or approve the agent's actions

The agent doesn't just look things up — it **reasons about what it knows and what it still needs to find out**. It can chain multiple graph queries, combine graph results with vector search, ask clarifying questions when the query is ambiguous, and explain its reasoning by showing the graph paths it traversed.

This is the frontier of applied agentic AI: systems that are grounded in verifiable knowledge structures, that can explain their reasoning, and that keep a human in the loop for oversight.