# Module 04: Knowledge Graphs

## Why Graphs

Relational databases store data in tables. Documents store data in blobs of text. Both work well for their intended purpose, but neither naturally represents *relationships* between entities.

Consider: "Alice manages Bob, who works on Project Atlas, which uses the Nova 7 workstation, which was designed by the Hardware Division, which Alice also oversees."

In a relational database, this requires joining 4-5 tables. In a document store, this information might be scattered across separate documents with no explicit connection. In a **knowledge graph**, this is a direct traversal:

```
Alice -[MANAGES]→ Bob -[WORKS_ON]→ Project Atlas -[USES]→ Nova 7 -[DESIGNED_BY]→ Hardware Division -[OVERSEEN_BY]→ Alice
```

The relationships *are* the data. You don't compute them at query time — they're stored explicitly, and traversing them is the database's core operation.

---

## The Property Graph Model

Neo4j uses the **property graph model**, which has three building blocks:

### Nodes
Entities — things that exist. Each node has:
- One or more **labels** (like types): `Person`, `Project`, `Product`
- Zero or more **properties** (key-value pairs): `{name: "Alice", title: "VP Engineering"}`

### Relationships
Connections between nodes. Each relationship has:
- A **type**: `MANAGES`, `WORKS_ON`, `USES`
- A **direction**: from one node to another (though you can traverse in either direction when querying)
- Zero or more **properties**: `{since: "2023-01", role: "lead"}`

### Properties
Key-value data attached to nodes or relationships. Values can be strings, numbers, booleans, or arrays of these.

This model maps naturally to how humans think about domains. When someone describes their organisation, they talk about *people* who *manage* other *people* who *work on* *projects*. That sentence structure IS the graph structure.

---

## Cypher: The Query Language

Cypher is Neo4j's declarative query language. It uses ASCII art to represent graph patterns:

```cypher
// Find a node
MATCH (p:Person {name: "Alice"}) RETURN p

// Find a relationship
MATCH (a:Person)-[:MANAGES]->(b:Person) RETURN a.name, b.name

// Multi-hop: find who works on projects managed by Alice's reports
MATCH (alice:Person {name: "Alice"})-[:MANAGES]->(report)-[:WORKS_ON]->(project)
RETURN report.name, project.name

// Create nodes and relationships
CREATE (p:Person {name: "Alice", title: "VP Engineering"})
CREATE (alice)-[:MANAGES {since: "2023"}]->(bob)
```

The visual pattern-matching syntax is Cypher's great strength. You draw the pattern you're looking for, and the database finds all matches.

### Key Cypher Patterns

| Pattern | Meaning |
|---|---|
| `(n)` | Any node |
| `(n:Person)` | Node with label Person |
| `(n {name: "X"})` | Node with property name = "X" |
| `-[r:MANAGES]->` | Outgoing relationship of type MANAGES |
| `<-[:WORKS_ON]-` | Incoming relationship |
| `-[:KNOWS*1..3]-` | Variable-length path (1 to 3 hops) |

---

## Why Knowledge Graphs Matter for AI

### 1. Multi-Hop Reasoning
LLMs struggle with questions that require following chains of relationships. "Who manages the person who designed the product used by Project Atlas?" requires 4 hops. A knowledge graph answers this in milliseconds with a Cypher query. An LLM trying to reason through this from unstructured text will often hallucinate connections.

### 2. Explainability
When a graph database returns an answer, it can also return the *path* — the exact chain of nodes and relationships that connects the question to the answer. This is auditable, verifiable, and explainable in a way that "the LLM said so" is not.

### 3. Consistency
Graphs enforce structure. If you define that `MANAGES` connects `Person` to `Person`, you can't accidentally create a `MANAGES` relationship between a `Person` and a `Product`. This structural consistency prevents the kind of contradictions and confabulations that plague pure LLM approaches.

### 4. Dynamic Updates
Adding new facts to a graph is an `O(1)` operation — create a node, create a relationship. You don't need to re-embed documents, rebuild indices, or retrain anything. The graph is immediately queryable with the new information.

---

## LLM + Knowledge Graph Integration Patterns

### Natural Language to Cypher
The most direct pattern: an LLM translates a natural language question into a Cypher query, which is executed against the graph, and the results are fed back to the LLM for a natural language answer.

```
User question → LLM (translate to Cypher) → Execute query → LLM (format answer) → Response
```

This works well when:
- The graph schema is well-defined and communicated to the LLM
- Questions map cleanly to graph traversals
- The LLM has enough context about the schema to generate valid Cypher

### Graph-Guided Retrieval
Instead of (or in addition to) vector similarity, use the graph to find relevant information by traversing relationships from known entities. "Tell me about Alice's team" → find Alice → traverse MANAGES → collect all report nodes → use that structured data as context for generation.

### Schema as Prompt Context
Providing the graph schema (node labels, relationship types, property names) to the LLM as system prompt context helps it understand what questions the graph can answer and how to formulate queries.

---

## Graph vs. Relational vs. Document

| Aspect | Relational DB | Document Store | Knowledge Graph |
|---|---|---|---|
| **Relationships** | Computed (JOINs) | Implicit (in text) | Stored explicitly |
| **Multi-hop queries** | Expensive (N JOINs) | Not supported | Native, fast |
| **Schema flexibility** | Rigid | Schema-free | Flexible labels |
| **Aggregation** | Excellent | Good | Good |
| **Full-text search** | Add-on | Native | Add-on |
| **Explainability** | Query plan | None | Path traversal |

Knowledge graphs don't replace other databases. They excel specifically at relationship-heavy, traversal-heavy workloads — exactly the kind of reasoning that LLMs struggle with.

---

## Key Concepts Summary

| Concept | What It Does |
|---|---|
| **Node** | Represents an entity (person, product, project) |
| **Relationship** | Named, directed connection between nodes |
| **Property** | Key-value data on nodes or relationships |
| **Label** | Type tag on a node (Person, Product) |
| **Cypher** | Pattern-matching query language for graphs |
| **Multi-hop traversal** | Following chains of relationships across entities |
| **NL-to-Cypher** | LLM translates natural language to graph queries |
