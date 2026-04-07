"""
Module 05: Graph RAG — Combining Graphs and Retrieval
=======================================================
Demonstrates: Entity extraction from text, knowledge graph construction,
hybrid retrieval (vector + graph), and a LangGraph agent that combines
both retrieval strategies.

This module ties together everything from Modules 03 (RAG) and 04 (Knowledge
Graphs) into a unified Graph RAG pipeline.

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
"""

import json
import re
import numpy as np
from typing import TypedDict
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# ─── Connections ──────────────────────────────────────────────────────────
llm = ChatOllama(model="qwen2.5:7b", temperature=0)
embeddings = OllamaEmbeddings(model="qwen2.5:7b")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: SOURCE DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════
# Unstructured text about a fictional company — no pre-existing graph.
# The system must discover entities and relationships from raw text.

SOURCE_DOCUMENTS = [
    {
        "id": "doc-1",
        "title": "Company Overview",
        "text": """NovaTech was founded in 2018 by Dr. Amara Osei in Austin, Texas.
The company specialises in high-performance computing hardware for AI workloads.
Dr. Osei serves as CEO and has grown the company to over 1,200 employees.
NovaTech's annual revenue reached $340 million in fiscal year 2024.""",
    },
    {
        "id": "doc-2",
        "title": "Engineering Team",
        "text": """The engineering division at NovaTech is led by Marcus Rivera,
VP of Engineering. Marcus oversees three teams: the Hardware Team led by
Yuki Tanaka, the Software Team led by Priya Sharma, and the AI Research
Team led by Dr. James Kim. The Hardware Team designed the Nova 7 workstation
and the NovaBook Pro laptop. The AI Research Team developed the NovaMind
prediction engine.""",
    },
    {
        "id": "doc-3",
        "title": "Product Lineup",
        "text": """NovaTech's flagship product is the Nova 7 workstation, priced at
$24,999. It features dual AMD EPYC processors and supports up to 4 NVIDIA
H100 GPUs. The NovaBook Pro is a portable workstation priced at $3,499 with
an NVIDIA RTX 5080 GPU. The NovaMind prediction engine is a software product
that runs on both the Nova 7 and NovaBook Pro platforms.""",
    },
    {
        "id": "doc-4",
        "title": "Current Projects",
        "text": """Project Atlas is NovaTech's most ambitious initiative, aiming to
build a next-generation AI training platform. The project is led by Marcus
Rivera with support from Dr. James Kim's AI Research Team. Project Atlas
uses the Nova 7 workstation for development. Project Beacon focuses on
edge computing and is led by Priya Sharma. Project Beacon uses the NovaBook
Pro for field testing.""",
    },
    {
        "id": "doc-5",
        "title": "Recent Developments",
        "text": """Dr. Amara Osei announced a strategic partnership between NovaTech
and GlobalChip Corp to develop custom AI accelerators. Yuki Tanaka's
Hardware Team will lead the integration effort. The partnership aims to
enhance the Nova 7 workstation with custom silicon by 2026. Dr. James Kim
presented NovaMind's latest results at the International AI Conference,
demonstrating a 40% improvement in prediction accuracy.""",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: ENTITY EXTRACTION WITH LLM
# ═══════════════════════════════════════════════════════════════════════════
# Schema-guided extraction: tell the LLM exactly what kinds of entities
# and relationships to look for.

EXTRACTION_PROMPT = """You are an entity extraction specialist. Extract entities and
relationships from the text below.

Target entity types:
  - Person (name, title)
  - Organisation (name)
  - Product (name, price if mentioned)
  - Project (name, description)
  - Team (name)

Target relationship types:
  - FOUNDED (Person → Organisation)
  - LEADS (Person → Team/Project)
  - WORKS_AT (Person → Organisation)
  - DESIGNED (Team → Product)
  - DEVELOPED (Team → Product)
  - USES (Project → Product)
  - PART_OF (Team → Organisation)
  - OVERSEES (Person → Team)
  - PARTNERS_WITH (Organisation → Organisation)

Return a JSON object with this structure:
{
    "entities": [
        {"name": "...", "type": "Person", "properties": {"title": "..."}}
    ],
    "relationships": [
        {"from": "entity name", "to": "entity name", "type": "LEADS"}
    ]
}

Rules:
- Extract ONLY what is explicitly stated in the text
- Do NOT infer relationships that aren't stated
- Use consistent entity names (full names for people)
- Return ONLY the JSON, no explanation"""


def extract_entities(text: str) -> dict:
    """Extract entities and relationships from a text chunk."""
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=text),
    ]
    result = llm.invoke(messages)

    # Parse JSON from response
    content = result.content.strip()
    try:
        # Try direct parse
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON block
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return {"entities": [], "relationships": []}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: ENTITY RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════
# Merge duplicate entities that appear across different documents.
# Simple normalisation-based approach.

def normalise_name(name: str) -> str:
    """Normalise entity names for deduplication."""
    # Remove common prefixes
    name = re.sub(r"^(Dr\.\s*|Mr\.\s*|Ms\.\s*|Mrs\.\s*)", "", name.strip())
    return name.strip()


def resolve_entities(all_extractions: list[dict]) -> dict:
    """Merge entities across multiple extraction results."""
    entity_map = {}  # normalised_name → entity
    all_relationships = []

    for extraction in all_extractions:
        for entity in extraction.get("entities", []):
            key = normalise_name(entity["name"]).lower()
            if key in entity_map:
                # Merge properties
                existing = entity_map[key]
                for prop, val in entity.get("properties", {}).items():
                    if val and prop not in existing.get("properties", {}):
                        existing.setdefault("properties", {})[prop] = val
            else:
                entity_map[key] = entity

        for rel in extraction.get("relationships", []):
            all_relationships.append(rel)

    return {
        "entities": list(entity_map.values()),
        "relationships": all_relationships,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: LOAD INTO NEO4J
# ═══════════════════════════════════════════════════════════════════════════

def load_into_neo4j(resolved: dict):
    """Load extracted entities and relationships into Neo4j."""
    with driver.session() as session:
        # Clean slate
        session.run("MATCH (n) DETACH DELETE n")

        # Create entities
        for entity in resolved["entities"]:
            label = entity.get("type", "Entity")
            name = entity["name"]
            props = entity.get("properties", {})
            props["name"] = name

            # Build property string
            prop_parts = []
            for k, v in props.items():
                if isinstance(v, (int, float)):
                    prop_parts.append(f"{k}: {v}")
                else:
                    # Escape single quotes
                    safe_v = str(v).replace("'", "\\'")
                    prop_parts.append(f"{k}: '{safe_v}'")
            prop_str = ", ".join(prop_parts)

            cypher = f"MERGE (n:{label} {{name: '{name.replace(chr(39), chr(92)+chr(39))}'}}) SET n += {{{prop_str}}}"
            try:
                session.run(cypher)
            except Exception as e:
                print(f"    ⚠ Could not create entity {name}: {e}")

        # Create relationships
        created_rels = 0
        for rel in resolved["relationships"]:
            from_name = rel["from"].replace("'", "\\'")
            to_name = rel["to"].replace("'", "\\'")
            rel_type = rel["type"].replace(" ", "_").upper()

            cypher = f"""
                MATCH (a {{name: '{from_name}'}})
                MATCH (b {{name: '{to_name}'}})
                MERGE (a)-[:{rel_type}]->(b)
            """
            try:
                result = session.run(cypher)
                summary = result.consume()
                created_rels += summary.counters.relationships_created
            except Exception as e:
                print(f"    ⚠ Could not create rel {from_name}-[{rel_type}]->{to_name}: {e}")

    entity_count = len(resolved["entities"])
    print(f"  ✓ Loaded {entity_count} entities and {created_rels} relationships into Neo4j")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: VECTOR INDEX (in-memory)
# ═══════════════════════════════════════════════════════════════════════════
# Embed the source documents for vector retrieval alongside graph retrieval.

print("  Embedding source documents...")
DOC_TEXTS = [d["text"] for d in SOURCE_DOCUMENTS]
DOC_VECTORS = np.array(embeddings.embed_documents(DOC_TEXTS))
print(f"  Embedded {len(DOC_VECTORS)} documents, dimension: {DOC_VECTORS.shape[1]}")


def vector_retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Retrieve most similar documents via cosine similarity."""
    query_vec = np.array(embeddings.embed_query(query))
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = DOC_VECTORS / np.linalg.norm(DOC_VECTORS, axis=1, keepdims=True)
    similarities = doc_norms @ query_norm
    top_idx = np.argsort(similarities)[::-1][:top_k]
    return [
        {"doc": SOURCE_DOCUMENTS[i], "score": float(similarities[i])}
        for i in top_idx
    ]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: GRAPH RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

ENTITY_DETECT_PROMPT = """Identify named entities in the following question.
Return a JSON array of entity names. Only include proper nouns and specific names.
Return ONLY the JSON array, nothing else.

Example: ["Marcus Rivera", "Project Atlas", "Nova 7"]"""


def detect_question_entities(question: str) -> list[str]:
    """Use LLM to detect entities in the question."""
    messages = [
        SystemMessage(content=ENTITY_DETECT_PROMPT),
        HumanMessage(content=question),
    ]
    result = llm.invoke(messages)
    try:
        entities = json.loads(result.content.strip())
        return entities if isinstance(entities, list) else []
    except json.JSONDecodeError:
        return []


def graph_retrieve(entities: list[str], max_hops: int = 2) -> list[dict]:
    """Traverse the graph starting from detected entities."""
    results = []
    with driver.session() as session:
        for entity_name in entities:
            safe_name = entity_name.replace("'", "\\'")
            # Find the entity and its 1-2 hop neighbourhood
            cypher = f"""
                MATCH (start {{name: '{safe_name}'}})
                OPTIONAL MATCH path = (start)-[*1..{max_hops}]-(connected)
                WITH start, connected, 
                     [r IN relationships(path) | type(r)] AS rel_types,
                     [n IN nodes(path) | n.name] AS node_names
                RETURN start.name AS source, 
                       connected.name AS connected_entity,
                       labels(connected) AS connected_labels,
                       rel_types,
                       node_names
                LIMIT 20
            """
            try:
                records = session.run(cypher)
                for record in records:
                    results.append(dict(record))
            except Exception:
                pass

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: LANGGRAPH GRAPH-RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
# Combines vector retrieval and graph retrieval into a single agent.

class GraphRAGState(TypedDict):
    question: str
    detected_entities: list[str]
    vector_results: list[dict]
    graph_results: list[dict]
    merged_context: str
    answer: str
    steps: list[str]


def detect_entities_node(state: GraphRAGState) -> dict:
    """Detect entities in the question for graph lookup."""
    entities = detect_question_entities(state["question"])
    return {
        "detected_entities": entities,
        "steps": state.get("steps", []) + [f"detected entities: {entities}"],
    }


def vector_retrieve_node(state: GraphRAGState) -> dict:
    """Retrieve relevant documents via vector similarity."""
    results = vector_retrieve(state["question"], top_k=2)
    return {
        "vector_results": results,
        "steps": state["steps"] + [f"vector retrieved {len(results)} docs"],
    }


def graph_retrieve_node(state: GraphRAGState) -> dict:
    """Retrieve related entities via graph traversal."""
    results = graph_retrieve(state["detected_entities"])
    return {
        "graph_results": results,
        "steps": state["steps"] + [f"graph retrieved {len(results)} connections"],
    }


def merge_context_node(state: GraphRAGState) -> dict:
    """Merge vector and graph retrieval results into a unified context."""
    parts = []

    # Vector results: relevant text chunks
    parts.append("=== RELEVANT DOCUMENTS ===")
    for vr in state.get("vector_results", []):
        parts.append(f"[{vr['doc']['title']}] (similarity: {vr['score']:.3f})")
        parts.append(vr["doc"]["text"])
        parts.append("")

    # Graph results: structured relationships
    parts.append("=== KNOWLEDGE GRAPH CONNECTIONS ===")
    seen = set()
    for gr in state.get("graph_results", []):
        if gr.get("connected_entity"):
            key = (gr.get("source", ""), gr["connected_entity"])
            if key not in seen:
                seen.add(key)
                labels = ", ".join(gr.get("connected_labels", []))
                rels = " → ".join(gr.get("rel_types", []))
                path = " → ".join(gr.get("node_names", []))
                parts.append(f"  {path} (via {rels}) [{labels}]")

    merged = "\n".join(parts)
    return {
        "merged_context": merged,
        "steps": state["steps"] + ["merged vector + graph context"],
    }


GRAPHRAG_PROMPT = """You are an expert analyst. Answer the question using BOTH the
retrieved documents AND the knowledge graph connections provided.

When graph connections reveal relationships (e.g., who leads what, what uses what),
use those facts directly. When documents provide details (prices, dates, descriptions),
cite those.

Be thorough but concise. If information comes from the knowledge graph, mention
the relationship chain."""


def generate_node(state: GraphRAGState) -> dict:
    """Generate an answer from the merged context."""
    messages = [
        SystemMessage(content=GRAPHRAG_PROMPT),
        HumanMessage(content=f"Context:\n{state['merged_context']}\n\nQuestion: {state['question']}"),
    ]
    result = llm.invoke(messages)
    return {
        "answer": result.content,
        "steps": state["steps"] + ["generated answer"],
    }


def build_graphrag_pipeline():
    """Build the Graph RAG pipeline."""
    builder = StateGraph(GraphRAGState)

    builder.add_node("detect_entities", detect_entities_node)
    builder.add_node("vector_retrieve", vector_retrieve_node)
    builder.add_node("graph_retrieve", graph_retrieve_node)
    builder.add_node("merge_context", merge_context_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "detect_entities")
    # After entity detection, do both retrievals
    builder.add_edge("detect_entities", "vector_retrieve")
    builder.add_edge("vector_retrieve", "graph_retrieve")
    builder.add_edge("graph_retrieve", "merge_context")
    builder.add_edge("merge_context", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: RUN EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    # Direct lookup — both RAG and Graph RAG handle this
    "What is the Nova 7 workstation and how much does it cost?",
    # Multi-hop — Graph RAG excels here
    "Who leads the team that designed the Nova 7?",
    # Cross-document — requires connecting info from multiple sources
    "What projects does Marcus Rivera's AI Research Team member lead?",
    # Entity-centric — need everything connected to one entity
    "Tell me everything about Project Atlas — who's involved, what products it uses, and who leads it.",
    # Relationship chain
    "What is the connection between Dr. Amara Osei and the Nova 7 workstation?",
]


def main():
    print("=" * 64)
    print("  Module 05: Graph RAG")
    print("  Entity Extraction → KG Construction → Hybrid Retrieval")
    print("=" * 64)

    # ── Phase 1: Extract and build the knowledge graph ──
    print("\n  Phase 1: Extracting entities from source documents...")
    all_extractions = []
    for doc in SOURCE_DOCUMENTS:
        print(f"    Processing: {doc['title']}...")
        extraction = extract_entities(doc["text"])
        entity_count = len(extraction.get("entities", []))
        rel_count = len(extraction.get("relationships", []))
        print(f"      Found {entity_count} entities, {rel_count} relationships")
        all_extractions.append(extraction)

    # ── Phase 2: Resolve and load ──
    print("\n  Phase 2: Resolving entities and loading into Neo4j...")
    resolved = resolve_entities(all_extractions)
    print(f"    Resolved to {len(resolved['entities'])} unique entities")
    print(f"    Total relationships: {len(resolved['relationships'])}")
    load_into_neo4j(resolved)

    # ── Phase 3: Query with Graph RAG ──
    pipeline = build_graphrag_pipeline()

    print(f"\n{'=' * 64}")
    print("  Phase 3: Graph RAG Queries")
    print("  (Hybrid: Vector Similarity + Graph Traversal)")
    print("=" * 64)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'━' * 64}")
        print(f"  Question {i}: {question}")
        print(f"{'━' * 64}")

        config = {"configurable": {"thread_id": f"graphrag-{i}"}}
        result = pipeline.invoke(
            {"question": question, "steps": []},
            config,
        )

        print(f"\n  Detected entities: {result.get('detected_entities', [])}")

        print(f"\n  Vector results:")
        for vr in result.get("vector_results", []):
            print(f"    [{vr['doc']['title']}] score={vr['score']:.3f}")

        print(f"\n  Graph connections: {len(result.get('graph_results', []))} paths found")
        for gr in result.get("graph_results", [])[:5]:
            if gr.get("connected_entity"):
                path = " → ".join(gr.get("node_names", []))
                print(f"    {path}")

        print(f"\n  Answer: {result['answer']}")
        print(f"\n  Pipeline: {result['steps']}")

    # ── Summary ──
    print(f"\n{'=' * 64}")
    print("  Key observations:")
    print("  • Entity extraction turns unstructured text into a queryable graph")
    print("  • Entity resolution merges duplicates across documents")
    print("  • Vector search finds relevant TEXT, graph traversal finds STRUCTURE")
    print("  • Multi-hop questions benefit most from graph augmentation")
    print("  • The merged context gives the LLM BOTH similarity and relationships")
    print("  • This is the full pipeline: text → entities → graph → hybrid retrieval")
    print("=" * 64)

    driver.close()


if __name__ == "__main__":
    main()
