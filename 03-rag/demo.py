"""
Module 03: Retrieval-Augmented Generation (RAG)
=================================================
Demonstrates: Document chunking, embedding with Ollama, cosine similarity
retrieval, context injection, and a self-corrective RAG pipeline in LangGraph.

Uses a fictional company knowledge base to show how RAG grounds LLM answers
in real documents.
"""

import json
import numpy as np
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# ─── LLM and Embeddings ──────────────────────────────────────────────────
llm = ChatOllama(model="qwen2.5:7b", temperature=0)
embeddings = OllamaEmbeddings(model="qwen2.5:7b")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: THE KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════
# A small corpus about a fictional company. In production this would be
# thousands of documents from databases, wikis, PDFs, etc.

DOCUMENTS = [
    {
        "id": "policy-refund",
        "title": "Refund Policy",
        "content": """NovaTech offers a 30-day refund policy on all hardware products.
Software licenses are non-refundable once activated. To initiate a refund,
customers must contact support with their order number and reason for return.
Refunds are processed within 5-7 business days after the returned item is
received and inspected. Shipping costs for returns are the customer's
responsibility unless the item was defective or incorrectly shipped.""",
    },
    {
        "id": "policy-shipping",
        "title": "Shipping Information",
        "content": """NovaTech ships to all 50 US states and 12 international markets.
Standard shipping takes 5-7 business days domestically and 10-14 days
internationally. Express shipping (2-day) is available for an additional
$15.99. Free shipping is offered on orders over $100. All orders include
tracking information sent via email within 24 hours of shipment.""",
    },
    {
        "id": "product-nova7",
        "title": "Nova 7 Workstation",
        "content": """The Nova 7 is NovaTech's flagship workstation, designed for
AI and machine learning workloads. It features dual AMD EPYC 9654 processors
(192 cores total), 512GB DDR5 ECC RAM, and supports up to 4 NVIDIA H100 GPUs.
Storage options include NVMe RAID arrays up to 30TB. The Nova 7 starts at
$24,999 and is available in rack-mount and tower configurations. Power
consumption is rated at 2.8kW under full load.""",
    },
    {
        "id": "product-novabook",
        "title": "NovaBook Pro Laptop",
        "content": """The NovaBook Pro is a portable developer workstation with a
16-inch 4K display, Intel Core Ultra 9 processor, 64GB LPDDR5X RAM, and
NVIDIA RTX 5080 mobile GPU. Battery life is rated at 8 hours for development
workloads. It weighs 2.1kg and includes Thunderbolt 5 connectivity. The
NovaBook Pro is priced at $3,499 and comes with a 2-year warranty.""",
    },
    {
        "id": "support-hours",
        "title": "Support Hours and Contact",
        "content": """NovaTech technical support is available Monday through Friday,
8 AM to 8 PM Eastern Time. Premium support customers have access to 24/7
phone support. All customers can submit tickets through the online portal
at any time. Average response time for standard tickets is 4 hours during
business hours. Critical hardware issues receive priority escalation with
a 1-hour response commitment for premium customers.""",
    },
    {
        "id": "company-history",
        "title": "About NovaTech",
        "content": """NovaTech was founded in 2018 by Dr. Amara Osei and James
Rodriguez in Austin, Texas. The company started as a custom PC builder
for data scientists and grew into a full hardware manufacturer. NovaTech
went public in 2023 and currently employs 1,200 people across offices
in Austin, Portland, and Dublin. Revenue for fiscal year 2024 was
$340 million.""",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: CHUNKING
# ═══════════════════════════════════════════════════════════════════════════
# Our documents are small enough to be chunks themselves.  In production
# you'd split large documents into overlapping paragraphs.  Here we
# demonstrate the concept with a simple sentence-boundary chunker.

def chunk_document(doc: dict, max_sentences: int = 4) -> list[dict]:
    """Split a document into chunks of at most max_sentences sentences."""
    sentences = [s.strip() for s in doc["content"].replace("\n", " ").split(".") if s.strip()]
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        chunk_text = ". ".join(chunk_sentences) + "."
        chunks.append({
            "id": f"{doc['id']}-chunk-{i // max_sentences}",
            "source": doc["title"],
            "text": chunk_text,
        })
    return chunks


# Build the chunk corpus
ALL_CHUNKS = []
for doc in DOCUMENTS:
    ALL_CHUNKS.extend(chunk_document(doc))

print(f"  Chunked {len(DOCUMENTS)} documents into {len(ALL_CHUNKS)} chunks")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════
# Convert each chunk into a numerical vector.  We also embed query strings
# at retrieval time using the same model.

print("  Embedding chunks (this may take a moment)...")

CHUNK_TEXTS = [c["text"] for c in ALL_CHUNKS]
CHUNK_VECTORS = np.array(embeddings.embed_documents(CHUNK_TEXTS))

print(f"  Embedded {len(CHUNK_VECTORS)} chunks, vector dimension: {CHUNK_VECTORS.shape[1]}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: VECTOR RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════
# Simple cosine similarity search — no external vector store needed.

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Embed a query and return the top-k most similar chunks."""
    query_vector = np.array(embeddings.embed_query(query))
    similarities = cosine_similarity(query_vector, CHUNK_VECTORS)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "chunk": ALL_CHUNKS[idx],
            "score": float(similarities[idx]),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: LANGGRAPH RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
# A 3-node self-corrective RAG pipeline:
#   retrieve_docs → evaluate_relevance → generate_answer
#                        ↓ (if poor)
#                   rephrase_query → retrieve_docs (retry once)

class RAGState(TypedDict):
    question: str              # user's question
    retrieved_chunks: list     # chunks from retrieval
    relevance: str             # "sufficient" or "insufficient"
    rephrased: str             # rephrased question (if needed)
    answer: str                # final answer
    attempt: int               # retrieval attempt number
    steps: list[str]           # audit trail


def retrieve_docs(state: RAGState) -> dict:
    """Retrieve relevant chunks for the current question."""
    # Use rephrased question on retry, original otherwise
    query = state.get("rephrased") or state["question"]
    results = retrieve(query, top_k=3)
    return {
        "retrieved_chunks": results,
        "attempt": state.get("attempt", 0) + 1,
        "steps": state.get("steps", []) + [
            f"retrieved {len(results)} chunks (attempt {state.get('attempt', 0) + 1})"
        ],
    }


RELEVANCE_PROMPT = """You are a relevance evaluator. Given a question and retrieved
context chunks, determine if the chunks contain enough information to answer the question.

Respond with EXACTLY one word: sufficient or insufficient"""


def evaluate_relevance(state: RAGState) -> dict:
    """Evaluate whether retrieved chunks are relevant to the question."""
    chunks_text = "\n---\n".join(
        f"[{r['chunk']['source']}] {r['chunk']['text']}" for r in state["retrieved_chunks"]
    )
    messages = [
        SystemMessage(content=RELEVANCE_PROMPT),
        HumanMessage(content=f"Question: {state['question']}\n\nRetrieved context:\n{chunks_text}"),
    ]
    result = llm.invoke(messages)
    relevance = "sufficient" if "sufficient" in result.content.strip().lower() else "insufficient"
    return {
        "relevance": relevance,
        "steps": state["steps"] + [f"evaluated relevance: {relevance}"],
    }


REPHRASE_PROMPT = """You are a query optimiser. The original question did not retrieve
useful documents. Rephrase the question to be more specific and use different keywords
that might match relevant documents better.

Return ONLY the rephrased question, nothing else."""


def rephrase_query(state: RAGState) -> dict:
    """Rephrase the question to improve retrieval on retry."""
    messages = [
        SystemMessage(content=REPHRASE_PROMPT),
        HumanMessage(content=f"Original question: {state['question']}"),
    ]
    result = llm.invoke(messages)
    return {
        "rephrased": result.content.strip(),
        "steps": state["steps"] + [f"rephrased to: {result.content.strip()[:80]}"],
    }


GENERATE_PROMPT = """You are a helpful customer support assistant for NovaTech.
Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so honestly.
Be concise and direct. Cite which document the information comes from."""


def generate_answer(state: RAGState) -> dict:
    """Generate an answer grounded in the retrieved context."""
    chunks_text = "\n---\n".join(
        f"[Source: {r['chunk']['source']}] {r['chunk']['text']}"
        for r in state["retrieved_chunks"]
    )
    messages = [
        SystemMessage(content=GENERATE_PROMPT),
        HumanMessage(content=f"Context:\n{chunks_text}\n\nQuestion: {state['question']}"),
    ]
    result = llm.invoke(messages)
    return {
        "answer": result.content,
        "steps": state["steps"] + ["generated answer"],
    }


def route_after_evaluation(state: RAGState) -> str:
    """Route based on relevance evaluation: retry once or generate."""
    if state["relevance"] == "insufficient" and state["attempt"] < 2:
        return "rephrase_query"
    return "generate_answer"


def build_rag_pipeline():
    """Build the self-corrective RAG pipeline."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("evaluate_relevance", evaluate_relevance)
    builder.add_node("rephrase_query", rephrase_query)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "retrieve_docs")
    builder.add_edge("retrieve_docs", "evaluate_relevance")

    # Conditional: either rephrase and retry, or generate
    builder.add_conditional_edges(
        "evaluate_relevance",
        route_after_evaluation,
        {"rephrase_query": "rephrase_query", "generate_answer": "generate_answer"},
    )
    builder.add_edge("rephrase_query", "retrieve_docs")  # loop back
    builder.add_edge("generate_answer", END)

    return builder.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: RUN THE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    "What is NovaTech's refund policy for hardware?",
    "How much does the Nova 7 workstation cost?",
    "What are the support hours?",
    "Does NovaTech ship internationally and how long does it take?",
    "Who founded NovaTech and when?",
]


def main():
    pipeline = build_rag_pipeline()

    print("\n" + "=" * 64)
    print("  Module 03: Retrieval-Augmented Generation")
    print("  Self-Corrective RAG Pipeline with LangGraph")
    print("=" * 64)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'━' * 64}")
        print(f"  Question {i}: {question}")
        print(f"{'━' * 64}")

        config = {"configurable": {"thread_id": f"rag-{i}"}}
        result = pipeline.invoke(
            {"question": question, "steps": [], "attempt": 0},
            config,
        )

        # Show retrieved sources
        print(f"\n  Retrieved chunks:")
        for r in result["retrieved_chunks"]:
            print(f"    [{r['chunk']['source']}] score={r['score']:.3f}")
            print(f"      {r['chunk']['text'][:100]}...")

        print(f"\n  Relevance: {result['relevance']}")
        print(f"  Retrieval attempts: {result['attempt']}")
        print(f"\n  Answer: {result['answer']}")
        print(f"\n  Pipeline steps: {result['steps']}")

    # ── Demonstrate retrieval WITHOUT generation (inspection) ──
    print(f"\n{'=' * 64}")
    print("  BONUS: Raw retrieval inspection")
    print("=" * 64)
    test_query = "laptop battery life"
    results = retrieve(test_query, top_k=3)
    print(f"\n  Query: '{test_query}'")
    for r in results:
        print(f"    score={r['score']:.3f} [{r['chunk']['source']}] {r['chunk']['text'][:80]}...")

    print(f"\n{'=' * 64}")
    print("  Key observations:")
    print("  • Chunks are embedded ONCE, queries embedded at retrieval time")
    print("  • Cosine similarity finds semantically similar text, not keyword matches")
    print("  • The evaluate_relevance node enables SELF-CORRECTION")
    print("  • 'Answer ONLY from context' prevents hallucination bleed")
    print("  • Small, focused chunks outperform dumping entire documents")
    print("=" * 64)


if __name__ == "__main__":
    main()
