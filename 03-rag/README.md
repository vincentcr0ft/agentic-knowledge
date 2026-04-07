# Module 03: Retrieval-Augmented Generation (RAG)

## What RAG Solves

Large language models know what they were trained on — and nothing else. They cannot access your company's internal docs, yesterday's meeting notes, or the latest API changelog. When asked about information outside their training data, they either admit ignorance or, worse, confabulate a plausible-sounding answer.

**Retrieval-Augmented Generation** bridges this gap. Instead of hoping the model knows the answer, you *retrieve* relevant documents first, then *generate* an answer grounded in that retrieved context.

The pattern:  
`Question → Retrieve relevant chunks → Inject into prompt → Generate grounded answer`

This is not fine-tuning. The model's weights never change. You are changing what the model *sees*, not what it *knows*.

---

## The RAG Pipeline, Step by Step

### 1. Chunking

Raw documents are too long to fit in a prompt. Even if they did fit, models struggle with long contexts — the "lost in the middle" problem is well documented. Relevant information buried in the middle of a 50-page document is often ignored.

**Chunking** splits documents into smaller, digestible pieces. Each chunk should be:
- **Self-contained**: makes sense without reading the surrounding chunks
- **Focused**: covers one topic or one idea
- **Appropriately sized**: large enough for context, small enough for relevance (typically 200-1000 tokens)

Chunking strategies:
- **Fixed-size**: split every N characters/tokens. Simple but crude — often cuts mid-sentence
- **Sentence-based**: split on sentence boundaries. Better coherence
- **Paragraph-based**: split on paragraph boundaries. Best coherence for well-structured text
- **Semantic**: split when the topic changes (using embeddings to detect shifts). Most sophisticated but computationally expensive
- **Overlapping**: chunks share some text at boundaries to avoid losing context at split points

The chunking strategy you choose directly impacts retrieval quality. Bad chunks → bad retrieval → bad answers.

### 2. Embedding

To find which chunks are relevant to a question, you need a way to measure **semantic similarity** — not just keyword matching. The sentence "What is our refund policy?" should match a chunk about "returns and exchanges" even though they share no words.

**Embeddings** convert text into dense numerical vectors (arrays of floating-point numbers, typically 384-4096 dimensions). Texts with similar meanings produce vectors that are close together in this high-dimensional space.

Key properties:
- Semantic similarity maps to geometric proximity (cosine similarity, dot product)
- The same model must embed both the chunks and the query — mixing models produces incompatible vector spaces
- Embedding models are separate from generation models. They are smaller, faster, and trained specifically for this task

In our sovereign stack, Ollama can produce embeddings from any model. We use `qwen2.5:7b` for both embedding and generation. Dedicated embedding models (like `nomic-embed-text`) are better for production but require a separate download.

### 3. Vector Storage and Retrieval

Once chunks are embedded, the vectors are stored in a **vector store** — a database optimised for nearest-neighbour search in high-dimensional space. Given a query vector, it efficiently finds the K most similar chunk vectors.

For production: Chroma, FAISS, Pinecone, Weaviate, pgvector.  
For learning: a Python list and `numpy` cosine similarity is perfectly sufficient.

The retrieval step:
1. Embed the user's question using the same embedding model
2. Compute similarity between the question vector and all chunk vectors
3. Return the top-K most similar chunks
4. Feed those chunks into the generation prompt as context

### 4. Generation with Retrieved Context

The retrieved chunks are injected into the prompt as context:

```
System: You are a helpful assistant. Answer the question based ONLY on the
provided context. If the context doesn't contain the answer, say so.

Context:
{retrieved_chunks}

Question: {user_question}
```

The constraint "based ONLY on the provided context" is critical. Without it, the model will happily blend retrieved facts with its own training data, producing answers that are partially grounded and partially hallucinated — the worst outcome because it's hard to tell which parts are real.

---

## The "Lost in the Middle" Problem

Research (Liu et al., 2023) showed that when relevant information is placed in the middle of a long context, models perform significantly worse than when it appears at the beginning or end. This is another reason chunking matters — you want to retrieve *only* the relevant chunks, not dump an entire document into the prompt.

RAG with 3-5 highly relevant chunks consistently outperforms RAG with 20 loosely relevant chunks.

---

## Where RAG Falls Short

RAG has real limitations that motivate the knowledge graph approaches in later modules:

- **No relational reasoning**: "Who reports to the manager of the person who wrote this document?" requires following relationships across entities. Vector similarity doesn't capture relationships — it captures topical similarity
- **No aggregation**: "How many products launched this quarter?" requires counting across multiple documents, not retrieving the most similar one
- **Chunk boundary problems**: If the answer spans two chunks that didn't get retrieved together, RAG misses it
- **Contradictory sources**: When multiple chunks contain conflicting information, RAG has no mechanism to resolve conflicts or determine which source is authoritative
- **No explanation of reasoning**: RAG retrieves and generates, but cannot show you the chain of facts that led to an answer

These limitations are not bugs — they're inherent to the "find similar text" paradigm. Knowledge graphs address them by encoding *structure* and *relationships*, not just similarity.

---

## RAG in a LangGraph Agent

In an agentic setting, the RAG pipeline becomes *nodes in a graph*:

```
START → retrieve → evaluate_relevance → generate → END
                        ↓
                   (if irrelevant)
                        ↓
                   rephrase_query → retrieve  (loop)
```

This gives you capabilities that a simple RAG pipeline lacks:
- **Self-correction**: the agent can evaluate whether retrieved chunks actually answer the question and retry with a rephrased query
- **Multi-step retrieval**: break a complex question into sub-questions, retrieve for each, then synthesise
- **Source evaluation**: a node can assess whether the retrieved context is sufficient before generating

The demo builds exactly this pattern.

---

## Key Concepts Summary

| Concept | What It Does |
|---|---|
| **Chunking** | Splits documents into retrievable units |
| **Embedding** | Converts text to numerical vectors for similarity comparison |
| **Vector search** | Finds the K most similar chunks to a query |
| **Context injection** | Places retrieved chunks into the generation prompt |
| **Grounding constraint** | "Answer ONLY from context" prevents hallucination bleed |
| **Self-corrective RAG** | Agent evaluates retrieval quality and retries if needed |
