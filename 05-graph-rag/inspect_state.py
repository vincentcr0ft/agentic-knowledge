"""
Module 05 — Agent Run Record
═════════════════════════════
Runs the Graph RAG pipeline and prints a human-readable record
of every step: what the agent was given, what it did, and why.
"""

from demo import (
    build_graphrag_pipeline,
    extract_entities,
    resolve_entities,
    load_into_neo4j,
    SOURCE_DOCUMENTS,
    driver,
)


def _narrate_run(pipeline, question: str, thread_id: str):
    """Run one question and print a step-by-step narrative."""
    config = {"configurable": {"thread_id": thread_id}}
    result = pipeline.invoke({"question": question, "steps": []}, config)

    print(f"\n{'═' * 70}")
    print(f"  AGENT RUN RECORD")
    print(f"  Question: \"{question}\"")
    print(f"{'═' * 70}")

    history = list(reversed(list(pipeline.get_state_history(config))))

    step_num = 0
    prev_values = {}

    for cp in history:
        vals = cp.values
        steps = vals.get("steps", [])
        if not steps and not vals.get("detected_entities"):
            print(f"\n  ── Input ──────────────────────────────────────────────────────")
            print(f"  The pipeline received the question and an empty audit trail.")
            print(f"  It will detect entities, do vector retrieval, do graph")
            print(f"  retrieval, merge both contexts, and generate an answer.")
            prev_values = dict(vals)
            continue

        latest_step = steps[-1] if steps else None
        if latest_step and latest_step == (prev_values.get("steps", []) or [""])[-1]:
            prev_values = dict(vals)
            continue

        step_num += 1

        # ── detect_entities ───────────────────────────────────────
        if vals.get("detected_entities") is not None and prev_values.get("detected_entities") is None:
            entities = vals["detected_entities"]
            print(f"\n  ── Step {step_num}: Detect Entities ─────────────────────────────────")
            print(f"  Given:   The question: \"{question}\"")
            print(f"  Action:  Sent the question to the LLM with a prompt asking it")
            print(f"           to identify proper nouns and named entities. The LLM")
            print(f"           returns a JSON array of entity names.")
            print(f"  Result:  Found {len(entities)} entities: {entities}")
            print(f"  Why:     Entity detection drives the graph retrieval path.")
            print(f"           These names will be matched against Neo4j nodes to")
            print(f"           find structural connections (who leads what, what uses")
            print(f"           what) that vector search alone would miss.")

        # ── vector_retrieve ───────────────────────────────────────
        elif vals.get("vector_results") and not prev_values.get("vector_results"):
            vr = vals["vector_results"]
            print(f"\n  ── Step {step_num}: Vector Retrieval ────────────────────────────────")
            print(f"  Given:   The question as a text query.")
            print(f"  Action:  Embedded the question with Ollama, computed cosine")
            print(f"           similarity against {len(SOURCE_DOCUMENTS)} source document")
            print(f"           embeddings, and returned the top 2.")
            print(f"  Result:  {len(vr)} documents retrieved:")
            for j, v in enumerate(vr, 1):
                doc = v.get("doc", {})
                score = v.get("score", 0)
                print(f"           {j}. [{doc.get('title', '?')}] score={score:.3f}")
            print(f"  Why:     Vector retrieval finds documents that are semantically")
            print(f"           similar to the question — good for factual details")
            print(f"           like prices, dates, and descriptions.")

        # ── graph_retrieve ────────────────────────────────────────
        elif vals.get("graph_results") is not None and prev_values.get("graph_results") is None:
            gr = vals["graph_results"]
            entities = vals.get("detected_entities", [])
            print(f"\n  ── Step {step_num}: Graph Retrieval ─────────────────────────────────")
            print(f"  Given:   The detected entities: {entities}")
            print(f"  Action:  For each entity, ran a Neo4j query to find the matching")
            print(f"           node and traverse up to 2 hops outward, collecting all")
            print(f"           connected nodes and relationship types.")
            print(f"  Result:  {len(gr)} graph connections found:")
            seen = set()
            for conn in gr[:6]:
                if conn.get("connected_entity"):
                    path = " → ".join(conn.get("node_names", []))
                    rels = " → ".join(conn.get("rel_types", []))
                    key = (conn.get("source", ""), conn["connected_entity"])
                    if key not in seen:
                        seen.add(key)
                        print(f"           {path}  (via {rels})")
            if len(gr) > 6:
                print(f"           … and {len(gr) - 6} more connections")
            print(f"  Why:     Graph retrieval finds structural relationships that")
            print(f"           vector search cannot — e.g. who manages whom, which")
            print(f"           team designed which product. Multi-hop traversal reveals")
            print(f"           indirect connections across the organisation.")

        # ── merge_context ─────────────────────────────────────────
        elif vals.get("merged_context") and not prev_values.get("merged_context"):
            ctx = vals["merged_context"]
            doc_section = ctx.find("=== RELEVANT DOCUMENTS ===")
            graph_section = ctx.find("=== KNOWLEDGE GRAPH CONNECTIONS ===")
            print(f"\n  ── Step {step_num}: Merge Context ───────────────────────────────────")
            print(f"  Given:   Vector results ({len(vals.get('vector_results', []))} docs)")
            print(f"           and graph results ({len(vals.get('graph_results', []))} connections).")
            print(f"  Action:  Combined both into a single text block with two sections:")
            print(f"           '=== RELEVANT DOCUMENTS ===' (document text + scores)")
            print(f"           '=== KNOWLEDGE GRAPH CONNECTIONS ===' (entity paths).")
            print(f"  Result:  Merged context: {len(ctx)} characters total")
            if graph_section > 0:
                print(f"           Documents section: {graph_section - doc_section} chars")
                print(f"           Graph section:     {len(ctx) - graph_section} chars")
            print(f"  Why:     The LLM needs both sources in a single prompt. Documents")
            print(f"           provide detail (prices, specs); graph connections provide")
            print(f"           structure (who leads what). Neither alone is sufficient.")

        # ── generate ──────────────────────────────────────────────
        elif vals.get("answer") and not prev_values.get("answer"):
            print(f"\n  ── Step {step_num}: Generate Answer ─────────────────────────────────")
            print(f"  Given:   The merged context ({len(vals.get('merged_context', ''))} chars)")
            print(f"           and the original question.")
            print(f"  Action:  Sent both to the LLM with a prompt that says 'use BOTH")
            print(f"           the retrieved documents AND the knowledge graph connections'.")
            print(f"           When graph connections reveal relationships, use those")
            print(f"           facts directly. When documents provide details, cite those.")
            answer = vals["answer"]
            print(f"  Result:  \"{answer[:150]}{'…' if len(answer) > 150 else ''}\"")
            print(f"  Why:     This is the final node — the hybrid context gives the LLM")
            print(f"           access to both semantic similarity (documents) and")
            print(f"           structural knowledge (graph). The answer should reflect both.")

        prev_values = dict(vals)

    # ── Audit trail ───────────────────────────────────────────────
    print(f"\n  ── Audit Trail ───────────────────────────────────────────────")
    for i, step in enumerate(result.get("steps", []), 1):
        print(f"    {i}. {step}")
    print()


def main():
    print("=" * 70)
    print("  Module 05: Graph RAG — Agent Run Records")
    print("  A readable log of what the agent was given, did, and why.")
    print("=" * 70)

    # Build the knowledge graph first
    print("\n  Building knowledge graph from source documents …")
    all_extractions = []
    for doc in SOURCE_DOCUMENTS:
        extraction = extract_entities(doc["text"])
        all_extractions.append(extraction)
    resolved = resolve_entities(all_extractions)
    load_into_neo4j(resolved)
    print(f"  Loaded {len(resolved['entities'])} entities and"
          f" {len(resolved['relationships'])} relationships.\n")

    pipeline = build_graphrag_pipeline()

    questions = [
        "Who leads the team that designed the Nova 7?",
        "Tell me everything about Project Atlas — who's involved and what it uses.",
    ]

    for i, q in enumerate(questions, 1):
        _narrate_run(pipeline, q, f"run-{i}")

    driver.close()


if __name__ == "__main__":
    main()
