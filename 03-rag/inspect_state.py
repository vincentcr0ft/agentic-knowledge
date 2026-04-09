"""
Module 03 — Agent Run Record
═════════════════════════════
Runs the self-corrective RAG pipeline and prints a human-readable
record of every step: what the agent was given, what it did, and why.
"""

from demo import build_rag_pipeline


def _narrate_run(pipeline, question: str, thread_id: str):
    """Run one question and print a step-by-step narrative."""
    config = {"configurable": {"thread_id": thread_id}}
    result = pipeline.invoke(
        {"question": question, "steps": [], "attempt": 0},
        config,
    )

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
        if not steps and not vals.get("retrieved_chunks"):
            print(f"\n  ── Input ──────────────────────────────────────────────────────")
            print(f"  The pipeline received the question, attempt counter at 0,")
            print(f"  and an empty audit trail. It will retrieve chunks, evaluate")
            print(f"  their relevance, and either generate an answer or rephrase")
            print(f"  and retry (up to 2 attempts).")
            prev_values = dict(vals)
            continue

        latest_step = steps[-1] if steps else None
        if latest_step and latest_step == (prev_values.get("steps", []) or [""])[-1]:
            prev_values = dict(vals)
            continue

        step_num += 1
        attempt = vals.get("attempt", 0)

        # ── retrieve_docs ─────────────────────────────────────────
        if (vals.get("retrieved_chunks")
                and len(vals.get("retrieved_chunks", [])) != len(prev_values.get("retrieved_chunks", []))):
            chunks = vals["retrieved_chunks"]
            query_used = vals.get("rephrased") or vals["question"]
            print(f"\n  ── Step {step_num}: Retrieve Documents (attempt {attempt}) ──────────")
            print(f"  Given:   Query: \"{query_used}\"")
            if vals.get("rephrased") and attempt > 1:
                print(f"           (rephrased from original: \"{vals['question']}\")")
            print(f"  Action:  Embedded the query with Ollama, computed cosine")
            print(f"           similarity against all {len(chunks)}-chunk index, returned top 3.")
            print(f"  Result:  {len(chunks)} chunks retrieved:")
            for j, c in enumerate(chunks, 1):
                chunk = c.get("chunk", c)
                score = c.get("score", "?")
                source = chunk.get("source", "?") if isinstance(chunk, dict) else "?"
                text = (chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk))[:80]
                print(f"           {j}. [{source}] score={score:.3f}" if isinstance(score, float) else f"           {j}. [{source}]")
                print(f"              \"{text}…\"")
            print(f"  Why:     Retrieval is always the first step — the pipeline needs")
            print(f"           relevant context before it can generate an answer.")

        # ── evaluate_relevance ────────────────────────────────────
        elif vals.get("relevance") and not prev_values.get("relevance"):
            rel = vals["relevance"]
            print(f"\n  ── Step {step_num}: Evaluate Relevance ──────────────────────────────")
            print(f"  Given:   The question and the {len(vals.get('retrieved_chunks', []))} chunks")
            print(f"           retrieved in the previous step.")
            print(f"  Action:  Sent both to the LLM with a prompt asking it to judge")
            print(f"           whether the chunks contain enough information to answer")
            print(f"           the question. Response is one word: sufficient/insufficient.")
            print(f"  Result:  relevance = \"{rel}\"")
            if rel == "sufficient":
                print(f"  Why:     The chunks contain relevant information → proceed to")
                print(f"           generate an answer. No rephrase needed.")
            else:
                if attempt < 2:
                    print(f"  Why:     The chunks don't have enough relevant information.")
                    print(f"           Since this is attempt {attempt} (< 2), the pipeline")
                    print(f"           will rephrase the query and try retrieval again.")
                else:
                    print(f"  Why:     The chunks still lack relevance, but max attempts")
                    print(f"           reached. The pipeline will generate the best answer")
                    print(f"           it can with what it has.")

        # ── rephrase_query ────────────────────────────────────────
        elif vals.get("rephrased") and vals.get("rephrased") != prev_values.get("rephrased"):
            print(f"\n  ── Step {step_num}: Rephrase Query ──────────────────────────────────")
            print(f"  Given:   Original question: \"{vals['question']}\"")
            print(f"  Action:  Asked the LLM to rephrase the question using different")
            print(f"           keywords that might match relevant documents better.")
            print(f"  Result:  Rephrased to: \"{vals['rephrased']}\"")
            print(f"  Why:     The previous retrieval returned chunks judged as")
            print(f"           'insufficient'. Rephrasing is the self-correction")
            print(f"           mechanism — different wording may hit different chunks")
            print(f"           in the vector index.")

        # ── Handle second retrieval after rephrase ────────────────
        elif (vals.get("relevance") and prev_values.get("relevance")
              and vals.get("relevance") != prev_values.get("relevance")):
            # This catches the second evaluate after rephrase
            rel = vals["relevance"]
            print(f"\n  ── Step {step_num}: Re-evaluate Relevance (attempt {attempt}) ────────")
            print(f"  Given:   The rephrased query's retrieval results.")
            print(f"  Result:  relevance = \"{rel}\"")
            if rel == "sufficient":
                print(f"  Why:     The rephrased query found better chunks → generating.")
            else:
                print(f"  Why:     Still insufficient, but max retries reached → generating.")

        # ── generate_answer ───────────────────────────────────────
        elif vals.get("answer") and not prev_values.get("answer"):
            print(f"\n  ── Step {step_num}: Generate Answer ─────────────────────────────────")
            chunks = vals.get("retrieved_chunks", [])
            sources = []
            for c in chunks:
                chunk = c.get("chunk", c)
                if isinstance(chunk, dict):
                    sources.append(chunk.get("source", "?"))
            print(f"  Given:   The question and {len(chunks)} retrieved chunks from:")
            print(f"           {', '.join(sources)}")
            print(f"  Action:  Sent the chunks as context to the LLM with a prompt")
            print(f"           that says 'answer ONLY from the provided context' and")
            print(f"           'cite which document the information comes from'.")
            answer = vals["answer"]
            print(f"  Result:  \"{answer[:150]}{'…' if len(answer) > 150 else ''}\"")
            print(f"  Why:     This is the final node. The answer is grounded in the")
            print(f"           retrieved context — the LLM cannot hallucinate beyond")
            print(f"           what the chunks provide. Total attempts: {attempt}.")

        prev_values = dict(vals)

    # ── Audit trail ───────────────────────────────────────────────
    print(f"\n  ── Audit Trail ───────────────────────────────────────────────")
    for i, step in enumerate(result.get("steps", []), 1):
        print(f"    {i}. {step}")
    print()


def main():
    pipeline = build_rag_pipeline()

    print("=" * 70)
    print("  Module 03: RAG — Agent Run Records")
    print("  A readable log of what the agent was given, did, and why.")
    print("=" * 70)

    questions = [
        "What is NovaTech's refund policy for hardware?",
        "How do things work around here?",
    ]

    for i, q in enumerate(questions, 1):
        _narrate_run(pipeline, q, f"run-{i}")


if __name__ == "__main__":
    main()
