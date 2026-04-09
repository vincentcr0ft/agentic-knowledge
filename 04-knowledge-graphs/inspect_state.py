"""
Module 04 — Agent Run Record
═════════════════════════════
Runs the NL-to-Cypher pipeline and prints a human-readable record
of every step: what the agent was given, what it did, and why.
"""

from demo import build_nl_cypher_pipeline, build_graph, driver


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
        if not steps and not vals.get("cypher"):
            print(f"\n  ── Input ──────────────────────────────────────────────────────")
            print(f"  The pipeline received a natural-language question and an")
            print(f"  empty audit trail. It will translate the question to Cypher,")
            print(f"  execute it against Neo4j, and format the results.")
            prev_values = dict(vals)
            continue

        latest_step = steps[-1] if steps else None
        if latest_step and latest_step == (prev_values.get("steps", []) or [""])[-1]:
            prev_values = dict(vals)
            continue

        step_num += 1

        # ── generate_cypher ───────────────────────────────────────
        if vals.get("cypher") and not prev_values.get("cypher"):
            print(f"\n  ── Step {step_num}: Generate Cypher ──────────────────────────────────")
            print(f"  Given:   The question: \"{question}\"")
            print(f"           Plus the full graph schema (node labels, properties,")
            print(f"           relationship types) embedded in the system prompt.")
            print(f"  Action:  Sent both to the LLM (temperature=0) with instructions")
            print(f"           to return ONLY a Cypher query — no explanation. The")
            print(f"           prompt specifies exact property casing and OPTIONAL MATCH")
            print(f"           guidance.")
            cypher = vals["cypher"]
            print(f"  Result:  Generated Cypher:")
            for line in cypher.strip().split("\n"):
                print(f"             {line.strip()}")
            print(f"  Why:     The LLM acts as a translator — it knows the graph schema")
            print(f"           and converts the English question into a precise Cypher")
            print(f"           query. Temperature=0 ensures consistent translations.")

        # ── execute_query ─────────────────────────────────────────
        elif vals.get("query_result") is not None and prev_values.get("query_result") is None:
            records = vals.get("query_result", [])
            error = vals.get("error", "")
            print(f"\n  ── Step {step_num}: Execute Query ───────────────────────────────────")
            print(f"  Given:   The Cypher query generated in Step 1.")
            print(f"  Action:  Executed the Cypher against the Neo4j database")
            print(f"           at bolt://localhost:7687.")
            if error:
                print(f"  Result:  ERROR — {error}")
                print(f"  Why:     The generated Cypher had a syntax or logic error.")
                print(f"           The error message is captured in state for the")
                print(f"           answer node to communicate to the user.")
            else:
                print(f"  Result:  {len(records)} record(s) returned:")
                for rec in records[:5]:
                    print(f"             {rec}")
                if len(records) > 5:
                    print(f"             … and {len(records) - 5} more")
                print(f"  Why:     The Cypher was valid and Neo4j returned results.")
                print(f"           These raw records will be formatted into natural")
                print(f"           language in the next step.")

        # ── format_answer ─────────────────────────────────────────
        elif vals.get("answer") and not prev_values.get("answer"):
            error = vals.get("error", "")
            records = vals.get("query_result", [])
            print(f"\n  ── Step {step_num}: Format Answer ───────────────────────────────────")
            if error:
                print(f"  Given:   The query error: \"{error[:80]}\"")
                print(f"  Action:  Asked the LLM to explain the failure and suggest")
                print(f"           rephrasing.")
            else:
                print(f"  Given:   The question and {len(records)} raw database record(s).")
                print(f"  Action:  Sent the raw JSON results to the LLM with a prompt")
                print(f"           saying 'provide a clear natural language answer'.")
            answer = vals["answer"]
            print(f"  Result:  \"{answer[:150]}{'…' if len(answer) > 150 else ''}\"")
            print(f"  Why:     Raw database records aren't user-friendly. This node")
            print(f"           translates structured data back into English.")
            print(f"           This is the final node — the pipeline ends here.")

        prev_values = dict(vals)

    # ── Audit trail ───────────────────────────────────────────────
    print(f"\n  ── Audit Trail ───────────────────────────────────────────────")
    for i, step in enumerate(result.get("steps", []), 1):
        print(f"    {i}. {step}")
    print()


def main():
    print("=" * 70)
    print("  Module 04: Knowledge Graphs — Agent Run Records")
    print("  A readable log of what the agent was given, did, and why.")
    print("=" * 70)

    print("\n  Building knowledge graph …")
    build_graph()

    pipeline = build_nl_cypher_pipeline()

    questions = [
        "Who does Bob Martinez manage?",
        "What is the full management chain above Dave Wilson?",
        "What products are used by active projects?",
    ]

    for i, q in enumerate(questions, 1):
        _narrate_run(pipeline, q, f"run-{i}")

    driver.close()


if __name__ == "__main__":
    main()
