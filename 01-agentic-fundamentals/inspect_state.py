"""
Module 01 — Agent Run Record
═════════════════════════════
Runs the customer-service agent and prints a human-readable record
of every step: what the agent was given, what it did, and why.
"""

from demo import build_agent, ORDER_DATABASE


# ─── Narrative helpers ────────────────────────────────────────────────────

def _describe_route(classification: str) -> str:
    """Human-readable explanation of the routing decision."""
    explanations = {
        "greeting": (
            "The conditional edge checked the classification field and "
            "routed to 'handle_greeting' because the query was a greeting."
        ),
        "order_query": (
            "The conditional edge checked the classification field and "
            "routed to 'lookup_order' because an order query was detected."
        ),
        "general": (
            "The conditional edge checked the classification field and "
            "routed to 'handle_general' because the query didn't match "
            "greeting or order patterns."
        ),
    }
    return explanations.get(classification, "Unknown classification — defaulted to general handler.")


def _narrate_run(agent, query: str, thread_id: str):
    """Run a query and print a step-by-step narrative."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"query": query, "steps": []}, config)

    # Collect checkpoints in chronological order (oldest → newest)
    history = list(reversed(list(agent.get_state_history(config))))

    print(f"\n{'═' * 70}")
    print(f"  AGENT RUN RECORD")
    print(f"  Query: \"{query}\"")
    print(f"{'═' * 70}")

    step_num = 0
    prev_values = {}

    for cp in history:
        vals = cp.values
        steps = vals.get("steps", [])
        if not steps and not vals.get("classification"):
            # Initial state — nothing happened yet
            print(f"\n  ── Input ──────────────────────────────────────────────────────")
            print(f"  The agent received the query and an empty audit trail.")
            print(f"  No processing has occurred yet.")
            prev_values = dict(vals)
            continue

        latest_step = steps[-1] if steps else None
        if latest_step and latest_step == (prev_values.get("steps", []) or [""])[-1]:
            prev_values = dict(vals)
            continue  # duplicate checkpoint

        step_num += 1

        # ── classify ──────────────────────────────────────────────
        if vals.get("classification") and not prev_values.get("classification"):
            print(f"\n  ── Step {step_num}: Classify ────────────────────────────────────────")
            print(f"  Given:   The raw query: \"{vals['query']}\"")
            print(f"  Action:  Sent the query to the LLM with a system prompt that")
            print(f"           asks it to pick one of three categories (greeting,")
            print(f"           order_query, general) and extract any order ID.")
            print(f"  Result:  category = \"{vals['classification']}\"", end="")
            if vals.get("order_id"):
                print(f",  order_id = \"{vals['order_id']}\"")
            else:
                print(f",  no order ID found")
            print(f"  Why:     Every query enters the graph at the 'classify' node.")
            print(f"           Classification drives routing — the next node depends")
            print(f"           entirely on this result.")
            print(f"\n  Routing: {_describe_route(vals['classification'])}")

        # ── lookup_order (tool call) ──────────────────────────────
        elif vals.get("tool_result") and not prev_values.get("tool_result"):
            print(f"\n  ── Step {step_num}: Lookup Order (tool call) ─────────────────────")
            print(f"  Given:   order_id = \"{vals.get('order_id', '?')}\"")
            print(f"  Action:  Called the lookup_order tool — a Python function that")
            print(f"           queries the order database (a simple dict of")
            print(f"           {len(ORDER_DATABASE)} known orders).")
            print(f"  Result:  {vals['tool_result']}")
            found = vals.get("order_id", "") in ORDER_DATABASE
            if found:
                print(f"  Why:     The order exists in the database, so the tool returned")
                print(f"           its status, item, and ETA. This factual result will be")
                print(f"           passed to the next node to compose a response.")
            else:
                print(f"  Why:     The order was NOT found in the database. The tool")
                print(f"           returned a 'not found' message. The response node")
                print(f"           will communicate this to the user.")

        # ── respond_with_order ────────────────────────────────────
        elif (vals.get("response") and not prev_values.get("response")
              and prev_values.get("tool_result")):
            print(f"\n  ── Step {step_num}: Respond with Order Info ──────────────────────")
            print(f"  Given:   The original query and the tool result:")
            print(f"           \"{vals.get('tool_result', '')}\"")
            print(f"  Action:  Sent both to the LLM with a system prompt asking it")
            print(f"           to write a helpful, concise customer-service response")
            print(f"           incorporating the order information.")
            print(f"  Result:  \"{vals['response'][:120]}{'…' if len(vals.get('response','')) > 120 else ''}\"")
            print(f"  Why:     The tool provided raw data; this node turns it into a")
            print(f"           natural-language answer suitable for the customer.")
            print(f"           This is the final node — the graph ends here.")

        # ── handle_greeting ───────────────────────────────────────
        elif (vals.get("response") and not prev_values.get("response")
              and vals.get("classification") == "greeting"):
            print(f"\n  ── Step {step_num}: Handle Greeting ──────────────────────────────")
            print(f"  Given:   The original query: \"{vals['query']}\"")
            print(f"  Action:  Sent the query to the LLM with a system prompt asking")
            print(f"           for a warm, one-sentence greeting response.")
            print(f"  Result:  \"{vals['response'][:120]}{'…' if len(vals.get('response','')) > 120 else ''}\"")
            print(f"  Why:     Classified as a greeting — no tools or data lookups")
            print(f"           needed. A friendly reply is sufficient.")
            print(f"           This is the final node — the graph ends here.")

        # ── handle_general ────────────────────────────────────────
        elif (vals.get("response") and not prev_values.get("response")
              and vals.get("classification") == "general"):
            print(f"\n  ── Step {step_num}: Handle General Query ─────────────────────────")
            print(f"  Given:   The original query: \"{vals['query']}\"")
            print(f"  Action:  Sent the query to the LLM with a system prompt asking")
            print(f"           for a concise 1-2 sentence answer.")
            print(f"  Result:  \"{vals['response'][:120]}{'…' if len(vals.get('response','')) > 120 else ''}\"")
            print(f"  Why:     Classified as a general query — no order lookup or")
            print(f"           greeting logic applies. The LLM answers directly.")
            print(f"           This is the final node — the graph ends here.")

        prev_values = dict(vals)

    # ── Audit trail ───────────────────────────────────────────────
    print(f"\n  ── Audit Trail ───────────────────────────────────────────────")
    for i, step in enumerate(result.get("steps", []), 1):
        print(f"    {i}. {step}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    agent = build_agent()

    test_cases = [
        ("Hey there, how's it going?", "run-greeting"),
        ("Where is my order ORD-1001?", "run-order-found"),
        ("What's your return policy?", "run-general"),
        ("Can you check on ORD-9999?", "run-order-missing"),
    ]

    print("=" * 70)
    print("  Module 01: Agentic Fundamentals — Agent Run Records")
    print("  A readable log of what the agent was given, did, and why.")
    print("=" * 70)

    for query, thread_id in test_cases:
        _narrate_run(agent, query, thread_id)


if __name__ == "__main__":
    main()
