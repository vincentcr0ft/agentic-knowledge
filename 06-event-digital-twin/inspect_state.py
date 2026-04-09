"""
Module 06 — Agent Run Record
═════════════════════════════
Runs the ingest and query pipelines and prints a human-readable
record of every step: what the agent was given, what it did, and why.

The interview phase is interactive and is best demonstrated via
  python demo.py <statement_file>
"""

from schema import (
    linearise_graph,
    run_schema_completeness,
    prioritise_gaps,
    NODE_TYPES,
)
from ingest import build_ingest_graph, ingest_statement, driver
from query import build_query_graph


# ─── Sample statement ────────────────────────────────────────────────────
STATEMENT = (
    "I was walking along King Street at approximately 2:15 PM on Tuesday "
    "when I heard a loud crash. I turned and saw a red car had collided "
    "with a cyclist at the junction of King Street and Queen's Road. The "
    "driver got out — a tall man wearing a dark jacket. He looked at the "
    "cyclist who was on the ground and then got back in his car and drove "
    "off heading north on Queen's Road. Another woman who was nearby "
    "called an ambulance. I stayed with the cyclist until the paramedics "
    "arrived about ten minutes later."
)


def _narrate_ingest():
    """Run ingest and narrate each step."""
    print(f"\n{'═' * 70}")
    print(f"  AGENT RUN RECORD — INGEST PIPELINE")
    print(f"  Statement → Extraction → Coreference Resolution → Graph Load")
    print(f"{'═' * 70}")

    print(f"\n  ── Input ──────────────────────────────────────────────────────")
    print(f"  The pipeline received a {len(STATEMENT)}-character witness statement.")
    print(f"  It will parse it into sentences, extract entities/relationships")
    print(f"  using the ontology schema, resolve coreferences, and load the")
    print(f"  result into Neo4j.")

    result = ingest_statement(STATEMENT)

    # Walk through what happened
    steps = result.get("steps", [])
    extracted = result.get("extracted", {})
    resolved = result.get("resolved", {})

    step_num = 0
    for step in steps:
        step_num += 1
        if "parse_statement" in step:
            seg_count = len(result.get("segments", []))
            print(f"\n  ── Step {step_num}: Parse Statement ─────────────────────────────────")
            print(f"  Given:   The raw statement text ({len(STATEMENT)} chars).")
            print(f"  Action:  Split the text on sentence boundaries (period/question/")
            print(f"           exclamation followed by whitespace).")
            print(f"  Result:  {seg_count} segments created. Each segment becomes a")
            print(f"           provenance reference — we can trace any extracted fact")
            print(f"           back to the sentence it came from.")
            print(f"  Why:     Sentence-level provenance is essential for auditability.")

        elif "extract_entities" in step:
            n_ents = len(extracted.get("entities", []))
            n_rels = len(extracted.get("relationships", []))
            print(f"\n  ── Step {step_num}: Extract Entities ────────────────────────────────")
            print(f"  Given:   All {len(result.get('segments', []))} segments joined as full text.")
            print(f"  Action:  Sent the text to the LLM with a schema-guided prompt")
            print(f"           listing all {len(NODE_TYPES)} node types and their required")
            print(f"           properties (Event, Person, Vehicle, Location, Time, etc).")
            print(f"           The prompt also lists valid relationship types and rules")
            print(f"           like 'resolve pronouns' and 'break compound events'.")
            print(f"  Result:  {n_ents} entities, {n_rels} relationships extracted.")
            for ent in extracted.get("entities", [])[:5]:
                label = ent.get("label", "?")
                props = ent.get("properties", {})
                desc = (props.get("description") or props.get("name_or_description")
                        or props.get("value") or props.get("summary") or "?")
                print(f"           {label:20s}: {desc}")
            if n_ents > 5:
                print(f"           … and {n_ents - 5} more")
            print(f"  Why:     Schema-guided extraction uses the ontology as a contract —")
            print(f"           the LLM can only produce entity/relationship types that")
            print(f"           exist in our schema. This prevents hallucinated structures.")

        elif "resolve_entities" in step:
            orig_count = len(extracted.get("entities", []))
            resolved_count = len(resolved.get("entities", []))
            print(f"\n  ── Step {step_num}: Coreference Resolution ─────────────────────────")
            print(f"  Given:   {orig_count} extracted entities and the original text.")
            print(f"  Action:  Asked the LLM to identify co-referent entities (e.g.")
            print(f'           "the driver" and "a tall man" = same Person). The prompt')
            print(f"           forbids merging across labels or merging different events.")
            print(f"  Result:  {orig_count} → {resolved_count} entities", end="")
            if orig_count != resolved_count:
                print(f" ({orig_count - resolved_count} merged)")
            else:
                print(f" (no merges needed)")
            if orig_count > 0 and resolved_count < orig_count * 0.4:
                print(f"  Safety:  Over-merge safeguard triggered — kept originals.")
            print(f"  Why:     Without resolution, 'the driver' and 'he' would appear")
            print(f"           as separate people in the graph, creating false gaps.")

        elif "load_to_graph" in step:
            print(f"\n  ── Step {step_num}: Load to Neo4j ──────────────────────────────────")
            load_summary = result.get("load_summary", "")
            resolved_ents = len(resolved.get("entities", []))
            resolved_rels = len(resolved.get("relationships", []))
            print(f"  Given:   {resolved_ents} resolved entities, {resolved_rels} relationships.")
            print(f"  Action:  Used parameterised MERGE queries (no string interpolation)")
            print(f"           to load each entity and relationship into Neo4j. Every node")
            print(f"           gets provenance properties: source text, source_type,")
            print(f"           extracted_at timestamp, and confidence level.")
            print(f"  Result:  {load_summary}")
            print(f"  Why:     MERGE ensures idempotency — running ingest twice won't")
            print(f"           duplicate nodes. Provenance lets us trace any graph fact")
            print(f"           back to the text and extraction that created it.")

    # Graph state
    triples = linearise_graph(driver)
    triple_count = triples.count("\n") + 1 if triples != "(empty graph)" else 0
    print(f"\n  ── Result: Graph State ───────────────────────────────────────")
    print(f"  {triple_count} triples in the graph:")
    for line in triples.split("\n"):
        print(f"    {line}")

    # Gap analysis
    gaps = run_schema_completeness(driver)
    gaps = prioritise_gaps(gaps)
    print(f"\n  ── Gap Analysis ──────────────────────────────────────────────")
    print(f"  {len(gaps)} schema gaps remaining:")
    for g in gaps[:8]:
        print(f"    [{g.priority:8s}] {g.gap_description}")
    if len(gaps) > 8:
        print(f"    … and {len(gaps) - 8} more")

    print(f"\n  ── Audit Trail ───────────────────────────────────────────────")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")
    print()


def _narrate_query():
    """Run a query and narrate each step."""
    question = "What happened at the junction of King Street and Queen's Road?"

    print(f"\n{'═' * 70}")
    print(f"  AGENT RUN RECORD — QUERY PIPELINE")
    print(f"  Question → Entity Detection → Graph Retrieval → Grounded Answer")
    print(f"{'═' * 70}")

    print(f"\n  ── Input ──────────────────────────────────────────────────────")
    print(f"  Question: \"{question}\"")

    query_graph = build_query_graph()
    q_config = {"configurable": {"thread_id": "inspect-q1"}}

    q_result = query_graph.invoke(
        {
            "question": question,
            "detected_entities": [],
            "subgraph_triples": "",
            "full_graph_triples": "",
            "provenance": [],
            "answer": "",
            "reasoning_path": "",
            "steps": [],
        },
        q_config,
    )

    history = list(reversed(list(query_graph.get_state_history(q_config))))

    step_num = 0
    prev_values = {}

    for cp in history:
        vals = cp.values
        steps = vals.get("steps", [])
        if not steps and not vals.get("detected_entities"):
            prev_values = dict(vals)
            continue

        latest_step = steps[-1] if steps else None
        if latest_step and latest_step == (prev_values.get("steps", []) or [""])[-1]:
            prev_values = dict(vals)
            continue

        step_num += 1

        if vals.get("detected_entities") and not prev_values.get("detected_entities"):
            entities = vals["detected_entities"]
            print(f"\n  ── Step {step_num}: Detect Entities ─────────────────────────────────")
            print(f"  Given:   The question text.")
            print(f"  Action:  Asked the LLM to identify key concepts: people, vehicles,")
            print(f"           locations, events, times, and objects mentioned. Returns")
            print(f"           a JSON array of search terms.")
            print(f"  Result:  {len(entities)} entities: {entities}")
            print(f"  Why:     Entity detection drives targeted graph retrieval. Instead")
            print(f"           of dumping the whole graph, we find the relevant subgraph.")

        elif vals.get("subgraph_triples") and not prev_values.get("subgraph_triples"):
            sub = vals["subgraph_triples"]
            prov = vals.get("provenance", [])
            sub_count = sub.count("\n") + 1 if sub != "(no matches)" else 0
            print(f"\n  ── Step {step_num}: Retrieve Subgraph ──────────────────────────────")
            print(f"  Given:   Detected entities: {vals.get('detected_entities', [])}")
            print(f"  Action:  For each entity, searched Neo4j for nodes with matching")
            print(f"           descriptions (case-insensitive substring), then expanded")
            print(f"           to a 2-hop neighbourhood. Also collected provenance records")
            print(f"           (source text for each fact).")
            print(f"  Result:  {sub_count} relevant triples, {len(prov)} provenance records")
            print(f"  Why:     Targeted retrieval provides focused context to the LLM.")
            print(f"           The full graph is also provided as fallback context.")

        elif vals.get("answer") and not prev_values.get("answer"):
            answer = vals["answer"]
            print(f"\n  ── Step {step_num}: Generate Grounded Answer ────────────────────────")
            print(f"  Given:   The relevant subgraph triples + full graph as context.")
            print(f"  Action:  Sent both to the LLM with a prompt requiring:")
            print(f"           - Answer ONLY from graph facts (no LLM knowledge)")
            print(f"           - Cite facts as [FACT: ...]")
            print(f"           - Show reasoning path through the graph")
            print(f"           - Distinguish certain vs inferred facts")
            print(f"  Result:  \"{answer[:120]}{'…' if len(answer) > 120 else ''}\"")
            print(f"  Why:     Grounding prevents hallucination. Every claim must trace")
            print(f"           back to a graph triple. The reasoning path shows exactly")
            print(f"           which nodes and relationships were traversed.")

        prev_values = dict(vals)

    print(f"\n  ── Audit Trail ───────────────────────────────────────────────")
    for i, s in enumerate(q_result.get("steps", []), 1):
        print(f"    {i}. {s}")
    print()


def main():
    print("=" * 70)
    print("  Module 06: Event Digital Twin — Agent Run Records")
    print("  A readable log of what the agent was given, did, and why.")
    print("=" * 70)

    _narrate_ingest()
    _narrate_query()

    driver.close()


if __name__ == "__main__":
    main()