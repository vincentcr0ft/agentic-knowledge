"""
06 · Event Digital Twin — Interview Pipeline
═════════════════════════════════════════════

The core innovation: a conversational loop that analyses the knowledge graph
for gaps, generates targeted follow-up questions, collects answers from a
human, extracts new facts, and updates the graph — repeating until the graph
reaches completeness or the witness has no more information.

Pipeline (LangGraph with interrupt):

  analyse_gaps  →  generate_questions  →  [INTERRUPT: human answers]
       ↑                                          │
       └──── update_graph ◄── extract_from_answers ┘

Uses LangGraph's interrupt() mechanism for human-in-the-loop.

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b
  - Graph already populated by ingest.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from neo4j import GraphDatabase

from schema import (
    INVESTIGATIVE_COMPLETENESS_PROMPT,
    NARRATIVE_COHERENCE_PROMPT,
    QUESTION_GENERATION_PROMPT,
    SCHEMA_COMPLETENESS_RULES,
    Gap,
    build_extraction_prompt,
    linearise_graph,
    prioritise_gaps,
    run_schema_completeness,
    NODE_TYPES,
    RELATIONSHIP_TYPES,
)


# ─── Connections ──────────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ═══════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════

class InterviewState(TypedDict):
    gaps: list[dict]                     # current gap analysis results
    questions: list[dict]                # generated questions
    user_answers: list[str]              # answers from human
    interview_round: int                 # current round number
    max_rounds: int                      # termination limit
    is_complete: bool                    # whether graph is complete
    graph_snapshot: str                  # linearised triples
    update_summary: str                  # what was added this round
    steps: list[str]                     # audit trail


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — analyse_gaps
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_array(content: str) -> list | None:
    """Parse a JSON array from LLM output, tolerating markdown fences."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find outermost [ … ]
    start = content.find("[")
    end = content.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(content[start:end])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def analyse_gaps(state: InterviewState) -> dict:
    """Three-level gap analysis against the current graph.

    Level 1: Schema completeness — Cypher queries against structural rules
    Level 2: Narrative coherence — LLM checks temporal/spatial/participant consistency
    Level 3: Investigative completeness — LLM identifies what an investigator would need
    """
    round_num = state.get("interview_round", 0) + 1
    triples = linearise_graph(driver)

    print(f"\n  ── Interview Round {round_num} ──")
    print(f"  ▸ Current graph ({triples.count(chr(10)) + 1} triples):")
    for line in triples.split("\n")[:10]:
        print(f"    {line}")
    if triples.count("\n") > 9:
        print(f"    … ({triples.count(chr(10)) + 1 - 10} more)")

    all_gaps: list[dict] = []

    # ── Level 1: Schema completeness ────────────────────────────────────
    print("  ▸ Level 1: Schema completeness …")
    schema_gaps = run_schema_completeness(driver)
    for gap in schema_gaps:
        all_gaps.append({
            "rule_id": gap.rule_id,
            "priority": gap.priority,
            "entity_label": gap.entity_label,
            "entity_desc": gap.entity_desc,
            "gap_description": gap.gap_description,
        })
    print(f"    Found {len(schema_gaps)} schema gaps")

    # ── Level 2: Narrative coherence (only on first round + every 2nd) ──
    if round_num <= 2:
        print("  ▸ Level 2: Narrative coherence …")
        prompt = NARRATIVE_COHERENCE_PROMPT.format(triples=triples)
        result = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Analyse now."),
        ])
        narrative_gaps = _parse_json_array(result.content)
        if narrative_gaps:
            all_gaps.extend(narrative_gaps)
            print(f"    Found {len(narrative_gaps)} narrative gaps")
        else:
            print("    No narrative gaps found (or parse failed)")

    # ── Level 3: Investigative completeness (first round only) ──────────
    if round_num == 1:
        print("  ▸ Level 3: Investigative completeness …")
        prompt = INVESTIGATIVE_COMPLETENESS_PROMPT.format(triples=triples)
        result = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Analyse now."),
        ])
        inv_gaps = _parse_json_array(result.content)
        if inv_gaps:
            all_gaps.extend(inv_gaps)
            print(f"    Found {len(inv_gaps)} investigative gaps")
        else:
            print("    No investigative gaps found (or parse failed)")

    # ── Deduplicate and prioritise ──────────────────────────────────────
    seen = set()
    unique_gaps = []
    for gap in all_gaps:
        key = gap.get("gap_description", "")
        if key not in seen:
            seen.add(key)
            unique_gaps.append(gap)

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    unique_gaps.sort(key=lambda g: priority_order.get(g.get("priority", "low"), 99))

    print(f"  ▸ Total unique gaps: {len(unique_gaps)}")
    for g in unique_gaps[:5]:
        print(f"    [{g.get('priority', '?'):8s}] {g.get('gap_description', '?')}")
    if len(unique_gaps) > 5:
        print(f"    … and {len(unique_gaps) - 5} more")

    is_complete = len(unique_gaps) == 0

    return {
        "gaps": unique_gaps,
        "interview_round": round_num,
        "is_complete": is_complete,
        "graph_snapshot": triples,
        "steps": state.get("steps", []) + [
            f"analyse_gaps (round {round_num}): {len(unique_gaps)} gaps found"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — generate_questions
# ═══════════════════════════════════════════════════════════════════════════

MAX_QUESTIONS_PER_ROUND = 4


def generate_questions(state: InterviewState) -> dict:
    """Generate targeted follow-up questions from the prioritised gaps.

    Questions should be specific, non-leading, and reference what IS known.
    """
    gaps = state.get("gaps", [])
    triples = state.get("graph_snapshot", "")

    if not gaps:
        return {
            "questions": [],
            "steps": state.get("steps", []) + [
                "generate_questions: no gaps, no questions needed"
            ],
        }

    # Format gaps for the prompt
    gap_lines = []
    for i, gap in enumerate(gaps[:10], 1):  # Cap at 10 for prompt size
        gap_lines.append(
            f"{i}. [{gap.get('priority', '?')}] {gap.get('gap_description', '?')}"
        )
    gap_text = "\n".join(gap_lines)

    prompt = QUESTION_GENERATION_PROMPT.format(
        gaps=gap_text,
        triples=triples,
        max_questions=MAX_QUESTIONS_PER_ROUND,
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate the questions now."),
    ])

    questions = _parse_json_array(result.content)

    if not questions:
        # Fallback: generate simple questions from top gaps
        questions = []
        for gap in gaps[:MAX_QUESTIONS_PER_ROUND]:
            questions.append({
                "question": f"Can you provide more detail about: {gap.get('gap_description', '')}?",
                "targets_gaps": [gap.get("rule_id", "unknown")],
            })
        print("  ⚠ LLM question generation failed — using gap-based fallback")

    # Cap at max
    questions = questions[:MAX_QUESTIONS_PER_ROUND]

    print(f"  ▸ Generated {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        print(f"    Q{i}: {q.get('question', '?')}")

    return {
        "questions": questions,
        "steps": state.get("steps", []) + [
            f"generate_questions: {len(questions)} questions"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — collect_answers (INTERRUPT — human-in-the-loop)
# ═══════════════════════════════════════════════════════════════════════════

def collect_answers(state: InterviewState) -> dict:
    """Pause execution and present questions to the human.

    Uses LangGraph's interrupt() to suspend the graph. The caller
    resumes with a Command(resume=<answers>) to continue.
    """
    questions = state.get("questions", [])

    if not questions:
        return {
            "user_answers": [],
            "steps": state.get("steps", []) + [
                "collect_answers: no questions to ask"
            ],
        }

    # Format questions for display
    question_texts = []
    for i, q in enumerate(questions, 1):
        question_texts.append(f"Q{i}: {q.get('question', '?')}")

    # ── INTERRUPT — pause here and wait for human input ─────────────────
    answers = interrupt({
        "questions": question_texts,
        "instruction": (
            "Please answer the questions above. You can answer "
            "'I don't know' for any question. Provide your answers "
            "as a list, one per question."
        ),
    })

    # answers comes back from Command(resume=...)
    if isinstance(answers, str):
        user_answers = [answers]
    elif isinstance(answers, list):
        user_answers = answers
    else:
        user_answers = [str(answers)]

    print(f"  ▸ Received {len(user_answers)} answers from witness")

    return {
        "user_answers": user_answers,
        "steps": state.get("steps", []) + [
            f"collect_answers: received {len(user_answers)} answers"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — extract_from_answers
# ═══════════════════════════════════════════════════════════════════════════

ANSWER_EXTRACTION_PROMPT = """\
You are extracting new facts from a witness's follow-up answers to add to \
an existing knowledge graph.

EXISTING GRAPH (linearised triples):
{triples}

QUESTIONS ASKED:
{questions}

WITNESS ANSWERS:
{answers}

Extract any NEW entities and relationships from the answers. Use the same \
schema as the original extraction:

{schema}

RULES:
- Only extract facts from the ANSWERS — do not repeat existing graph content
- If the witness says "I don't know" or similar, extract nothing for that question
- Link new entities to existing ones where appropriate
- For entity ids, use a prefix to distinguish from original: a_p1, a_e1, a_v1 etc.
- Assign source_type: "interview_round_{round_num}"

Return valid JSON:
{{
  "entities": [...],
  "relationships": [...]
}}

If no new facts can be extracted, return: {{"entities": [], "relationships": []}}
"""


def _parse_json_obj(content: str) -> dict | None:
    """Parse a JSON object from LLM output."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return None


def extract_from_answers(state: InterviewState) -> dict:
    """Extract new entities/relationships from the witness's answers."""
    answers = state.get("user_answers", [])
    questions = state.get("questions", [])
    triples = state.get("graph_snapshot", "")
    round_num = state.get("interview_round", 1)

    if not answers or all(
        a.lower().strip() in ("", "i don't know", "idk", "no", "n/a", "none")
        for a in answers
    ):
        print("  ▸ No actionable answers — skipping extraction")
        return {
            "update_summary": "No new facts extracted",
            "steps": state.get("steps", []) + [
                "extract_from_answers: no actionable answers"
            ],
        }

    # Format Q&A pairs
    qa_lines = []
    for i, q in enumerate(questions):
        answer = answers[i] if i < len(answers) else "(no answer)"
        qa_lines.append(f"Q: {q.get('question', '?')}\nA: {answer}")
    qa_text = "\n\n".join(qa_lines)

    question_text = "\n".join(
        f"Q{i}: {q.get('question', '?')}" for i, q in enumerate(questions, 1)
    )

    prompt = ANSWER_EXTRACTION_PROMPT.format(
        triples=triples,
        questions=question_text,
        answers=qa_text,
        schema=build_extraction_prompt(),
        round_num=round_num,
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Extract new facts now."),
    ])

    extracted = _parse_json_obj(result.content)

    if extracted is None:
        print("  ⚠ Answer extraction returned invalid JSON")
        return {
            "update_summary": "Extraction failed",
            "steps": state.get("steps", []) + [
                "extract_from_answers: JSON parse failed"
            ],
        }

    n_ents = len(extracted.get("entities", []))
    n_rels = len(extracted.get("relationships", []))
    print(f"  ▸ Extracted from answers: {n_ents} entities, {n_rels} relationships")

    # ── Load new facts into Neo4j ───────────────────────────────────────
    summary = _merge_new_facts(extracted, round_num)

    return {
        "update_summary": summary,
        "steps": state.get("steps", []) + [
            f"extract_from_answers: {summary}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 5 — update_graph (merge new facts)
# ═══════════════════════════════════════════════════════════════════════════

# Safe property name pattern
SAFE_PROP_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
ALLOWED_LABELS = set(NODE_TYPES.keys())
ALLOWED_REL_TYPES = set(RELATIONSHIP_TYPES.keys())

MERGE_KEYS = {
    "Event":               "description",
    "Person":              "name_or_description",
    "Vehicle":             "description",
    "Location":            "description",
    "Time":                "value",
    "Object":              "description",
    "PhysicalDescription": "summary",
    "Observation":         "description",
}


def _merge_new_facts(extracted: dict, round_num: int) -> str:
    """Merge newly extracted entities and relationships into Neo4j.

    Uses MERGE for idempotency — won't duplicate existing nodes.
    """
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])
    timestamp = datetime.now(timezone.utc).isoformat()
    created_nodes = 0
    created_rels = 0

    with driver.session() as session:
        # ── Merge entity nodes ──────────────────────────────────────────
        id_to_desc = {}
        for entity in entities:
            label = entity.get("label", "Entity")
            if label not in ALLOWED_LABELS:
                continue

            entity_id = entity.get("id", "")
            props = dict(entity.get("properties", {}))

            desc_key = (
                props.get("description")
                or props.get("name_or_description")
                or props.get("name")
                or props.get("value")
                or props.get("summary")
                or f"unnamed_{entity_id}"
            )
            id_to_desc[entity_id] = desc_key

            # Provenance
            props["source_type"] = f"interview_round_{round_num}"
            props["extracted_at"] = timestamp
            props["confidence"] = "medium"

            safe_props = {
                k: v for k, v in props.items()
                if SAFE_PROP_RE.match(k) and v is not None
            }

            merge_key = MERGE_KEYS.get(label, "description")
            merge_val = safe_props.get(merge_key, desc_key)

            cypher = (
                f"MERGE (n:{label} {{{merge_key}: $merge_val}}) "
                f"SET n += $props"
            )
            try:
                session.run(cypher, merge_val=merge_val, props=safe_props)
                created_nodes += 1
            except Exception as e:
                print(f"    ⚠ Failed to merge {label} '{desc_key}': {e}")

        # ── Merge relationships ─────────────────────────────────────────
        for rel in relationships:
            rel_type = rel.get("rel_type", "")
            from_id = rel.get("from_id", "")
            to_id = rel.get("to_id", "")

            if rel_type not in ALLOWED_REL_TYPES:
                continue

            from_desc = id_to_desc.get(from_id)
            to_desc = id_to_desc.get(to_id)

            if not from_desc or not to_desc:
                # Try matching against existing graph nodes
                continue

            cypher = (
                f"MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                f"a.name, a.value, a.summary) = $from_desc "
                f"MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                f"b.name, b.value, b.summary) = $to_desc "
                f"MERGE (a)-[:{rel_type}]->(b)"
            )
            try:
                result = session.run(cypher, from_desc=from_desc, to_desc=to_desc)
                summary = result.consume()
                created_rels += summary.counters.relationships_created
            except Exception as e:
                print(f"    ⚠ Failed to merge rel: {e}")

    msg = f"Merged {created_nodes} nodes, {created_rels} relationships (round {round_num})"
    print(f"  ✓ {msg}")
    return msg


# ═══════════════════════════════════════════════════════════════════════════
# Routing — should we continue the interview loop?
# ═══════════════════════════════════════════════════════════════════════════

def should_continue(state: InterviewState) -> str:
    """Decide whether to loop back to gap analysis or end."""
    if state.get("is_complete", False):
        print("  ✓ Graph is complete — ending interview")
        return "done"

    round_num = state.get("interview_round", 0)
    max_rounds = state.get("max_rounds", 5)
    if round_num >= max_rounds:
        print(f"  ⚠ Reached max rounds ({max_rounds}) — ending interview")
        return "done"

    # Check if last round produced any new facts
    summary = state.get("update_summary", "")
    if "Merged 0 nodes, 0 relationships" in summary:
        print("  ▸ No new facts last round — ending interview")
        return "done"

    return "continue"


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_interview_graph() -> StateGraph:
    """Build the interview loop as a LangGraph with human-in-the-loop."""
    builder = StateGraph(InterviewState)

    builder.add_node("analyse_gaps", analyse_gaps)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("collect_answers", collect_answers)
    builder.add_node("extract_from_answers", extract_from_answers)

    # Entry → gap analysis
    builder.add_edge(START, "analyse_gaps")

    # Gap analysis → conditional: if gaps, generate questions; else done
    builder.add_conditional_edges(
        "analyse_gaps",
        lambda s: "done" if s.get("is_complete") else "has_gaps",
        {"has_gaps": "generate_questions", "done": END},
    )

    # Questions → collect answers (INTERRUPT happens here)
    builder.add_edge("generate_questions", "collect_answers")

    # Answers → extract new facts
    builder.add_edge("collect_answers", "extract_from_answers")

    # After extraction → check if we should continue
    builder.add_conditional_edges(
        "extract_from_answers",
        should_continue,
        {"continue": "analyse_gaps", "done": END},
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════
# CLI — interactive interview loop
# ═══════════════════════════════════════════════════════════════════════════

def run_interview(max_rounds: int = 5, thread_id: str = "interview-1"):
    """Run the full interview loop interactively.

    The graph will pause at each collect_answers node, present questions
    to the user, and resume when answers are provided.
    """
    graph = build_interview_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: InterviewState = {
        "gaps": [],
        "questions": [],
        "user_answers": [],
        "interview_round": 0,
        "max_rounds": max_rounds,
        "is_complete": False,
        "graph_snapshot": "",
        "update_summary": "",
        "steps": [],
    }

    print(f"{'═' * 70}")
    print(f"  EVENT DIGITAL TWIN — INTERVIEW PHASE")
    print(f"{'═' * 70}")
    print(f"  Analysing graph for gaps and generating questions …")
    print(f"  (type 'quit' or 'done' to end early)\n")

    # ── First invocation — runs until the interrupt ─────────────────────
    result = None
    for event in graph.stream(initial_state, config, stream_mode="values"):
        result = event

    # ── Interview loop — resume after each interrupt ────────────────────
    while True:
        # Check the current state
        snapshot = graph.get_state(config)

        # If the graph has finished, we're done
        if not snapshot.next:
            print(f"\n{'─' * 70}")
            print("  Interview complete.")
            break

        # We're at an interrupt — present questions and collect answers
        # The interrupt value contains the questions
        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_data = task.interrupts[0].value
                    questions = interrupt_data.get("questions", [])

                    print(f"\n{'─' * 70}")
                    print("  Follow-up questions for the witness:\n")
                    for q in questions:
                        print(f"    {q}")
                    print()
                    print(f"  {interrupt_data.get('instruction', '')}")
                    print(f"{'─' * 70}\n")

        # Collect answers from the user
        answers = []
        questions = result.get("questions", []) if result else []
        n_questions = len(questions)

        for i in range(max(n_questions, 1)):
            try:
                answer = input(f"  Answer {i + 1}: ").strip()
            except (EOFError, KeyboardInterrupt):
                answer = "done"

            if answer.lower() in ("quit", "done", "exit"):
                print("\n  Ending interview early.")
                # Print final state
                _print_final_summary(graph, config)
                return

            answers.append(answer)

            # If user answered fewer than expected, that's fine
            if i >= n_questions - 1:
                break

        # Resume the graph with the answers
        result = None
        for event in graph.stream(
            Command(resume=answers), config, stream_mode="values"
        ):
            result = event

    _print_final_summary(graph, config)


def _print_final_summary(graph, config):
    """Print the final graph state and audit trail."""
    final_state = graph.get_state(config)
    state_values = final_state.values

    print(f"\n{'═' * 70}")
    print("  FINAL GRAPH STATE")
    print(f"{'═' * 70}")

    triples = linearise_graph(driver)
    print(f"\n  Graph ({triples.count(chr(10)) + 1} triples):")
    for line in triples.split("\n"):
        print(f"    {line}")

    print(f"\n{'─' * 70}")
    print("  Audit trail:")
    for step in state_values.get("steps", []):
        print(f"    • {step}")

    rounds = state_values.get("interview_round", 0)
    gaps_remaining = len(state_values.get("gaps", []))
    print(f"\n  Rounds completed: {rounds}")
    print(f"  Gaps remaining:  {gaps_remaining}")
    print(f"{'═' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_interview(max_rounds=5)
