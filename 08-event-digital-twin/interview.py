"""
08 · Digital Twin — Interview Pipeline
═══════════════════════════════════════

Conversational loop that analyses the knowledge graph for gaps, generates
follow-up questions, collects answers from a human, extracts new facts,
and updates the graph. Uses LangGraph interrupt() for human-in-the-loop.

Pipeline:
  analyse_gaps → self_resolve → generate_questions → [INTERRUPT] →
  extract_from_answers → [loop back]

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
    Gap,
    QUESTION_GENERATION_PROMPT,
    get_active_spec,
    get_node_types,
    get_relationship_types,
    get_ontology_id,
    linearise_graph,
    materialise_provenance,
    prioritise_gaps,
    run_schema_completeness,
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
    gaps: list[dict]
    addressed_gap_ids: list[str]
    questions: list[dict]
    user_answers: list[str]
    interview_round: int
    max_rounds: int
    is_complete: bool
    graph_snapshot: str
    update_summary: str
    steps: list[str]
    original_statement: str
    all_answers: list[str]
    transcript: list[dict]


# ═══════════════════════════════════════════════════════════════════════════
# JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_array(content: str) -> list | None:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
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


def _parse_json_obj(content: str) -> dict | None:
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


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — analyse_gaps
# ═══════════════════════════════════════════════════════════════════════════

NARRATIVE_COHERENCE_PROMPT = """\
You are an investigative analyst reviewing a knowledge graph built from a \
witness statement. The graph is represented as linearised triples below.

GRAPH TRIPLES:
{triples}

Analyse the graph for narrative coherence problems: temporal consistency, \
spatial consistency, participant consistency, causal plausibility.

Return a JSON array of gaps:
[{{"rule_id": "narrative_<type>", "priority": "medium", "entity_label": "<label>", \
"entity_desc": "<entity>", "gap_description": "<what is inconsistent>"}}]

If no issues found, return: []
"""

INVESTIGATIVE_COMPLETENESS_PROMPT = """\
You are an experienced police investigator reviewing a knowledge graph \
from a witness statement (linearised triples below).

GRAPH TRIPLES:
{triples}

Identify information gaps that matter for investigation: identification, \
sequence of events, implied participants, environmental conditions, \
witness position, directions, physical evidence.

Return a JSON array of gaps:
[{{"rule_id": "investigative_<type>", "priority": "<critical|high|medium|low>", \
"entity_label": "<label>", "entity_desc": "<entity>", \
"gap_description": "<what an investigator would want to know>"}}]

If no gaps found, return: []
"""


def analyse_gaps(state: InterviewState) -> dict:
    round_num = state.get("interview_round", 0) + 1
    triples = linearise_graph(driver)

    print(f"\n  ── Interview Round {round_num} ──")
    print(f"  ▸ Current graph ({triples.count(chr(10)) + 1} triples)")

    all_gaps: list[dict] = []

    # Level 1: Schema completeness
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

    # Level 2: Narrative coherence (rounds 1-2)
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

    # Level 3: Investigative completeness (round 1 only)
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

    # Deduplicate and prioritise
    addressed = set(state.get("addressed_gap_ids", []))
    seen = set()
    unique_gaps = []
    for gap in all_gaps:
        key = gap.get("gap_description", "")
        composite_key = f"{gap.get('rule_id', '')}::{gap.get('entity_desc', '')}"
        if key not in seen and composite_key not in addressed:
            seen.add(key)
            unique_gaps.append(gap)

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    unique_gaps.sort(key=lambda g: priority_order.get(g.get("priority", "low"), 99))

    print(f"  ▸ Total unique gaps: {len(unique_gaps)}")
    for g in unique_gaps[:5]:
        print(f"    [{g.get('priority', '?'):8s}] {g.get('gap_description', '?')}")

    return {
        "gaps": unique_gaps,
        "interview_round": round_num,
        "is_complete": len(unique_gaps) == 0,
        "graph_snapshot": triples,
        "steps": state.get("steps", []) + [
            f"analyse_gaps (round {round_num}): {len(unique_gaps)} gaps"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — attempt_self_resolution
# ═══════════════════════════════════════════════════════════════════════════

SELF_RESOLUTION_PROMPT = """\
You are an analyst reviewing a knowledge graph built from a witness statement.
The graph has gaps. Check whether the ORIGINAL TEXT already contains \
information that fills these gaps.

ORIGINAL STATEMENT:
{statement}

PREVIOUS FOLLOW-UP ANSWERS:
{previous_answers}

CURRENT GRAPH:
{triples}

GAPS TO CHECK:
{gaps}

For EACH gap, decide "resolvable" or "unresolvable".
For resolvable gaps, extract entities and relationships.

Return valid JSON:
{{
  "resolutions": [
    {{
      "gap_index": 0,
      "status": "resolvable",
      "reasoning": "...",
      "entities": [...],
      "relationships": [...]
    }},
    {{
      "gap_index": 1,
      "status": "unresolvable",
      "reasoning": "..."
    }}
  ]
}}

RULES:
- For relationships to EXISTING nodes, use the node's description as the id
- DO NOT create new Event nodes
- DO NOT invent facts not in the text
- A Time or Location mentioned once CAN apply to multiple events in the same scene
"""


def attempt_self_resolution(state: InterviewState) -> dict:
    gaps = state.get("gaps", [])
    triples = state.get("graph_snapshot", "")
    statement = state.get("original_statement", "")
    prev_answers = state.get("all_answers", [])
    round_num = state.get("interview_round", 1)
    addressed = list(state.get("addressed_gap_ids", []))
    transcript = list(state.get("transcript", []))

    if not gaps:
        return {"gaps": [], "steps": state.get("steps", []) + ["self_resolution: no gaps"]}

    batch = gaps[:10]
    gap_lines = [f"{i}. [{g.get('priority', '?')}] {g.get('gap_description', '?')}"
                 for i, g in enumerate(batch)]

    answers_text = "\n".join(f"A{i+1}: {a}" for i, a in enumerate(prev_answers)) if prev_answers else "(none)"

    prompt = SELF_RESOLUTION_PROMPT.format(
        statement=statement, previous_answers=answers_text,
        triples=triples, gaps="\n".join(gap_lines),
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Analyse each gap and resolve what you can."),
    ])

    parsed = _parse_json_obj(result.content)
    resolved_count = 0
    resolved_indices = set()

    if parsed and "resolutions" in parsed:
        for res in parsed["resolutions"]:
            idx = res.get("gap_index", -1)
            if res.get("status") == "resolvable" and 0 <= idx < len(batch):
                gap = batch[idx]
                entities = res.get("entities", [])
                relationships = res.get("relationships", [])
                if entities or relationships:
                    _merge_new_facts({"entities": entities, "relationships": relationships}, round_num)
                    resolved_count += 1
                    resolved_indices.add(idx)
                    composite_key = f"{gap.get('rule_id', '')}::{gap.get('entity_desc', '')}"
                    if composite_key not in addressed:
                        addressed.append(composite_key)
                    transcript.append({
                        "type": "self_resolution", "round": round_num,
                        "gap": gap.get("gap_description", ""),
                        "reasoning": res.get("reasoning", ""),
                    })
                    print(f"    ✓ Self-resolved: {gap.get('gap_description', '?')[:60]}")

    remaining = [g for i, g in enumerate(gaps) if i not in resolved_indices]
    remaining = [g for g in remaining
                 if f"{g.get('rule_id', '')}::{g.get('entity_desc', '')}" not in set(addressed)]

    print(f"  ▸ Self-resolution: {resolved_count} filled, {len(remaining)} remain")

    updated_triples = linearise_graph(driver) if resolved_count > 0 else triples

    return {
        "gaps": remaining,
        "addressed_gap_ids": addressed,
        "graph_snapshot": updated_triples,
        "is_complete": len(remaining) == 0,
        "transcript": transcript,
        "steps": state.get("steps", []) + [
            f"self_resolution (round {round_num}): {resolved_count} resolved"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — generate_questions
# ═══════════════════════════════════════════════════════════════════════════

def generate_questions(state: InterviewState) -> dict:
    gaps = state.get("gaps", [])
    triples = state.get("graph_snapshot", "")

    if not gaps:
        return {"questions": [], "steps": state.get("steps", []) + ["no gaps, no questions"]}

    top_gap = gaps[0]
    gap_lines = [f"{i+1}. [{g.get('priority', '?')}] {g.get('gap_description', '?')}"
                 for i, g in enumerate(gaps[:5])]

    prompt = QUESTION_GENERATION_PROMPT.format(
        gaps="\n".join(gap_lines), triples=triples, max_questions=1,
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate exactly ONE question targeting the first gap."),
    ])

    questions = _parse_json_array(result.content)
    if not questions:
        questions = [{"question": f"Can you provide more detail about: {top_gap.get('gap_description', '')}?",
                      "targets_gaps": [top_gap.get("rule_id", "unknown")]}]

    question = questions[0]
    composite_key = f"{top_gap.get('rule_id', '')}::{top_gap.get('entity_desc', '')}"
    question["_gap_composite_key"] = composite_key
    question["_reason"] = top_gap.get("gap_description", "")

    print(f"  ▸ Question: {question.get('question', '?')}")

    return {
        "questions": [question],
        "steps": state.get("steps", []) + [
            f"generate_questions: targeting '{top_gap.get('rule_id', '?')}'"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — collect_answers (INTERRUPT)
# ═══════════════════════════════════════════════════════════════════════════

def collect_answers(state: InterviewState) -> dict:
    questions = state.get("questions", [])
    if not questions:
        return {"user_answers": [], "steps": state.get("steps", []) + ["no questions"]}

    question_text = questions[0].get("question", "?")
    reason = questions[0].get("_reason", "")

    answer = interrupt({
        "question": question_text,
        "reason": reason,
        "instruction": "Please answer. Say 'I don't know' to skip.",
    })

    user_answer = answer[0] if isinstance(answer, list) else str(answer)
    print(f"  ▸ Received answer: {user_answer[:80]}")

    return {"user_answers": [user_answer], "steps": state.get("steps", []) + ["collected answer"]}


# ═══════════════════════════════════════════════════════════════════════════
# Node 5 — extract_from_answers
# ═══════════════════════════════════════════════════════════════════════════

ANSWER_EXTRACTION_PROMPT = """\
You are updating an existing knowledge graph with new details from a \
witness's follow-up answer. ENRICH existing entities, do NOT create new Events.

EXISTING GRAPH:
{triples}

QUESTION: {questions}
ANSWER: {answers}
GAP BEING FILLED: {gap_description}

Return valid JSON: {{"entities": [...], "relationships": [...]}}
If no useful facts, return: {{"entities": [], "relationships": []}}
"""


def extract_from_answers(state: InterviewState) -> dict:
    answers = state.get("user_answers", [])
    questions = state.get("questions", [])
    triples = state.get("graph_snapshot", "")
    round_num = state.get("interview_round", 1)
    addressed = list(state.get("addressed_gap_ids", []))
    all_answers = list(state.get("all_answers", []))
    transcript = list(state.get("transcript", []))

    gaps = state.get("gaps", [])
    gap_desc = gaps[0].get("gap_description", "") if gaps else ""

    if questions:
        composite_key = questions[0].get("_gap_composite_key", "")
        if composite_key and composite_key not in addressed:
            addressed.append(composite_key)

    answer_text = answers[0] if answers else ""
    question_text = questions[0].get("question", "?") if questions else "?"

    if answer_text:
        all_answers.append(answer_text)

    no_info = ("", "i don't know", "idk", "no", "n/a", "none", "not sure",
               "can't remember", "don't remember")
    if not answer_text or answer_text.lower().strip() in no_info:
        transcript.append({"type": "question_answer", "round": round_num,
                           "question": question_text, "answer": answer_text or "(none)",
                           "outcome": "No new facts"})
        return {
            "update_summary": "No new facts (gap addressed)",
            "addressed_gap_ids": addressed, "all_answers": all_answers,
            "transcript": transcript,
            "steps": state.get("steps", []) + ["no actionable answer"],
        }

    prompt = ANSWER_EXTRACTION_PROMPT.format(
        triples=triples, questions=question_text,
        answers=f"Q: {question_text}\nA: {answer_text}",
        gap_description=gap_desc,
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Extract facts now."),
    ])

    extracted = _parse_json_obj(result.content)
    if extracted is None:
        transcript.append({"type": "question_answer", "round": round_num,
                           "question": question_text, "answer": answer_text,
                           "outcome": "Extraction failed"})
        return {
            "update_summary": "Extraction failed (gap addressed)",
            "addressed_gap_ids": addressed, "all_answers": all_answers,
            "transcript": transcript,
            "steps": state.get("steps", []) + ["extraction parse failed"],
        }

    n_ents = len(extracted.get("entities", []))
    n_rels = len(extracted.get("relationships", []))
    print(f"  ▸ Extracted: {n_ents} entities, {n_rels} relationships")

    summary = _merge_new_facts(extracted, round_num)
    transcript.append({"type": "question_answer", "round": round_num,
                       "question": question_text, "answer": answer_text,
                       "outcome": summary})

    return {
        "update_summary": summary,
        "addressed_gap_ids": addressed, "all_answers": all_answers,
        "transcript": transcript,
        "steps": state.get("steps", []) + [f"extracted: {summary}"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Merge helper
# ═══════════════════════════════════════════════════════════════════════════

SAFE_PROP_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

MERGE_KEYS = {
    "Event": "description", "Process": "description",
    "Person": "name_or_description", "Actor": "name_or_description",
    "Agent": "name_or_description",
    "Vehicle": "description", "Location": "description",
    "Place": "description", "SpatialRegion": "description",
    "Time": "value", "TemporalRegion": "value",
    "Object": "description", "Role": "description",
    "PhysicalDescription": "summary", "Observation": "description",
    "InformationContentEntity": "description",
}


def _merge_new_facts(extracted: dict, round_num: int) -> str:
    entities = extracted.get("entities", [])
    relationships = extracted.get("relationships", [])
    timestamp = datetime.now(timezone.utc).isoformat()
    allowed_labels = set(get_node_types().keys())
    allowed_rels = set(get_relationship_types().keys())
    ontology_id = get_ontology_id()
    created_nodes = 0
    created_rels = 0

    with driver.session() as session:
        id_to_desc = {}
        for entity in entities:
            label = entity.get("label", "Entity")
            if label not in allowed_labels:
                continue
            entity_id = entity.get("id", "")
            props = dict(entity.get("properties", {}))
            desc_key = (props.get("description") or props.get("name_or_description")
                        or props.get("name") or props.get("value")
                        or props.get("summary") or f"unnamed_{entity_id}")
            id_to_desc[entity_id] = desc_key

            props["source_type"] = f"interview_round_{round_num}"
            props["extracted_at"] = timestamp
            props["confidence"] = "medium"
            props["ontology_id"] = ontology_id

            safe_props = {k: v for k, v in props.items()
                          if SAFE_PROP_RE.match(k) and v is not None}

            merge_key = MERGE_KEYS.get(label, "description")
            merge_val = safe_props.get(merge_key, desc_key)

            try:
                session.run(
                    f"MERGE (n:{label} {{{merge_key}: $merge_val}}) SET n += $props",
                    merge_val=merge_val, props=safe_props,
                )
                created_nodes += 1
            except Exception as e:
                print(f"    ⚠ Failed to merge {label}: {e}")

        for rel in relationships:
            rel_type = rel.get("rel_type", "")
            if rel_type not in allowed_rels:
                continue
            from_desc = id_to_desc.get(rel.get("from_id", ""))
            to_desc = id_to_desc.get(rel.get("to_id", ""))
            if not from_desc or not to_desc:
                continue
            try:
                result = session.run(
                    f"MATCH (a) WHERE coalesce(a.description, a.name_or_description, "
                    f"a.name, a.value, a.summary) = $from_desc "
                    f"MATCH (b) WHERE coalesce(b.description, b.name_or_description, "
                    f"b.name, b.value, b.summary) = $to_desc "
                    f"MERGE (a)-[:{rel_type}]->(b)",
                    from_desc=from_desc, to_desc=to_desc,
                )
                created_rels += result.consume().counters.relationships_created
            except Exception as e:
                print(f"    ⚠ Failed to merge rel: {e}")

    msg = f"Merged {created_nodes} nodes, {created_rels} rels (round {round_num})"
    print(f"  ✓ {msg}")

    if created_nodes > 0 or created_rels > 0:
        materialise_provenance(
            driver, source_type=f"interview_round_{round_num}",
            observation_desc=f"Follow-up interview round {round_num}",
            observation_type="follow_up",
        )

    return msg


# ═══════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════

def should_continue(state: InterviewState) -> str:
    if state.get("is_complete"):
        print("  ✓ Graph complete — ending interview")
        return "done"
    if state.get("interview_round", 0) >= state.get("max_rounds", 5):
        print(f"  ⚠ Max rounds reached — ending")
        return "done"
    if "Merged 0 nodes, 0 rel" in state.get("update_summary", ""):
        print("  ▸ No new facts — ending")
        return "done"
    return "continue"


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_interview_graph() -> StateGraph:
    builder = StateGraph(InterviewState)
    builder.add_node("analyse_gaps", analyse_gaps)
    builder.add_node("attempt_self_resolution", attempt_self_resolution)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("collect_answers", collect_answers)
    builder.add_node("extract_from_answers", extract_from_answers)

    builder.add_edge(START, "analyse_gaps")
    builder.add_conditional_edges(
        "analyse_gaps",
        lambda s: "done" if s.get("is_complete") else "has_gaps",
        {"has_gaps": "attempt_self_resolution", "done": END},
    )
    builder.add_conditional_edges(
        "attempt_self_resolution",
        lambda s: "done" if s.get("is_complete") else "has_gaps",
        {"has_gaps": "generate_questions", "done": END},
    )
    builder.add_edge("generate_questions", "collect_answers")
    builder.add_edge("collect_answers", "extract_from_answers")
    builder.add_conditional_edges(
        "extract_from_answers", should_continue,
        {"continue": "analyse_gaps", "done": END},
    )

    return builder.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_interview(
    max_rounds: int = 5,
    thread_id: str = "interview-1",
    statement: str = "",
    transcript_path: str | None = None,
):
    graph = build_interview_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: InterviewState = {
        "gaps": [], "addressed_gap_ids": [], "questions": [],
        "user_answers": [], "interview_round": 0, "max_rounds": max_rounds,
        "is_complete": False, "graph_snapshot": "", "update_summary": "",
        "steps": [], "original_statement": statement, "all_answers": [],
        "transcript": [],
    }

    print(f"{'═' * 70}")
    print(f"  DIGITAL TWIN — INTERVIEW (ontology: {get_active_spec().name})")
    print(f"{'═' * 70}")
    print(f"  Type 'quit' or 'done' to end early.\n")

    for event in graph.stream(initial_state, config, stream_mode="values"):
        pass

    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break

        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    data = task.interrupts[0].value
                    print(f"\n{'─' * 70}")
                    if data.get("reason"):
                        print(f"  [Reason: {data['reason']}]")
                    print(f"  {data.get('question', '?')}")
                    print(f"{'─' * 70}\n")

        try:
            answer = input("  Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "done"

        if answer.lower() in ("quit", "done", "exit"):
            print("\n  Ending interview early.")
            break

        for event in graph.stream(Command(resume=answer), config, stream_mode="values"):
            pass

    # Final summary
    final_state = graph.get_state(config).values
    triples = linearise_graph(driver)
    print(f"\n{'═' * 70}")
    print(f"  FINAL GRAPH ({triples.count(chr(10)) + 1} triples)")
    print(f"{'═' * 70}")
    for line in triples.split("\n"):
        print(f"    {line}")
    print(f"\n  Rounds: {final_state.get('interview_round', 0)}")
    print(f"  Gaps remaining: {len(final_state.get('gaps', []))}")

    if transcript_path:
        _write_transcript(final_state, transcript_path)


def _write_transcript(state_values: dict, transcript_path: str):
    from pathlib import Path
    lines = ["INTERVIEW TRANSCRIPT",
             f"Generated: {datetime.now(timezone.utc).isoformat()}",
             f"Ontology: {get_active_spec().name}", ""]

    statement = state_values.get("original_statement", "")
    if statement:
        lines.extend(["WITNESS STATEMENT", "-" * 40, statement, ""])

    transcript = state_values.get("transcript", [])
    if transcript:
        lines.extend(["INTERVIEW", "-" * 40])
        for entry in transcript:
            if entry.get("type") == "self_resolution":
                lines.extend(["", "[SYSTEM resolved from existing text]",
                              f"Gap: {entry.get('gap', '?')}",
                              f"Reasoning: {entry.get('reasoning', '?')}"])
            elif entry.get("type") == "question_answer":
                lines.extend(["", f"Q: {entry.get('question', '?')}",
                              f"A: {entry.get('answer', '?')}"])

    lines.extend(["", "SUMMARY", "-" * 40,
                   f"Rounds: {state_values.get('interview_round', 0)}",
                   f"Gaps remaining: {len(state_values.get('gaps', []))}"])

    Path(transcript_path).write_text("\n".join(lines) + "\n")
    print(f"\n  Transcript written to: {transcript_path}")


if __name__ == "__main__":
    run_interview(max_rounds=5)
