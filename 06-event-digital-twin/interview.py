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
    addressed_gap_ids: list[str]         # gap rule_ids already asked about
    questions: list[dict]                # generated questions
    user_answers: list[str]              # answers from human
    interview_round: int                 # current round number
    max_rounds: int                      # termination limit
    is_complete: bool                    # whether graph is complete
    graph_snapshot: str                  # linearised triples
    update_summary: str                  # what was added this round
    steps: list[str]                     # audit trail
    original_statement: str              # full text of the initial statement
    all_answers: list[str]               # accumulated answers across all rounds
    transcript: list[dict]               # records of each interaction for output


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

    # ── Deduplicate, filter addressed, and prioritise ─────────────────
    addressed = set(state.get("addressed_gap_ids", []))
    seen = set()
    unique_gaps = []
    for gap in all_gaps:
        key = gap.get("gap_description", "")
        rule_id = gap.get("rule_id", "")
        # Build a composite key: rule_id + entity for schema gaps
        composite_key = f"{rule_id}::{gap.get('entity_desc', '')}"
        if key not in seen and composite_key not in addressed:
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
# Node 2 — attempt_self_resolution
# ═══════════════════════════════════════════════════════════════════════════

SELF_RESOLUTION_PROMPT = """\
You are an analyst reviewing a knowledge graph built from a witness statement.
The graph has gaps (missing information). Your job is to check whether the
ORIGINAL TEXT already contains information that fills these gaps — the
extraction may have missed it, or the information may apply to multiple events.

For example, if the witness says "at about 2:15 PM" once and there are
multiple events, that time likely applies to all events in the same scene.
Similarly, a location mentioned once may apply to all nearby events.

ORIGINAL STATEMENT:
{statement}

PREVIOUS FOLLOW-UP ANSWERS:
{previous_answers}

CURRENT GRAPH (linearised triples):
{triples}

GAPS TO CHECK:
{gaps}

For EACH gap, decide:
  - "resolvable": the text already contains information to fill this gap
  - "unresolvable": the text genuinely does not contain this information

For resolvable gaps, extract the entities and relationships needed.

Return valid JSON:
{{
  "resolutions": [
    {{
      "gap_index": 0,
      "status": "resolvable",
      "reasoning": "The witness mentioned 2:15 PM which applies to this event too",
      "entities": [
        {{"id": "t_resolved", "label": "Time", "properties": {{"value": "approximately 2:15 PM", "precision": "approximate"}}}}
      ],
      "relationships": [
        {{"from_id": "existing_event_desc", "rel_type": "OCCURRED_AT_TIME", "to_id": "t_resolved"}}
      ]
    }},
    {{
      "gap_index": 1,
      "status": "unresolvable",
      "reasoning": "The text does not mention a registration number"
    }}
  ]
}}

CRITICAL RULES:
- For relationships to EXISTING nodes, use the node's description as the id
  (match against the graph triples above)
- DO NOT create new Event nodes — events are already in the graph
- DO NOT invent facts not present in the text
- A Time or Location mentioned once in the text CAN apply to multiple events
  in the same scene — this is the most common resolution
- Check the FULL text carefully, including follow-up answers
"""


def attempt_self_resolution(state: InterviewState) -> dict:
    """Try to fill gaps from existing text before asking the witness.

    Many gaps arise because the extraction stage creates separate events
    but links contextual information (time, place) to only one of them.
    The original text often already contains the answer.
    """
    gaps = state.get("gaps", [])
    triples = state.get("graph_snapshot", "")
    statement = state.get("original_statement", "")
    prev_answers = state.get("all_answers", [])
    round_num = state.get("interview_round", 1)
    addressed = list(state.get("addressed_gap_ids", []))
    transcript = list(state.get("transcript", []))

    if not gaps:
        return {
            "gaps": [],
            "steps": state.get("steps", []) + [
                "self_resolution: no gaps to resolve"
            ],
        }

    # Format gaps for the prompt (batch up to 10)
    batch = gaps[:10]
    gap_lines = []
    for i, gap in enumerate(batch):
        gap_lines.append(
            f"{i}. [{gap.get('priority', '?')}] {gap.get('gap_description', '?')}"
        )
    gap_text = "\n".join(gap_lines)

    answers_text = "\n".join(
        f"A{i+1}: {a}" for i, a in enumerate(prev_answers)
    ) if prev_answers else "(none yet)"

    prompt = SELF_RESOLUTION_PROMPT.format(
        statement=statement,
        previous_answers=answers_text,
        triples=triples,
        gaps=gap_text,
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
            status = res.get("status", "")
            reasoning = res.get("reasoning", "")

            if status == "resolvable" and 0 <= idx < len(batch):
                gap = batch[idx]
                entities = res.get("entities", [])
                relationships = res.get("relationships", [])

                if entities or relationships:
                    extracted = {
                        "entities": entities,
                        "relationships": relationships,
                    }
                    _merge_new_facts(extracted, round_num)
                    resolved_count += 1
                    resolved_indices.add(idx)

                    # Mark as addressed
                    composite_key = f"{gap.get('rule_id', '')}::{gap.get('entity_desc', '')}"
                    if composite_key not in addressed:
                        addressed.append(composite_key)

                    # Record in transcript
                    transcript.append({
                        "type": "self_resolution",
                        "round": round_num,
                        "gap": gap.get("gap_description", ""),
                        "reasoning": reasoning,
                        "entities_added": len(entities),
                        "relationships_added": len(relationships),
                    })

                    print(f"    ✓ Self-resolved: {gap.get('gap_description', '?')[:60]}")
                    print(f"      Reason: {reasoning[:80]}")

    # Remove resolved gaps from the list
    remaining_gaps = [
        g for i, g in enumerate(gaps)
        if i not in resolved_indices
    ]
    # Also remove any gaps beyond batch that were already addressed
    remaining_gaps = [
        g for g in remaining_gaps
        if f"{g.get('rule_id', '')}::{g.get('entity_desc', '')}" not in set(addressed)
    ]

    print(f"  ▸ Self-resolution: {resolved_count} gaps filled from existing text,"
          f" {len(remaining_gaps)} remain")

    # Refresh the graph snapshot after self-resolution
    updated_triples = linearise_graph(driver) if resolved_count > 0 else triples

    return {
        "gaps": remaining_gaps,
        "addressed_gap_ids": addressed,
        "graph_snapshot": updated_triples,
        "is_complete": len(remaining_gaps) == 0,
        "transcript": transcript,
        "steps": state.get("steps", []) + [
            f"self_resolution (round {round_num}): resolved {resolved_count},"
            f" {len(remaining_gaps)} remain"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — generate_questions
# ═══════════════════════════════════════════════════════════════════════════

MAX_QUESTIONS_PER_ROUND = 1


def generate_questions(state: InterviewState) -> dict:
    """Generate ONE targeted follow-up question for the highest-priority gap.

    Asks the single most important question per round so the witness
    can focus. The gap being targeted is tracked so it won't be
    re-asked regardless of the answer quality.
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

    # Pick the single highest-priority gap
    top_gap = gaps[0]

    # Format for the prompt — provide context of the top gap + a few more
    gap_lines = []
    for i, gap in enumerate(gaps[:5], 1):
        gap_lines.append(
            f"{i}. [{gap.get('priority', '?')}] {gap.get('gap_description', '?')}"
        )
    gap_text = "\n".join(gap_lines)

    prompt = QUESTION_GENERATION_PROMPT.format(
        gaps=gap_text,
        triples=triples,
        max_questions=1,
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate exactly ONE question targeting the first gap."),
    ])

    questions = _parse_json_array(result.content)

    if not questions:
        # Fallback: generate a direct question from the top gap
        questions = [{
            "question": f"Can you provide more detail about: {top_gap.get('gap_description', '')}?",
            "targets_gaps": [top_gap.get("rule_id", "unknown")],
        }]
        print("  ⚠ LLM question generation failed — using gap-based fallback")

    # Take only the first question
    question = questions[0]

    # Ensure targets_gaps includes the top gap's composite key
    if "targets_gaps" not in question:
        question["targets_gaps"] = []
    composite_key = f"{top_gap.get('rule_id', '')}::{top_gap.get('entity_desc', '')}"
    question["_gap_composite_key"] = composite_key
    question["_reason"] = top_gap.get("gap_description", "")

    print(f"  ▸ Question (targeting: {top_gap.get('gap_description', '?')[:60]}):")
    print(f"    Why:  {top_gap.get('gap_description', '?')}")
    print(f"    Ask:  {question.get('question', '?')}")

    return {
        "questions": [question],
        "steps": state.get("steps", []) + [
            f"generate_questions: 1 question targeting '{top_gap.get('rule_id', '?')}'"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — collect_answers (INTERRUPT — human-in-the-loop)
# ═══════════════════════════════════════════════════════════════════════════

def collect_answers(state: InterviewState) -> dict:
    """Pause execution and present ONE question to the human.

    Uses LangGraph's interrupt() to suspend the graph. The caller
    resumes with a Command(resume=<answer>) to continue.
    """
    questions = state.get("questions", [])

    if not questions:
        return {
            "user_answers": [],
            "steps": state.get("steps", []) + [
                "collect_answers: no questions to ask"
            ],
        }

    question_text = questions[0].get("question", "?")
    reason = questions[0].get("_reason", "")

    # ── INTERRUPT — pause here and wait for human input ─────────────────
    answer = interrupt({
        "question": question_text,
        "reason": reason,
        "instruction": (
            "Please answer the question above. If you don't know, "
            "say 'I don't know' and we'll move on."
        ),
    })

    # answer comes back from Command(resume=...)
    if isinstance(answer, list):
        user_answer = answer[0] if answer else ""
    else:
        user_answer = str(answer)

    print(f"  ▸ Received answer: {user_answer[:80]}")

    return {
        "user_answers": [user_answer],
        "steps": state.get("steps", []) + [
            f"collect_answers: received answer"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — extract_from_answers
# ═══════════════════════════════════════════════════════════════════════════

ANSWER_EXTRACTION_PROMPT = """\
You are updating an existing knowledge graph with new details from a \
witness's follow-up answer. Your job is to ENRICH existing entities, \
NOT create new Event nodes.

EXISTING GRAPH (linearised triples):
{triples}

QUESTION ASKED:
{questions}

WITNESS ANSWER:
{answers}

THE GAP BEING FILLED:
{gap_description}

Your task: extract facts from the answer that fill the gap described above.

CRITICAL RULES:
- DO NOT create new Event nodes — the events are already in the graph
- Instead, add PROPERTIES to existing entities or create RELATIONSHIPS \
between existing entities and new detail entities
- For example, if the gap is "Vehicle has no registration" and the answer \
is "I think it was AB12 CDE", update the existing Vehicle node's \
registration property — do NOT create a new Vehicle
- You CAN create new nodes for: PhysicalDescription, Time, Location, Object \
— these are detail nodes that attach to existing entities
- Reference existing entities by matching their description from the graph
- If the answer doesn't provide useful information, return empty arrays

HOW TO STRUCTURE UPDATES:

1. To add a property to an existing node, return it as an entity with the \
SAME description/identifier and the new properties:
   {{"id": "existing_v1", "label": "Vehicle", "properties": {{"description": \
"a red car", "registration": "AB12 CDE"}}}}

2. To link a new detail node to an existing entity, create the detail node \
and a relationship:
   {{"id": "pd1", "label": "PhysicalDescription", "properties": {{"summary": \
"tall, dark hair, about 30"}}}}
   relationship: {{"from_id": "existing_p1", "rel_type": "DESCRIBED_AS", \
"to_id": "pd1"}}

3. To add a time to an existing event, create a Time node and link it:
   {{"id": "t1", "label": "Time", "properties": {{"value": "about 2:15 PM", \
"precision": "approximate"}}}}
   relationship: {{"from_id": "existing_e1", "rel_type": "OCCURRED_AT_TIME", \
"to_id": "t1"}}

Return valid JSON:
{{
  "entities": [...],
  "relationships": [...]
}}

If no useful facts can be extracted, return: {{"entities": [], "relationships": []}}
"""


def extract_from_answers(state: InterviewState) -> dict:
    """Extract new facts from the witness's answer and mark the gap as addressed.

    Even if extraction fails or the answer is unhelpful, the gap is marked
    as addressed so it won't be re-asked.
    """
    answers = state.get("user_answers", [])
    questions = state.get("questions", [])
    triples = state.get("graph_snapshot", "")
    round_num = state.get("interview_round", 1)
    addressed = list(state.get("addressed_gap_ids", []))
    all_answers = list(state.get("all_answers", []))
    transcript = list(state.get("transcript", []))

    # Mark the targeted gap as addressed regardless of answer quality
    gap_desc = ""
    gaps = state.get("gaps", [])
    if gaps:
        gap_desc = gaps[0].get("gap_description", "")

    if questions:
        composite_key = questions[0].get("_gap_composite_key", "")
        if composite_key and composite_key not in addressed:
            addressed.append(composite_key)

    answer_text = answers[0] if answers else ""
    question_text = questions[0].get("question", "?") if questions else "?"

    # Accumulate this answer for future self-resolution rounds
    if answer_text:
        all_answers.append(answer_text)

    if not answer_text or answer_text.lower().strip() in (
        "", "i don't know", "idk", "no", "n/a", "none", "not sure",
        "can't remember", "don't remember",
    ):
        print(f"  ▸ No actionable answer — gap marked as addressed, moving on")
        transcript.append({
            "type": "question_answer",
            "round": round_num,
            "question": question_text,
            "reason": gap_desc,
            "answer": answer_text or "(no answer)",
            "outcome": "No new facts extracted",
        })
        return {
            "update_summary": "No new facts extracted (gap marked addressed)",
            "addressed_gap_ids": addressed,
            "all_answers": all_answers,
            "transcript": transcript,
            "steps": state.get("steps", []) + [
                "extract_from_answers: no actionable answer, gap addressed"
            ],
        }

    prompt = ANSWER_EXTRACTION_PROMPT.format(
        triples=triples,
        questions=question_text,
        answers=f"Q: {question_text}\nA: {answer_text}",
        gap_description=gap_desc,
    )

    result = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Extract facts to fill the gap now."),
    ])

    extracted = _parse_json_obj(result.content)

    if extracted is None:
        print("  ⚠ Answer extraction returned invalid JSON — gap marked addressed")
        transcript.append({
            "type": "question_answer",
            "round": round_num,
            "question": question_text,
            "reason": gap_desc,
            "answer": answer_text,
            "outcome": "Extraction failed (parse error)",
        })
        return {
            "update_summary": "Extraction failed (gap marked addressed)",
            "addressed_gap_ids": addressed,
            "all_answers": all_answers,
            "transcript": transcript,
            "steps": state.get("steps", []) + [
                "extract_from_answers: JSON parse failed, gap addressed"
            ],
        }

    n_ents = len(extracted.get("entities", []))
    n_rels = len(extracted.get("relationships", []))
    print(f"  ▸ Extracted from answer: {n_ents} entities, {n_rels} relationships")

    # ── Load new facts into Neo4j ───────────────────────────────────────
    summary = _merge_new_facts(extracted, round_num)

    transcript.append({
        "type": "question_answer",
        "round": round_num,
        "question": question_text,
        "reason": gap_desc,
        "answer": answer_text,
        "outcome": summary,
    })

    return {
        "update_summary": summary,
        "addressed_gap_ids": addressed,
        "all_answers": all_answers,
        "transcript": transcript,
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
    """Build the interview loop as a LangGraph with human-in-the-loop.

    Pipeline:
      analyse_gaps → attempt_self_resolution → [if gaps remain]
        generate_questions → collect_answers → extract_from_answers → [loop]
    """
    builder = StateGraph(InterviewState)

    builder.add_node("analyse_gaps", analyse_gaps)
    builder.add_node("attempt_self_resolution", attempt_self_resolution)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("collect_answers", collect_answers)
    builder.add_node("extract_from_answers", extract_from_answers)

    # Entry → gap analysis
    builder.add_edge(START, "analyse_gaps")

    # Gap analysis → conditional: if gaps, try self-resolution; else done
    builder.add_conditional_edges(
        "analyse_gaps",
        lambda s: "done" if s.get("is_complete") else "has_gaps",
        {"has_gaps": "attempt_self_resolution", "done": END},
    )

    # Self-resolution → conditional: if gaps remain, ask human; else done
    builder.add_conditional_edges(
        "attempt_self_resolution",
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

def run_interview(
    max_rounds: int = 5,
    thread_id: str = "interview-1",
    statement: str = "",
    transcript_path: str | None = None,
):
    """Run the full interview loop interactively.

    Args:
        max_rounds: Maximum interview rounds before stopping.
        thread_id: LangGraph thread identifier.
        statement: The original witness statement text (for self-resolution).
        transcript_path: If set, write a plain-text transcript on completion.
    """
    graph = build_interview_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: InterviewState = {
        "gaps": [],
        "addressed_gap_ids": [],
        "questions": [],
        "user_answers": [],
        "interview_round": 0,
        "max_rounds": max_rounds,
        "is_complete": False,
        "graph_snapshot": "",
        "update_summary": "",
        "steps": [],
        "original_statement": statement,
        "all_answers": [],
        "transcript": [],
    }

    print(f"{'═' * 70}")
    print(f"  EVENT DIGITAL TWIN — INTERVIEW PHASE")
    print(f"{'═' * 70}")
    print(f"  The system will first try to fill gaps from the existing text,")
    print(f"  then ask follow-up questions for anything it can't resolve.")
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

        # We're at an interrupt — present the question and collect one answer
        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_data = task.interrupts[0].value
                    question = interrupt_data.get("question", "")
                    reason = interrupt_data.get("reason", "")

                    print(f"\n{'─' * 70}")
                    if reason:
                        print(f"  [Reason: {reason}]")
                    print(f"  {question}")
                    print(f"\n  {interrupt_data.get('instruction', '')}")
                    print(f"{'─' * 70}\n")

        # Collect one answer from the user
        try:
            answer = input("  Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "done"

        if answer.lower() in ("quit", "done", "exit"):
            print("\n  Ending interview early.")
            _print_final_summary(graph, config, transcript_path)
            return

        # Resume the graph with the answer
        result = None
        for event in graph.stream(
            Command(resume=answer), config, stream_mode="values"
        ):
            result = event

    _print_final_summary(graph, config, transcript_path)


def _write_transcript(state_values: dict, transcript_path: str):
    """Write a plain-text transcript of the full interview session."""
    from pathlib import Path

    lines = []
    lines.append("=" * 70)
    lines.append("  EVENT DIGITAL TWIN — INTERVIEW TRANSCRIPT")
    lines.append(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 70)

    # Original statement
    statement = state_values.get("original_statement", "")
    if statement:
        lines.append("")
        lines.append("─" * 70)
        lines.append("  ORIGINAL STATEMENT")
        lines.append("─" * 70)
        lines.append(f"  {statement}")

    # Transcript entries
    transcript = state_values.get("transcript", [])
    if transcript:
        lines.append("")
        lines.append("─" * 70)
        lines.append("  INTERVIEW RECORD")
        lines.append("─" * 70)

        for i, entry in enumerate(transcript, 1):
            lines.append("")
            entry_type = entry.get("type", "unknown")
            round_num = entry.get("round", "?")

            if entry_type == "self_resolution":
                lines.append(f"  [{i}] SELF-RESOLVED (Round {round_num})")
                lines.append(f"      Gap:       {entry.get('gap', '?')}")
                lines.append(f"      Reasoning: {entry.get('reasoning', '?')}")
                lines.append(f"      Added:     {entry.get('entities_added', 0)} entities,"
                             f" {entry.get('relationships_added', 0)} relationships")
            elif entry_type == "question_answer":
                lines.append(f"  [{i}] QUESTION (Round {round_num})")
                lines.append(f"      Asked:     {entry.get('question', '?')}")
                lines.append(f"      Reason:    {entry.get('reason', '?')}")
                lines.append(f"      Answer:    {entry.get('answer', '?')}")
                lines.append(f"      Outcome:   {entry.get('outcome', '?')}")

    # Final graph
    triples = linearise_graph(driver)
    lines.append("")
    lines.append("─" * 70)
    lines.append("  FINAL GRAPH STATE")
    lines.append("─" * 70)
    for line in triples.split("\n"):
        lines.append(f"  {line}")

    # Audit trail
    lines.append("")
    lines.append("─" * 70)
    lines.append("  AUDIT TRAIL")
    lines.append("─" * 70)
    for step in state_values.get("steps", []):
        lines.append(f"    • {step}")

    rounds = state_values.get("interview_round", 0)
    gaps_remaining = len(state_values.get("gaps", []))
    lines.append("")
    lines.append(f"  Rounds completed: {rounds}")
    lines.append(f"  Gaps remaining:  {gaps_remaining}")
    lines.append("=" * 70)

    text = "\n".join(lines) + "\n"
    Path(transcript_path).write_text(text)
    print(f"\n  Transcript written to: {transcript_path}")


def _print_final_summary(graph, config, transcript_path: str | None = None):
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

    if transcript_path:
        _write_transcript(state_values, transcript_path)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_interview(max_rounds=5)
