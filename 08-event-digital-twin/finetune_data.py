"""
08 · Digital Twin — Fine-Tuning Data Generation
════════════════════════════════════════════════

Generates training datasets for fine-tuning the extraction and Q&A models.

Outputs:
  - SFT training data  (statement → JSON extraction pairs)
  - DPO preference pairs (chosen vs rejected extractions)
  - PyKEEN triples      (Neo4j graph → TSV for KG embedding training)

Usage:
  python finetune_data.py --generate-sft              # SFT training set
  python finetune_data.py --generate-dpo              # DPO preference pairs
  python finetune_data.py --export-triples            # PyKEEN TSV export
  python finetune_data.py --augment                   # Paraphrase augmentation
  python finetune_data.py --all                       # All of the above

Prerequisites:
  - Neo4j running on bolt://localhost:7687 (neo4j / cabbage123)
  - Ollama running with qwen2.5:7b (for augmentation)
  - Graph populated via demo.py (for triple export)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

# ─── Paths ────────────────────────────────────────────────────────────────

STATEMENTS_DIR = Path(__file__).parent / "statements"
TRAINING_DIR = Path(__file__).parent / "training_data"
GOLD_DIR = TRAINING_DIR / "gold_annotations"

# ─── Connections ──────────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cabbage123"


# ═══════════════════════════════════════════════════════════════════════════
# Gold annotations — hand-corrected extraction targets per source
# ═══════════════════════════════════════════════════════════════════════════

GOLD_ANNOTATIONS: dict[str, dict[str, Any]] = {
    "king_street_collision.txt": {
        "entities": [
            {"id": "e1", "label": "Event", "properties": {"description": "Vehicle-cyclist collision at King Street/Queen's Road junction", "type": "collision"}},
            {"id": "p1", "label": "Person", "properties": {"name_or_description": "Witness on King Street", "role": "witness"}},
            {"id": "p2", "label": "Person", "properties": {"name_or_description": "Driver of red car", "role": "driver"}},
            {"id": "p3", "label": "Person", "properties": {"name_or_description": "Cyclist", "role": "victim"}},
            {"id": "p4", "label": "Person", "properties": {"name_or_description": "Woman who called ambulance", "role": "bystander"}},
            {"id": "v1", "label": "Vehicle", "properties": {"description": "Red car"}},
            {"id": "l1", "label": "Location", "properties": {"description": "King Street"}},
            {"id": "l2", "label": "Location", "properties": {"description": "Junction of King Street and Queen's Road"}},
            {"id": "l3", "label": "Location", "properties": {"description": "Queen's Road"}},
            {"id": "t1", "label": "Time", "properties": {"value": "approximately 2:15 PM Tuesday"}},
            {"id": "t2", "label": "Time", "properties": {"value": "ten minutes later"}},
        ],
        "relationships": [
            {"from_id": "e1", "rel_type": "OCCURRED_AT", "to_id": "l2"},
            {"from_id": "e1", "rel_type": "OCCURRED_AT_TIME", "to_id": "t1"},
            {"from_id": "p1", "rel_type": "WITNESSED", "to_id": "e1"},
            {"from_id": "p2", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p3", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p2", "rel_type": "DROVE", "to_id": "v1"},
            {"from_id": "v1", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p4", "rel_type": "WITNESSED", "to_id": "e1"},
        ],
    },
    "queen_road_witness.txt": {
        "entities": [
            {"id": "e1", "label": "Event", "properties": {"description": "Red hatchback strikes cyclist at King Street/Queen's Road junction", "type": "collision"}},
            {"id": "p1", "label": "Person", "properties": {"name_or_description": "Witness at bus stop on Queen's Road", "role": "witness"}},
            {"id": "p2", "label": "Person", "properties": {"name_or_description": "Driver - tall man", "role": "driver"}},
            {"id": "p3", "label": "Person", "properties": {"name_or_description": "Man on bicycle", "role": "victim"}},
            {"id": "p4", "label": "Person", "properties": {"name_or_description": "Woman in green coat", "role": "bystander"}},
            {"id": "v1", "label": "Vehicle", "properties": {"description": "Red hatchback, partial plate KV"}},
            {"id": "l1", "label": "Location", "properties": {"description": "Bus stop on Queen's Road south of King Street junction"}},
            {"id": "l2", "label": "Location", "properties": {"description": "King Street/Queen's Road junction"}},
            {"id": "t1", "label": "Time", "properties": {"value": "around quarter past two Tuesday afternoon"}},
        ],
        "relationships": [
            {"from_id": "e1", "rel_type": "OCCURRED_AT", "to_id": "l2"},
            {"from_id": "e1", "rel_type": "OCCURRED_AT_TIME", "to_id": "t1"},
            {"from_id": "p1", "rel_type": "WITNESSED", "to_id": "e1"},
            {"from_id": "p2", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p3", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p2", "rel_type": "DROVE", "to_id": "v1"},
            {"from_id": "v1", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p4", "rel_type": "WITNESSED", "to_id": "e1"},
        ],
    },
    "cctv_log.txt": {
        "entities": [
            {"id": "e1", "label": "Event", "properties": {"description": "Vehicle strikes cyclist at junction centre", "type": "collision"}},
            {"id": "p1", "label": "Person", "properties": {"name_or_description": "Male driver (tall, dark jacket)", "role": "driver"}},
            {"id": "p2", "label": "Person", "properties": {"name_or_description": "Cyclist crossing junction", "role": "victim"}},
            {"id": "p3", "label": "Person", "properties": {"name_or_description": "Female pedestrian (999 caller)", "role": "bystander"}},
            {"id": "p4", "label": "Person", "properties": {"name_or_description": "Second pedestrian (witness 1)", "role": "witness"}},
            {"id": "v1", "label": "Vehicle", "properties": {"description": "Red hatchback, partial plate KV68"}},
            {"id": "l1", "label": "Location", "properties": {"description": "Junction of King Street / Queen's Road"}},
            {"id": "t1", "label": "Time", "properties": {"value": "14:13:42"}},
            {"id": "t2", "label": "Time", "properties": {"value": "14:13:45"}},
            {"id": "t3", "label": "Time", "properties": {"value": "14:14:22"}},
            {"id": "t4", "label": "Time", "properties": {"value": "14:24:38"}},
        ],
        "relationships": [
            {"from_id": "e1", "rel_type": "OCCURRED_AT", "to_id": "l1"},
            {"from_id": "e1", "rel_type": "OCCURRED_AT_TIME", "to_id": "t2"},
            {"from_id": "p1", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p2", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p1", "rel_type": "DROVE", "to_id": "v1"},
            {"from_id": "v1", "rel_type": "INVOLVED_IN", "to_id": "e1"},
        ],
    },
    "paramedic_report.txt": {
        "entities": [
            {"id": "e1", "label": "Event", "properties": {"description": "Road traffic collision at King Street/Queen's Road junction", "type": "collision"}},
            {"id": "p1", "label": "Person", "properties": {"name_or_description": "Mr James Chen, age 34", "role": "victim"}},
            {"id": "p2", "label": "Person", "properties": {"name_or_description": "Female 999 caller", "role": "bystander"}},
            {"id": "p3", "label": "Person", "properties": {"name_or_description": "Male bystander", "role": "witness"}},
            {"id": "p4", "label": "Person", "properties": {"name_or_description": "Sarah Mitchell, PM-2847", "role": "paramedic"}},
            {"id": "l1", "label": "Location", "properties": {"description": "Junction of King Street and Queen's Road"}},
            {"id": "l2", "label": "Location", "properties": {"description": "Royal Infirmary"}},
            {"id": "t1", "label": "Time", "properties": {"value": "14:15"}},
            {"id": "t2", "label": "Time", "properties": {"value": "14:24"}},
            {"id": "t3", "label": "Time", "properties": {"value": "14:41"}},
            {"id": "o1", "label": "Observation", "properties": {"description": "Suspected fractured left clavicle, laceration to left knee, road rash to left forearm", "observation_type": "medical_assessment"}},
        ],
        "relationships": [
            {"from_id": "e1", "rel_type": "OCCURRED_AT", "to_id": "l1"},
            {"from_id": "e1", "rel_type": "OCCURRED_AT_TIME", "to_id": "t1"},
            {"from_id": "p1", "rel_type": "INVOLVED_IN", "to_id": "e1"},
            {"from_id": "p4", "rel_type": "RESPONDED_TO", "to_id": "e1"},
            {"from_id": "o1", "rel_type": "MADE_BY", "to_id": "p4"},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# SFT data generation
# ═══════════════════════════════════════════════════════════════════════════

def _load_statement(filename: str) -> str | None:
    """Load a statement file from the statements directory."""
    path = STATEMENTS_DIR / filename
    if path.exists():
        return path.read_text().strip()
    return None


def _build_system_prompt() -> str:
    """Build the system prompt from the active ontology."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06-ontologies"))
    from ontology_spec import OntologySpec  # noqa: E402
    from schema_org_event import SCHEMA_ORG_EVENT  # noqa: E402
    return SCHEMA_ORG_EVENT.build_extraction_prompt()


def generate_sft_dataset() -> list[dict]:
    """Generate SFT training data from gold annotations.

    Each example is a conversation: system prompt + user statement → assistant JSON.
    """
    system_prompt = _build_system_prompt()
    dataset = []

    for filename, gold in GOLD_ANNOTATIONS.items():
        statement = _load_statement(filename)
        if statement is None:
            print(f"  ⚠ Statement file not found: {filename}")
            continue

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": statement},
                {"role": "assistant", "content": json.dumps(gold, indent=2)},
            ],
            "source_file": filename,
        }
        dataset.append(example)
        print(f"  ✓ SFT example from {filename} ({len(gold['entities'])} entities, {len(gold['relationships'])} rels)")

    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# DPO preference pair generation
# ═══════════════════════════════════════════════════════════════════════════

def _inject_hallucination(gold: dict) -> dict:
    """Create a rejected example by injecting common extraction errors."""
    import copy
    rejected = copy.deepcopy(gold)
    entities = rejected.get("entities", [])

    # Error 1: Add a hallucinated entity not in the source
    entities.append({
        "id": f"h{len(entities)+1}",
        "label": "Person",
        "properties": {"name_or_description": "Unknown third-party witness", "role": "informant"},
    })

    # Error 2: Use a wrong label for an existing entity
    if entities:
        entities[0] = dict(entities[0])
        entities[0]["label"] = "Observation"  # Wrong label for an Event

    # Error 3: Add a hallucinated relationship
    rels = rejected.get("relationships", [])
    if len(entities) >= 2:
        rels.append({
            "from_id": entities[0]["id"],
            "rel_type": "CAUSED_BY",  # Not in schema
            "to_id": entities[1]["id"],
        })

    return rejected


def _make_incomplete(gold: dict) -> dict:
    """Create a rejected example by removing entities and relationships."""
    import copy
    rejected = copy.deepcopy(gold)

    # Remove ~40% of entities
    entities = rejected.get("entities", [])
    cutoff = max(1, len(entities) * 3 // 5)
    rejected["entities"] = entities[:cutoff]

    # Remove relationships that reference deleted entities
    kept_ids = {e["id"] for e in rejected["entities"]}
    rejected["relationships"] = [
        r for r in rejected.get("relationships", [])
        if r["from_id"] in kept_ids and r["to_id"] in kept_ids
    ]

    return rejected


def generate_dpo_dataset() -> list[dict]:
    """Generate DPO preference pairs: gold (chosen) vs corrupted (rejected)."""
    system_prompt = _build_system_prompt()
    dataset = []

    for filename, gold in GOLD_ANNOTATIONS.items():
        statement = _load_statement(filename)
        if statement is None:
            continue

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{statement}<|im_end|>\n<|im_start|>assistant\n"

        # Pair 1: gold vs hallucinated
        dataset.append({
            "prompt": prompt,
            "chosen": json.dumps(gold, indent=2),
            "rejected": json.dumps(_inject_hallucination(gold), indent=2),
            "error_type": "hallucination",
            "source_file": filename,
        })

        # Pair 2: gold vs incomplete
        dataset.append({
            "prompt": prompt,
            "chosen": json.dumps(gold, indent=2),
            "rejected": json.dumps(_make_incomplete(gold), indent=2),
            "error_type": "incomplete",
            "source_file": filename,
        })

        print(f"  ✓ DPO pairs from {filename} (2 pairs: hallucination + incomplete)")

    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# Paraphrase augmentation
# ═══════════════════════════════════════════════════════════════════════════

PARAPHRASE_STYLES = [
    ("formal", "Rewrite this witness statement in formal police report language. "
     "Keep ALL factual details identical. Return ONLY the rewritten text."),
    ("colloquial", "Rewrite this witness statement as if spoken informally. "
     "Keep ALL factual details identical. Return ONLY the rewritten text."),
    ("fragmented", "Rewrite this witness statement as brief, fragmented notes "
     "a witness might jot down quickly. Keep ALL factual details. Return ONLY the notes."),
]


def generate_augmented_dataset() -> list[dict]:
    """Generate paraphrased versions of statements for data augmentation.

    Requires Ollama running with qwen2.5:7b.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("  ⚠ langchain-ollama not installed — skipping augmentation")
        return []

    llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)
    system_prompt = _build_system_prompt()
    dataset = []

    for filename, gold in GOLD_ANNOTATIONS.items():
        statement = _load_statement(filename)
        if statement is None:
            continue

        for style_name, style_prompt in PARAPHRASE_STYLES:
            print(f"  ▸ Paraphrasing {filename} → {style_name}...")
            messages = [
                SystemMessage(content=style_prompt),
                HumanMessage(content=statement),
            ]
            try:
                result = llm.invoke(messages)
                paraphrased = result.content.strip()

                dataset.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": paraphrased},
                        {"role": "assistant", "content": json.dumps(gold, indent=2)},
                    ],
                    "source_file": filename,
                    "augmentation": style_name,
                })
                print(f"    ✓ {style_name}: {len(paraphrased)} chars")
            except Exception as e:
                print(f"    ⚠ Failed: {e}")

    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# PyKEEN triple export
# ═══════════════════════════════════════════════════════════════════════════

def export_triples_for_pykeen() -> Path:
    """Export the Neo4j graph as a TSV file for PyKEEN training.

    Format: head<TAB>relation<TAB>tail
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    triples = []

    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (a)-[r]->(b) "
                "WHERE NOT a:GraphVersion AND NOT b:GraphVersion "
                "RETURN labels(a)[0] + ':' + "
                "  coalesce(a.description, a.name_or_description, a.name, "
                "    a.value, a.summary, 'unknown') AS head, "
                "type(r) AS relation, "
                "labels(b)[0] + ':' + "
                "  coalesce(b.description, b.name_or_description, b.name, "
                "    b.value, b.summary, 'unknown') AS tail"
            )
            for rec in result:
                triples.append(f"{rec['head']}\t{rec['relation']}\t{rec['tail']}")
    finally:
        driver.close()

    if not triples:
        print("  ⚠ No triples found in graph — run the ingest pipeline first")
        return Path("")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAINING_DIR / "event_triples.tsv"
    out_path.write_text("\n".join(triples) + "\n")

    print(f"  ✓ Exported {len(triples)} triples → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Scenario generation templates
# ═══════════════════════════════════════════════════════════════════════════

SCENARIO_TEMPLATES = [
    {
        "type": "workplace_incident",
        "prompt": (
            "Generate a realistic witness statement about a workplace accident. "
            "Include: at least 3 people (witness, injured person, supervisor), "
            "a location, specific times, a sequence of events, and an emergency response. "
            "Make it 100-200 words. Return ONLY the statement text."
        ),
    },
    {
        "type": "environmental_spill",
        "prompt": (
            "Generate a realistic witness statement about a chemical spill at "
            "an industrial site. Include: at least 3 people, specific locations, "
            "times, vehicles, and an emergency response. "
            "Make it 100-200 words. Return ONLY the statement text."
        ),
    },
    {
        "type": "assault",
        "prompt": (
            "Generate a realistic witness statement about an assault in a public place. "
            "Include: at least 3 people (witness, victim, suspect), "
            "physical descriptions, specific times and location. "
            "Make it 100-200 words. Return ONLY the statement text."
        ),
    },
    {
        "type": "traffic_pedestrian",
        "prompt": (
            "Generate a realistic witness statement about a vehicle hitting a pedestrian "
            "at a crosswalk. Include: witness, driver, pedestrian, vehicle details, "
            "specific times, bystander response. "
            "Make it 100-200 words. Return ONLY the statement text."
        ),
    },
]


def generate_synthetic_scenarios(num_per_type: int = 2) -> list[dict]:
    """Generate synthetic incident scenarios using the LLM.

    Each generated statement is paired with a placeholder gold annotation
    that must be reviewed by a human before training.

    Requires Ollama running.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("  ⚠ langchain-ollama not installed — skipping scenario generation")
        return []

    llm = ChatOllama(model="qwen2.5:7b", temperature=0.8)
    system_prompt = _build_system_prompt()
    scenarios = []

    for template in SCENARIO_TEMPLATES:
        for i in range(num_per_type):
            print(f"  ▸ Generating {template['type']} scenario {i+1}/{num_per_type}...")
            try:
                # Step 1: Generate statement
                result = llm.invoke([HumanMessage(content=template["prompt"])])
                statement = result.content.strip()

                # Step 2: Extract with current pipeline (becomes draft, not gold)
                extract_result = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=statement),
                ])
                content = extract_result.content.strip()
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
                try:
                    draft_extraction = json.loads(content)
                except json.JSONDecodeError:
                    draft_extraction = {"entities": [], "relationships": [], "_parse_error": True}

                scenarios.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": statement},
                        {"role": "assistant", "content": json.dumps(draft_extraction, indent=2)},
                    ],
                    "scenario_type": template["type"],
                    "needs_review": True,
                })
                n_ents = len(draft_extraction.get("entities", []))
                print(f"    ✓ Generated ({len(statement)} chars, {n_ents} entities) — NEEDS HUMAN REVIEW")
            except Exception as e:
                print(f"    ⚠ Failed: {e}")

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _save_jsonl(data: list[dict], filename: str) -> Path:
    """Save a list of dicts as JSONL."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAINING_DIR / filename
    with open(out_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(data)} examples → {out_path}")
    return out_path


def _save_gold_annotations() -> None:
    """Save gold annotations as individual JSON files for review."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for filename, gold in GOLD_ANNOTATIONS.items():
        statement = _load_statement(filename)
        record = {
            "source_file": filename,
            "statement": statement or "(file not found)",
            "gold_extraction": gold,
        }
        out_path = GOLD_DIR / filename.replace(".txt", ".json")
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"  ✓ Saved {len(GOLD_ANNOTATIONS)} gold annotations → {GOLD_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning data generation")
    parser.add_argument("--generate-sft", action="store_true", help="Generate SFT training set")
    parser.add_argument("--generate-dpo", action="store_true", help="Generate DPO preference pairs")
    parser.add_argument("--export-triples", action="store_true", help="Export graph triples for PyKEEN")
    parser.add_argument("--augment", action="store_true", help="Generate paraphrased augmentations (requires Ollama)")
    parser.add_argument("--scenarios", action="store_true", help="Generate synthetic scenarios (requires Ollama)")
    parser.add_argument("--scenarios-per-type", type=int, default=2, help="Scenarios per incident type")
    parser.add_argument("--all", action="store_true", help="Generate all training data")
    parser.add_argument("--save-gold", action="store_true", help="Save gold annotations as JSON files")
    args = parser.parse_args()

    if not any([args.generate_sft, args.generate_dpo, args.export_triples,
                args.augment, args.scenarios, args.all, args.save_gold]):
        parser.print_help()
        return

    print(f"\n{'═' * 60}")
    print(f"  Fine-Tuning Data Generation")
    print(f"{'═' * 60}\n")

    if args.save_gold or args.all:
        print("── Gold Annotations ──")
        _save_gold_annotations()
        print()

    if args.generate_sft or args.all:
        print("── SFT Training Data ──")
        sft_data = generate_sft_dataset()
        _save_jsonl(sft_data, "sft_extraction.jsonl")
        print()

    if args.generate_dpo or args.all:
        print("── DPO Preference Pairs ──")
        dpo_data = generate_dpo_dataset()
        _save_jsonl(dpo_data, "dpo_pairs.jsonl")
        print()

    if args.augment or args.all:
        print("── Paraphrase Augmentation ──")
        aug_data = generate_augmented_dataset()
        if aug_data:
            _save_jsonl(aug_data, "sft_augmented.jsonl")
        print()

    if args.scenarios or args.all:
        print("── Synthetic Scenarios ──")
        scenario_data = generate_synthetic_scenarios(args.scenarios_per_type)
        if scenario_data:
            _save_jsonl(scenario_data, "sft_scenarios_draft.jsonl")
            print("  ⚠ Draft scenarios need human review before use in training!")
        print()

    if args.export_triples or args.all:
        print("── PyKEEN Triple Export ──")
        export_triples_for_pykeen()
        print()

    print(f"{'═' * 60}")
    print(f"  Complete. Training data in: {TRAINING_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
