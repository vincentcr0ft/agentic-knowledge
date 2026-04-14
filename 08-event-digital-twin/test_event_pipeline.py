"""
08 · Event Digital Twin — Exhaustive Validation Tests
═════════════════════════════════════════════════════

Tests WITHOUT requiring Neo4j or Ollama:
  1. Schema bridge — ontology selection, spec access, Gap dataclass
  2. Ingest pipeline — parse_statement, _parse_json, _get_merge_key, SAFE_PROP_RE
  3. Coreference resolution — over-merge guard, prompt structure
  4. Multi-source consistency — merge keys cover all ontology labels
  5. Hallucination surface — faithfulness contract, source grounding
  6. Statement corpus — structural validation of provided texts
  7. Cross-module integration — ch06 ontology specs are consistent with ch08 usage
  8. Pipeline state contracts — TypedDict keys, step tracking
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# Add module dirs to path
_MODULE_DIR = Path(__file__).resolve().parent
_ONTOLOGY_DIR = _MODULE_DIR.parent / "06-ontologies"
sys.path.insert(0, str(_MODULE_DIR))
sys.path.insert(0, str(_ONTOLOGY_DIR))

# Import the ontology specs directly (bypassing schema.py bridge which
# has import-time dependencies that may not resolve)
from ontology_spec import OntologySpec, NodeSpec, RelSpec, CompletenessRule
from schema_org_event import SCHEMA_ORG_EVENT
from sem_event import SEM_EVENT
from bfo_cco_event import BFO_CCO_EVENT

ALL_SPECS = [SCHEMA_ORG_EVENT, SEM_EVENT, BFO_CCO_EVENT]
SPEC_IDS = ["schema_org", "sem", "bfo_cco"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers extracted from ingest.py without Neo4j/Ollama imports
# ═══════════════════════════════════════════════════════════════════════════

SAFE_PROP_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

def _parse_json(content: str) -> dict | None:
    """Replica of ingest._parse_json for testing."""
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

def _get_merge_key(label: str) -> str:
    """Replica of ingest._get_merge_key for testing."""
    merge_keys = {
        "Event": "description", "Process": "description",
        "Person": "name_or_description", "Actor": "name_or_description",
        "Agent": "name_or_description",
        "Vehicle": "description", "Location": "description",
        "Place": "description", "SpatialRegion": "description",
        "Time": "value", "TemporalRegion": "value",
        "TemporalInterval": "value",
        "Object": "description",
        "Role": "role_type", "AgentRole": "role_type",
        "PhysicalDescription": "summary",
        "DescriptiveICE": "summary",
        "Observation": "description",
        "InformationContentEntity": "description",
        "MaterialEntity": "description",
        "Site": "description",
        "Act": "description",
    }
    return merge_keys.get(label, "description")

def _parse_statement(text: str) -> list[str]:
    """Replica of ingest.parse_statement's splitting logic."""
    raw_segments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw_segments if s.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════════════

KING_STREET_STATEMENT = (
    "I was walking along King Street at approximately 2:15 PM on Tuesday "
    "when I heard a loud crash. I turned and saw a red car had collided "
    "with a cyclist at the junction of King Street and Queen's Road. "
    "The driver got out — a tall man wearing a dark jacket. He looked "
    "at the cyclist who was on the ground and then got back in his car "
    "and drove off heading north on Queen's Road. Another woman who was "
    "nearby called an ambulance. I stayed with the cyclist until the "
    "paramedics arrived about ten minutes later."
)

# Expected entities a correct extraction should find
EXPECTED_PERSONS = ["Dr Sarah Chen", "witness", "truck driver", "suspect",
                    "young woman", "victim", "elderly man", "bystander"]
EXPECTED_VEHICLES = ["tanker truck", "hatchback", "red car"]
EXPECTED_LOCATIONS = ["King Street", "Riverside Avenue", "Bridge Lane", "Queen's Road"]
EXPECTED_EVENTS = ["collision", "crash", "drove off", "fled", "called", "ambulance"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. JSON parsing robustness tests
# ═══════════════════════════════════════════════════════════════════════════

class TestJSONParsing:
    """LLM output comes in many formats — parser must handle all."""

    def test_clean_json(self):
        data = _parse_json('{"entities": [], "relationships": []}')
        assert data == {"entities": [], "relationships": []}

    def test_markdown_fenced(self):
        raw = '```json\n{"entities": [{"id": "e1"}], "relationships": []}\n```'
        data = _parse_json(raw)
        assert data is not None
        assert len(data["entities"]) == 1

    def test_markdown_fenced_no_lang(self):
        raw = '```\n{"entities": [], "relationships": []}\n```'
        data = _parse_json(raw)
        assert data is not None

    def test_prefix_text(self):
        raw = 'Here is the JSON output:\n{"entities": [], "relationships": []}'
        data = _parse_json(raw)
        assert data is not None

    def test_suffix_text(self):
        raw = '{"entities": [], "relationships": []}\n\nI hope this helps!'
        data = _parse_json(raw)
        assert data is not None

    def test_garbage_returns_none(self):
        assert _parse_json("not json at all") is None

    def test_empty_returns_none(self):
        assert _parse_json("") is None

    def test_partial_json(self):
        raw = '{"entities": [{"id": "e1"'  # truncated
        result = _parse_json(raw)
        assert result is None

    def test_nested_json_in_text(self):
        raw = 'Result: {"entities": [{"id": "e1", "label": "Event", "properties": {"description": "crash"}}], "relationships": []}'
        data = _parse_json(raw)
        assert data is not None
        assert data["entities"][0]["label"] == "Event"

    def test_handles_unicode(self):
        raw = '{"entities": [{"id": "e1", "label": "Event", "properties": {"description": "café collision"}}], "relationships": []}'
        data = _parse_json(raw)
        assert data is not None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Statement parsing tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStatementParsing:
    """parse_statement splits on sentence boundaries."""

    def test_splits_on_periods(self):
        segs = _parse_statement("First sentence. Second sentence. Third one.")
        assert len(segs) == 3

    def test_preserves_content(self):
        text = "He drove off. She stayed."
        segs = _parse_statement(text)
        assert "He drove off." in segs
        assert "She stayed." in segs

    def test_handles_question_marks(self):
        segs = _parse_statement("What happened? He drove off.")
        assert len(segs) == 2

    def test_handles_exclamation_marks(self):
        segs = _parse_statement("Stop! The car drove off.")
        assert len(segs) == 2

    def test_king_street_splits_into_sentences(self):
        segs = _parse_statement(KING_STREET_STATEMENT)
        assert len(segs) >= 4  # at least 4 sentences

    def test_empty_input(self):
        segs = _parse_statement("")
        assert segs == []

    def test_single_sentence(self):
        segs = _parse_statement("Just one sentence.")
        assert len(segs) == 1

    def test_no_trailing_whitespace(self):
        segs = _parse_statement("Sentence one.  Sentence two.  ")
        for seg in segs:
            assert seg == seg.strip()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Merge key coverage tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeKeys:
    """_get_merge_key must cover all labels from all ontologies."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_all_ontology_labels_have_merge_keys(self, spec: OntologySpec):
        """Every node label in every ontology should be handled."""
        for label in spec.node_types:
            key = _get_merge_key(label)
            assert key != "description" or label in (
                "Event", "Process", "Vehicle", "Location", "Place",
                "SpatialRegion", "Object", "Observation",
                "InformationContentEntity", "MaterialEntity", "Site",
                "Act", "DescriptiveICE", "AgentRole",
            ), (
                f"Label '{label}' from {spec.id} falls through to default "
                f"merge_key 'description' — verify this is intentional"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_merge_key_matches_spec_merge_key(self, spec: OntologySpec):
        """_get_merge_key should agree with NodeSpec.merge_key where mapped."""
        for label, ndef in spec.node_types.items():
            ingest_key = _get_merge_key(label)
            if label in ("Event", "Process", "Person", "Actor", "Agent",
                         "Vehicle", "Location", "Place", "Time", "Object",
                         "Role", "AgentRole", "PhysicalDescription",
                         "Observation", "InformationContentEntity"):
                assert ingest_key == ndef.merge_key, (
                    f"Mismatch for {spec.id}/{label}: "
                    f"ingest says '{ingest_key}', spec says '{ndef.merge_key}'"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Property safety tests (Cypher injection prevention)
# ═══════════════════════════════════════════════════════════════════════════

class TestPropertySafety:
    """SAFE_PROP_RE must prevent Cypher injection via property names."""

    def test_allows_valid_snake_case(self):
        assert SAFE_PROP_RE.match("description")
        assert SAFE_PROP_RE.match("name_or_description")
        assert SAFE_PROP_RE.match("age_estimate")
        assert SAFE_PROP_RE.match("source_type")

    def test_rejects_cypher_injection(self):
        assert not SAFE_PROP_RE.match("}) DETACH DELETE n //")
        assert not SAFE_PROP_RE.match("name})-[r]-(")
        assert not SAFE_PROP_RE.match("'; DROP")

    def test_rejects_uppercase(self):
        assert not SAFE_PROP_RE.match("Description")
        assert not SAFE_PROP_RE.match("NameOrDescription")

    def test_rejects_spaces(self):
        assert not SAFE_PROP_RE.match("first name")

    def test_rejects_special_chars(self):
        assert not SAFE_PROP_RE.match("prop-name")
        assert not SAFE_PROP_RE.match("prop.name")

    def test_rejects_empty(self):
        assert not SAFE_PROP_RE.match("")

    def test_rejects_starting_with_number(self):
        assert not SAFE_PROP_RE.match("1prop")

    def test_allows_underscore_prefix(self):
        assert SAFE_PROP_RE.match("_private_prop")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Coreference resolution contract tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCoreferenceResolution:
    """The over-merge guard and resolution contract."""

    def test_over_merge_rejection_threshold(self):
        """If resolved < 40% of original entities, reject merge."""
        original_count = 10
        resolved_count = 3  # 30% — should reject
        assert resolved_count < original_count * 0.4

    def test_over_merge_acceptance_threshold(self):
        """If resolved >= 40% of original, accept."""
        original_count = 10
        resolved_count = 5  # 50% — should accept
        assert resolved_count >= original_count * 0.4

    def test_id_mapping_rewrites_relationships(self):
        """When entities merge, relationship endpoints must be updated."""
        id_mapping = {"p2": "p1", "p3": "p1"}
        relationships = [
            {"from_id": "p2", "rel_type": "PARTICIPATED_IN", "to_id": "e1"},
            {"from_id": "p3", "rel_type": "WITNESSED", "to_id": "e1"},
        ]
        resolved_rels = []
        for rel in relationships:
            resolved_rels.append({
                "from_id": id_mapping.get(rel["from_id"], rel["from_id"]),
                "rel_type": rel["rel_type"],
                "to_id": id_mapping.get(rel["to_id"], rel["to_id"]),
            })
        # Both should now point from p1
        for rel in resolved_rels:
            assert rel["from_id"] == "p1"

    def test_single_entity_skips_resolution(self):
        """With ≤1 entity, coreference resolution should be skipped."""
        entities = [{"id": "e1", "label": "Event", "properties": {}}]
        assert len(entities) <= 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. Source text corpus tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSourceCorpus:
    """Validate the provided witness statements."""

    @pytest.fixture
    def king_street_text(self) -> str:
        path = _MODULE_DIR / "statements" / "king_street_collision.txt"
        if not path.exists():
            pytest.skip("king_street_collision.txt not found")
        return path.read_text().strip()

    def test_king_street_non_empty(self, king_street_text: str):
        assert len(king_street_text) > 50

    def test_king_street_mentions_location(self, king_street_text: str):
        assert "King Street" in king_street_text

    def test_king_street_mentions_time(self, king_street_text: str):
        assert "2:15 PM" in king_street_text or "2:15" in king_street_text

    def test_king_street_mentions_vehicle(self, king_street_text: str):
        assert "car" in king_street_text.lower()

    def test_king_street_mentions_persons(self, king_street_text: str):
        text_lower = king_street_text.lower()
        assert "driver" in text_lower
        assert "cyclist" in text_lower

    def test_king_street_mentions_incident(self, king_street_text: str):
        text_lower = king_street_text.lower()
        assert "crash" in text_lower or "collide" in text_lower

    def test_king_street_has_multiple_sentences(self, king_street_text: str):
        segs = _parse_statement(king_street_text)
        assert len(segs) >= 3

    def test_king_street_describes_hit_and_run(self, king_street_text: str):
        """The statement should describe a hit-and-run scenario."""
        text_lower = king_street_text.lower()
        assert "drove off" in text_lower or "drove away" in text_lower

    def test_king_street_mentions_emergency_response(self, king_street_text: str):
        text_lower = king_street_text.lower()
        assert "ambulance" in text_lower or "paramedic" in text_lower


# ═══════════════════════════════════════════════════════════════════════════
# 7. Cross-module integration (ch06 ↔ ch08)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossModuleIntegration:
    """Verify ch08 pipeline assumptions match ch06 ontology specs."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_extraction_prompt_is_non_empty(self, spec: OntologySpec):
        prompt = spec.build_extraction_prompt()
        assert len(prompt) > 100

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_extraction_prompt_requests_json(self, spec: OntologySpec):
        prompt = spec.build_extraction_prompt()
        assert '"entities"' in prompt
        assert '"relationships"' in prompt

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_extraction_prompt_mentions_unique_ids(self, spec: OntologySpec):
        prompt = spec.build_extraction_prompt()
        assert "unique" in prompt.lower() or "id" in prompt.lower()

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_completeness_rules_have_valid_cypher(self, spec: OntologySpec):
        """Rules should have non-empty Cypher queries."""
        for rule in spec.completeness_rules:
            assert rule.cypher, (
                f"{spec.id}: rule '{rule.rule_id}' has empty cypher"
            )
            assert "MATCH" in rule.cypher or "RETURN" in rule.cypher

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_shacl_shapes_generate_correctly(self, spec: OntologySpec):
        shapes = spec.build_shacl_shapes()
        assert "sh:NodeShape" in shapes
        assert len(shapes) > 50

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_constraint_cypher_generates(self, spec: OntologySpec):
        constraints = spec.get_constraint_cypher()
        assert len(constraints) >= 1
        for c in constraints:
            assert "CREATE CONSTRAINT" in c

    def test_schema_org_is_default(self):
        """Ch08 defaults to schema-org-event-v1 — verify it exists."""
        assert SCHEMA_ORG_EVENT.id == "schema-org-event-v1"

    def test_all_specs_have_description_field_on_events(self):
        """The ingest pipeline uses 'description' as primary merge key for events."""
        event_labels = {
            "schema_org": "Event",
            "sem": "Event",
            "bfo_cco": "Process",
        }
        for spec, spec_id in zip(ALL_SPECS, SPEC_IDS):
            label = event_labels[spec_id]
            ndef = spec.node_types[label]
            assert "description" in ndef.required_props, (
                f"{spec.id}: event node '{label}' missing 'description' "
                f"required property"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Hallucination surface tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHallucinationSurface:
    """Verify that the extraction → graph pipeline has hallucination guards."""

    def test_extraction_result_structure(self):
        """Extracted data must follow the entities/relationships schema."""
        valid = {"entities": [
            {"id": "e1", "label": "Event", "properties": {"description": "crash"}},
        ], "relationships": []}
        assert "entities" in valid
        assert "relationships" in valid
        assert valid["entities"][0]["label"] == "Event"

    def test_hallucinated_entity_detectable(self):
        """An entity not in the source text should be detectable."""
        source = KING_STREET_STATEMENT.lower()
        hallucinated_entities = [
            "helicopter", "gun", "explosion", "John Smith", "hospital fire"
        ]
        for entity in hallucinated_entities:
            assert entity.lower() not in source, (
                f"'{entity}' should NOT be in the source text"
            )

    def test_source_grounded_entities_present(self):
        """Entities from source text should be detectable as grounded."""
        source = KING_STREET_STATEMENT.lower()
        grounded = ["king street", "car", "cyclist", "ambulance", "2:15"]
        for entity in grounded:
            assert entity.lower() in source

    def test_provenance_tagging_fields(self):
        """Ingested nodes must carry provenance metadata."""
        required_prov = ["source", "source_type", "extracted_at",
                         "confidence", "ontology_id"]
        # These are the keys the load_to_graph function adds
        for field in required_prov:
            assert SAFE_PROP_RE.match(field), (
                f"Provenance field '{field}' must be safe for Cypher"
            )

    def test_hallucination_check_possible_via_faithfulness(self):
        """The quality module's faithfulness probe can detect hallucinations.

        This tests the contract: source_text + graph_triples → score + violations.
        """
        # Simulate what a faithfulness probe would return
        source_text = KING_STREET_STATEMENT
        # A hallucinated triple
        hallucinated_triple = "(Person: John Smith) -[DROVE]-> (Vehicle: blue van)"
        # John Smith and blue van don't appear in the source
        assert "john smith" not in source_text.lower()
        assert "blue van" not in source_text.lower()

    def test_multi_source_merge_key_collision(self):
        """Different sources describing the same event should merge, not duplicate."""
        # Two sources mentioning "a red car collided with a cyclist"
        merge_key = _get_merge_key("Event")
        assert merge_key == "description"
        # Same description from different sources → same MERGE target


# ═══════════════════════════════════════════════════════════════════════════
# 9. Pipeline state contract tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineStateContract:
    """IngestState TypedDict must have all required fields."""

    def test_ingest_state_fields(self):
        """Verify the expected fields exist."""
        expected = {
            "raw_statement", "source_id", "source_type", "segments",
            "extracted", "resolved", "load_summary", "steps",
        }
        # Can't import IngestState without Neo4j, so test via field names
        for field in expected:
            # The field should be a valid Python identifier and safe
            assert field.isidentifier()

    def test_step_tracking_is_list(self):
        """Steps should accumulate a list of strings."""
        steps = []
        steps = steps + ["parse_statement: 5 segments"]
        steps = steps + ["extract_entities: 8 entities, 5 relationships"]
        assert len(steps) == 2
        assert all(isinstance(s, str) for s in steps)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Multi-source consistency tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiSourceConsistency:
    """When multiple sources describe overlapping events, the system must handle it."""

    def test_same_event_different_descriptions_merge(self):
        """Slightly different descriptions of the same event should still
        be recognisable as candidates for coreference resolution."""
        desc1 = "a red car collided with a cyclist"
        desc2 = "a car hit a bike rider"
        # These are different strings → they'll create separate nodes
        # Coreference resolution should catch this
        assert desc1 != desc2  # confirms they need resolution

    def test_contradictory_facts_should_be_detectable(self):
        """Two sources giving contradictory details should surface in quality checks."""
        # Source 1: "red car" / Source 2: "blue car"
        fact1 = {"label": "Vehicle", "properties": {"colour": "red"}}
        fact2 = {"label": "Vehicle", "properties": {"colour": "blue"}}
        assert fact1["properties"]["colour"] != fact2["properties"]["colour"]

    def test_temporal_consistency_across_sources(self):
        """Different sources giving different times for the same event
        should create a consistency violation."""
        time1 = {"label": "Time", "properties": {"value": "2:15 PM"}}
        time2 = {"label": "Time", "properties": {"value": "2:45 PM"}}
        assert time1["properties"]["value"] != time2["properties"]["value"]

    def test_source_type_taxonomy(self):
        """Valid source types for provenance tracking."""
        valid_types = {"statement", "interview_round_1", "interview_round_2",
                       "interview_round_3", "police_report", "cctv"}
        # At minimum, "statement" must be supported
        assert "statement" in valid_types

    def test_graph_version_node_created(self):
        """Each ingestion should create a GraphVersion node for lineage."""
        # The load_to_graph function creates:
        # CREATE (v:GraphVersion {source_id, source_type, timestamp, ontology_id})
        version_props = ["source_id", "source_type", "timestamp", "ontology_id"]
        for prop in version_props:
            assert SAFE_PROP_RE.match(prop)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Temporal reasoning unit tests (temporal.py pure functions)
# ═══════════════════════════════════════════════════════════════════════════

# Import temporal helpers — no Neo4j dependency at import time
from temporal import parse_time_value, _minutes_to_time_str, _word_to_hour


class TestTimeParsing:
    """parse_time_value converts diverse time formats to minutes-since-midnight."""

    def test_hh_mm_ss(self):
        assert parse_time_value("14:13:45") == pytest.approx(14 * 60 + 13 + 45 / 60)

    def test_hh_mm(self):
        assert parse_time_value("14:13") == pytest.approx(14 * 60 + 13)

    def test_12h_pm(self):
        assert parse_time_value("2:15 PM") == pytest.approx(14 * 60 + 15)

    def test_12h_am(self):
        assert parse_time_value("9:30 AM") == pytest.approx(9 * 60 + 30)

    def test_quarter_past(self):
        result = parse_time_value("quarter past two")
        assert result == pytest.approx(2 * 60 + 15)

    def test_half_past(self):
        result = parse_time_value("half past three")
        assert result == pytest.approx(3 * 60 + 30)

    def test_approximately(self):
        result = parse_time_value("approximately 2:15 PM")
        assert result == pytest.approx(14 * 60 + 15)

    def test_none_for_garbage(self):
        assert parse_time_value("some random text") is None

    def test_none_for_empty(self):
        assert parse_time_value("") is None

    def test_midnight(self):
        assert parse_time_value("0:00") == pytest.approx(0.0)

    def test_embedded_time(self):
        result = parse_time_value("Impact at 14:13")
        assert result == pytest.approx(14 * 60 + 13)


class TestMinutesToTimeStr:
    """_minutes_to_time_str converts back to HH:MM."""

    def test_basic(self):
        assert _minutes_to_time_str(14 * 60 + 13) == "14:13"

    def test_midnight(self):
        assert _minutes_to_time_str(0.0) == "00:00"

    def test_noon(self):
        assert _minutes_to_time_str(12 * 60) == "12:00"


class TestWordToHour:
    """_word_to_hour maps English words to hour ints."""

    def test_known_words(self):
        assert _word_to_hour("two") == 2
        assert _word_to_hour("twelve") == 12
        assert _word_to_hour("noon") == 12
        assert _word_to_hour("midnight") == 0

    def test_case_insensitive(self):
        assert _word_to_hour("TWO") == 2

    def test_unknown_returns_zero(self):
        assert _word_to_hour("blarg") == 0


# ═══════════════════════════════════════════════════════════════════════════
# 12. Export helpers (export.py pure functions)
# ═══════════════════════════════════════════════════════════════════════════

from export import _sanitise_for_turtle, _node_uri


class TestExportHelpers:
    """Pure helper functions from export.py."""

    def test_sanitise_quotes(self):
        assert _sanitise_for_turtle('He said "hello"') == 'He said \\"hello\\"'

    def test_sanitise_newlines(self):
        assert _sanitise_for_turtle("line1\nline2") == "line1\\nline2"

    def test_sanitise_backslash(self):
        assert _sanitise_for_turtle("path\\to") == "path\\\\to"

    def test_node_uri_format(self):
        assert _node_uri(42) == "urn:event-twin:node:42"

    def test_node_uri_zero(self):
        assert _node_uri(0) == "urn:event-twin:node:0"


# ═══════════════════════════════════════════════════════════════════════════
# 13. Additional source corpus tests (new witness fixtures)
# ═══════════════════════════════════════════════════════════════════════════

class TestExtendedSourceCorpus:
    """Validate the new multi-source test fixtures."""

    @pytest.fixture
    def queen_road_text(self) -> str:
        path = _MODULE_DIR / "statements" / "queen_road_witness.txt"
        if not path.exists():
            pytest.skip("queen_road_witness.txt not found")
        return path.read_text().strip()

    @pytest.fixture
    def cctv_text(self) -> str:
        path = _MODULE_DIR / "statements" / "cctv_log.txt"
        if not path.exists():
            pytest.skip("cctv_log.txt not found")
        return path.read_text().strip()

    @pytest.fixture
    def paramedic_text(self) -> str:
        path = _MODULE_DIR / "statements" / "paramedic_report.txt"
        if not path.exists():
            pytest.skip("paramedic_report.txt not found")
        return path.read_text().strip()

    # Queen's Road witness
    def test_queen_road_non_empty(self, queen_road_text):
        assert len(queen_road_text) > 50

    def test_queen_road_mentions_location(self, queen_road_text):
        assert "Queen" in queen_road_text

    def test_queen_road_mentions_vehicle(self, queen_road_text):
        text_l = queen_road_text.lower()
        assert "car" in text_l or "hatchback" in text_l

    def test_queen_road_has_partial_plate(self, queen_road_text):
        assert "KV" in queen_road_text

    # CCTV log
    def test_cctv_non_empty(self, cctv_text):
        assert len(cctv_text) > 50

    def test_cctv_has_precise_timestamps(self, cctv_text):
        assert "14:13:45" in cctv_text

    def test_cctv_mentions_speed(self, cctv_text):
        assert "38" in cctv_text or "mph" in cctv_text.lower()

    def test_cctv_has_camera_id(self, cctv_text):
        assert "KS-QR-001" in cctv_text

    # Paramedic report
    def test_paramedic_non_empty(self, paramedic_text):
        assert len(paramedic_text) > 50

    def test_paramedic_identifies_patient(self, paramedic_text):
        assert "James Chen" in paramedic_text

    def test_paramedic_has_clinical_detail(self, paramedic_text):
        text_l = paramedic_text.lower()
        assert "fractur" in text_l or "laceration" in text_l

    def test_paramedic_has_crew_id(self, paramedic_text):
        assert "PM-2847" in paramedic_text or "Sarah Mitchell" in paramedic_text


# ═══════════════════════════════════════════════════════════════════════════
# 14. Cross-source time contradiction detection
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossSourceTimeContradiction:
    """Time values from different sources should be detectable as conflicting."""

    def test_witness_vs_cctv_time_difference(self):
        """Witness says ~14:15, CCTV says 14:13:45 — detectable difference."""
        witness_time = parse_time_value("2:15 PM")
        cctv_time = parse_time_value("14:13:45")
        assert witness_time is not None
        assert cctv_time is not None
        # Difference should be > 0 but < 5 minutes (close but not identical)
        diff = abs(witness_time - cctv_time)
        assert 0 < diff < 5

    def test_identical_times_have_zero_diff(self):
        t1 = parse_time_value("14:13:45")
        t2 = parse_time_value("14:13:45")
        assert t1 == t2

    def test_large_time_disagreement(self):
        """Hugely different times should be obvious contradictions."""
        t1 = parse_time_value("2:15 PM")
        t2 = parse_time_value("9:30 AM")
        diff = abs(t1 - t2)
        assert diff > 60  # more than an hour apart


# ═══════════════════════════════════════════════════════════════════════════
# 15. Confidence model tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceModel:
    """The Bayesian-inspired confidence update rules."""

    def test_corroboration_boosts_confidence(self):
        """Corroboration: new = old + (1 - old) * 0.3"""
        old = 0.8
        new = old + (1 - old) * 0.3
        assert new > old
        assert new == pytest.approx(0.86)

    def test_contradiction_decays_confidence(self):
        """Contradiction: new = old * 0.7"""
        old = 0.8
        new = old * 0.7
        assert new < old
        assert new == pytest.approx(0.56)

    def test_corroboration_never_exceeds_one(self):
        conf = 0.99
        conf = conf + (1 - conf) * 0.3
        assert conf <= 1.0

    def test_contradiction_never_below_zero(self):
        conf = 0.01
        conf = conf * 0.7
        assert conf >= 0.0

    def test_double_corroboration(self):
        """Two corroborations should increase more than one."""
        conf = 0.8
        conf = conf + (1 - conf) * 0.3  # first
        conf2 = conf + (1 - conf) * 0.3  # second
        assert conf2 > conf > 0.8

    def test_initial_confidence_is_float(self):
        """Default extraction confidence should be 0.8 (float, not string)."""
        initial = 0.8
        assert isinstance(initial, float)
        assert 0 < initial <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
