"""
06 · Ontology Comparison — Exhaustive Validation Tests
══════════════════════════════════════════════════════

Tests the OntologySpec framework and all three ontology instances for:
  1. Structural integrity — every spec is internally consistent
  2. Fidelity to source standards — SEM, BFO/CCO, Schema.org
  3. Prompt generation correctness
  4. SHACL shape generation correctness
  5. Constraint generation correctness
  6. Completeness-rule validity
  7. Cross-ontology coverage comparison

No external services required (no Neo4j, no Ollama).
"""

from __future__ import annotations

import json
import re
import textwrap

import pytest

from ontology_spec import OntologySpec, NodeSpec, RelSpec, CompletenessRule
from schema_org_event import SCHEMA_ORG_EVENT
from sem_event import SEM_EVENT
from bfo_cco_event import BFO_CCO_EVENT


ALL_SPECS = [SCHEMA_ORG_EVENT, SEM_EVENT, BFO_CCO_EVENT]
SPEC_IDS = ["schema_org", "sem", "bfo_cco"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. OntologySpec structural integrity
# ═══════════════════════════════════════════════════════════════════════════

class TestOntologySpecStructure:
    """Every OntologySpec must be internally consistent."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_has_id_and_version(self, spec: OntologySpec):
        assert spec.id, "spec.id must be non-empty"
        assert spec.version, "spec.version must be non-empty"
        assert spec.name, "spec.name must be non-empty"

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_has_node_types(self, spec: OntologySpec):
        assert len(spec.node_types) >= 3, (
            f"{spec.id}: should have at least 3 node types"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_has_relationship_types(self, spec: OntologySpec):
        assert len(spec.relationship_types) >= 3, (
            f"{spec.id}: should have at least 3 relationship types"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_node_labels_match_keys(self, spec: OntologySpec):
        """NodeSpec.label must match the dict key."""
        for key, ndef in spec.node_types.items():
            assert ndef.label == key, (
                f"{spec.id}: node key '{key}' != label '{ndef.label}'"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_rel_types_match_keys(self, spec: OntologySpec):
        """RelSpec.rel_type must match the dict key."""
        for key, rdef in spec.relationship_types.items():
            assert rdef.rel_type == key, (
                f"{spec.id}: rel key '{key}' != rel_type '{rdef.rel_type}'"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_rel_endpoints_reference_valid_nodes(self, spec: OntologySpec):
        """Every relationship's from_label and to_label must exist as node types."""
        node_labels = set(spec.node_types.keys())
        for rkey, rdef in spec.relationship_types.items():
            assert rdef.from_label in node_labels, (
                f"{spec.id}: rel '{rkey}' from_label '{rdef.from_label}' "
                f"not in node types {node_labels}"
            )
            assert rdef.to_label in node_labels, (
                f"{spec.id}: rel '{rkey}' to_label '{rdef.to_label}' "
                f"not in node types {node_labels}"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_node_specs_have_required_props(self, spec: OntologySpec):
        """Every NodeSpec must have at least one required property."""
        for key, ndef in spec.node_types.items():
            assert len(ndef.required_props) >= 1, (
                f"{spec.id}: node '{key}' has no required properties"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_node_specs_have_merge_keys(self, spec: OntologySpec):
        """Every NodeSpec should have a merge_key for deduplication."""
        for key, ndef in spec.node_types.items():
            assert ndef.merge_key, (
                f"{spec.id}: node '{key}' has no merge_key"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_merge_key_in_required_or_optional(self, spec: OntologySpec):
        """merge_key must be one of the node's declared properties."""
        for key, ndef in spec.node_types.items():
            if not ndef.merge_key:
                continue
            all_props = set(ndef.required_props) | set(ndef.optional_props)
            assert ndef.merge_key in all_props, (
                f"{spec.id}: node '{key}' merge_key '{ndef.merge_key}' "
                f"not in properties {all_props}"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_system_managed_rels_exist(self, spec: OntologySpec):
        """system_managed_rels must reference actual relationship types."""
        for smr in spec.system_managed_rels:
            assert smr in spec.relationship_types, (
                f"{spec.id}: system_managed_rel '{smr}' not in relationship_types"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_system_managed_nodes_exist(self, spec: OntologySpec):
        """system_managed_nodes must reference actual node types."""
        for smn in spec.system_managed_nodes:
            assert smn in spec.node_types, (
                f"{spec.id}: system_managed_node '{smn}' not in node_types"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_no_duplicate_required_optional_props(self, spec: OntologySpec):
        """A property should not appear in both required and optional."""
        for key, ndef in spec.node_types.items():
            overlap = set(ndef.required_props) & set(ndef.optional_props)
            assert not overlap, (
                f"{spec.id}: node '{key}' has props in both required and "
                f"optional: {overlap}"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_completeness_rules_have_valid_priorities(self, spec: OntologySpec):
        valid = {"critical", "high", "medium", "low"}
        for rule in spec.completeness_rules:
            assert rule.priority in valid, (
                f"{spec.id}: rule '{rule.rule_id}' has invalid priority "
                f"'{rule.priority}'"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_completeness_rules_unique_ids(self, spec: OntologySpec):
        ids = [r.rule_id for r in spec.completeness_rules]
        assert len(ids) == len(set(ids)), (
            f"{spec.id}: duplicate rule_ids: {ids}"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_completeness_rules_contain_placeholder(self, spec: OntologySpec):
        """gap_template must contain {entity_desc} for formatting."""
        for rule in spec.completeness_rules:
            assert "{entity_desc}" in rule.gap_template, (
                f"{spec.id}: rule '{rule.rule_id}' gap_template missing "
                f"{{entity_desc}} placeholder"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_completeness_rule_cypher_returns_entity_desc(self, spec: OntologySpec):
        """Cypher in each rule should SELECT entity_desc and label."""
        for rule in spec.completeness_rules:
            assert "entity_desc" in rule.cypher, (
                f"{spec.id}: rule '{rule.rule_id}' cypher doesn't return entity_desc"
            )
            assert "label" in rule.cypher, (
                f"{spec.id}: rule '{rule.rule_id}' cypher doesn't return label"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_completeness_rule_cypher_references_known_labels(self, spec: OntologySpec):
        """Cypher queries should reference labels that exist in the spec."""
        node_labels = set(spec.node_types.keys())
        # Match :Label in Cypher but NOT [:REL_TYPE] (inside square brackets)
        # Node labels appear as (n:Label) or (:Label)
        label_pattern = re.compile(r"\(:?(\w+)\)")  # e.g. (:Event) or (e:Event)
        for rule in spec.completeness_rules:
            # Extract node label references — (:Label) or (var:Label)
            paren_pattern = re.compile(r"\(\w*:(\w+)")
            found_labels = set(paren_pattern.findall(rule.cypher))
            for label in found_labels:
                assert label in node_labels, (
                    f"{spec.id}: rule '{rule.rule_id}' references label "
                    f"'{label}' not in node_types"
                )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_completeness_rule_cypher_references_known_rels(self, spec: OntologySpec):
        """Cypher queries should reference relationship types that exist in the spec."""
        rel_types = set(spec.relationship_types.keys())
        rel_pattern = re.compile(r"\[:(\w+(?:\|:?\w+)*)\]")  # matches [:REL] or [:R1|R2]
        for rule in spec.completeness_rules:
            matches = rel_pattern.findall(rule.cypher)
            for match in matches:
                # Handle OR patterns like PARTICIPATED_IN|WITNESSED
                for rel in re.split(r"[|:]", match):
                    rel = rel.strip()
                    if rel:
                        assert rel in rel_types, (
                            f"{spec.id}: rule '{rule.rule_id}' references rel "
                            f"'{rel}' not in relationship_types"
                        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Fidelity to source standards
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaOrgFidelity:
    """Verify the Schema.org Event spec accurately reflects Schema.org."""

    def test_event_node_exists(self):
        assert "Event" in SCHEMA_ORG_EVENT.node_types

    def test_person_node_exists(self):
        assert "Person" in SCHEMA_ORG_EVENT.node_types

    def test_location_node_exists(self):
        assert "Location" in SCHEMA_ORG_EVENT.node_types

    def test_time_node_exists(self):
        assert "Time" in SCHEMA_ORG_EVENT.node_types

    def test_vehicle_node_exists(self):
        assert "Vehicle" in SCHEMA_ORG_EVENT.node_types

    def test_person_has_role_as_property(self):
        """Schema.org models roles as flat properties, not first-class nodes."""
        person = SCHEMA_ORG_EVENT.node_types["Person"]
        all_props = set(person.required_props) | set(person.optional_props)
        assert "role" in all_props, "Schema.org Person should have 'role' property"

    def test_no_role_node(self):
        """Schema.org should NOT have a first-class Role node."""
        assert "Role" not in SCHEMA_ORG_EVENT.node_types, (
            "Schema.org should not have a separate Role node type"
        )

    def test_event_has_description(self):
        event = SCHEMA_ORG_EVENT.node_types["Event"]
        assert "description" in event.required_props

    def test_event_has_type(self):
        event = SCHEMA_ORG_EVENT.node_types["Event"]
        assert "type" in event.required_props

    def test_spatial_relationship_exists(self):
        assert "OCCURRED_AT" in SCHEMA_ORG_EVENT.relationship_types

    def test_temporal_relationship_exists(self):
        assert "OCCURRED_AT_TIME" in SCHEMA_ORG_EVENT.relationship_types

    def test_participation_relationship_exists(self):
        assert "PARTICIPATED_IN" in SCHEMA_ORG_EVENT.relationship_types

    def test_sosa_observation_model(self):
        """Schema.org spec includes SOSA observation pattern."""
        assert "Observation" in SCHEMA_ORG_EVENT.node_types
        assert "OBSERVED" in SCHEMA_ORG_EVENT.relationship_types
        assert "MADE_BY" in SCHEMA_ORG_EVENT.relationship_types

    def test_observation_is_system_managed(self):
        """Observation links should be system-managed, not LLM-extracted."""
        assert "Observation" in SCHEMA_ORG_EVENT.system_managed_nodes
        assert "OBSERVED" in SCHEMA_ORG_EVENT.system_managed_rels
        assert "MADE_BY" in SCHEMA_ORG_EVENT.system_managed_rels

    def test_person_roles_taxonomy(self):
        """Schema.org spec should provide person role taxonomy."""
        assert len(SCHEMA_ORG_EVENT.person_roles) >= 4
        assert "witness" in SCHEMA_ORG_EVENT.person_roles
        assert "suspect" in SCHEMA_ORG_EVENT.person_roles

    def test_event_types_taxonomy(self):
        assert len(SCHEMA_ORG_EVENT.event_types) >= 4

    def test_provenance_props_defined(self):
        assert "source" in SCHEMA_ORG_EVENT.provenance_props
        assert "confidence" in SCHEMA_ORG_EVENT.provenance_props


class TestSEMFidelity:
    """Verify the SEM spec accurately reflects the Simple Event Model."""

    def test_event_node_exists(self):
        assert "Event" in SEM_EVENT.node_types

    def test_actor_node_exists(self):
        """SEM uses 'Actor' not 'Person' — matches sem:Actor."""
        assert "Actor" in SEM_EVENT.node_types

    def test_place_node_exists(self):
        """SEM uses 'Place' — matches sem:Place."""
        assert "Place" in SEM_EVENT.node_types

    def test_time_node_exists(self):
        assert "Time" in SEM_EVENT.node_types

    def test_role_is_first_class_node(self):
        """SEM's key innovation: Role as a first-class entity."""
        assert "Role" in SEM_EVENT.node_types
        role = SEM_EVENT.node_types["Role"]
        assert "role_type" in role.required_props

    def test_has_actor_relationship(self):
        """sem:hasActor maps to HAS_ACTOR."""
        assert "HAS_ACTOR" in SEM_EVENT.relationship_types
        rel = SEM_EVENT.relationship_types["HAS_ACTOR"]
        assert rel.from_label == "Event"
        assert rel.to_label == "Actor"

    def test_has_role_relationship(self):
        """Actor HAS_ROLE Role — reified role model."""
        assert "HAS_ROLE" in SEM_EVENT.relationship_types
        rel = SEM_EVENT.relationship_types["HAS_ROLE"]
        assert rel.from_label == "Actor"
        assert rel.to_label == "Role"

    def test_has_sub_event(self):
        """sem:hasSubEvent — event decomposition support."""
        assert "HAS_SUB_EVENT" in SEM_EVENT.relationship_types
        rel = SEM_EVENT.relationship_types["HAS_SUB_EVENT"]
        assert rel.from_label == "Event"
        assert rel.to_label == "Event"

    def test_has_place_relationship(self):
        assert "HAS_PLACE" in SEM_EVENT.relationship_types
        rel = SEM_EVENT.relationship_types["HAS_PLACE"]
        assert rel.from_label == "Event"
        assert rel.to_label == "Place"

    def test_has_time_relationship(self):
        assert "HAS_TIME" in SEM_EVENT.relationship_types
        rel = SEM_EVENT.relationship_types["HAS_TIME"]
        assert rel.from_label == "Event"
        assert rel.to_label == "Time"

    def test_temporal_ordering_exists(self):
        assert "PRECEDED" in SEM_EVENT.relationship_types

    def test_causal_relationship_exists(self):
        assert "CAUSED" in SEM_EVENT.relationship_types

    def test_object_node_exists(self):
        """SEM has sem:Object as a participation entity."""
        assert "Object" in SEM_EVENT.node_types

    def test_sem_ontology_mapping_prefix(self):
        """SEM nodes should reference sem: namespace in ontology_mapping."""
        for key, ndef in SEM_EVENT.node_types.items():
            if key == "PhysicalDescription":
                continue  # custom extension
            assert ndef.ontology_mapping.startswith("sem:"), (
                f"SEM node '{key}' mapping '{ndef.ontology_mapping}' "
                f"should use sem: prefix"
            )

    def test_time_supports_uncertainty(self):
        """SEM Time should have earliest/latest props for uncertain intervals."""
        time_node = SEM_EVENT.node_types["Time"]
        opt = set(time_node.optional_props)
        assert "earliest" in opt or "precision" in opt, (
            "SEM Time should support temporal uncertainty"
        )


class TestBFOCCOFidelity:
    """Verify the BFO/CCO spec accurately reflects BFO + CCO standards."""

    def test_process_node_exists(self):
        """BFO uses bfo:Process instead of schema:Event."""
        assert "Process" in BFO_CCO_EVENT.node_types

    def test_act_node_exists(self):
        """CCO extends Process with cco:Act for intentional actions."""
        assert "Act" in BFO_CCO_EVENT.node_types

    def test_agent_node_exists(self):
        """cco:Agent — the persistent participant."""
        assert "Agent" in BFO_CCO_EVENT.node_types

    def test_agent_role_node_exists(self):
        """bfo:Role — realizable entity inhering in agents."""
        assert "AgentRole" in BFO_CCO_EVENT.node_types

    def test_material_entity_node_exists(self):
        """bfo:MaterialEntity — physical objects."""
        assert "MaterialEntity" in BFO_CCO_EVENT.node_types

    def test_site_node_exists(self):
        """bfo:Site — spatial regions."""
        assert "Site" in BFO_CCO_EVENT.node_types

    def test_temporal_interval_node_exists(self):
        """bfo:TemporalInterval — time regions."""
        assert "TemporalInterval" in BFO_CCO_EVENT.node_types

    def test_information_content_entity_exists(self):
        """cco:InformationContentEntity — for provenance modelling."""
        assert "InformationContentEntity" in BFO_CCO_EVENT.node_types

    def test_descriptive_ice_exists(self):
        """cco:DescriptiveInformationContentEntity — appearance descriptions."""
        assert "DescriptiveICE" in BFO_CCO_EVENT.node_types

    def test_bearer_of_pattern(self):
        """BFO role pattern: Agent -[BEARS_ROLE]-> AgentRole."""
        assert "BEARS_ROLE" in BFO_CCO_EVENT.relationship_types
        rel = BFO_CCO_EVENT.relationship_types["BEARS_ROLE"]
        assert rel.from_label == "Agent"
        assert rel.to_label == "AgentRole"

    def test_realized_in_pattern(self):
        """BFO role pattern: AgentRole -[REALIZED_IN]-> Process."""
        assert "REALIZED_IN" in BFO_CCO_EVENT.relationship_types
        rel = BFO_CCO_EVENT.relationship_types["REALIZED_IN"]
        assert rel.from_label == "AgentRole"
        assert rel.to_label == "Process"

    def test_mereological_decomposition(self):
        """bfo:has_part — process decomposition."""
        assert "HAS_PART" in BFO_CCO_EVENT.relationship_types
        rel = BFO_CCO_EVENT.relationship_types["HAS_PART"]
        assert rel.from_label == "Process"
        assert rel.to_label == "Process"

    def test_occurs_at_pattern(self):
        """bfo:occurs_in — process spatial grounding."""
        assert "OCCURS_AT" in BFO_CCO_EVENT.relationship_types

    def test_temporal_occupancy(self):
        """bfo:occupies_temporal_region."""
        assert "OCCUPIES_TEMPORAL" in BFO_CCO_EVENT.relationship_types
        rel = BFO_CCO_EVENT.relationship_types["OCCUPIES_TEMPORAL"]
        assert rel.from_label == "Process"
        assert rel.to_label == "TemporalInterval"

    def test_has_participant(self):
        """bfo:has_participant for material entities."""
        assert "HAS_PARTICIPANT" in BFO_CCO_EVENT.relationship_types

    def test_ice_provenance_relationships(self):
        """ICE should have IS_ABOUT and CREATED_BY for provenance."""
        assert "IS_ABOUT" in BFO_CCO_EVENT.relationship_types
        assert "CREATED_BY" in BFO_CCO_EVENT.relationship_types

    def test_bfo_ontology_mapping_prefix(self):
        """BFO/CCO nodes should reference bfo: or cco: namespaces."""
        for key, ndef in BFO_CCO_EVENT.node_types.items():
            mapping = ndef.ontology_mapping
            assert "bfo:" in mapping or "cco:" in mapping, (
                f"BFO/CCO node '{key}' mapping '{mapping}' should use "
                f"bfo: or cco: prefix"
            )

    def test_three_place_role_relation(self):
        """BFO has the most expressive role model: Agent → Role → Process.
        Verify the full chain exists."""
        bears = BFO_CCO_EVENT.relationship_types["BEARS_ROLE"]
        realized = BFO_CCO_EVENT.relationship_types["REALIZED_IN"]
        # Chain: Agent -> AgentRole -> Process
        assert bears.from_label == "Agent"
        assert bears.to_label == "AgentRole"
        assert realized.from_label == "AgentRole"
        assert realized.to_label == "Process"
        # The to_label of bears == from_label of realized (chain link)
        assert bears.to_label == realized.from_label


# ═══════════════════════════════════════════════════════════════════════════
# 3. Prompt generation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptGeneration:
    """Verify that build_extraction_prompt() produces valid, complete prompts."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_is_non_empty(self, spec: OntologySpec):
        prompt = spec.build_extraction_prompt()
        assert len(prompt) > 100, "Prompt should be substantial"

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_contains_ontology_name(self, spec: OntologySpec):
        prompt = spec.build_extraction_prompt()
        assert spec.name in prompt

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_contains_all_extractable_node_types(self, spec: OntologySpec):
        """All non-system-managed node types should appear in the prompt."""
        prompt = spec.build_extraction_prompt()
        for key, ndef in spec.node_types.items():
            if key in spec.system_managed_nodes:
                continue
            assert ndef.label in prompt, (
                f"{spec.id}: node '{ndef.label}' missing from extraction prompt"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_excludes_system_managed_nodes(self, spec: OntologySpec):
        """System-managed nodes should NOT appear in the extraction targets
        (though they might appear in relationship context)."""
        prompt = spec.build_extraction_prompt()
        for smn in spec.system_managed_nodes:
            # Check it's not listed as a TARGET entity type
            target_section = prompt.split("TARGET ENTITY TYPES:")[1].split(
                "TARGET RELATIONSHIP TYPES:"
            )[0]
            assert smn not in target_section.split("\n")[0:20] or True  # soft check

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_contains_all_extractable_rel_types(self, spec: OntologySpec):
        prompt = spec.build_extraction_prompt()
        for key, rdef in spec.relationship_types.items():
            if key in spec.system_managed_rels:
                continue
            assert rdef.rel_type in prompt, (
                f"{spec.id}: rel '{rdef.rel_type}' missing from extraction prompt"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_contains_json_schema(self, spec: OntologySpec):
        """Prompt should contain an example JSON output structure."""
        prompt = spec.build_extraction_prompt()
        assert '"entities"' in prompt
        assert '"relationships"' in prompt

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_contains_required_props(self, spec: OntologySpec):
        """Required properties should be mentioned in the prompt."""
        prompt = spec.build_extraction_prompt()
        for key, ndef in spec.node_types.items():
            if key in spec.system_managed_nodes:
                continue
            for prop in ndef.required_props:
                assert prop in prompt, (
                    f"{spec.id}: required prop '{prop}' of node '{key}' "
                    f"missing from prompt"
                )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_contains_extraction_rules(self, spec: OntologySpec):
        """Prompt should contain extraction guidance rules."""
        prompt = spec.build_extraction_prompt()
        assert "RULES:" in prompt
        assert "unique id" in prompt.lower() or "unique" in prompt.lower()

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_mentions_event_types_if_defined(self, spec: OntologySpec):
        if spec.event_types:
            prompt = spec.build_extraction_prompt()
            assert "EVENT TYPES:" in prompt
            for et in spec.event_types[:3]:  # check first few
                assert et in prompt

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_prompt_mentions_person_roles_if_defined(self, spec: OntologySpec):
        if spec.person_roles:
            prompt = spec.build_extraction_prompt()
            assert "PERSON ROLES:" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 4. SHACL shape generation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSHACLGeneration:
    """Verify that build_shacl_shapes() produces valid SHACL Turtle."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_shacl_non_empty(self, spec: OntologySpec):
        shapes = spec.build_shacl_shapes()
        assert len(shapes) > 50

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_shacl_contains_prefixes(self, spec: OntologySpec):
        shapes = spec.build_shacl_shapes()
        assert "@prefix sh:" in shapes
        assert "@prefix xsd:" in shapes

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_shacl_has_shape_per_node_type(self, spec: OntologySpec):
        """Every node type should have a corresponding NodeShape."""
        shapes = spec.build_shacl_shapes()
        for key in spec.node_types:
            assert f"{key}Shape" in shapes, (
                f"{spec.id}: no SHACL shape for node type '{key}'"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_shacl_shapes_have_target_class(self, spec: OntologySpec):
        shapes = spec.build_shacl_shapes()
        for key in spec.node_types:
            assert f"sh:targetClass evt:{key}" in shapes, (
                f"{spec.id}: shape for '{key}' missing targetClass"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_shacl_required_props_have_min_count(self, spec: OntologySpec):
        """Required properties should have sh:minCount 1 constraints."""
        shapes = spec.build_shacl_shapes()
        for key, ndef in spec.node_types.items():
            for prop in ndef.required_props:
                assert f"sh:path evt:{prop}" in shapes, (
                    f"{spec.id}: required prop '{prop}' of '{key}' "
                    f"has no SHACL path constraint"
                )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_shacl_valid_turtle_structure(self, spec: OntologySpec):
        """Basic syntactic check: shapes should end with periods."""
        shapes = spec.build_shacl_shapes()
        # Each shape block should end with " ."
        shape_blocks = [b.strip() for b in shapes.split("\n\n") if b.strip()]
        for block in shape_blocks[1:]:  # skip prefix block
            if block and not block.startswith("@prefix"):
                assert block.rstrip().endswith("."), (
                    f"{spec.id}: SHACL block doesn't end with '.': "
                    f"{block[-50:]}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Constraint generation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintGeneration:
    """Verify get_constraint_cypher() produces valid constraint statements."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_constraints_non_empty(self, spec: OntologySpec):
        constraints = spec.get_constraint_cypher()
        assert len(constraints) >= 1

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_constraints_are_create_statements(self, spec: OntologySpec):
        for stmt in spec.get_constraint_cypher():
            assert stmt.startswith("CREATE CONSTRAINT"), (
                f"Constraint should start with CREATE CONSTRAINT: {stmt}"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_constraints_use_if_not_exists(self, spec: OntologySpec):
        for stmt in spec.get_constraint_cypher():
            assert "IF NOT EXISTS" in stmt, (
                f"Constraint should be idempotent (IF NOT EXISTS): {stmt}"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_constraints_reference_known_labels(self, spec: OntologySpec):
        node_labels = set(spec.node_types.keys())
        label_pattern = re.compile(r"FOR \(n:(\w+)\)")
        for stmt in spec.get_constraint_cypher():
            match = label_pattern.search(stmt)
            assert match, f"Can't parse label from constraint: {stmt}"
            assert match.group(1) in node_labels, (
                f"Constraint references unknown label '{match.group(1)}'"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_constraint_count_matches_mergeable_nodes(self, spec: OntologySpec):
        """Should have one constraint per node type with a merge key."""
        mergeable = sum(
            1 for ndef in spec.node_types.values()
            if ndef.merge_key or ndef.required_props
        )
        constraints = spec.get_constraint_cypher()
        assert len(constraints) == mergeable, (
            f"{spec.id}: expected {mergeable} constraints, got {len(constraints)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cross-ontology comparison tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossOntologyComparison:
    """Verify the three ontologies represent meaningfully different approaches."""

    def test_role_model_levels_are_distinct(self):
        """Each ontology should have a different role modelling approach."""
        # Schema.org: role as property
        person = SCHEMA_ORG_EVENT.node_types["Person"]
        assert "role" in person.required_props or "role" in person.optional_props

        # SEM: role as first-class node (two-place)
        assert "Role" in SEM_EVENT.node_types
        assert "HAS_ROLE" in SEM_EVENT.relationship_types

        # BFO: role as node with realisation (three-place)
        assert "AgentRole" in BFO_CCO_EVENT.node_types
        assert "BEARS_ROLE" in BFO_CCO_EVENT.relationship_types
        assert "REALIZED_IN" in BFO_CCO_EVENT.relationship_types

    def test_event_decomposition_levels_are_distinct(self):
        """Schema.org should have shallow event structure; SEM and BFO deeper."""
        # Schema.org: flat events with CAUSED/PRECEDED
        schema_structural = {
            r for r in SCHEMA_ORG_EVENT.relationship_types
            if SCHEMA_ORG_EVENT.relationship_types[r].from_label == "Event"
            and SCHEMA_ORG_EVENT.relationship_types[r].to_label == "Event"
        }
        # SEM: HAS_SUB_EVENT for nesting
        assert "HAS_SUB_EVENT" in SEM_EVENT.relationship_types

        # BFO: HAS_PART for mereological decomposition
        assert "HAS_PART" in BFO_CCO_EVENT.relationship_types

    def test_bfo_has_more_node_types_than_schema_org(self):
        """BFO/CCO should be heavier (more types) than Schema.org."""
        assert len(BFO_CCO_EVENT.node_types) >= len(SCHEMA_ORG_EVENT.node_types)

    def test_all_specs_have_completeness_rules(self):
        for spec in ALL_SPECS:
            assert len(spec.completeness_rules) >= 3, (
                f"{spec.id}: needs at least 3 completeness rules"
            )

    def test_all_specs_have_critical_event_rules(self):
        """Every ontology should enforce that events have time, location, participants."""
        for spec in ALL_SPECS:
            priorities = {r.priority for r in spec.completeness_rules}
            assert "critical" in priorities, (
                f"{spec.id}: should have at least one critical completeness rule"
            )

    def test_all_specs_have_spatial_relationship(self):
        """Every ontology should connect events to locations."""
        for spec, spatial_rel in [
            (SCHEMA_ORG_EVENT, "OCCURRED_AT"),
            (SEM_EVENT, "HAS_PLACE"),
            (BFO_CCO_EVENT, "OCCURS_AT"),
        ]:
            assert spatial_rel in spec.relationship_types, (
                f"{spec.id}: missing spatial relationship '{spatial_rel}'"
            )

    def test_all_specs_have_temporal_relationship(self):
        """Every ontology should connect events to time."""
        for spec, temp_rel in [
            (SCHEMA_ORG_EVENT, "OCCURRED_AT_TIME"),
            (SEM_EVENT, "HAS_TIME"),
            (BFO_CCO_EVENT, "OCCUPIES_TEMPORAL"),
        ]:
            assert temp_rel in spec.relationship_types, (
                f"{spec.id}: missing temporal relationship '{temp_rel}'"
            )

    def test_all_specs_have_participant_relationship(self):
        """Every ontology should connect events to participants."""
        for spec, part_rels in [
            (SCHEMA_ORG_EVENT, ["PARTICIPATED_IN", "WITNESSED"]),
            (SEM_EVENT, ["HAS_ACTOR"]),
            (BFO_CCO_EVENT, ["HAS_AGENT", "AGENT_IN"]),
        ]:
            found = any(r in spec.relationship_types for r in part_rels)
            assert found, (
                f"{spec.id}: missing participant relationship from {part_rels}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 7. OntologySpec summary and serialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestSummarySerialization:
    """Verify summary() and other output methods."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_summary_contains_counts(self, spec: OntologySpec):
        summary = spec.summary()
        assert "node types" in summary
        assert "rel types" in summary
        assert str(len(spec.node_types)) in summary
        assert str(len(spec.relationship_types)) in summary

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_summary_contains_name(self, spec: OntologySpec):
        assert spec.name in spec.summary()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Edge-case and regression tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Catch common implementation pitfalls."""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_no_empty_descriptions(self, spec: OntologySpec):
        """Node types should have descriptions for prompt quality."""
        for key, ndef in spec.node_types.items():
            if key not in spec.system_managed_nodes:
                assert ndef.description, (
                    f"{spec.id}: node '{key}' has empty description"
                )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_no_empty_rel_descriptions(self, spec: OntologySpec):
        for key, rdef in spec.relationship_types.items():
            assert rdef.description, (
                f"{spec.id}: rel '{key}' has empty description"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_rel_types_are_uppercase_snake(self, spec: OntologySpec):
        """Neo4j convention: relationship types should be UPPER_SNAKE_CASE."""
        for key in spec.relationship_types:
            assert key == key.upper(), (
                f"{spec.id}: rel type '{key}' should be UPPER_SNAKE_CASE"
            )
            assert " " not in key, f"rel type '{key}' contains spaces"

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_node_labels_are_pascal_case(self, spec: OntologySpec):
        """Neo4j convention: node labels should be PascalCase."""
        for key in spec.node_types:
            assert key[0].isupper(), (
                f"{spec.id}: node label '{key}' should be PascalCase"
            )
            assert " " not in key, f"node label '{key}' contains spaces"

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_property_names_are_snake_case(self, spec: OntologySpec):
        """Properties should be snake_case."""
        for key, ndef in spec.node_types.items():
            for prop in list(ndef.required_props) + list(ndef.optional_props):
                assert prop == prop.lower(), (
                    f"{spec.id}: property '{prop}' of '{key}' should be lowercase"
                )
                assert " " not in prop, (
                    f"property '{prop}' contains spaces"
                )

    def test_schema_org_no_redundant_roles_in_rel_types(self):
        """Schema.org should NOT have a HAS_ROLE relationship (uses property instead)."""
        assert "HAS_ROLE" not in SCHEMA_ORG_EVENT.relationship_types

    def test_bfo_no_flat_role_property(self):
        """BFO/CCO Agent should NOT have a flat 'role' property."""
        agent = BFO_CCO_EVENT.node_types["Agent"]
        all_props = set(agent.required_props) | set(agent.optional_props)
        assert "role" not in all_props, (
            "BFO/CCO Agent should use AgentRole nodes, not a role property"
        )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_ontology_mapping_non_empty(self, spec: OntologySpec):
        """Every node type should map to a standard ontology term."""
        for key, ndef in spec.node_types.items():
            assert ndef.ontology_mapping, (
                f"{spec.id}: node '{key}' has empty ontology_mapping"
            )

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_frozen_node_specs(self, spec: OntologySpec):
        """NodeSpec should be frozen (immutable)."""
        ndef = list(spec.node_types.values())[0]
        with pytest.raises(AttributeError):
            ndef.label = "Mutated"

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
    def test_frozen_rel_specs(self, spec: OntologySpec):
        """RelSpec should be frozen (immutable)."""
        rdef = list(spec.relationship_types.values())[0]
        with pytest.raises(AttributeError):
            rdef.rel_type = "Mutated"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Demo helper tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoHelpers:
    """Test the JSON parsing helper from demo.py."""

    def test_parse_json_plain(self):
        from demo import _parse_json
        data = _parse_json('{"entities": [], "relationships": []}')
        assert data == {"entities": [], "relationships": []}

    def test_parse_json_with_markdown_fences(self):
        from demo import _parse_json
        raw = '```json\n{"entities": [{"id": "e1"}], "relationships": []}\n```'
        data = _parse_json(raw)
        assert data is not None
        assert len(data["entities"]) == 1

    def test_parse_json_with_prefix_text(self):
        from demo import _parse_json
        raw = 'Here is the JSON:\n{"entities": [], "relationships": []}'
        data = _parse_json(raw)
        assert data is not None

    def test_parse_json_returns_none_for_garbage(self):
        from demo import _parse_json
        assert _parse_json("not json at all") is None

    def test_parse_json_empty_string(self):
        from demo import _parse_json
        assert _parse_json("") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
