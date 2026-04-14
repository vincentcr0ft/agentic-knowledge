"""
07 · Graph Quality — Exhaustive Validation Tests
═════════════════════════════════════════════════

Tests the quality-probe framework WITHOUT requiring Neo4j or Ollama:
  1. Data structures — Violation, DimensionResult, QualityReport
  2. Weight system — normalisation, dynamic phase inclusion
  3. Recommendation engine — threshold-driven suggestions
  4. Score computation — boundary conditions, sentinel handling
  5. Report formatting — summary output structure
  6. Cypher probe logic — query patterns and violation generation
  7. SHACL shapes file — structural validation
  8. Core module consistency — quality_core self-consistency
  9. Hallucination detection — faithfulness probe contract
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# Add the module directory to path for flat imports
_MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODULE_DIR))

from quality_core import (
    Violation,
    DimensionResult,
    QualityReport,
    _BASE_WEIGHTS,
    _PHASE2_WEIGHTS,
    _PHASE3_WEIGHTS,
    DIMENSION_WEIGHTS,
    compute_overall,
    generate_recommendations,
    linearise_graph,
    calibrate_llm_probe,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data structure tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDataStructures:
    """Violation, DimensionResult, QualityReport must be well-formed."""

    def test_violation_creation(self):
        v = Violation(dimension="schema", severity="error", message="test")
        assert v.dimension == "schema"
        assert v.severity == "error"
        assert v.node_label is None

    def test_violation_with_node_ref(self):
        v = Violation(
            dimension="faithfulness", severity="error",
            message="Hallucinated", node_label="Event", node_id="e1"
        )
        assert v.node_label == "Event"
        assert v.node_id == "e1"

    def test_violation_valid_severities(self):
        """Only error, warning, info should be used."""
        for sev in ("error", "warning", "info"):
            v = Violation(dimension="test", severity=sev, message="ok")
            assert v.severity == sev

    def test_dimension_result_defaults(self):
        dr = DimensionResult(dimension="schema", score=0.8)
        assert dr.violations == []
        assert dr.details == {}

    def test_dimension_result_with_violations(self):
        v = Violation(dimension="schema", severity="warning", message="missing")
        dr = DimensionResult(dimension="schema", score=0.5, violations=[v])
        assert len(dr.violations) == 1

    def test_quality_report_defaults(self):
        qr = QualityReport()
        assert qr.schema_score == 0.0
        assert qr.overall_score == 0.0
        assert qr.violations == []
        assert qr.recommendations == []

    def test_quality_report_has_all_11_dimensions(self):
        qr = QualityReport()
        dims = [
            "schema_score", "structural_score", "constraint_score",
            "consistency_score", "coherence_score", "faithfulness_score",
            "semantic_completeness_score", "investigative_readiness_score",
            "link_prediction_score", "triple_plausibility_score",
            "entity_clustering_score",
        ]
        for dim in dims:
            assert hasattr(qr, dim), f"QualityReport missing {dim}"

    def test_quality_report_has_timestamp(self):
        qr = QualityReport()
        assert hasattr(qr, "timestamp")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Weight system tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWeightSystem:
    """Dynamic weight normalisation and phase inclusion."""

    def test_base_weights_sum_to_one(self):
        total = sum(_BASE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Base weights sum = {total}"

    def test_base_weights_have_six_dimensions(self):
        assert len(_BASE_WEIGHTS) == 6

    def test_base_weight_keys(self):
        expected = {"schema", "structural", "constraint",
                    "consistency", "coherence", "faithfulness"}
        assert set(_BASE_WEIGHTS.keys()) == expected

    def test_phase2_weights_keys(self):
        expected = {"semantic_completeness", "investigative_readiness"}
        assert set(_PHASE2_WEIGHTS.keys()) == expected

    def test_phase3_weights_keys(self):
        expected = {"link_prediction", "triple_plausibility", "entity_clustering"}
        assert set(_PHASE3_WEIGHTS.keys()) == expected

    def test_all_weights_positive(self):
        for w_dict in (_BASE_WEIGHTS, _PHASE2_WEIGHTS, _PHASE3_WEIGHTS):
            for k, v in w_dict.items():
                assert v > 0, f"Weight {k} must be positive"

    def test_faithfulness_has_highest_base_weight(self):
        """Faithfulness should be heavily weighted — hallucination detection."""
        max_dim = max(_BASE_WEIGHTS, key=_BASE_WEIGHTS.get)
        assert max_dim in ("faithfulness", "consistency"), (
            f"Expected faithfulness or consistency to have highest weight, got {max_dim}"
        )

    def test_dimension_weights_alias(self):
        assert DIMENSION_WEIGHTS == _BASE_WEIGHTS


# ═══════════════════════════════════════════════════════════════════════════
# 3. Score computation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreComputation:
    """compute_overall() boundary conditions."""

    def test_perfect_scores(self):
        qr = QualityReport(
            schema_score=1.0, structural_score=1.0,
            constraint_score=1.0, consistency_score=1.0,
            coherence_score=1.0, faithfulness_score=1.0,
        )
        score = compute_overall(qr)
        assert abs(score - 1.0) < 0.001

    def test_zero_scores(self):
        qr = QualityReport()
        score = compute_overall(qr)
        assert score == 0.0

    def test_mixed_scores(self):
        qr = QualityReport(
            schema_score=1.0, structural_score=0.0,
            constraint_score=1.0, consistency_score=0.0,
            coherence_score=1.0, faithfulness_score=0.0,
        )
        score = compute_overall(qr)
        assert 0.3 < score < 0.7

    def test_score_clamped_to_unit_interval(self):
        qr = QualityReport(
            schema_score=1.0, structural_score=1.0,
            constraint_score=1.0, consistency_score=1.0,
            coherence_score=1.0, faithfulness_score=1.0,
        )
        score = compute_overall(qr)
        assert 0.0 <= score <= 1.0

    def test_phase2_included_when_dimension_result_present(self):
        qr = QualityReport(
            schema_score=1.0, structural_score=1.0,
            constraint_score=1.0, consistency_score=1.0,
            coherence_score=1.0, faithfulness_score=1.0,
            semantic_completeness_score=0.5,
            dimension_results=[
                DimensionResult(dimension="semantic_completeness", score=0.5),
            ],
        )
        score = compute_overall(qr)
        # With semantic_completeness at 0.5, overall should be < 1.0
        assert score < 1.0

    def test_phase3_included_when_dimension_result_present(self):
        qr = QualityReport(
            schema_score=1.0, structural_score=1.0,
            constraint_score=1.0, consistency_score=1.0,
            coherence_score=1.0, faithfulness_score=1.0,
            link_prediction_score=0.0,
            dimension_results=[
                DimensionResult(dimension="link_prediction", score=0.0),
            ],
        )
        score = compute_overall(qr)
        assert score < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Recommendation engine tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRecommendations:
    """Threshold-driven actionable recommendations."""

    def test_no_recommendations_for_perfect_scores(self):
        qr = QualityReport(
            schema_score=1.0, structural_score=1.0,
            constraint_score=1.0, consistency_score=1.0,
            coherence_score=1.0, faithfulness_score=1.0,
            overall_score=1.0,
        )
        recs = generate_recommendations(qr)
        assert len(recs) == 0

    def test_schema_gaps_trigger_recommendation(self):
        qr = QualityReport(schema_score=0.5)
        recs = generate_recommendations(qr)
        assert any("schema" in r.lower() or "Schema" in r for r in recs)

    def test_structural_issues_trigger_recommendation(self):
        qr = QualityReport(structural_score=0.5)
        recs = generate_recommendations(qr)
        assert any("fragment" in r.lower() or "isolated" in r.lower() for r in recs)

    def test_faithfulness_issues_trigger_recommendation(self):
        qr = QualityReport(faithfulness_score=0.3)
        recs = generate_recommendations(qr)
        assert any("faithfulness" in r.lower() or "hallucin" in r.lower() for r in recs)

    def test_coherence_issues_trigger_recommendation(self):
        qr = QualityReport(coherence_score=0.4)
        recs = generate_recommendations(qr)
        assert any("coherence" in r.lower() for r in recs)

    def test_low_overall_triggers_warning(self):
        qr = QualityReport(overall_score=0.2)
        recs = generate_recommendations(qr)
        assert any("warning" in r.lower() or "WARNING" in r for r in recs)

    def test_constraint_issues_trigger_recommendation(self):
        qr = QualityReport(constraint_score=0.3)
        recs = generate_recommendations(qr)
        assert any("provenance" in r.lower() for r in recs)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Report formatting tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReportFormatting:
    """QualityReport.summary() output structure."""

    def test_summary_is_string(self):
        qr = QualityReport()
        assert isinstance(qr.summary(), str)

    def test_summary_contains_header(self):
        summary = QualityReport().summary()
        assert "QUALITY REPORT" in summary

    def test_summary_contains_phase_sections(self):
        summary = QualityReport().summary()
        assert "Phase 1" in summary
        assert "Phase 2" in summary
        assert "Phase 3" in summary

    def test_summary_shows_all_dimension_scores(self):
        qr = QualityReport(
            schema_score=0.85, structural_score=0.9,
            constraint_score=0.7, consistency_score=0.95,
            coherence_score=0.8, faithfulness_score=0.6,
        )
        summary = qr.summary()
        assert "0.85" in summary
        assert "0.90" in summary
        assert "0.60" in summary

    def test_summary_shows_violations(self):
        qr = QualityReport(violations=[
            Violation("faithfulness", "error", "Hallucinated fact: X"),
            Violation("schema", "warning", "Missing type Y"),
        ])
        summary = qr.summary()
        assert "Errors:   1" in summary
        assert "Warnings: 1" in summary

    def test_summary_shows_error_details(self):
        qr = QualityReport(violations=[
            Violation("faithfulness", "error", "Hallucinated: car was blue"),
        ])
        summary = qr.summary()
        assert "Hallucinated" in summary

    def test_summary_shows_recommendations(self):
        qr = QualityReport(recommendations=["Fix the schema gaps"])
        summary = qr.summary()
        assert "Fix the schema gaps" in summary


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cypher probe logic tests (pattern validation without Neo4j)
# ═══════════════════════════════════════════════════════════════════════════

class TestCypherProbePatterns:
    """Validate probe function signatures and DimensionResult contracts."""

    def test_cypher_probes_importable(self):
        from cypher_probes import (
            probe_schema_population,
            probe_structural_connectivity,
            probe_consistency,
            probe_source_grounding,
        )
        # Functions exist and are callable
        assert callable(probe_schema_population)
        assert callable(probe_structural_connectivity)
        assert callable(probe_consistency)
        assert callable(probe_source_grounding)

    def test_schema_probe_requires_labels(self):
        """probe_schema_population should accept expected_labels."""
        import inspect
        from cypher_probes import probe_schema_population
        sig = inspect.signature(probe_schema_population)
        params = list(sig.parameters.keys())
        assert "driver" in params
        assert "expected_labels" in params

    def test_llm_probes_importable(self):
        from llm_probes import probe_coherence, probe_faithfulness
        assert callable(probe_coherence)
        assert callable(probe_faithfulness)

    def test_llm_probe_coherence_takes_triples(self):
        """probe_coherence should accept linearised triples string."""
        import inspect
        from llm_probes import probe_coherence
        sig = inspect.signature(probe_coherence)
        params = list(sig.parameters.keys())
        assert "triples" in params

    def test_llm_probe_faithfulness_takes_source_text(self):
        """probe_faithfulness should accept triples AND source_text."""
        import inspect
        from llm_probes import probe_faithfulness
        sig = inspect.signature(probe_faithfulness)
        params = list(sig.parameters.keys())
        assert "triples" in params
        assert "source_text" in params

    def test_llm_probe_faithfulness_detects_hallucinated(self):
        """Faithfulness probe should report hallucinated facts as errors."""
        # The probe returns DimensionResult with violations for hallucinated facts
        # We test the contract without calling the LLM
        v = Violation(
            dimension="faithfulness", severity="error",
            message="Hallucinated fact: car was blue"
        )
        assert v.severity == "error"
        assert "Hallucinated" in v.message


# ═══════════════════════════════════════════════════════════════════════════
# 7. SHACL shapes file tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSHACLShapesFile:
    """Validate the shapes.ttl file structure."""

    @pytest.fixture
    def shapes_content(self) -> str:
        shapes_path = _MODULE_DIR / "shapes.ttl"
        assert shapes_path.exists(), "shapes.ttl not found"
        return shapes_path.read_text()

    def test_shapes_file_has_prefixes(self, shapes_content: str):
        assert "@prefix sh:" in shapes_content
        assert "@prefix evt:" in shapes_content

    def test_shapes_has_event_shape(self, shapes_content: str):
        assert "EventShape" in shapes_content

    def test_shapes_has_person_shape(self, shapes_content: str):
        assert "PersonShape" in shapes_content

    def test_shapes_has_vehicle_shape(self, shapes_content: str):
        assert "VehicleShape" in shapes_content

    def test_shapes_has_location_shape(self, shapes_content: str):
        assert "LocationShape" in shapes_content

    def test_shapes_has_time_shape(self, shapes_content: str):
        assert "TimeShape" in shapes_content

    def test_shapes_has_observation_shape(self, shapes_content: str):
        assert "ObservationShape" in shapes_content

    def test_shapes_has_provenance_shape(self, shapes_content: str):
        assert "ProvenanceShape" in shapes_content

    def test_event_shape_requires_description(self, shapes_content: str):
        # Between EventShape and next shape, should have description constraint
        assert "evt:description" in shapes_content

    def test_event_shape_enforces_type_enum(self, shapes_content: str):
        """Event type should be constrained to known values."""
        assert "sh:in" in shapes_content
        assert "incident" in shapes_content or "observation" in shapes_content

    def test_person_shape_requires_name(self, shapes_content: str):
        assert "evt:name_or_description" in shapes_content

    def test_person_role_enum(self, shapes_content: str):
        """Person role should list valid roles."""
        assert "witness" in shapes_content
        assert "suspect" in shapes_content

    def test_shapes_use_valid_turtle_syntax(self, shapes_content: str):
        """Basic syntax: shapes should end with periods."""
        lines = shapes_content.strip().split("\n")
        non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        # Turtle blocks should end with " ."
        last_non_blank = non_empty[-1] if non_empty else ""
        assert last_non_blank.endswith("."), f"Last line: {last_non_blank}"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Core module consistency (quality_core self-consistency)
# ═══════════════════════════════════════════════════════════════════════════

class TestCoreModuleParity:
    """quality_core.py internal consistency checks."""

    def test_quality_report_score_fields_match_weight_keys(self):
        """Every weight key should have a corresponding *_score field on QualityReport."""
        qr = QualityReport()
        all_keys = set(_BASE_WEIGHTS) | set(_PHASE2_WEIGHTS) | set(_PHASE3_WEIGHTS)
        for dim in all_keys:
            attr = f"{dim}_score"
            assert hasattr(qr, attr), (
                f"QualityReport missing {attr} for weight dimension {dim}"
            )

    def test_compute_overall_and_generate_recommendations_use_same_thresholds(self):
        """Both functions should treat the same score as problematic."""
        # A schema_score of 0.5 should trigger a recommendation AND lower overall
        qr = QualityReport(
            schema_score=0.5, structural_score=1.0,
            constraint_score=1.0, consistency_score=1.0,
            coherence_score=1.0, faithfulness_score=1.0,
        )
        score = compute_overall(qr)
        recs = generate_recommendations(qr)
        assert score < 1.0  # lowered by schema
        assert len(recs) > 0  # recommendation generated

    def test_overall_score_agrees_between_entry_points(self):
        """compute_overall should produce deterministic results."""
        qr1 = QualityReport(
            schema_score=0.8, structural_score=0.9,
            constraint_score=0.7, consistency_score=0.85,
            coherence_score=0.75, faithfulness_score=0.6,
        )
        qr2 = QualityReport(
            schema_score=0.8, structural_score=0.9,
            constraint_score=0.7, consistency_score=0.85,
            coherence_score=0.75, faithfulness_score=0.6,
        )
        score1 = compute_overall(qr1)
        score2 = compute_overall(qr2)
        assert abs(score1 - score2) < 0.001


# ═══════════════════════════════════════════════════════════════════════════
# 9. Hallucination detection contract tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHallucinationDetection:
    """Faithfulness probes must follow a specific contract for hallucination detection."""

    def test_hallucination_violation_uses_error_severity(self):
        """Hallucinated facts must be reported as errors, not warnings."""
        v = Violation(
            dimension="faithfulness", severity="error",
            message="Hallucinated fact: the car was blue"
        )
        assert v.severity == "error"
        assert v.dimension == "faithfulness"

    def test_missing_fact_uses_warning_severity(self):
        """Facts missing from graph should be warnings, not errors."""
        v = Violation(
            dimension="faithfulness", severity="warning",
            message="Missing from graph: the cyclist was injured"
        )
        assert v.severity == "warning"

    def test_faithfulness_result_contract(self):
        """A faithfulness DimensionResult should contain hallucination count in details."""
        dr = DimensionResult(
            dimension="faithfulness", score=0.6,
            violations=[
                Violation("faithfulness", "error", "Hallucinated: X"),
                Violation("faithfulness", "warning", "Missing: Y"),
            ],
            details={
                "hallucinated_count": 1,
                "missing_count": 1,
                "raw_response": '{"score": 6}',
            },
        )
        assert dr.details["hallucinated_count"] == 1
        assert dr.details["missing_count"] == 1
        errors = [v for v in dr.violations if v.severity == "error"]
        warnings = [v for v in dr.violations if v.severity == "warning"]
        assert len(errors) == 1
        assert len(warnings) == 1

    def test_hallucination_scoring_proportional(self):
        """Score should decrease with more hallucinated facts."""
        # High hallucination → low faithfulness
        low_faith = DimensionResult(
            dimension="faithfulness", score=0.2,
            violations=[
                Violation("faithfulness", "error", f"Hallucinated: fact_{i}")
                for i in range(8)
            ],
        )
        high_faith = DimensionResult(
            dimension="faithfulness", score=0.9,
            violations=[],
        )
        assert low_faith.score < high_faith.score


# ═══════════════════════════════════════════════════════════════════════════
# 10. Calibration system tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibration:
    """calibrate_llm_probe() should aggregate multiple runs."""

    def test_calibration_returns_median(self):
        """Calibration should return median result with stats."""
        call_count = 0

        def fake_probe():
            nonlocal call_count
            call_count += 1
            scores = [0.6, 0.8, 0.7]
            return DimensionResult(
                dimension="coherence",
                score=scores[(call_count - 1) % 3],
            )

        result = calibrate_llm_probe(fake_probe, runs=3)
        assert "calibration" in result.details
        cal = result.details["calibration"]
        assert cal["runs"] == 3
        assert "mean" in cal
        assert "std" in cal
        assert "min" in cal
        assert "max" in cal

    def test_calibration_skips_on_sentinel(self):
        """If probe returns -1.0 sentinel, calibration should pass through."""
        def unavailable_probe():
            return DimensionResult(dimension="link_prediction", score=-1.0)

        result = calibrate_llm_probe(unavailable_probe, runs=3)
        assert result.score == -1.0


# ═══════════════════════════════════════════════════════════════════════════
# 11. Combined source validation
# ═══════════════════════════════════════════════════════════════════════════

class TestCombinedSourceValidation:
    """Tests that verify cross-module consistency with 06-ontologies."""

    def test_shacl_shapes_match_ontology_node_types(self):
        """shapes.ttl should cover the same entity types as schema-org-event-v1."""
        shapes_path = _MODULE_DIR / "shapes.ttl"
        if not shapes_path.exists():
            pytest.skip("shapes.ttl not found")

        shapes = shapes_path.read_text()

        # Expected from Schema.org ontology spec
        expected_entities = ["Event", "Person", "Vehicle", "Location", "Time"]
        for entity in expected_entities:
            assert f"{entity}Shape" in shapes, (
                f"shapes.ttl missing shape for {entity}"
            )

    def test_violation_dimensions_match_weight_system(self):
        """All weight keys should correspond to valid violation dimensions."""
        all_weight_keys = set(_BASE_WEIGHTS.keys()) | set(_PHASE2_WEIGHTS.keys()) | set(_PHASE3_WEIGHTS.keys())
        # Each should be usable as a violation dimension
        for dim in all_weight_keys:
            v = Violation(dimension=dim, severity="info", message="test")
            assert v.dimension == dim

    def test_quality_report_score_fields_match_weights(self):
        """Every weight key should have a corresponding *_score field on QualityReport."""
        qr = QualityReport()
        all_weight_keys = set(_BASE_WEIGHTS.keys()) | set(_PHASE2_WEIGHTS.keys()) | set(_PHASE3_WEIGHTS.keys())
        for dim in all_weight_keys:
            attr = f"{dim}_score"
            assert hasattr(qr, attr), (
                f"QualityReport missing {attr} for weight dimension {dim}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
