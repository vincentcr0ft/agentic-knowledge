"""
Phase 2 — DeepEval-based structured quality probes.

Requires: pip install deepeval

Uses DeepEval's G-Eval (LLM-as-judge) framework to provide standardised,
research-backed evaluation metrics with explanations.  Falls back to the
native LLM probes in llm_probes.py if DeepEval is not installed.

Metrics:
  - NarrativeCoherence:       Does the graph form a coherent story?
  - ExtractionFaithfulness:   Does the graph faithfully represent the source?
  - SemanticCompleteness:     Are important source facts captured in the graph?
  - InvestigativeReadiness:   Is the graph useful for investigative purposes?
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quality_probe.core import DimensionResult, Violation
from schema import linearise_graph


def _deepeval_available() -> bool:
    """Check if DeepEval is installed and compatible with this Python version."""
    try:
        import deepeval  # noqa: F401
        return True
    except Exception:
        # ImportError if not installed; TypeError on Python <3.10
        # (deepeval uses X | None union syntax requiring 3.10+)
        return False


def _get_ollama_model():
    """Create a DeepEval-compatible wrapper around the local Ollama model."""
    from deepeval.models import DeepEvalBaseLLM

    class OllamaModel(DeepEvalBaseLLM):
        """Wrap the local qwen2.5:7b model served by Ollama."""

        def __init__(self):
            self._model_name = "qwen2.5:7b"
            super().__init__(model=self._model_name)

        def load_model(self):
            from langchain_ollama import ChatOllama
            return ChatOllama(model=self._model_name, temperature=0)

        def generate(self, prompt: str, **kwargs) -> str:
            from langchain_core.messages import HumanMessage
            resp = self.model.invoke([HumanMessage(content=prompt)])
            return resp.content

        async def a_generate(self, prompt: str, **kwargs) -> str:
            from langchain_core.messages import HumanMessage
            resp = await self.model.ainvoke([HumanMessage(content=prompt)])
            return resp.content

        def get_model_name(self) -> str:
            return self._model_name

    return OllamaModel()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Narrative Coherence (G-Eval)
# ═══════════════════════════════════════════════════════════════════════════════

def probe_coherence_deepeval(driver) -> DimensionResult:
    """Score narrative coherence using DeepEval G-Eval.

    Falls back to the native LLM probe if DeepEval is not installed.
    """
    if not _deepeval_available():
        return DimensionResult(
            dimension="coherence",
            score=-1.0,  # sentinel: caller should fall back
            violations=[Violation(
                dimension="coherence",
                severity="info",
                message="DeepEval not installed — falling back to native LLM probe",
            )],
            details={"skipped": True, "reason": "deepeval not installed"},
        )

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    triples = linearise_graph(driver)
    if triples == "(empty graph)":
        return DimensionResult(
            dimension="coherence",
            score=0.0,
            violations=[Violation(
                dimension="coherence",
                severity="error",
                message="Cannot assess coherence — graph is empty",
            )],
        )

    coherence_metric = GEval(
        name="NarrativeCoherence",
        model=_get_ollama_model(),
        criteria=(
            "Evaluate whether this knowledge graph, extracted from a witness "
            "statement, represents a coherent and internally consistent "
            "narrative of a witnessed event. Consider:\n"
            "1. Do events form a logical, understandable sequence?\n"
            "2. Are participants consistently referenced across events?\n"
            "3. Do spatial movements between locations make physical sense?\n"
            "4. Are causal links plausible and complete?\n"
            "5. Can you reconstruct a clear, unambiguous story from this graph?\n"
            "Score 0.0 for incoherent, 1.0 for perfectly coherent."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
    )

    test_case = LLMTestCase(
        input="Evaluate knowledge graph narrative coherence",
        actual_output=triples,
    )

    try:
        coherence_metric.measure(test_case)
        score = float(coherence_metric.score)
        reason = coherence_metric.reason or ""
    except Exception as e:
        return DimensionResult(
            dimension="coherence",
            score=0.5,
            violations=[Violation(
                dimension="coherence",
                severity="info",
                message=f"DeepEval coherence metric failed: {e}",
            )],
            details={"error": str(e)},
        )

    violations = []
    if score < 0.6:
        violations.append(Violation(
            dimension="coherence",
            severity="warning",
            message=f"Low narrative coherence ({score:.2f}): {reason}",
        ))

    return DimensionResult(
        dimension="coherence",
        score=max(0.0, min(1.0, score)),
        violations=violations,
        details={
            "deepeval_score": score,
            "reason": reason,
            "metric": "GEval/NarrativeCoherence",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Extraction Faithfulness (G-Eval)
# ═══════════════════════════════════════════════════════════════════════════════

def probe_faithfulness_deepeval(driver, source_text: str) -> DimensionResult:
    """Score extraction faithfulness using DeepEval G-Eval.

    Checks whether the extracted graph is faithful to the source text,
    without hallucinated facts.

    Falls back to the native LLM probe if DeepEval is not installed.
    """
    if not _deepeval_available():
        return DimensionResult(
            dimension="faithfulness",
            score=-1.0,
            violations=[Violation(
                dimension="faithfulness",
                severity="info",
                message="DeepEval not installed — falling back to native LLM probe",
            )],
            details={"skipped": True, "reason": "deepeval not installed"},
        )

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    triples = linearise_graph(driver)
    if triples == "(empty graph)":
        return DimensionResult(
            dimension="faithfulness",
            score=0.0,
            violations=[Violation(
                dimension="faithfulness",
                severity="error",
                message="Cannot assess faithfulness — graph is empty",
            )],
        )

    faithfulness_metric = GEval(
        name="ExtractionFaithfulness",
        model=_get_ollama_model(),
        criteria=(
            "Compare the source witness statement (the 'input') against the "
            "extracted knowledge graph (the 'actual_output'). Evaluate:\n"
            "1. Are there facts in the graph NOT stated or implied in the source? "
            "(Hallucinations — most serious)\n"
            "2. Are facts in the graph distorted from what the source says? "
            "(Wrong time, wrong person, exaggerated details)\n"
            "3. Is the graph a faithful representation of the source text?\n"
            "Score 0.0 for entirely hallucinated, 1.0 for perfectly faithful."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.8,
    )

    test_case = LLMTestCase(
        input=source_text,
        actual_output=triples,
    )

    try:
        faithfulness_metric.measure(test_case)
        score = float(faithfulness_metric.score)
        reason = faithfulness_metric.reason or ""
    except Exception as e:
        return DimensionResult(
            dimension="faithfulness",
            score=0.5,
            violations=[Violation(
                dimension="faithfulness",
                severity="info",
                message=f"DeepEval faithfulness metric failed: {e}",
            )],
            details={"error": str(e)},
        )

    violations = []
    if score < 0.8:
        violations.append(Violation(
            dimension="faithfulness",
            severity="warning" if score >= 0.5 else "error",
            message=f"Faithfulness concern ({score:.2f}): {reason}",
        ))

    return DimensionResult(
        dimension="faithfulness",
        score=max(0.0, min(1.0, score)),
        violations=violations,
        details={
            "deepeval_score": score,
            "reason": reason,
            "metric": "GEval/ExtractionFaithfulness",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Semantic Completeness (G-Eval)
# ═══════════════════════════════════════════════════════════════════════════════

def probe_semantic_completeness(driver, source_text: str) -> DimensionResult:
    """Score whether the graph captures all important facts from the source.

    Complements schema completeness (which checks node-type population)
    with a semantic check: are there important facts in the source text
    that the graph fails to represent?
    """
    if not _deepeval_available():
        return DimensionResult(
            dimension="semantic_completeness",
            score=-1.0,
            violations=[Violation(
                dimension="semantic_completeness",
                severity="info",
                message="DeepEval not installed — semantic completeness skipped",
            )],
            details={"skipped": True},
        )

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    triples = linearise_graph(driver)
    if triples == "(empty graph)":
        return DimensionResult(
            dimension="semantic_completeness",
            score=0.0,
            violations=[Violation(
                dimension="semantic_completeness",
                severity="error",
                message="Cannot assess completeness — graph is empty",
            )],
        )

    completeness_metric = GEval(
        name="SemanticCompleteness",
        model=_get_ollama_model(),
        criteria=(
            "Compare the source witness statement (the 'input') against the "
            "extracted knowledge graph (the 'actual_output'). Evaluate how "
            "completely the graph captures facts from the source:\n"
            "1. Are all people mentioned in the source represented?\n"
            "2. Are all events and actions captured?\n"
            "3. Are all times, locations, and objects included?\n"
            "4. Are physical descriptions and vehicle details preserved?\n"
            "5. Are relationships between entities correctly represented?\n"
            "Score 0.0 for nothing captured, 1.0 for every fact represented."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=source_text,
        actual_output=triples,
    )

    try:
        completeness_metric.measure(test_case)
        score = float(completeness_metric.score)
        reason = completeness_metric.reason or ""
    except Exception as e:
        return DimensionResult(
            dimension="semantic_completeness",
            score=0.5,
            violations=[Violation(
                dimension="semantic_completeness",
                severity="info",
                message=f"DeepEval completeness metric failed: {e}",
            )],
            details={"error": str(e)},
        )

    violations = []
    if score < 0.7:
        violations.append(Violation(
            dimension="semantic_completeness",
            severity="warning" if score >= 0.4 else "error",
            message=f"Semantic completeness gap ({score:.2f}): {reason}",
        ))

    return DimensionResult(
        dimension="semantic_completeness",
        score=max(0.0, min(1.0, score)),
        violations=violations,
        details={
            "deepeval_score": score,
            "reason": reason,
            "metric": "GEval/SemanticCompleteness",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Investigative Readiness (G-Eval)
# ═══════════════════════════════════════════════════════════════════════════════

def probe_investigative_readiness(driver) -> DimensionResult:
    """Score whether the graph is useful for investigative purposes.

    Evaluates from a police investigator's perspective: would this graph
    help reconstruct what happened and identify next steps?
    """
    if not _deepeval_available():
        return DimensionResult(
            dimension="investigative_readiness",
            score=-1.0,
            violations=[Violation(
                dimension="investigative_readiness",
                severity="info",
                message="DeepEval not installed — investigative readiness skipped",
            )],
            details={"skipped": True},
        )

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    triples = linearise_graph(driver)
    if triples == "(empty graph)":
        return DimensionResult(
            dimension="investigative_readiness",
            score=0.0,
            violations=[Violation(
                dimension="investigative_readiness",
                severity="error",
                message="Cannot assess readiness — graph is empty",
            )],
        )

    readiness_metric = GEval(
        name="InvestigativeReadiness",
        model=_get_ollama_model(),
        criteria=(
            "Evaluate this knowledge graph from a police investigator's "
            "perspective. Score how useful it would be for:\n"
            "1. Reconstructing the sequence of events\n"
            "2. Identifying suspects and witnesses\n"
            "3. Establishing a timeline with specific times/dates\n"
            "4. Locating the scene and relevant places\n"
            "5. Identifying physical evidence to corroborate accounts\n"
            "6. Spotting gaps that need follow-up investigation\n"
            "Score 0.0 for useless, 1.0 for investigation-ready."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )

    test_case = LLMTestCase(
        input="Evaluate knowledge graph investigative readiness",
        actual_output=triples,
    )

    try:
        readiness_metric.measure(test_case)
        score = float(readiness_metric.score)
        reason = readiness_metric.reason or ""
    except Exception as e:
        return DimensionResult(
            dimension="investigative_readiness",
            score=0.5,
            violations=[Violation(
                dimension="investigative_readiness",
                severity="info",
                message=f"DeepEval readiness metric failed: {e}",
            )],
            details={"error": str(e)},
        )

    violations = []
    if score < 0.5:
        violations.append(Violation(
            dimension="investigative_readiness",
            severity="warning",
            message=f"Low investigative readiness ({score:.2f}): {reason}",
        ))

    return DimensionResult(
        dimension="investigative_readiness",
        score=max(0.0, min(1.0, score)),
        violations=violations,
        details={
            "deepeval_score": score,
            "reason": reason,
            "metric": "GEval/InvestigativeReadiness",
        },
    )
