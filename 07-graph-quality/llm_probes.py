"""
LLM-powered quality probes — coherence and faithfulness.

Uses the local Ollama model to judge whether the graph content
is internally coherent and faithful to the original source text.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from quality_core import DimensionResult, Violation

_MODEL = "qwen2.5:7b"


def _ask_llm(prompt: str) -> str:
    llm = ChatOllama(model=_MODEL, temperature=0)
    return llm.invoke(prompt).content


def probe_coherence(triples: str) -> DimensionResult:
    """Ask the LLM whether the graph triples form a coherent narrative."""
    prompt = (
        "You are a knowledge-graph quality auditor.\n"
        "Below are triples extracted from a knowledge graph.\n"
        "Rate the overall COHERENCE of these triples on a 0-10 scale.\n"
        "Coherence means: the triples form a consistent, non-contradictory narrative.\n"
        "Return ONLY a JSON object: {\"score\": <0-10>, \"issues\": [\"...\"]}\n\n"
        f"TRIPLES:\n{triples}"
    )
    raw = _ask_llm(prompt)

    import json
    try:
        result = json.loads(raw)
        score = float(result.get("score", 5)) / 10.0
        issues = result.get("issues", [])
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        issues = [f"LLM returned unparseable response: {raw[:200]}"]

    violations = [
        Violation(dimension="coherence", severity="info", message=issue)
        for issue in issues
    ]
    return DimensionResult(
        dimension="coherence", score=score,
        violations=violations,
        details={"raw_response": raw[:500]},
    )


def probe_faithfulness(triples: str, source_text: str) -> DimensionResult:
    """Judge whether graph triples are faithful to the original source text."""
    prompt = (
        "You are a knowledge-graph quality auditor.\n"
        "TASK: Compare the knowledge-graph triples against the original source text.\n"
        "Rate FAITHFULNESS on a 0-10 scale.\n"
        "Faithfulness means: every triple is supported by the source text "
        "(no hallucinated facts).\n"
        "Return ONLY a JSON object: "
        "{\"score\": <0-10>, \"hallucinated\": [\"...\"], \"missing\": [\"...\"]}\n\n"
        f"SOURCE TEXT:\n{source_text[:3000]}\n\n"
        f"TRIPLES:\n{triples}"
    )
    raw = _ask_llm(prompt)

    import json
    try:
        result = json.loads(raw)
        score = float(result.get("score", 5)) / 10.0
        hallucinated = result.get("hallucinated", [])
        missing = result.get("missing", [])
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        hallucinated = []
        missing = [f"LLM returned unparseable response: {raw[:200]}"]

    violations = []
    for h in hallucinated:
        violations.append(Violation(
            dimension="faithfulness", severity="error",
            message=f"Hallucinated fact: {h}",
        ))
    for m in missing:
        violations.append(Violation(
            dimension="faithfulness", severity="warning",
            message=f"Missing from graph: {m}",
        ))

    return DimensionResult(
        dimension="faithfulness", score=score,
        violations=violations,
        details={
            "hallucinated_count": len(hallucinated),
            "missing_count": len(missing),
            "raw_response": raw[:500],
        },
    )
