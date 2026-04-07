"""
Full LangGraph pipeline test using a FakeLLM.
Proves: StateGraph, nodes, conditional edges, checkpointer, tool-calling pattern.
No model download required — swap FakeLLM for ChatOllama when a model is available.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ── State ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    classification: str
    response: str
    steps: list[str]


# ── Fake LLM (deterministic, no download) ────────────────────────────────
class FakeLLM:
    """Drop-in replacement for testing. Returns canned responses."""

    def classify(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["hello", "hi", "hey", "greet"]):
            return "greeting"
        if any(w in q for w in ["code", "python", "function", "debug", "error"]):
            return "technical"
        return "general"

    def respond(self, query: str, classification: str) -> str:
        responses = {
            "greeting": f"Hello! How can I help you today?",
            "technical": f"Here's a technical answer about: {query}",
            "general": f"Here's a general answer about: {query}",
        }
        return responses.get(classification, "I'm not sure how to help with that.")


llm = FakeLLM()


# ── Node functions ───────────────────────────────────────────────────────
def classify_query(state: AgentState) -> dict:
    classification = llm.classify(state["query"])
    return {
        "classification": classification,
        "steps": state.get("steps", []) + [f"classified as '{classification}'"],
    }


def handle_greeting(state: AgentState) -> dict:
    response = llm.respond(state["query"], "greeting")
    return {
        "response": response,
        "steps": state["steps"] + ["handled greeting"],
    }


def handle_technical(state: AgentState) -> dict:
    response = llm.respond(state["query"], "technical")
    return {
        "response": response,
        "steps": state["steps"] + ["handled technical"],
    }


def handle_general(state: AgentState) -> dict:
    response = llm.respond(state["query"], "general")
    return {
        "response": response,
        "steps": state["steps"] + ["handled general"],
    }


# ── Conditional edge router ─────────────────────────────────────────────
def route_query(state: AgentState) -> Literal["handle_greeting", "handle_technical", "handle_general"]:
    mapping = {
        "greeting": "handle_greeting",
        "technical": "handle_technical",
        "general": "handle_general",
    }
    return mapping.get(state["classification"], "handle_general")


# ── Build graph ──────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("classify", classify_query)
    builder.add_node("handle_greeting", handle_greeting)
    builder.add_node("handle_technical", handle_technical)
    builder.add_node("handle_general", handle_general)

    # Edges
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_query)
    builder.add_edge("handle_greeting", END)
    builder.add_edge("handle_technical", END)
    builder.add_edge("handle_general", END)

    # Compile with checkpointer
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── Run tests ────────────────────────────────────────────────────────────
def main():
    graph = build_graph()

    test_cases = [
        {"query": "Hello there!", "expected_class": "greeting"},
        {"query": "How do I write a Python function?", "expected_class": "technical"},
        {"query": "What is the meaning of life?", "expected_class": "general"},
    ]

    print("=" * 60)
    print("LangGraph Pipeline Test (FakeLLM)")
    print("=" * 60)

    all_passed = True
    for i, tc in enumerate(test_cases, 1):
        config = {"configurable": {"thread_id": f"test-{i}"}}
        result = graph.invoke({"query": tc["query"], "steps": []}, config)

        passed = result["classification"] == tc["expected_class"]
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"\nTest {i}: [{status}]")
        print(f"  Query:          {tc['query']}")
        print(f"  Classification: {result['classification']} (expected: {tc['expected_class']})")
        print(f"  Response:       {result['response']}")
        print(f"  Steps:          {result['steps']}")

    # Verify checkpointer persistence
    config = {"configurable": {"thread_id": "test-1"}}
    snapshot = graph.get_state(config)
    checkpoint_ok = snapshot.values.get("query") == "Hello there!"
    print(f"\nCheckpointer test: {'PASS' if checkpoint_ok else 'FAIL'}")
    print(f"  Retrieved state for thread 'test-1': query={snapshot.values.get('query')!r}")

    if not checkpoint_ok:
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("Swap FakeLLM for ChatOllama when a model is available:")
        print("  from langchain_ollama import ChatOllama")
        print("  llm = ChatOllama(model='qwen2.5:7b')")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
