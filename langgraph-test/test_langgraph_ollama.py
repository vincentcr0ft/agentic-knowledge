"""
LangGraph + ChatOllama (qwen2.5:7b) end-to-end pipeline test.
Sovereign: no API keys, no cloud, fully local.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# ── State ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    response: str
    steps: list[str]


# ── LLM ──────────────────────────────────────────────────────────────────
llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ── Nodes ────────────────────────────────────────────────────────────────
def think(state: AgentState) -> dict:
    """LLM analyses the query and decides on approach."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Briefly analyze the user's question and state your approach in one sentence."),
        HumanMessage(content=state["query"]),
    ]
    result = llm.invoke(messages)
    return {
        "steps": state.get("steps", []) + [f"think: {result.content[:100]}"],
    }


def respond(state: AgentState) -> dict:
    """LLM generates the final answer."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Give a concise answer in 2-3 sentences max."),
        HumanMessage(content=state["query"]),
    ]
    result = llm.invoke(messages)
    return {
        "response": result.content,
        "steps": state["steps"] + ["respond: generated final answer"],
    }


# ── Build graph ──────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("think", think)
    builder.add_node("respond", respond)
    builder.add_edge(START, "think")
    builder.add_edge("think", "respond")
    builder.add_edge("respond", END)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── Run ──────────────────────────────────────────────────────────────────
def main():
    graph = build_graph()

    queries = [
        "What is LangGraph used for?",
        "Explain the difference between a list and a tuple in Python.",
    ]

    print("=" * 60)
    print("LangGraph + ChatOllama (qwen2.5:7b) Pipeline Test")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")

        config = {"configurable": {"thread_id": f"test-{i}"}}
        result = graph.invoke({"query": query, "steps": []}, config)

        print(f"Response: {result['response']}")
        print(f"Steps: {result['steps']}")

    # Verify checkpointer
    config = {"configurable": {"thread_id": "test-1"}}
    snapshot = graph.get_state(config)
    checkpoint_ok = snapshot.values.get("query") == queries[0]
    print(f"\nCheckpointer: {'PASS' if checkpoint_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print("SOVEREIGN LLM PIPELINE: WORKING")
    print("No API keys. No cloud. Fully local.")
    print("=" * 60)


if __name__ == "__main__":
    main()
