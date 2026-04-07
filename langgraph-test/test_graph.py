#!/usr/bin/env python3
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class SimpleState(TypedDict):
    count: int

def inc(state: SimpleState):
    return {"count": state.get("count", 0) + 1}

def main():
    builder = StateGraph(SimpleState)
    builder.add_node("inc", inc)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-thread"}}
    result = graph.invoke({"count": 0}, config)
    print("Invoke result:\n", result)

if __name__ == "__main__":
    main()
