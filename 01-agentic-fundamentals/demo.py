"""
Module 01: Agentic Fundamentals
================================
Demonstrates: StateGraph, typed state, nodes, conditional edges,
tool calling, checkpointing, and state inspection.

A customer service agent that classifies queries, routes to the
right handler, looks up orders via a tool, and checkpoints every step.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ─── LLM ─────────────────────────────────────────────────────────────────
llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ─── Step 1: Define rich typed state ─────────────────────────────────────
# This is NOT just a message list. Each field carries specific meaning
# and drives routing decisions.

class AgentState(TypedDict):
    query: str               # the user's original input
    classification: str      # greeting | order_query | general
    order_id: str             # extracted from query if present
    tool_result: str          # result from order lookup
    response: str             # final response to the user
    steps: list[str]          # audit trail of what happened


# ─── Step 2: Define a tool ───────────────────────────────────────────────
# This is a regular Python function. The agent decides when to call it.
# In production, this would hit a real database or API.

ORDER_DATABASE = {
    "ORD-1001": {"status": "shipped", "eta": "tomorrow", "item": "Mechanical Keyboard"},
    "ORD-1002": {"status": "processing", "eta": "3 days", "item": "USB-C Hub"},
    "ORD-1003": {"status": "delivered", "eta": "n/a", "item": "Monitor Stand"},
}

def lookup_order(order_id: str) -> str:
    """Look up an order in the database."""
    order = ORDER_DATABASE.get(order_id)
    if order:
        return f"Order {order_id}: {order['item']} — status: {order['status']}, ETA: {order['eta']}"
    return f"Order {order_id} not found."


# ─── Step 3: Define node functions ───────────────────────────────────────
# Each node is a function that receives state and returns a partial update.
# The framework merges the update into the existing state.

def classify(state: AgentState) -> dict:
    """Node 1: Classify the user's query and extract order ID if present."""
    messages = [
        SystemMessage(content=(
            "Classify the user's message into exactly one category:\n"
            "- greeting: if the user is saying hello or making small talk\n"
            "- order_query: if the user is asking about an order (look for ORD-XXXX pattern)\n"
            "- general: anything else\n\n"
            "Also extract any order ID (format: ORD-XXXX) if present.\n\n"
            "Respond in this exact format:\n"
            "CATEGORY: <category>\n"
            "ORDER_ID: <order_id or none>"
        )),
        HumanMessage(content=state["query"]),
    ]
    result = llm.invoke(messages)
    text = result.content.strip()

    # Parse the structured response
    classification = "general"
    order_id = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("CATEGORY:"):
            classification = line.split(":", 1)[1].strip().lower()
        elif line.startswith("ORDER_ID:"):
            val = line.split(":", 1)[1].strip()
            if val.lower() != "none":
                order_id = val

    return {
        "classification": classification,
        "order_id": order_id,
        "steps": state.get("steps", []) + [f"classified as '{classification}'"],
    }


def handle_greeting(state: AgentState) -> dict:
    """Node 2a: Respond to a greeting."""
    messages = [
        SystemMessage(content="You are a friendly customer service agent. Respond to the greeting warmly in one sentence."),
        HumanMessage(content=state["query"]),
    ]
    result = llm.invoke(messages)
    return {
        "response": result.content,
        "steps": state["steps"] + ["handled as greeting"],
    }


def lookup_order_node(state: AgentState) -> dict:
    """Node 2b: Use the tool to look up order information."""
    # This is where the agent ACTS — calling an external tool
    tool_result = lookup_order(state["order_id"])
    return {
        "tool_result": tool_result,
        "steps": state["steps"] + [f"looked up order {state['order_id']}"],
    }


def respond_with_order_info(state: AgentState) -> dict:
    """Node 2c: Generate a response incorporating tool results."""
    messages = [
        SystemMessage(content="You are a customer service agent. Use the order information to give a helpful, concise response."),
        HumanMessage(content=f"Customer asked: {state['query']}\n\nOrder info: {state['tool_result']}"),
    ]
    result = llm.invoke(messages)
    return {
        "response": result.content,
        "steps": state["steps"] + ["generated order response"],
    }


def handle_general(state: AgentState) -> dict:
    """Node 2d: Handle general queries."""
    messages = [
        SystemMessage(content="You are a helpful customer service agent. Answer concisely in 1-2 sentences."),
        HumanMessage(content=state["query"]),
    ]
    result = llm.invoke(messages)
    return {
        "response": result.content,
        "steps": state["steps"] + ["handled as general query"],
    }


# ─── Step 4: Define conditional routing ──────────────────────────────────
# This is the decision point. The function inspects state and returns
# the name of the next node. This is deterministic code, not LLM output.

def route_by_classification(state: AgentState) -> Literal["handle_greeting", "lookup_order", "handle_general"]:
    """Route to the appropriate handler based on classification."""
    routes = {
        "greeting": "handle_greeting",
        "order_query": "lookup_order",
    }
    return routes.get(state["classification"], "handle_general")


# ─── Step 5: Build the graph ─────────────────────────────────────────────

def build_agent():
    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("classify", classify)
    builder.add_node("handle_greeting", handle_greeting)
    builder.add_node("lookup_order", lookup_order_node)
    builder.add_node("respond_with_order", respond_with_order_info)
    builder.add_node("handle_general", handle_general)

    # Wire the edges
    builder.add_edge(START, "classify")                            # always start by classifying
    builder.add_conditional_edges("classify", route_by_classification)  # then route
    builder.add_edge("handle_greeting", END)                       # greetings end
    builder.add_edge("lookup_order", "respond_with_order")         # tool result → response
    builder.add_edge("respond_with_order", END)                    # then end
    builder.add_edge("handle_general", END)                        # general queries end

    # Compile with a checkpointer — every node execution is persisted
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ─── Step 6: Run the agent ───────────────────────────────────────────────

def main():
    agent = build_agent()

    test_cases = [
        "Hey there, how's it going?",
        "Where is my order ORD-1001?",
        "What's your return policy?",
        "Can you check on ORD-9999?",  # order not found
    ]

    print("=" * 64)
    print("  Module 01: Agentic Fundamentals")
    print("  Customer Service Agent with Tools + Conditional Routing")
    print("=" * 64)

    for i, query in enumerate(test_cases, 1):
        config = {"configurable": {"thread_id": f"customer-{i}"}}
        result = agent.invoke({"query": query, "steps": []}, config)

        print(f"\n{'─' * 64}")
        print(f"  Query {i}: {query}")
        print(f"  Classification: {result['classification']}")
        if result.get("order_id"):
            print(f"  Order ID: {result['order_id']}")
        if result.get("tool_result"):
            print(f"  Tool Result: {result['tool_result']}")
        print(f"  Response: {result['response']}")
        print(f"  Steps: {result['steps']}")

    # ─── Step 7: Demonstrate checkpointing ───────────────────────────────
    print(f"\n{'=' * 64}")
    print("  Checkpointing Demo")
    print("=" * 64)

    # Retrieve state from thread "customer-2" (the order lookup)
    config = {"configurable": {"thread_id": "customer-2"}}
    snapshot = agent.get_state(config)

    print(f"\n  Retrieved state for thread 'customer-2':")
    print(f"    query: {snapshot.values.get('query')}")
    print(f"    classification: {snapshot.values.get('classification')}")
    print(f"    order_id: {snapshot.values.get('order_id')}")
    print(f"    tool_result: {snapshot.values.get('tool_result')}")
    print(f"    steps: {snapshot.values.get('steps')}")
    print(f"\n  This state was checkpointed automatically after each node.")
    print(f"  In production, this would survive process restarts,")
    print(f"  enable time-travel debugging, and support multi-turn flows.")


if __name__ == "__main__":
    main()
