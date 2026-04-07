# Agentic Fundamentals

## What makes AI "agentic"

A conventional LLM interaction is stateless: you send a prompt, get a response, it's over. An **agentic** system differs in three ways:

**1. It maintains state across steps.** The system carries structured data — not just a message log — from one reasoning step to the next. That state might track what's been collected, what's been validated, what decisions have been made, and what remains to do.

**2. It makes decisions about its own control flow.** Rather than following a fixed sequence, an agent inspects its current state and decides which step to take next. It might call a tool, ask a human for input, retry a failed validation, or skip ahead — based on conditions, not a script.

**3. It uses tools.** An agent can reach outside the LLM to query a database, call an API, read a file, or run a calculation. The LLM decides *when* to use a tool and *what arguments* to pass, then incorporates the result into its next reasoning step.

The combination of persistent state, conditional branching, and tool use separates an agent from a chatbot. A chatbot generates text. An agent *does things*.

## The graph metaphor

Most agentic frameworks model the workflow as a **directed graph**. Each node is a processing step (call the LLM, run a tool, validate data, ask a human). Each edge defines what can happen next. Some edges are unconditional ("after extraction, always validate"). Others are **conditional** — a routing function inspects the current state and returns the name of the next node.

This is powerful because it lets you express complex logic — loops, branches, fallbacks, parallel paths — in a structure the framework can checkpoint, resume, and inspect. The graph *is* the program.

```
[START]
   │
   ▼
[Classify Query]
   │
   ├── greeting ──────► [Handle Greeting] ──► [END]
   │
   ├── order_query ───► [Lookup Order] ──► [Respond with Info] ──► [END]
   │
   └── general ───────► [General Response] ──► [END]
```

## State: the backbone

In a simple chatbot, state is a list of messages. In an agentic system, state is a **structured object with multiple typed fields**: the user's query, extracted entities, validation flags, collected form data, the current step, error counts, and so on.

This matters because conditional edges route based on state. If `email` is empty, route to the collection node. If `validation_errors` is non-empty, route to the retry node. If `approval_status` is "rejected", route to the escalation node. The richer your state schema, the more sophisticated your control flow.

## Tool calling

Modern LLMs support **function/tool calling** natively. You describe available tools as JSON schemas (name, description, parameters with types). The LLM decides when to invoke a tool and generates the arguments. The framework intercepts the tool call, executes it, and feeds the result back into the conversation.

This is fundamentally different from prompting the LLM to "write Python code" — the tool execution happens outside the LLM, in deterministic code you control. The LLM is the decision-maker; the tools are the hands.

## The ReAct pattern

ReAct (Reason + Act) is the dominant pattern in modern agents:

1. **Reason**: The LLM thinks about the current state and what to do next
2. **Act**: It invokes a tool with specific arguments
3. **Observe**: The tool result is returned
4. **Loop**: The LLM reasons about the result and decides on the next action — or returns a final answer

The loop continues until the LLM decides it has enough information to answer, or a maximum iteration limit is reached. The explicit reasoning step forces the model to articulate its plan, which reduces hallucination and creates inspection points for debugging.

## Checkpointing

After each node executes, the entire state is serialised and stored (in memory, SQLite, Postgres, or Neo4j). This enables:

- **Durability**: Process crashes → state survives → resume from last checkpoint
- **Time travel**: Retrieve any previous state for debugging
- **Thread isolation**: Multiple conversations run concurrently, each with independent state
- **Human-in-the-loop**: The graph pauses mid-execution, saves state, waits for human input, then resumes exactly where it stopped

## The demo

`demo.py` builds a customer service agent that:

1. Classifies incoming queries (greeting, order lookup, general)
2. Routes to the appropriate handler via conditional edges
3. Uses a tool to look up order information
4. Maintains rich typed state throughout
5. Demonstrates checkpointing with state retrieval

Run it:
```bash
cd 01-agentic-fundamentals
../langgraph-test/langgraph-env/bin/python demo.py
```
