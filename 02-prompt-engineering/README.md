# Prompt Engineering in Agentic Settings

## Why agentic prompting is different

When you prompt a standalone LLM, you craft a single instruction and get a single response. In an agentic system, you're designing a **system of prompts** — each serving a different node in a graph, each with a narrow responsibility. The quality of the whole system depends on how well these prompts work together.

## System prompts as behavioural contracts

Each node has its own system prompt that acts as a **specification**:

- **Role**: "You are a data validator" / "You are a query classifier"
- **Output format**: "Respond with a JSON object matching this schema" / "Respond with exactly one word"
- **Constraints**: "Only use information from the provided context" / "Never include personal opinions"
- **Scope boundaries**: "If you cannot determine the answer, return null for that field"

The more constrained the prompt, the more reliable the output. Agents need predictability, not creativity. Broad, do-everything prompts are the enemy.

## Prompt decomposition

The core principle: take a complex task and decompose it into subtasks that each have a clear input-output contract.

**Bad** (single mega-prompt):
> "Read this email. Extract the sender's name, role, and years of experience. Classify them as qualified or unqualified. Write a polite response."

The LLM tries to do everything at once, often skipping steps or mixing concerns.

**Good** (decomposed across nodes):
1. **Extract node**: "Extract name, role, and years of experience. Return JSON."
2. **Classify node**: "Given this profile, classify as qualified or unqualified. Return one word."
3. **Respond node**: "Write a polite response appropriate for this classification."

Each prompt is testable in isolation. If extraction breaks, you fix that prompt without touching classification or response generation.

## Structured output

Agents need LLMs to produce structured data. Two mechanisms:

**Tool/function calling**: The LLM receives tool schemas with typed parameters and generates invocation arguments. The framework validates the output against the schema.

**Direct JSON output**: The prompt instructs the LLM to respond in a specific JSON format. Less reliable than native tool calling — the response may include trailing text, miss required fields, or produce invalid syntax.

The failure modes are important to understand:
- Valid JSON that doesn't match your schema (missing fields)
- Correct schema but hallucinated values
- Text wrapped around the JSON ("Here's the result: {...}")
- Inconsistent formatting across invocations

## Temperature in agents

For intermediate processing nodes (classification, extraction, routing), temperature should be **0** or near-zero. Non-determinism in these steps causes inconsistent routing, failed validations, and irreproducible bugs.

For user-facing generation nodes, temperature can be higher (0.3–0.7) to allow natural variation. But even here, lower is usually better for agents — you want reliability over novelty.

## The demo

`demo.py` processes a job application email through a three-node pipeline:

1. **Extract**: Pulls structured data (name, role, experience) with a constrained JSON prompt
2. **Classify**: Determines qualification status with a single-word response prompt
3. **Respond**: Generates an appropriate response using the extracted data and classification

It then contrasts this with a single mega-prompt doing the same task, showing how decomposition improves reliability.

Run it:
```bash
cd 02-prompt-engineering
../langgraph-test/langgraph-env/bin/python demo.py
```
