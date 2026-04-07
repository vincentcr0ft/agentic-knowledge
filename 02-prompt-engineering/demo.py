"""
Module 02: Prompt Engineering in Agentic Settings
===================================================
Demonstrates: Prompt decomposition, system prompts as contracts,
structured output extraction, and the difference between a single
mega-prompt vs a multi-node pipeline.

Processes a job application email through a decomposed pipeline,
then compares with a single-prompt approach.
"""

import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ─── LLM ─────────────────────────────────────────────────────────────────
# temperature=0 for extraction and classification (deterministic)
llm_precise = ChatOllama(model="qwen2.5:7b", temperature=0)
# slightly higher for the response node (natural language)
llm_creative = ChatOllama(model="qwen2.5:7b", temperature=0.3)


# ─── Sample data ─────────────────────────────────────────────────────────

APPLICATION_EMAILS = [
    """Dear Hiring Manager,

My name is Sarah Chen and I'm writing to apply for the Senior Python Developer
position. I have 8 years of experience in software engineering, with the last
4 years focused on Python backend development. I've led teams of up to 6
developers and have extensive experience with FastAPI, PostgreSQL, and AWS.

I'm currently a Tech Lead at DataFlow Inc and am looking for a new challenge.

Best regards,
Sarah Chen""",

    """Hi,

I saw your job posting and wanted to throw my hat in the ring. I'm Jake, just
graduated from university with a CS degree. I did a 3-month internship at a
startup where I used Python a bit. I'm really enthusiastic about learning more!

Cheers,
Jake Martinez""",

    """To whom it may concern,

I am Dr. Priya Sharma, a principal engineer with 15 years of experience in
distributed systems. I hold a PhD in Computer Science from MIT and have
published 12 papers on fault-tolerant architectures. Most recently I designed
the real-time data pipeline at MegaCorp processing 2M events/second.

I believe I would be an excellent fit for your engineering leadership role.

Regards,
Dr. Priya Sharma""",
]


# ─── State ────────────────────────────────────────────────────────────────

class ApplicationState(TypedDict):
    email: str                 # raw email text
    extracted: dict            # structured data from extraction node
    classification: str        # qualified | unqualified
    response: str              # generated response letter
    steps: list[str]           # audit trail


# ─── Node 1: Extract ─────────────────────────────────────────────────────
# System prompt is a BEHAVIOURAL CONTRACT: specific role, specific format,
# specific constraints. The LLM has one job.

EXTRACT_PROMPT = """You are a data extraction specialist. Your ONLY job is to
extract structured information from a job application email.

Return a JSON object with exactly these fields:
{
    "name": "full name of the applicant",
    "current_role": "their current job title or 'student' or 'unknown'",
    "years_experience": <integer or 0 if not mentioned>,
    "key_skills": ["list", "of", "mentioned", "skills"],
    "notable_achievements": "one-sentence summary of standout achievements, or 'none mentioned'"
}

Rules:
- Extract ONLY what is explicitly stated. Do not infer or assume.
- If a field is not mentioned, use the default value shown above.
- Return ONLY the JSON object. No explanation, no preamble."""


def extract(state: ApplicationState) -> dict:
    """Extract structured data from the email using a constrained prompt."""
    messages = [
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=state["email"]),
    ]
    result = llm_precise.invoke(messages)

    # Parse JSON from the response
    try:
        extracted = json.loads(result.content.strip())
    except json.JSONDecodeError:
        # Fallback: try to find JSON in the response
        text = result.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        extracted = json.loads(text[start:end]) if start >= 0 else {"error": "parse failed"}

    return {
        "extracted": extracted,
        "steps": state.get("steps", []) + ["extracted structured data"],
    }


# ─── Node 2: Classify ────────────────────────────────────────────────────
# Even narrower prompt. ONE word output. Temperature=0.

CLASSIFY_PROMPT = """You are a hiring classifier. Based on the candidate profile below,
classify them as either "qualified" or "unqualified" for a Senior Python Developer role.

Requirements for "qualified":
- At least 5 years of experience
- Python mentioned as a skill or in their role
- Some leadership or team experience

Respond with EXACTLY one word: qualified or unqualified"""


def classify(state: ApplicationState) -> dict:
    """Classify the candidate based on extracted data."""
    profile_summary = json.dumps(state["extracted"], indent=2)
    messages = [
        SystemMessage(content=CLASSIFY_PROMPT),
        HumanMessage(content=f"Candidate profile:\n{profile_summary}"),
    ]
    result = llm_precise.invoke(messages)
    classification = result.content.strip().lower().rstrip(".")

    # Normalise to expected values
    if "qualified" in classification and "unqualified" not in classification:
        classification = "qualified"
    elif "unqualified" in classification:
        classification = "unqualified"
    else:
        classification = "unqualified"  # default safe

    return {
        "classification": classification,
        "steps": state["steps"] + [f"classified as {classification}"],
    }


# ─── Node 3: Respond ─────────────────────────────────────────────────────
# Slightly higher temperature for natural language. Prompt includes
# all context from previous nodes.

RESPOND_QUALIFIED = """You are a friendly hiring manager. Write a brief, warm response
(3-4 sentences) to a qualified candidate. Thank them, mention something specific
from their background, and let them know next steps (an interview will be scheduled).
Do not use placeholder brackets."""

RESPOND_UNQUALIFIED = """You are a kind hiring manager. Write a brief, respectful response
(2-3 sentences) to a candidate who doesn't meet the requirements. Thank them for
their interest, gently explain they're looking for more experience, and encourage
them to apply again in the future. Do not use placeholder brackets."""


def respond(state: ApplicationState) -> dict:
    """Generate a response appropriate to the classification."""
    prompt = RESPOND_QUALIFIED if state["classification"] == "qualified" else RESPOND_UNQUALIFIED
    profile = json.dumps(state["extracted"], indent=2)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Candidate profile:\n{profile}"),
    ]
    result = llm_creative.invoke(messages)
    return {
        "response": result.content,
        "steps": state["steps"] + ["generated response"],
    }


# ─── Build the decomposed pipeline ───────────────────────────────────────

def build_pipeline():
    builder = StateGraph(ApplicationState)
    builder.add_node("extract", extract)
    builder.add_node("classify", classify)
    builder.add_node("respond", respond)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "classify")
    builder.add_edge("classify", "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=MemorySaver())


# ─── Comparison: single mega-prompt ──────────────────────────────────────

MEGA_PROMPT = """Read the following job application email. Do ALL of the following:
1. Extract the applicant's name, current role, years of experience, and key skills
2. Classify them as "qualified" or "unqualified" for a Senior Python Developer role
   (requires 5+ years experience, Python skills, and leadership experience)
3. Write a brief response letter appropriate to their qualification status

Format your response as:
NAME: ...
ROLE: ...
EXPERIENCE: ...
SKILLS: ...
CLASSIFICATION: ...
RESPONSE: ..."""


def run_mega_prompt(email: str) -> str:
    """Process an application with a single do-everything prompt."""
    messages = [
        SystemMessage(content=MEGA_PROMPT),
        HumanMessage(content=email),
    ]
    result = llm_precise.invoke(messages)
    return result.content


# ─── Run ──────────────────────────────────────────────────────────────────

def main():
    pipeline = build_pipeline()

    print("=" * 64)
    print("  Module 02: Prompt Engineering in Agentic Settings")
    print("  Decomposed Pipeline vs Mega-Prompt")
    print("=" * 64)

    for i, email in enumerate(APPLICATION_EMAILS, 1):
        # Extract first line for display
        first_line = email.strip().split("\n")[0]

        print(f"\n{'━' * 64}")
        print(f"  Application {i}: {first_line}")
        print(f"{'━' * 64}")

        # ── Decomposed pipeline ──
        print(f"\n  ▸ DECOMPOSED PIPELINE (3 nodes, each with narrow prompt)")
        config = {"configurable": {"thread_id": f"app-{i}"}}
        result = pipeline.invoke({"email": email, "steps": []}, config)

        print(f"    Extracted: {json.dumps(result['extracted'], indent=6)}")
        print(f"    Classification: {result['classification']}")
        print(f"    Response: {result['response'][:200]}...")
        print(f"    Steps: {result['steps']}")

        # ── Single mega-prompt ──
        print(f"\n  ▸ MEGA-PROMPT (single prompt does everything)")
        mega_result = run_mega_prompt(email)
        print(f"    Raw output: {mega_result[:300]}...")

    print(f"\n{'=' * 64}")
    print("  Key observations:")
    print("  • Decomposed: each node has ONE job → reliable, testable")
    print("  • Mega-prompt: tries everything → fragile, hard to debug")
    print("  • Extract node uses temp=0 (deterministic)")
    print("  • Respond node uses temp=0.3 (natural variation)")
    print("  • Each prompt is a behavioural CONTRACT, not a suggestion")
    print("=" * 64)


if __name__ == "__main__":
    main()
