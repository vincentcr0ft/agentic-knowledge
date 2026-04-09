"""
Module 02 — Agent Run Record
═════════════════════════════
Runs the prompt-engineering pipeline and prints a human-readable
record of every step: what the agent was given, what it did, and why.
"""

import json

from demo import build_pipeline, APPLICATION_EMAILS as SAMPLE_EMAILS


def _narrate_run(pipeline, email: str, thread_id: str, label: str):
    """Run one email and print a step-by-step narrative."""
    config = {"configurable": {"thread_id": thread_id}}
    result = pipeline.invoke({"email": email, "steps": []}, config)

    first_line = email.strip().split("\n")[0]

    print(f"\n{'═' * 70}")
    print(f"  AGENT RUN RECORD — {label}")
    print(f"  Email: \"{first_line}…\"")
    print(f"{'═' * 70}")

    # Collect checkpoints chronologically
    history = list(reversed(list(pipeline.get_state_history(config))))

    step_num = 0
    prev_values = {}

    for cp in history:
        vals = cp.values
        steps = vals.get("steps", [])
        if not steps and not vals.get("extracted"):
            print(f"\n  ── Input ──────────────────────────────────────────────────────")
            print(f"  The pipeline received a {len(email)}-character email and an")
            print(f"  empty audit trail. The email will flow through three nodes:")
            print(f"  extract → classify → respond.")
            prev_values = dict(vals)
            continue

        latest_step = steps[-1] if steps else None
        if latest_step and latest_step == (prev_values.get("steps", []) or [""])[-1]:
            prev_values = dict(vals)
            continue

        step_num += 1

        # ── extract ───────────────────────────────────────────────
        if vals.get("extracted") and not prev_values.get("extracted"):
            ext = vals["extracted"]
            print(f"\n  ── Step {step_num}: Extract ──────────────────────────────────────────")
            print(f"  Given:   The raw email text ({len(email)} chars).")
            print(f"  Action:  Sent the email to the LLM (temperature=0) with a")
            print(f"           constrained system prompt that demands a JSON object")
            print(f"           with exactly five fields: name, current_role,")
            print(f"           years_experience, key_skills, notable_achievements.")
            print(f"           The prompt forbids inference — extract only explicit facts.")
            print(f"  Result:")
            print(f"           name:          {ext.get('name', '?')}")
            print(f"           current_role:  {ext.get('current_role', '?')}")
            print(f"           years_exp:     {ext.get('years_experience', '?')}")
            print(f"           key_skills:    {ext.get('key_skills', '?')}")
            print(f"           achievements:  {str(ext.get('notable_achievements', '?'))[:80]}")
            print(f"  Why:     Extraction is the first node because downstream nodes")
            print(f"           (classify, respond) operate on structured data, not")
            print(f"           raw text. This decomposition keeps each prompt narrow")
            print(f"           and testable — one job per node.")

        # ── classify ──────────────────────────────────────────────
        elif vals.get("classification") and not prev_values.get("classification"):
            ext = vals.get("extracted", {})
            print(f"\n  ── Step {step_num}: Classify ────────────────────────────────────────")
            print(f"  Given:   The structured profile extracted in Step 1:")
            print(f"           {ext.get('name', '?')}, {ext.get('years_experience', '?')} years,")
            print(f"           skills: {ext.get('key_skills', '?')}")
            print(f"  Action:  Sent the profile JSON to the LLM (temperature=0) with a")
            print(f"           prompt defining 'qualified' as: 5+ years experience,")
            print(f"           Python skills, and leadership experience. The LLM must")
            print(f"           respond with exactly one word.")
            print(f"  Result:  classification = \"{vals['classification']}\"")
            yrs = ext.get("years_experience", 0)
            skills = ext.get("key_skills", [])
            has_python = any("python" in str(s).lower() for s in skills) if skills else False
            print(f"  Why:     The candidate has {yrs} years of experience", end="")
            if has_python:
                print(f" and Python skills.", end="")
            else:
                print(f" but Python is {'not clearly listed' if not has_python else 'listed'}.", end="")
            if yrs >= 5 and has_python:
                print(f"\n           This meets the threshold → qualified.")
            elif yrs < 5:
                print(f"\n           This is below the 5-year threshold → unqualified.")
            else:
                print(f"\n           Missing key requirements → unqualified.")

        # ── respond ───────────────────────────────────────────────
        elif vals.get("response") and not prev_values.get("response"):
            cls = vals.get("classification", "?")
            print(f"\n  ── Step {step_num}: Respond ─────────────────────────────────────────")
            print(f"  Given:   The classification (\"{cls}\") and the structured profile.")
            if cls == "qualified":
                print(f"  Action:  Used the RESPOND_QUALIFIED prompt (temperature=0.3).")
                print(f"           The prompt asks for a warm 3-4 sentence reply that")
                print(f"           thanks the candidate, mentions something specific")
                print(f"           from their background, and offers next steps.")
            else:
                print(f"  Action:  Used the RESPOND_UNQUALIFIED prompt (temperature=0.3).")
                print(f"           The prompt asks for a kind 2-3 sentence reply that")
                print(f"           thanks them but explains more experience is needed.")
            resp = vals["response"]
            print(f"  Result:  \"{resp[:120]}{'…' if len(resp) > 120 else ''}\"")
            print(f"  Why:     The classification determines which response prompt is")
            print(f"           used. Temperature is raised to 0.3 here (vs 0 for")
            print(f"           extract/classify) to allow natural language variation")
            print(f"           while keeping the tone controlled.")
            print(f"           This is the final node — the pipeline ends here.")

        prev_values = dict(vals)

    # ── Audit trail ───────────────────────────────────────────────
    print(f"\n  ── Audit Trail ───────────────────────────────────────────────")
    for i, step in enumerate(result.get("steps", []), 1):
        print(f"    {i}. {step}")
    print()


def main():
    pipeline = build_pipeline()

    print("=" * 70)
    print("  Module 02: Prompt Engineering — Agent Run Records")
    print("  A readable log of what the agent was given, did, and why.")
    print("=" * 70)

    labels = ["Sarah Chen (experienced)", "Jake Martinez (new grad)", "Dr. Priya Sharma (senior)"]
    for i, (email, label) in enumerate(zip(SAMPLE_EMAILS, labels), 1):
        _narrate_run(pipeline, email, f"run-{i}", label)


if __name__ == "__main__":
    main()
