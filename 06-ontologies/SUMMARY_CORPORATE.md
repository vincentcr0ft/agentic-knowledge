# Ontologies for AI: A Business Perspective

## What Are Ontologies and Why Should You Care?

An **ontology** is a shared, structured vocabulary that defines the concepts in a domain and the relationships between them. Think of it as a **universal data dictionary** — but one that also captures *how things relate to each other*, not just what they are.

When your organisation extracts knowledge from documents, reports, interviews, or any unstructured text using AI, the results are only as good as the structure you impose on them. Ontologies provide that structure. Without one, every team, every system, and every AI model invents its own way of representing the same information — leading to silos, inconsistencies, and missed connections.

## The Business Problem

Consider a practical scenario: your organisation needs to reconstruct events from multiple witness statements, incident reports, or intelligence feeds. The AI reads the text and extracts *people*, *places*, *times*, and *events*. But:

- **Who** is the same person mentioned under different descriptions?
- **What role** did they play — witness, suspect, first responder?
- **How** do events relate — did one cause another?
- **Where** are the gaps — what should we know but don't?

The ontology you choose determines which of these questions your system can answer.

## Three Ontology Approaches Compared

| Approach | Analogy | Best For | Limitation |
|----------|---------|----------|------------|
| **Schema.org Event** | A simple spreadsheet | Rapid prototyping, search-engine-friendly apps, lightweight integrations | People can only have one role; no sub-events; shallow structure |
| **SEM (Simple Event Model)** | A relational database with proper normalisation | Multi-source investigation, media analysis, historical research | Less commercial tooling; smaller ecosystem |
| **BFO / CCO** | An ISO-certified engineering specification | Defence, intelligence, forensics, formal compliance, cross-agency interoperability | Heavier; requires specialist knowledge; harder for AI to extract cleanly |

## Why This Matters Now

### 1. AI Is Only as Smart as Its Schema
Large Language Models (LLMs) like GPT-4 or open-source alternatives can *read* text, but they need to be told *what to look for*. The ontology serves as that instruction set. A better ontology means more complete, more accurate extraction — and fewer blind spots.

### 2. Government and Defence Are Setting the Standard
In **January 2024**, the US Department of Defense and Intelligence Community formally adopted **BFO + CCO** (Basic Formal Ontology + Common Core Ontologies) as their baseline standard for all formal ontology work. This ISO-standard framework (ISO/IEC 21838-2:2021) is now the expected interoperability layer for defence and intelligence systems worldwide. Organisations working in or adjacent to these sectors will increasingly need to align.

### 3. Interoperability Is the Real ROI
The primary value of a well-chosen ontology is **interoperability** — different systems, teams, and even different AI models can share, compare, and merge their knowledge graphs because they speak the same structural language. This eliminates costly manual data reconciliation and enables cross-domain analysis.

### 4. Quality Assurance Becomes Automated
With ontologies, you can express *completeness rules* — for example, "every incident must have a location and at least one participant." These rules run automatically against extracted data, flagging gaps for human review. This turns knowledge extraction from a one-shot process into a **continuous quality loop**.

## What This Module Demonstrates

Our implementation provides a **pluggable ontology framework** — a single extraction pipeline that works with any of the three ontologies above. The same source text is processed through each, and the resulting knowledge graphs are compared side-by-side. This approach:

- **De-risks ontology choice** — you can evaluate trade-offs before committing
- **Enables migration** — switch ontologies as requirements evolve
- **Supports hybrid strategies** — use Schema.org for rapid intake, BFO/CCO for formal analysis

## Key Takeaway

Ontology selection is not a technical afterthought — it is a **strategic capability decision**. The right ontology determines what your AI systems can discover, how well they integrate with partners, and whether their outputs meet formal standards. As AI-driven knowledge extraction becomes central to decision-making, the organisations that invest in principled ontology design will have a significant structural advantage.

---

*Further reading: Smith et al., "Building Ontologies with Basic Formal Ontology" (MIT Press, 2015); DOD/IC Memorandum on BFO+CCO Adoption (February 2024); van Hage et al., "Design and use of the Simple Event Model" (Web Semantics, 2011).*
