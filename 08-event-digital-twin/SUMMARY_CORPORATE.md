# Event Digital Twins — A Corporate Deep Dive

## What Is a Digital Twin?

A digital twin is a live digital replica of something real — a jet engine, a factory floor, a city's traffic system — that stays continuously synchronised with its physical counterpart through data feeds. The concept was formalised by Michael Grieves at the University of Michigan in 2002 and named by NASA engineer John Vickers in 2010. NASA had been doing something similar since the 1960s, using ground-based simulators to mirror the state of spacecraft in flight; when Apollo 13's oxygen tank exploded in 1970, it was a ground-based replica that let engineers work out how to bring the crew home.

Every digital twin has three parts:

1. **The physical system** — the real thing being modelled
2. **The digital representation** — a structured model that mirrors the physical system's state
3. **The communication channel** (the "digital thread") — the data flow that keeps the two synchronised

The key word is *live*. A CAD drawing of a bridge is a model. A simulation of wind loads on that bridge is a simulation. But a digital twin of that bridge receives live sensor data from strain gauges and accelerometers, updates its model in real time, predicts when maintenance is needed, and can send control signals back (e.g. adjusting dampers). That bidirectional, continuous feedback loop is what distinguishes a twin from a model.

The digital twin market is projected to grow from $24.5 billion in 2025 to $259.3 billion by 2032 (Fortune Business Insights). 92% of companies deploying digital twins report returns above 10%.

## What Is an *Event* Digital Twin?

Traditional digital twins model physical objects — turbines, vehicles, buildings. This chapter introduces a different application: modelling **something that happened**.

An event digital twin reconstructs an incident — a collision, a crime, a disaster — from multiple evidence sources into a structured, queryable, evolving digital model.

| Traditional Digital Twin | Event Digital Twin |
|---|---|
| Sensors on a turbine | Witness statements, CCTV, forensic reports |
| Real-time telemetry | Batch evidence ingestion + investigative interview |
| Predicts mechanical failure | Reconstructs what happened, detects gaps |
| Models one physical object | Models one event, many perspectives |
| Continuous sensor stream | Discrete evidence documents |

The fundamental challenge is different: instead of continuous numerical data from reliable sensors, an event digital twin must combine **incomplete, subjective, sometimes contradictory human accounts** into a single coherent model — while tracking exactly which source said what and how confident we are in each claim.

## Why Knowledge Graphs?

An event doesn't have a single "correct" reading the way a turbine's RPM does. It involves:

- **Multiple actors** — a driver, a cyclist, witnesses, paramedics, each with different roles
- **Temporal sequences** — events happened in order, but different witnesses may report that order differently
- **Spatial relationships** — "at the junction of King Street and Queen's Road"
- **Provenance chains** — who said what, when, and how reliable are they?

A knowledge graph captures all of these naturally as **nodes** (entities: people, events, locations, times) connected by **edges** (relationships: participated in, occurred at, preceded by). Every node carries metadata: which source reported it, when it was extracted, and how confident the system is in that fact.

This structure enables formal reasoning: if Event A preceded Event B, and Event B preceded Event C, then Event A preceded Event C. If two sources agree on a fact, confidence increases. If they disagree, both claims are preserved and the contradiction is flagged.

## The Investigation Metaphor

The system models the pipeline a skilled investigator follows:

### 1. Collect
Gather all available evidence: witness statements, CCTV footage logs, paramedic reports, police reports. Each is ingested as a separate source.

### 2. Extract
An AI model (running locally, not in the cloud) reads each source and identifies who, what, when, and where — extracting entities (people, vehicles, locations, times, events) and relationships between them.

### 3. Corroborate
When the same fact appears in multiple sources, it's flagged as **corroborated** and its confidence score increases. This is the opposite of hallucination — it's evidence triangulation.

### 4. Detect Contradictions
When two sources disagree — Witness 1 says the collision happened at 2:15 PM, the CCTV log says 14:13:45 — both facts are preserved with their provenance, and the contradiction is explicitly recorded. The system does not silently pick one.

### 5. Build a Timeline
Events are placed in temporal order based on extracted timestamps and relative temporal markers ("then", "after", "before"). Temporal inconsistencies are detected automatically.

### 6. Fill Gaps
The system analyses the knowledge graph for missing information — events without timestamps, people without descriptions, vehicles without registration plates — and either resolves them from existing sources or generates targeted follow-up questions for investigators.

### 7. Assess Quality
How complete is the reconstruction? How coherent? How consistent across sources? The system produces a multi-dimensional quality score.

### 8. Query
Investigators can ask questions in plain language. Answers are grounded exclusively in the evidence graph with explicit citations — the system will never make up facts.

### 9. Simulate
"What if Witness 2 is unreliable? What changes?" The system can simulate removing a source and show which facts would be lost. This is what makes it a true *twin* — the ability to explore alternative scenarios.

## Business Value

| Capability | Value |
|---|---|
| **Multi-source fusion** | Single unified view from many accounts — no information silos |
| **Provenance tracking** | Every fact traceable to its exact source — audit trail |
| **Contradiction detection** | Conflicting accounts surfaced automatically — saves investigator time |
| **Gap analysis** | Systematic identification of what's still unknown — directs investigation |
| **Confidence scoring** | Quantitative measure of evidence strength — supports decision making |
| **Grounded Q&A** | Answers backed by evidence, not AI guesswork — trustworthy |
| **What-if simulation** | Test hypotheses without modifying the evidence — safe exploration |
| **Export & interoperability** | RDF/JSON-LD output for standards compliance — systems integration |

## Where Industry Is Going

- **Azure Digital Twins** (Microsoft) uses a graph-based modelling language (DTDL) with live execution environments and IoT integration
- **IBM Maximo** delivers AI-enabled asset management with digital twin technology
- **NVIDIA Omniverse** provides physics-based simulation for digital twin environments
- The **US Department of Defense** defines a digital twin as "an integrated multiphysics, multiscale, probabilistic simulation of an as-built system, enabled by a Digital Thread"

This chapter extends the digital twin paradigm from physical assets into investigative and forensic domains — an emerging application area with significant potential for law enforcement, insurance, emergency response, and judicial proceedings.

## Privacy and Ethics

Event digital twins raise important considerations:

- **Witness privacy** — statements may contain personally identifiable information
- **Bias in extraction** — AI models may interpret statements differently depending on language and cultural context
- **Confidence misuse** — numerical confidence scores should inform, not replace, human judgment
- **Chain of custody** — for forensic applications, the provenance chain must be tamper-evident

The system addresses these through explicit provenance tracking (every fact is attributed to its source), separation of extraction confidence from ground truth, and local model execution (no data leaves the organisation's infrastructure).
