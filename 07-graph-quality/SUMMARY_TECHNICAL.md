# Knowledge Graph Quality Assessment — Technical Derivation

> **Audience:** Data scientists, ML engineers, knowledge engineers.
> Theory, scoring derivations, and architecture — no line-by-line code
> walkthrough.

---

## 1. Problem Formulation

Let $G = (V, E, L, P)$ be a labelled property graph where $V$ is the set of
nodes, $E \subseteq V \times R \times V$ is the set of directed typed edges
with relation types $R$, $L: V \to 2^{\Lambda}$ assigns labels from a label
set $\Lambda$, and $P: V \cup E \to \text{Map}(\text{key}, \text{value})$
assigns properties.

Given a source document $D$ from which $G$ was extracted, the quality
assessment problem is to compute a quality vector:

$$
\mathbf{q}(G, D) = \langle q_1, q_2, \ldots, q_k \rangle \quad q_i \in [0, 1]
$$

where each $q_i$ measures a distinct quality dimension, and an overall score:

$$
Q(G, D) = \frac{\sum_{i \in \mathcal{A}} w_i \cdot q_i}{\sum_{i \in \mathcal{A}} w_i}
$$

where $\mathcal{A}$ is the set of *active* dimensions (those whose probes
actually ran) and $w_i$ are dimension-specific weights. Dynamic normalisation
ensures that skipped phases don't penalise the score.

---

## 2. Quality Dimensions — Theoretical Basis

The dimensions are grounded in the **ISO 25012** data quality model and
adapted to the knowledge graph context. The 11 dimensions decompose across
three evaluation paradigms:

### 2.1 Structural Dimensions (Graph-Theoretic)

**Schema Population ($q_{\text{schema}}$)**

Let $\Lambda_{\text{expected}}$ be the set of node labels defined by the
domain ontology, $\Lambda_{\text{populated}} = \{ \lambda \in \Lambda_{\text{expected}} : |\{v \in V : \lambda \in L(v)\}| > 0 \}$.

$$
q_{\text{schema}} = \max\!\left(0,\;\frac{|\Lambda_{\text{populated}}|}{|\Lambda_{\text{expected}}|} - 0.05 \cdot n_{\text{domain}}\right)
$$

where $n_{\text{domain}}$ is the number of domain-specific rule violations
(e.g., incident events with fewer than 2 participants).

**Structural Connectivity ($q_{\text{structural}}$)**

Let $V_{\text{isolated}} = \{v \in V : \deg(v) = 0\}$ and $\kappa$ be the
number of weakly connected components of $G$.

$$
q_{\text{structural}} = \text{clamp}\!\left(\frac{|V| - |V_{\text{isolated}}|}{|V|} - 0.1 \cdot \max(0, \kappa - 1),\; 0,\; 1\right)
$$

The first term penalises isolated nodes; the second penalises graph
fragmentation.

**Consistency ($q_{\text{consistency}}$)**

Four binary sub-checks $c_j \in \{0, 1\}$:

| Sub-check | Condition for $c_j = 1$ |
|-----------|------------------------|
| Temporal acyclicity | No directed cycles in the `PRECEDED` / `PRECEDES` subgraph |
| Time monotonicity | For every $(e_1, \texttt{PRECEDED}, e_2) \in E$: $\text{time}(e_1) \leq \text{time}(e_2)$ |
| Role exclusivity | No entity is simultaneously a witness and suspect of the same event |
| No duplicates | No two nodes share the same label and description |

$$
q_{\text{consistency}} = \frac{\sum_j c_j}{4}
$$

**Source Grounding ($q_{\text{constraint}}$)**

Let $V_{\text{content}} = V \setminus V_{\text{Observation}}$ (exclude
meta-observation nodes) and $V_{\text{orphan}} = \{v \in V_{\text{content}} :
P(v)[\texttt{source}] = \texttt{null}\}$.

$$
q_{\text{constraint}} = 1 - \frac{|V_{\text{orphan}}|}{|V_{\text{content}}|}
$$

### 2.2 Semantic Dimensions (LLM-as-Judge)

These dimensions use the **G-Eval** framework (Liu et al., NeurIPS 2023),
which formulates evaluation as a structured reasoning task for a language
model. The key insight is that LLMs can serve as proxies for human quality
judgements when given well-specified evaluation criteria.

The system supports two backends:

| Backend | Method | Advantage |
|---------|--------|-----------|
| **Native LLM** | Direct prompt → JSON response with score + issues | Simpler; fewer dependencies |
| **DeepEval G-Eval** | Structured evaluation with chain-of-thought, normalised scoring | More reproducible; includes reasoning traces |

**Graph Linearisation**

Before LLM evaluation, $G$ is linearised into a textual triple
representation:

$$
\text{linearise}(G) = \bigoplus_{(u, r, v) \in E} \text{"(} L(u)\text{:} d(u) \text{) -[} r \text{]-> (} L(v)\text{:} d(v) \text{)"}
$$

where $d(v) = P(v)[\texttt{description}] \oplus P(v)[\texttt{name}] \oplus
P(v)[\texttt{value}]$ (first non-null).

**Coherence ($q_{\text{coherence}}$)**

The LLM evaluates whether $\text{linearise}(G)$ forms a coherent narrative
across five criteria: logical sequencing, participant consistency, spatial
plausibility, causal completeness, and narrative reconstructability.

- Native mode: score ∈ {0, …, 10}, normalised to [0, 1]
- G-Eval mode: `NarrativeCoherence` metric with threshold 0.6

**Faithfulness ($q_{\text{faithfulness}}$)**

Given source document $D$, the LLM compares $\text{linearise}(G)$ against
$D$ and identifies:
- **Hallucinations** (severity: error): facts in $G$ not in $D$
- **Distortions** (severity: warning): facts changed from $D$
- **Omissions** (severity: info): facts in $D$ not in $G$

This is the most heavily weighted dimension ($w = 0.20$) because
hallucinated facts are the most dangerous failure mode in investigative
knowledge graphs.

**Semantic Completeness ($q_{\text{sem\_complete}}$)**

The complement of faithfulness — measures recall rather than precision. Uses
G-Eval to evaluate whether all important facts from $D$ appear in $G$.

**Investigative Readiness ($q_{\text{invest\_ready}}$)**

A domain-specific G-Eval metric that evaluates $G$ from a practitioner's
perspective: can an investigator use this graph to reconstruct events,
identify suspects, establish a timeline, and spot gaps?

### 2.3 Embedding Dimensions (Representation Learning)

These dimensions train a KG embedding model $f: V \cup R \to \mathbb{R}^d$
and use the learned representations to detect structural anomalies invisible
to graph queries.

**Model Choice: ComplEx**

The system uses the **RotatE** model (Sun et al., ICLR 2019), which
embeds entities and relations via rotations in complex space:

$$
t = h \circ r \quad \text{where} \quad r_k = e^{i\theta_k}
$$

Each relation is modelled as a rotation from head to tail in $\mathbb{C}^d$.
The scoring function is:

$$
\phi(h, r, t) = -\|h \circ r - t\|_2
$$

RotatE is chosen over ComplEx because:
- It handles **asymmetric** relations (PRECEDED, WITNESSED) via rotation
- It can model **composition** patterns (if A precedes B and B precedes C, then A precedes C)
- It is lightweight ($d=50$ gives good results on small graphs)
- It is well-supported by PyKEEN

ComplEx remains available as a fallback by changing `DEFAULT_MODEL` in
`embedding_probes.py`.

Training hyperparameters: $d=50$, 100 epochs, batch size =
$\min(64, |E|)$, seed 42 for reproducibility.

**Link Prediction ($q_{\text{link\_pred}}$)**

For a sample of $(h, r)$ pairs, predict the most plausible tail entities
$\hat{t} = \arg\max_t \phi(h, r, t)$. Predictions not in $E$ are candidate
missing links.

$$
q_{\text{link\_pred}} = \max\!\left(0,\; 1 - \frac{|\text{predicted missing}|}{|E|}\right)
$$

A high ratio of predicted missing links suggests structural incompleteness.

**Triple Plausibility ($q_{\text{plausibility}}$)**

Score every existing triple and flag those below a sigmoid-normalised
threshold $\tau = 0.3$:

$$
q_{\text{plausibility}} = \frac{|\{(h,r,t) \in E : \sigma(\phi(h,r,t)) \geq \tau\}|}{|E|}
$$

Low-scoring triples may be extraction errors or hallucinations — they
contradict the structural patterns learned from the rest of the graph.

**Entity Clustering ($q_{\text{cluster}}$)**

Group entities by label, compute per-group centroids $\mu_\lambda =
\frac{1}{|V_\lambda|}\sum_{v \in V_\lambda} f(v)$, flag outliers where
$\|f(v) - \mu_\lambda\| > \mu_d + 2\sigma_d$, and detect potential
duplicates where $\|f(v_i) - f(v_j)\| < 0.1$ for same-label pairs.

$$
q_{\text{cluster}} = \text{clamp}\!\left(1 - 0.15 \cdot n_{\text{outliers}},\; 0,\; 1\right)
$$

---

## 3. Weight Architecture

The weight system uses three tiers with dynamic normalisation:

| Tier | Dimensions | Raw weight sum |
|------|-----------|---------------|
| Phase 1 (structural + LLM) | schema, structural, constraint, consistency, coherence, faithfulness | 1.00 |
| Phase 2 (DeepEval extras) | semantic completeness, investigative readiness | 0.20 |
| Phase 3 (embeddings) | link prediction, triple plausibility, entity clustering | 0.20 |

When all three phases run, the raw sum is 1.40. Each weight is normalised:

$$
\hat{w}_i = \frac{w_i}{\sum_{j \in \mathcal{A}} w_j}
$$

This ensures the overall score always lies in $[0, 1]$ regardless of which
phases are active. When only Phase 1 runs, the weights already sum to 1.00
and are used directly.

The heaviest weights are consistency (0.20) and faithfulness (0.20),
reflecting the principle that **logical contradictions and hallucinations are
the most dangerous failure modes** in investigative KGs.

---

## 4. Evaluation of State-of-the-Art Alignment

### What aligns well

| Aspect | State of the art | This system |
|--------|-----------------|-------------|
| Multi-dimensional quality | ISO 25012, ISO 8000, GQM (Basili) | 11 dimensions across 3 paradigms |
| LLM-as-judge | G-Eval (Liu et al. 2023), MT-Bench (Zheng et al. 2023) | DeepEval G-Eval with local Ollama model |
| Constraint validation | W3C SHACL (2017) | pySHACL with custom Turtle shapes |
| KG embeddings | ComplEx (2016), RotatE (2019), TuckER (2019) | ComplEx via PyKEEN |
| Graceful degradation | — (not commonly addressed) | Modular phases with optional dependencies |
| Provenance tracking | PROV-O (W3C), FAIR principles | Source grounding probe checks `source` property |

### Recent advances not yet incorporated

| Technique | Paper/System | Potential value | Difficulty |
|-----------|-------------|-----------------|------------|
| **RotatE** (Sun et al. 2019) | Rotation-based embeddings | Already the default model | — (done) |
| **KG-BERT** (Yao et al. 2019) / **SimKGC** (Wang et al. 2022) | Text-aware KG completion | Uses entity descriptions for scoring, not just structure | Medium — requires transformer fine-tuning |
| **GNN-based quality** (R-GCN, CompGCN) | Graph neural networks for quality scoring | Learns quality signals from graph topology directly | High — requires training data of good/bad graphs |
| **LLM calibration** (e.g., verbalized confidence) | Tian et al. 2023 | Calibrated confidence intervals on LLM scores | — (done: `calibrate_llm_probe()`) |
| **Cross-source consistency** | Knowledge fusion literature | Detect contradictions between multiple witness statements about the same event | Medium — requires alignment step before comparison |
| **Ontology alignment metrics** | OntoClean, OQuaRE | Formal ontological quality measures | Medium — requires richer ontological metadata |

### Practical recommendations for future work

1. **Add structured output** — replace raw JSON prompts with Pydantic-
   constrained output (via `langchain` structured output) for more reliable
   LLM responses
2. **Cross-source consistency** — when Chapter 08 fuses multiple statements,
   add a probe that detects contradictions between sources
3. **GNN-based quality detection** — train a graph neural network to predict
   quality scores from topology directly

---

## 5. Comparison with Alternative Approaches

| Approach | Pros | Cons | When to use |
|----------|------|------|-------------|
| **This system** (multi-paradigm) | Catches structural, semantic, and latent issues | Requires LLM + optional GPU for embeddings | Production KG quality gates |
| **SHACL-only** | Standards-based, deterministic, fast | Only catches schema/constraint violations; no semantic checks | Data governance compliance |
| **LLM-only** (e.g., pure G-Eval) | Catches semantic issues humans would notice | Non-deterministic; misses structural issues; expensive at scale | Ad-hoc quality spot-checks |
| **Embedding-only** (e.g., AmpliGraph) | Good at finding missing links | Requires >50 triples; doesn't assess faithfulness | Large KGs, link prediction |
| **Unit testing** (Cypher assertions) | Fast, precise, deterministic | Brittle; needs manual test writing; no semantic coverage | CI/CD pipelines |

The value of this system is the **combination**: structural probes catch the
obvious issues cheaply, LLM probes catch meaning-level failures, and
embedding probes find hidden patterns. No single approach covers all three.

---

## 6. Known Limitations

1. **LLM scoring is non-deterministic.** Running the same coherence probe
   twice may give different scores. Temperature is set to 0 to minimise
   variance, but it doesn't eliminate it. Use `--calibrate` to quantify
   scoring variance across multiple runs.

2. **Small-graph bias in embeddings.** ComplEx needs ≥10 triples for link
   prediction and ≥15 for clustering. On very small graphs, embedding
   scores are either skipped or unreliable.

3. **No ground-truth calibration.** The scoring thresholds (0.6 for
   coherence, 0.8 for faithfulness) are heuristic. The `calibrate_llm_probe()`
   function quantifies variance but does not calibrate against human
   judgements on a held-out test set.

4. **RDF export is lossy.** The Neo4j → RDF conversion for SHACL validation
   uses a simplified mapping (all properties become string literals). Rich
   property types (lists, nested maps) are flattened.

5. **Single-statement scope.** The system evaluates one graph in isolation.
   When multiple witness statements are fused, cross-source consistency
   metrics are needed but not yet implemented.
