# Fine-Tuning Research Plan for Agentic AI & Knowledge Graphs

## 1. Assessment: Where Does Fine-Tuning Belong?

### The Disagreement

Two valid placements have been suggested:

| Placement | Rationale | Verdict |
|-----------|-----------|---------|
| **Folder 08** (Event Digital Twin) | Fine-tuning improves the *extraction* and *reasoning* models used mid-pipeline — it's an enhancement of the existing KG construction workflow | Domain-specific fine-tuning as a pipeline improvement |
| **Folder 09** (Deployment) | Fine-tuning produces *artefacts* (LoRA adapters, GGUF files) that must be served, versioned, and deployed — it's an operational concern | Adapter serving and model lifecycle |

### Recommendation: Both — Split Across Two Modules

**Fine-tuning spans both modules**, but with a clear separation of concerns:

| Module | Scope | Files |
|--------|-------|-------|
| **08-event-digital-twin** | *Why* and *how* to fine-tune: data generation, training scripts, evaluation against the existing pipeline, A/B comparison with the base model | `finetune_data.py`, `finetune_train.py`, `finetune_eval.py` |
| **09-deployment** | *Serving* fine-tuned models: GGUF quantisation, Ollama Modelfiles, adapter swapping, CI/CD for model artefacts | `modelfile_builder.py`, `adapter_serve.py`, deployment docs |

This mirrors the existing pattern: Module 06 defines ontologies, Module 07 tests them, Module 08 orchestrates them, Module 09 deploys them.

---

## 2. Why Fine-Tune? — Motivation in This Project

The current system uses `qwen2.5:7b` via Ollama with zero-shot/few-shot prompts for:

1. **Entity & relationship extraction** from witness statements (ingest.py)
2. **Coreference resolution** across extracted entities
3. **Interview question generation** for gap-filling
4. **Grounded Q&A** over the completed graph
5. **Quality assessment** (faithfulness, coherence probes)

### Current Limitations Addressable by Fine-Tuning

| Problem | Current Behaviour | Fine-Tuning Solution |
|---------|-------------------|----------------------|
| **Schema drift** | LLM invents labels outside the ontology despite schema-guided prompting | SFT on ontology-conformant examples locks the output vocabulary |
| **Extraction recall** | Implicit entities (locations, times, causal links) are missed | Domain-specific SFT improves recall on forensic/incident text |
| **JSON format failures** | LLM sometimes wraps JSON in markdown fences or adds commentary | SFT on clean JSON I/O eliminates format errors |
| **Coreference errors** | "the driver" ≠ "Mr Smith" resolution is inconsistent | SFT on coreference chains from annotated examples |
| **Confidence calibration** | Numerical confidence scores have no grounding | DPO/RLHF on preference pairs where annotators rank extraction quality |
| **Hallucinated relationships** | Relationships not supported by source text appear | Faithfulness-tuned model via DPO with positive/negative extraction pairs |

### Academic Support

- Zhu et al. (2023) "LLMs for KG Construction and Reasoning" — GPT-4 excels at reasoning but fine-tuned smaller models outperform on extraction tasks
- Yang et al. (2023) CP-KGC — even quantised fine-tuned models (Qwen-7B-Chat-int4) enhance KG completion over zero-shot baselines
- Kim et al. (2023) KG-GPT — multi-step sentence segmentation → graph retrieval → inference pipeline benefits from task-specific tuning at each stage

---

## 3. Fine-Tuning Methods — Taxonomy for This Project

### 3.1 Supervised Fine-Tuning (SFT)

**What**: Train the model on (input, expected output) pairs using next-token prediction loss.

**Applicable tasks in this project**:
- Entity/relationship extraction: `statement → JSON graph`
- Coreference resolution: `extracted entities → resolved entities`
- Schema-conformant output formatting

**Implementation approach**:
```
Training data format (conversational):
{
  "messages": [
    {"role": "system", "content": "<ontology schema + extraction instructions>"},
    {"role": "user", "content": "<witness statement text>"},
    {"role": "assistant", "content": "<valid JSON with entities and relationships>"}
  ]
}
```

**Tools**: HuggingFace TRL `SFTTrainer` + PEFT `LoraConfig`

**Key papers**:
- Howard & Ruder (2018) — ULMFiT: pretrain → domain-adapt → task-tune
- Hu et al. (2021) — LoRA: Low-Rank Adaptation (arXiv:2106.09685)
- Dettmers et al. (2023) — QLoRA: 4-bit quantised LoRA (arXiv:2305.14314)

### 3.2 Direct Preference Optimisation (DPO)

**What**: Align model outputs with human preferences without training a separate reward model. Uses pairs of (preferred, rejected) outputs.

**Applicable tasks in this project**:
- Faithfulness: prefer extractions grounded in source over hallucinated ones
- Completeness: prefer extractions that capture all entities over partial ones
- Confidence calibration: prefer responses with well-calibrated confidence scores
- Query answer quality: prefer grounded answers over speculative ones

**Implementation approach**:
```
Training data format:
{
  "prompt": "<system + user message with statement>",
  "chosen": "<complete, faithful JSON extraction>",
  "rejected": "<extraction with hallucinated entity or missing relationship>"
}
```

**Tools**: HuggingFace TRL `DPOTrainer`

**Key paper**: Rafailov et al. (2023) — Direct Preference Optimization (arXiv:2305.18290)

### 3.3 Knowledge Graph Embedding Fine-Tuning (PyKEEN)

**What**: Train vector representations of entities and relationships for link prediction, triple classification, and entity alignment.

**Already available**: PyKEEN 1.11.1 is installed in the project environment.

**Applicable tasks in this project**:
- Link prediction: predict missing relationships in the graph
- Entity alignment: match `POSSIBLY_SAME_AS` candidates via embedding similarity
- Anomaly detection: flag triples with low plausibility scores
- Graph completion: suggest missing entities based on structural patterns

**Implementation approach**:
```python
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Export Neo4j triples → (head, relation, tail) TSV
triples = TriplesFactory.from_path("event_triples.tsv")

result = pipeline(
    training=triples,
    model="ComplEx",      # or TransE, DistMult, RotatE
    training_loop="sLCWA",
    epochs=100,
)
```

**Key models to evaluate**: TransE, ComplEx, DistMult, RotatE, TuckER

### 3.4 Embedding Model Fine-Tuning

**What**: Fine-tune the embedding model used for vector similarity search in the RAG pipeline.

**Applicable tasks**:
- Improve retrieval of relevant chunks for grounded Q&A
- Better entity matching across sources in fusion
- Semantic similarity for coreference candidates

**Tools**: `sentence-transformers` with `SentenceTransformerTrainer`, contrastive learning on domain-specific pairs

### 3.5 Chain-of-Instructions (CoI) Tuning

**What**: Fine-tune on compositional multi-step instructions where each step's output feeds the next.

**Applicable tasks**: The ingest pipeline is inherently compositional:
`segment → extract → resolve → load`

CoI tuning would teach the model to handle the full chain, improving coherence across pipeline stages.

**Key paper**: Hayati et al. (2024) — Chain-of-Instructions (arXiv:2402.11532, AAAI 2025)

---

## 4. Implementation Plan — Module 08 (Training & Evaluation)

### Phase 1: Synthetic Training Data Generation

**Goal**: Generate labelled training data from the existing pipeline output and expert corrections.

| Step | Method | Output |
|------|--------|--------|
| 1a | Run existing pipeline on test corpus, capture raw extractions | `raw_extractions.jsonl` |
| 1b | Human review: correct, add missing entities, remove hallucinations | `gold_extractions.jsonl` |
| 1c | Augment with LLM-generated paraphrases of witness statements | `augmented_statements.jsonl` |
| 1d | Generate DPO preference pairs: gold vs. raw (raw as rejected) | `dpo_pairs.jsonl` |
| 1e | Export Neo4j graph to PyKEEN triple format | `event_triples.tsv` |

**File**: `finetune_data.py`

```
Commands:
  python finetune_data.py --generate-sft     # Create SFT training set
  python finetune_data.py --generate-dpo     # Create DPO preference pairs
  python finetune_data.py --export-triples   # Export graph for PyKEEN
  python finetune_data.py --augment          # Generate augmented statements
```

### Phase 2: Model Training

**File**: `finetune_train.py`

| Experiment | Base Model | Method | Hardware | Expected Outcome |
|------------|------------|--------|----------|------------------|
| SFT-extraction | Qwen2.5-7B | QLoRA (4-bit, rank 16) | Single GPU, ~16GB VRAM | Improved entity/relationship extraction F1 |
| SFT-coreference | Qwen2.5-7B | QLoRA | Single GPU | Better coreference resolution accuracy |
| DPO-faithfulness | SFT-extraction checkpoint | DPO with LoRA | Single GPU | Reduced hallucination rate |
| KGE-completion | N/A | PyKEEN ComplEx | CPU or GPU | Link prediction MRR > 0.5 |
| Embedding-retrieval | all-MiniLM-L6-v2 | Contrastive (sentence-transformers) | CPU/GPU | Improved retrieval precision@5 |

**QLoRA Configuration** (for SFT and DPO):
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Training Configuration**:
```python
from trl import SFTConfig

training_args = SFTConfig(
    output_dir="./sft-extraction",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    max_length=2048,
    packing=True,
    assistant_only_loss=True,
)
```

### Phase 3: Evaluation Framework

**File**: `finetune_eval.py`

Evaluate fine-tuned vs. base model on multiple dimensions:

| Metric | How Measured | Target |
|--------|-------------|--------|
| **Extraction F1** | Compare extracted entities/relationships against gold annotations | F1 > 0.85 |
| **Schema conformance** | % of outputs with valid labels from ontology spec | > 99% |
| **JSON validity** | % of outputs that parse as valid JSON | 100% |
| **Hallucination rate** | % of extracted triples not grounded in source text | < 5% |
| **Coreference accuracy** | % of correctly resolved entity mentions | > 0.80 |
| **Link prediction MRR** | PyKEEN mean reciprocal rank on held-out triples | MRR > 0.50 |
| **Graph quality score** | Run existing quality framework (07-graph-quality) on fine-tuned output | Score > 0.85 |
| **Latency** | Token generation speed: base vs. LoRA-merged vs. quantised | < 2x overhead |

**A/B Comparison Protocol**:
```
1. Run base model pipeline on test corpus → Graph_A
2. Run fine-tuned model pipeline on same corpus → Graph_B
3. Compare: quality scores, entity counts, relationship counts, timeline accuracy
4. Human evaluation: blind comparison of 20 random extractions
```

---

## 5. Implementation Plan — Module 09 (Serving & Deployment)

### Phase 4: Model Export & Quantisation

**Converting LoRA → GGUF for Ollama**:

```
Step 1: Merge LoRA adapter into base model
  → merged HuggingFace checkpoint

Step 2: Convert to GGUF using llama.cpp
  python convert_hf_to_gguf.py ./merged-model --outtype q4_K_M

Step 3: Create Ollama Modelfile
  FROM ./merged-model-q4_K_M.gguf
  PARAMETER temperature 0
  PARAMETER num_ctx 4096
  SYSTEM "You are a knowledge graph extraction assistant..."

Step 4: Register with Ollama
  ollama create kg-extraction -f Modelfile
```

**Alternative: Unsloth export** (2x faster training, direct GGUF export):
```python
from unsloth import FastLanguageModel
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
```

### Phase 5: Adapter Swapping Architecture

Design for the project's multi-task pipeline, where different nodes need different specialisations:

```
Pipeline Node          → Adapter/Model
─────────────────────────────────────────────
parse_statement        → base model (no adapter)
extract_entities       → kg-extraction adapter (SFT)
resolve_coreferences   → kg-coreference adapter (SFT)
interview questions    → base model (higher temperature)
grounded Q&A           → kg-qa adapter (DPO-faithfulness)
quality assessment     → base model
```

**Implementation**: Modify `ingest.py` and `query.py` to accept a `model` parameter, defaulting to base but switchable:
```python
llm_extract = ChatOllama(model="kg-extraction:latest", temperature=0)
llm_query = ChatOllama(model="kg-qa:latest", temperature=0.1)
```

### Phase 6: CI/CD for Model Artefacts

| Artefact | Versioning | Storage | Trigger |
|----------|-----------|---------|---------|
| LoRA adapters | Git LFS / HuggingFace Hub | `models/adapters/` | Training script completion |
| GGUF files | Semantic version tags | `models/gguf/` | Export script completion |
| Ollama Modelfiles | Alongside GGUF | `models/` | Manual review |
| PyKEEN checkpoints | `trained_model.pkl` | `models/kge/` | KGE training completion |
| Evaluation reports | JSON + Markdown | `reports/` | Every training run |

---

## 6. Training Data Strategy — Deep Dive

### 6.1 Gold Standard Annotation

The project already has 4 test sources with intentional overlaps and conflicts. This provides the nucleus for training data:

| Source File | Entities | Relationships | Conflicts |
|------------|----------|--------------|-----------|
| `king_street_collision.txt` | Primary scene description | Core event structure | — |
| `queen_road_witness.txt` | Secondary witness perspective | Overlapping entities | Timing conflicts |
| `cctv_log.txt` | Objective timestamps | Precise temporal relationships | Resolution vs. witness accounts |
| `paramedic_report.txt` | Medical entities, outcomes | RESPONDED_TO, TREATED relationships | Clinical vs. lay descriptions |

**Annotation process**:
1. Run extraction pipeline, capture output
2. Expert reviews each extraction, marking: correct / missing / hallucinated / wrong type
3. Produce gold JSON for each source under each ontology (Schema.org, SEM, BFO/CCO)
4. This gives 4 × 3 = 12 training examples minimum, expandable by augmentation

### 6.2 Synthetic Data Generation

To reach the ~200-500 examples typically needed for effective SFT:

| Method | Description | Volume |
|--------|-------------|--------|
| **Paraphrase augmentation** | Rewrite existing statements in different styles (formal, colloquial, fragmented) | 3-5x multiplier |
| **Scenario generation** | Use GPT-4/Claude to generate new incident scenarios (traffic, workplace, environmental) | 50-100 new scenarios |
| **Ontology permutation** | Same statement, extract under different ontologies | 3x multiplier |
| **Error injection** | Deliberately introduce extraction errors for DPO rejected examples | 1:1 with correct examples |
| **Cross-domain transfer** | Adapt forensic extraction patterns from public NER datasets (OntoNotes, ACE) | 50-100 examples |

### 6.3 PyKEEN Training Data

Export from Neo4j after a full pipeline run:

```cypher
MATCH (a)-[r]->(b)
WHERE type(r) <> 'POSSIBLY_SAME_AS'
RETURN labels(a)[0] + ':' + coalesce(a.name, a.description) AS head,
       type(r) AS relation,
       labels(b)[0] + ':' + coalesce(b.name, b.description) AS tail
```

Split: 80% train / 10% validation / 10% test

---

## 7. Risk Analysis & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting on 4 sources** | Model memorises specific entities, fails on new incidents | Augmentation + held-out evaluation + regularisation |
| **Catastrophic forgetting** | Fine-tuned model loses general capabilities | LoRA preserves base weights; merge only at deployment |
| **Ontology lock-in** | Model trained on Schema.org fails on SEM/BFO | Train separate adapters per ontology, or multi-ontology SFT |
| **GGUF conversion loss** | Quantisation degrades extraction quality | Evaluate at Q8, Q6_K, Q5_K_M, Q4_K_M; pick best accuracy/size tradeoff |
| **Evaluation gaming** | Model optimises for metrics but not real quality | Include human evaluation in every comparison |
| **Compute requirements** | QLoRA on 7B still needs ~16GB VRAM | Apple Silicon MPS support via Unsloth; or Google Colab T4/A100 |

---

## 8. Proposed File Structure

```
08-event-digital-twin/
  ...existing files...
  finetune_data.py          # Training data generation & export
  finetune_train.py         # SFT, DPO, KGE training scripts
  finetune_eval.py          # Evaluation framework & A/B comparison
  training_data/
    sft_extraction.jsonl    # SFT training set
    dpo_pairs.jsonl         # DPO preference pairs
    gold_annotations/       # Human-reviewed gold extractions
    event_triples.tsv       # PyKEEN graph export

09-deployment/
  ...existing placeholder...
  modelfile_builder.py      # Generate Ollama Modelfiles from adapters
  adapter_serve.py          # Adapter swapping configuration
  Modelfile.extraction      # Ollama Modelfile for extraction model
  Modelfile.qa              # Ollama Modelfile for Q&A model
  export_gguf.sh            # GGUF conversion pipeline
```

---

## 9. Suggested Order of Implementation

| Step | Module | Task | Dependencies |
|------|--------|------|-------------|
| 1 | 08 | Build `finetune_data.py` — synthetic data generation pipeline | Existing ingest pipeline working |
| 2 | 08 | Create 12+ gold annotations from existing test corpus | Human effort |
| 3 | 08 | Build `finetune_train.py` — QLoRA SFT training script | HuggingFace TRL, PEFT, bitsandbytes |
| 4 | 08 | Train extraction adapter, evaluate vs. base model | GPU access (MPS / Colab / cloud) |
| 5 | 08 | Build `finetune_eval.py` — evaluation framework | Steps 1-4 |
| 6 | 08 | Train DPO adapter for faithfulness | SFT checkpoint from step 4 |
| 7 | 08 | PyKEEN graph embedding training & evaluation | PyKEEN (already installed) |
| 8 | 09 | Build `modelfile_builder.py` — GGUF export pipeline | Trained adapters from steps 4-6 |
| 9 | 09 | Build `adapter_serve.py` — Ollama adapter swapping | GGUF files from step 8 |
| 10 | 09 | Write deployment documentation & CI/CD config | All previous steps |

---

## 10. Key References

### Fine-Tuning Methods
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314
- Rafailov et al. (2023). *Direct Preference Optimization*. arXiv:2305.18290
- Lialin et al. (2022). *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning*. arXiv:2303.15647
- Hayati et al. (2024). *Chain-of-Instructions: Compositional Instruction Tuning*. arXiv:2402.11532 (AAAI 2025)

### LLMs for Knowledge Graphs
- Zhu et al. (2023). *LLMs for Knowledge Graph Construction and Reasoning*. arXiv:2305.13168
- Kim et al. (2023). *KG-GPT: General Framework for Reasoning on KGs Using LLMs*. arXiv:2310.11220 (EMNLP 2023)
- Yang et al. (2023). *CP-KGC: Constrained Prompts for Knowledge Graph Completion*. arXiv:2310.08279

### KG Embeddings
- Bordes et al. (2013). *Translating Embeddings for Modeling Multi-relational Data* (TransE)
- Trouillon et al. (2016). *Complex Embeddings for Simple Link Prediction* (ComplEx)
- Ali et al. (2021). *PyKEEN 1.0: A Python Library for Training and Evaluating KG Embeddings*

### Training Infrastructure
- HuggingFace TRL — `SFTTrainer`, `DPOTrainer` documentation
- HuggingFace PEFT — `LoraConfig`, adapter merging
- Unsloth — 2x faster QLoRA training with GGUF export
- Ollama Modelfile specification — custom model serving

### Already Cited in Project
- Ramadan (2025) — Schema-guided extraction validated
- Spyropoulos (2023) — Forensic ontologies for LLM extraction
- Pandey (2020) — Event-centric KG design
- Fenton et al. (2020) — Bayesian evidence reasoning
