# PISA Cognitive Tutor v2.0 — Architecture Deep Dive

> **Built for NECTEC** (National Electronics and Computer Technology Center)  
> An AI-powered Socratic tutoring platform that measures student cognitive engagement using the **ICAP Framework** (Chi & Wylie, 2014) and the expanded **Computational Neuro-Software Engineering (C1–C8) Framework**.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [The ICAP Framework — The Core Theory](#2-the-icap-framework--the-core-theory)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Layer-by-Layer Breakdown](#4-layer-by-layer-breakdown)
   - [4A. Ingestion Layer](#4a-ingestion-layer)
   - [4B. Retrieval Layer](#4b-retrieval-layer)
   - [4C. Chatbot Layer](#4c-chatbot-layer)
   - [4D. Evaluation Layer](#4d-evaluation-layer)
   - [4E. Metrics Layer](#4e-metrics-layer)
   - [4F. API Layer](#4f-api-layer)
   - [4G. Frontend Layer](#4g-frontend-layer)
5. [Data Flow — End to End](#5-data-flow--end-to-end)
6. [Database Schemas](#6-database-schemas)
7. [The 8 Cognitive Dimensions (C1–C8)](#7-the-8-cognitive-dimensions-c1c8)
8. [Configuration Reference](#8-configuration-reference)
9. [Ensemble Voting — How It Works](#9-ensemble-voting--how-it-works)
10. [Critical Design Principles](#10-critical-design-principles)
11. [Deployment & Environment](#11-deployment--environment)
12. [File Manifest](#12-file-manifest)

---

## 1. What This System Does

PISA Cognitive Tutor v2.0 has two parallel jobs, kept deliberately separate:

| Job | When | Who does it |
|-----|------|-------------|
| **Socratic Tutoring** | Real-time, during student chat | `TutorEngine` (never assesses) |
| **Cognitive Assessment** | Offline, after sessions end | `AnalyzerPipeline` (never tutors) |

A student interacts with a Socratic tutor that guides them through PISA science questions using retrieved knowledge. After the session ends, an ensemble of 4 LLMs (running offline) scores every student turn across 8 cognitive dimensions. The scores are aggregated into per-student profiles and inter-rater reliability metrics, enabling research-grade analysis of cognitive engagement.

**The four research hypotheses being tested (H1–H4):**

| # | Hypothesis | Primary Metric |
| --- | --- | --- |
| **H1** | Higher C4–C8 scores correlate with longer response times (> 25 s) | Spearman ρ(C4-C8 mean, Duration_Sec) ≥ 0.35 |
| **H2** | ICAP state progresses toward Interactive over dialogue turns | LTA transition matrix: P(upgrade) > P(downgrade) |
| **H3** | 4-model ensemble achieves substantial agreement across all 8 dimensions | Gwet's AC2 ≥ 0.60 on C1–C4; AC2 ≥ 0.50 on C5–C8 |
| **H4** | Typhoon 2 shows higher agreement on Thai-language turns than English-only models | Δ agreement rate ≥ +5 pp on Thai-language turns |

---

## 2. The ICAP Framework — The Core Theory

The ICAP Framework (Chi & Wylie, 2014) classifies student learning behaviours into four engagement modes, ranked by cognitive depth:

```
Passive     — Student reads, watches, listens
    ↓
Active      — Student selects, highlights, repeats
    ↓
Constructive — Student explains, infers, connects (the critical leap)
    ↓
Interactive  — Student debates, critiques, builds on a partner
```

Each conversation turn is scored across 8 cognitive dimensions (C1–C8), which map onto these levels. A student's **dominant score** per dimension reflects habitual behaviour; their **max capability score** reflects their peak potential.

---

## 3. High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        INGESTION PIPELINE                           ║
║                    (runs once, offline)                              ║
║                                                                      ║
║  PISA PDFs ──► PDFProcessor ──► PIL Images (3-page shards)          ║
║                                       │                              ║
║                               VLMExtractor                           ║
║                       Qwen3-VL-8B (QLoRA 4-bit)                     ║
║                       (reads math, tables, Thai/Chinese)            ║
║                                       │                              ║
║                          JSON Validation & JSONL                     ║
║                                       │                              ║
║                    ┌──────────────────┴──────────────────┐          ║
║                    ▼                                      ▼          ║
║           knowledge_base.jsonl                  KnowledgeGraph       ║
║                    │                         (NetworkX DiGraph)      ║
║            BGE-M3 Embedder                    .gpickle file          ║
║          (dense 1024-d + sparse)                                     ║
║                    │                                                  ║
║            Qdrant Collection                                         ║
║           (hybrid vector DB)                                         ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                        ▼
   CHATBOT LAYER                           API LAYER
   ─────────────                           ─────────
   Chainlit (WebSocket UI)                FastAPI (REST + WS)
   TutorEngine                            (production endpoint)
   • Retrieves context (Qdrant)                   │
   • Socratic scaffolding                  Next.js Frontend
   • Streams from Ollama                   React 18 + TypeScript
   • Logs turns to SQLite                         │
   • NEVER assesses                        ChatWindow + Dashboard
          │
          ▼
   sessions.db (SQLite)
   sessions / turns tables
          │
          ▼
   OFFLINE EVALUATION PIPELINE
   ────────────────────────────
   AnalyzerPipeline (cron / manual)
   • Fetches unscored turns
   • 4 LLMs vote in parallel (Ollama)
   • Confidence-weighted majority vote
   • Writes to scores table
          │
          ▼
   UserProfiler → per-student profiles
          │
          ▼
   Metrics (Fleiss' κ, Gwet's AC2, Krippendorff's α, LTA)
```

---

## 4. Layer-by-Layer Breakdown

---

### 4A. Ingestion Layer

**Location:** `src/ingestion/`  
**When it runs:** Once (or incrementally when new PDFs are added).  
**Entry point:** `python -m src.ingestion.run_ingestion --config configs/model_configs.yaml`

The ingestion layer converts raw PISA PDF files into a searchable knowledge base. It has four components working in sequence.

---

#### `pdf_processor.py` — PDF to Images

**Why images, not text extraction?** PISA documents contain complex multi-column layouts, embedded diagrams, Thai/Chinese text, and mathematical equations. Traditional OCR and PDF text parsers fail on these. Converting pages to images and passing them to a Vision Language Model is far more reliable.

**What it does:**  
- Discovers all `*.pdf` files in `data/raw_pdfs/`
- Renders each page to a `PIL.Image.Image` at 200 DPI using **PyMuPDF**
- Groups pages into **shards of 3 pages each** (to fit within VLM context)
- Yields shards as an `AsyncGenerator[PageShard]`

**Key dataclass:**
```python
PageShard(
    pdf_path: Path,
    shard_index: int,
    page_range: Tuple[int, int],  # (start, end) inclusive
    images: List[Image.Image]
)
```

**Design note:** Rendering runs on CPU to leave GPU VRAM free for the VLM.

---

#### `vlm_extractor.py` — Images to Structured JSON

**Model:** `Qwen3-VL-8B-Instruct` (8B parameter Vision Language Model, loaded in QLoRA 4-bit via `bitsandbytes` — ~5 GB VRAM at runtime)

**What it does:**  
Takes 3-page image shards and asks the VLM to extract structured educational metadata in JSON format.

**The extraction prompt asks the VLM to identify:**
- `topic`: The scientific topic
- `pisa_competency`: Which PISA scientific competency is tested
- `question_summary`: The stimulus text (~800 chars)
- `correct_concept`: The scientifically correct concept
- `common_misconceptions`: Semicolon-separated misconceptions students typically have
- `key_vocabulary`: List of key terms
- `sub_questions`: List of sub-questions
- `pisa_knowledge_type`: Content / Process / Context
- `cognitive_demand`: Low / Medium / High
- `visual_description`: Description of any diagram or table

**Sampling parameters:** `temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0` — the official Qwen3-VL recommended values. Higher temperature than previous versions; consistent JSON output is ensured by the surgical bracket matcher rather than by suppressing sampling.

**JSON extraction:** Uses a "surgical bracket matcher" — it finds the outermost `[...]` in the raw output and attempts `json.loads()`. This handles the common LLM pattern of generating text before and after JSON.

**Validation (`_validate_record`):**  
- Requires: `topic, pisa_competency, question_summary, correct_concept, common_misconceptions`
- Normalises type mismatches (e.g., misconceptions returned as list → joined as semicolons)
- Rejects records missing required fields with a warning

---

#### `graph_rag_builder.py` — Knowledge Graph Construction

**Library:** NetworkX `DiGraph` (directed graph)

After the JSONL is built, the knowledge graph adds **structural relationships** on top of semantic vectors. This lets retrieval find misconceptions, related concepts, and prerequisite chains — things cosine similarity alone cannot express.

**Node types (4):**
| Node Type | Example |
|-----------|---------|
| `Concept` | "photosynthesis" |
| `Misconception` | "plants get energy from soil" |
| `PISACompetency` | "Scientific Enquiry" |
| `Question` | PISA question unit |

**Edge types (5):**
| Edge | Direction | Meaning |
|------|-----------|---------|
| `HAS_MISCONCEPTION` | Concept → Misconception | This concept is commonly misunderstood as... |
| `TESTS_COMPETENCY` | Concept → PISACompetency | This concept tests... |
| `RELATED_CONCEPT` | Concept → Concept | Vocabulary-derived relationships |
| `APPEARS_IN_QUESTION` | Question → Concept | This question tests this concept |
| `PREREQUISITE_OF` | (reserved) | For future multi-course expansion |

**`get_subgraph_context(seed_query, max_hops=2, max_nodes=20)`:**  
At query time, fuzzy-matches the seed concept and returns its BFS neighbourhood as a human-readable string. This gets injected into the tutor's system prompt alongside vector search results.

**Persistence:** Saved as `data/vector_db/knowledge_graph.gpickle` using Python's pickle at highest protocol for fast I/O.

---

#### `run_ingestion.py` — Pipeline Orchestrator

**What it does:**  
1. Discovers PDFs in `data/raw_pdfs/`
2. Reads existing JSONL to identify already-processed files (**smart resume** — no reprocessing)
3. For each unprocessed PDF: shards it, runs VLM extraction, appends records to JSONL
4. Builds the knowledge graph from the final JSONL
5. Ingests all records into Qdrant (embeddings computed at this step)

**Resumable design:** The pipeline can be interrupted at any point. On restart it skips PDFs whose `source_file` field already appears in `knowledge_base.jsonl`.

---

### 4B. Retrieval Layer

**Location:** `src/retrieval/`

The retrieval layer is what the chatbot and evaluator query at runtime. It answers the question: *"What does the knowledge base know about this student's message?"*

---

#### `embedder.py` — BGE-M3 Multilingual Embedder

**Model:** `BAAI/bge-m3` (570M parameters)

BGE-M3 is chosen over larger English-only embedding models for three reasons:
1. **Native multilingual support** (100+ languages, critical for Thai/Chinese PISA questions)
2. **Dual output** — one forward pass produces both dense and sparse embeddings
3. **Strong performance on Thai** — outperforms larger English models on cross-lingual retrieval

**Outputs:**
- **Dense:** 1024-dimensional L2-normalised vector (captures semantic meaning)
- **Sparse:** SPLADE-style `{token_id: weight}` dict (captures keyword importance)

**`record_to_text(record)`:**  
Converts a JSONL record into a canonical string for indexing:
```
topic | question_summary (800 chars) | correct_concept | common_misconceptions | key_vocabulary
```

---

#### `hybrid_search.py` — Qdrant Hybrid Search

**Database:** Qdrant running in local mode (`./data/vector_db/qdrant/`)

**Collection:** `pisa_knowledge`  
- Dense vector: 1024-d, Cosine distance  
- Sparse vector: on-disk SPLADE index  

**`search(query, top_k=8)`:**
1. Encodes the query with BGE-M3 (dense + sparse)
2. Runs Qdrant `query_points` with **dual prefetch**:
   - Dense prefetch: `limit = top_k × 2`
   - Sparse prefetch: `limit = top_k × 2`
3. Fuses results with **Reciprocal Rank Fusion (RRF)**: `score = Σ 1 / (rank + k)`
4. Filters by `score_threshold = 0.55`
5. Falls back to dense-only if hybrid query fails

**Why hybrid?** Dense search finds semantically similar content even with different wording. Sparse search finds exact keyword matches (important for technical vocabulary like chemical names and PISA competency codes). RRF combines both rank lists without requiring a learned fusion weight.

---

### 4C. Chatbot Layer

**Location:** `src/chatbot/`

The chatbot layer handles the student-facing experience. Its most important architectural constraint: **it never imports from `src.evaluation`** and never stores or retrieves cognitive scores.

---

#### `tutor_engine.py` — Socratic Response Generator

The engine that produces tutor responses. It is explicitly Socratic — it asks guiding questions rather than giving answers.

**`_retrieve_context(user_message)`:**
1. Calls `hybrid_search.search(user_message, top_k=5)`
2. Formats vector results as numbered blocks (`[KB-1]`, `[KB-2]`, ...)
3. If graph is loaded, calls `get_subgraph_context()` using the top result's concept
4. Combines: `vector context + graph summary`
5. Returns `(context_string, kb_used_flag)`

**`_build_messages(user_message, history, context, turn_number, file_context)`:**
- Loads the Socratic system prompt from `configs/prompts/socratic_tutor.txt`
- Injects retrieved context + any uploaded file descriptions
- Keeps the last **12 turns** of conversation history
- Appends the new user message

**`respond_stream(user_message, history, turn_number)`:**
- Retrieves context
- Builds message list
- Streams tokens from Ollama (`http://localhost:11434/api/chat`)
- Sanitises output (removes forbidden words via regex)
- Yields `(token, kb_used_flag)` tuples

**Forbidden word sanitisation:**  
Any token matching the following pattern is silently removed from the response:
```regex
\b(correct|incorrect|wrong|right answer|well done|good job|try again|
exactly|perfect|ถูกต้อง|ผิด|เก่งมาก|ดีมาก|ถูกแล้ว|ลองใหม่|
正确|不对|很好|对了)\b
```
This prevents the tutor from inadvertently providing assessment feedback — which would bias student behaviour and invalidate the cognitive measurement.

**LLM Backend:**

- Primary: Ollama local server (`qwen3:8b-q8_0`)
- Fallback: HuggingFace `transformers` (loads model directly)
- Config: `temperature=0.7, max_tokens=1024, top_p=0.9`

---

#### `session_manager.py` — Async SQLite Logger

Handles all persistent state for the chatbot layer. Uses `aiosqlite` for non-blocking database access with WAL mode for concurrent reads.

**`new_session(user_id, question_id) → session_id`:**  
Generates a UUID, inserts into `sessions` table, returns the ID.

**`record_turn(...)`:**  
Logs every student turn with timing. The `duration_sec` field is critical: it measures how long the student thought before responding, and serves as a **cognitive depth proxy**:

| Duration | Likely ICAP Level | Reasoning |
|----------|------------------|-----------|
| < 8 seconds | Passive | Surface recall, no deep processing |
| 8–25 seconds | Active | Reading and selecting |
| > 25 seconds | Constructive / Interactive | Reasoning and composing |

**`get_unscored_turns(batch_size=200)`:**  
Returns turns that have no corresponding entry in the `scores` table (via `LEFT JOIN`). Used by the AnalyzerPipeline to find work to do.

---

#### `file_processor.py` — Student File Upload Handler

Allows students to upload files during their session. The VLM describes them so the tutor can reference them in its Socratic questioning.

**Supported formats:**

| File Type | Processing |
|-----------|-----------|
| Images (PNG, JPG, WEBP, GIF) | VLM description (~600–800 words) |
| PDF | Render first 3 pages → VLM description |
| CSV / Excel | Pandas → Markdown table (max 50 rows) |
| Plain text / code | Truncated to 3000 chars |

**VLM description prompt requests:**
- Graph axes, scale, trends, data points
- Table structure, headers, values, merged cells
- Diagram components, spatial relationships, labels
- Equations in LaTeX notation
- All text transcribed with emphasis preserved

**Storage:** In-memory dict `_file_store[session_id]` — no files are written to disk. Cleared on session end via `cleanup_session_files(session_id)`.

**Injection format:** Descriptions are wrapped in XML tags and injected into the system prompt:
```
<STUDENT_UPLOADED_IMAGE>
filename.png — [VLM analysis]
</STUDENT_UPLOADED_IMAGE>
```

---

#### `app.py` — Chainlit WebSocket UI

The development-facing entry point. Runs as a WebSocket server for direct browser access.

**Lifecycle:**

```
@cl.on_chat_start
├─ Lazy-init: searcher, graph, session_manager (singletons)
├─ Generate: user_id (UUID), session_id
├─ Create: TutorEngine
├─ Store in cl.user_session (per-user state)
└─ Send: Welcome message

@cl.on_message
├─ Measure duration_sec (time since last AI message)
├─ Process any uploaded files
├─ engine.respond_stream() → stream tokens to UI
├─ Sanitise response
├─ Update history (keep last 12 turns in memory)
├─ session_manager.record_turn() → SQLite
└─ Auto-close after 12 turns

@cl.on_chat_end
└─ cleanup_session_files(session_id)
```

**12-turn limit:** Prevents sessions from becoming too long for reliable cognitive scoring. Research-validated cutoff.

---

### 4D. Evaluation Layer

**Location:** `src/evaluation/`  
**When it runs:** Offline — scheduled (cron every 15 min) or manually triggered.  
**Critical rule:** Never runs during a tutoring session.

---

#### `majority_voting.py` — 4-Model Ensemble

The heart of the cognitive assessment system. Four locally-running LLMs each independently score a student turn, then their votes are aggregated.

**The four models:**

| Model | Tag | Strength |
|-------|-----|---------|
| Qwen3-8B-Instruct | `qwen3:8b-q8_0` | General reasoning |
| Mistral-7B-Instruct-v0.2 | `mistral:7b-instruct-v0.2-q8_0` | Scientific reasoning |
| Llama3.1-8B-Instruct | `llama3.1:8b-instruct-q8_0` | Instruction following |
| Typhoon2-8B-Instruct | `typhoon2:8b-instruct-q4_K_M` | Thai-tuned |

All four run in parallel via `ThreadPoolExecutor`.

**Per-model output schema:**
```json
{
  "C1": 0,   "C2": 1,   "C3": 2,
  "C4": "NA", "C5": 1,  "C6": 0,
  "C7": 1,   "C8": 2,
  "confidence": 0.85,
  "icap_level": "Constructive",
  "reasoning": "Student explains the mechanism via..."
}
```

**Voting algorithm (per dimension):**
1. Collect scores from all responding models
2. Weight each vote: `model.kappa_weight × model.confidence`
3. **Plurality winner** = dominant score
4. **Controversial** if: `max_vote_weight / total_weight < 0.50`
5. **Max capability** = highest numeric score any model gave
6. **Consensus ratio** = fraction of total weight behind the winner

**Output per dimension:**
```json
{
  "C4": {
    "dominant": 1,
    "max_capability": 2,
    "vote_distribution": {"0": 0.10, "1": 0.55, "2": 0.35},
    "controversial": false,
    "consensus_ratio": 0.55
  }
}
```

**Kappa-adaptive weights:**  
After enough data accumulates, `reliability.py` computes each model's historical agreement with the majority vote. Weights are normalised to the range [0.5, 1.5] and persisted to `data/processed_jsonl/model_kappa_weights.json`. Models that historically agree more with the consensus get more voting power.

---

#### `analyzer_pipeline.py` — Offline Batch Assessment

**Entry point:** `python -m src.evaluation.analyzer_pipeline --config configs/model_configs.yaml --batch 100`

**Pipeline steps:**
1. Query `turns LEFT JOIN scores WHERE scores.id IS NULL` → unscored turns
2. For each turn, retrieve PISA context via Qdrant (`searcher.search(turn.user_input)`)
3. Call `voter.score_turn(user_input, context, turn_number, duration_sec)`
4. Flatten result into individual columns and insert into `scores` table
5. Loop until no unscored turns remain

**Why retrieve context at evaluation time too?**  
The PISA context gives the evaluator LLMs the rubric-relevant information (correct concept, common misconceptions) needed to score accurately. Without it, LLMs would be judging student responses without knowing what the correct answer is.

---

#### `user_profiler.py` — Per-Student Cognitive Profiles

Aggregates scored turns into a per-student profile.

**Profile computation:**

| Metric | How Computed |
|--------|-------------|
| Dominant score per dimension | Mode (most frequent value) |
| Max capability per dimension | Maximum value across all turns |
| ICAP level distribution | Fraction of turns at each level |
| Engagement trajectory | Avg C4–C8 score in first half vs. second half of turns |
| Duration–causal correlation | Spearman rank correlation: `duration_sec` ↔ `C4` score |
| Controversy rate | Fraction of dimension scores marked controversial |

**Engagement trajectory:**
- `improving` if second-half avg > first-half avg by +0.2
- `declining` if second-half avg < first-half avg by −0.2
- `stable` otherwise

**Output profile (per student):**
```json
{
  "user_id": "student_abc123",
  "n_turns_scored": 12,
  "dimension_profile": {
    "C1": { "dominant_score": 1, "max_capability": 2, "n_responses": 12, "score_distribution": {"0": 2, "1": 7, "2": 3} }
  },
  "icap_level_distribution": { "Passive": 0.10, "Active": 0.35, "Constructive": 0.40, "Interactive": 0.15 },
  "dominant_icap": "Constructive",
  "max_icap_achieved": "Interactive",
  "engagement_trajectory": "improving",
  "avg_duration_sec": 18.5,
  "duration_causal_correlation": 0.52,
  "controversy_rate": 0.08
}
```

---

### 4E. Metrics Layer

**Location:** `src/metrics/`

Provides research-grade inter-rater reliability measurements between the 4 ensemble models, ICAP state transition analysis, and calibration metrics.

---

#### `reliability.py`

Three complementary agreement statistics are computed, each addressing a different measurement concern.

---

**Fleiss' Kappa (`fleiss_kappa(rating_matrix)`):**  
Measures agreement among K raters on N items. Input: N×K matrix of ordinal scores (0/1/2).

| Range | Interpretation |
| --- | --- |
| < 0.00 | Poor (below chance) |
| 0.00–0.20 | Slight |
| 0.20–0.40 | Fair |
| 0.40–0.60 | Moderate |
| **0.60–0.80** | **Substantial** ← H3 target |
| 0.80–1.00 | Almost perfect |

**Limitation:** Fleiss' κ is artificially deflated when score distributions are skewed. In early sessions, most students score 0 on C7/C8 (higher-order dimensions). κ treats all-zeros as "chance agreement" even if every model unanimously agrees — this is the **Kappa Paradox**.

---

**Gwet's AC2 (`gwet_ac2(rating_matrix, categories=[0.0, 1.0, 2.0])`):**  
The **primary reliability statistic for H3**. Uses quadratic weights and a chance-correction formula that is robust to skewed distributions.

**Formula:**

```text
W[i,j] = 1 − ((v_i − v_j) / range)²   (quadratic weight matrix)
p_a    = mean weighted pairwise agreement across all items
T_w    = π @ W @ π                       (marginal-based chance agreement)
p_e    = T_w − Σ_k W[k,k]·π_k·(1−π_k) / (K−1)
AC2    = (p_a − p_e) / (1 − p_e)
```

Because `p_e` uses the actual marginal distribution rather than treating every category as equally likely, AC2 does not deflate when one category dominates. **For educational engagement data with skewed distributions, AC2 is strictly preferable to κ.**

**`interpret_ac2(v)`** uses the same thresholds as kappa but notes `"almost perfect (AI-validated)"` for v > 0.80, reflecting that AI evaluator ensembles typically achieve higher agreement than human raters on structured rubrics.

---

**Krippendorff's Alpha (`krippendorff_alpha(data, level_of_measurement="ordinal")`):**  
Handles missing data (when a model times out for a turn) and respects the ordinal scale structure. Useful as a cross-check alongside AC2.

---

**`compute_dimension_reliability(individual_votes_list)`:**  
Returns per-dimension breakdown for all C1–C8 dimensions:

```python
{
  "C4": {
    "fleiss_kappa": 0.6120,
    "gwet_ac2": 0.7450,        # primary metric (H3)
    "krippendorff_alpha": 0.5890,
    "n_items": 324,
    "pct_consensus": 0.712
  }
}
```

**`update_model_kappa_weights()`:**  
Recomputes each model's historical agreement with the majority vote. Normalises to [0.5, 1.5]. Persists to `data/processed_jsonl/model_kappa_weights.json`. This feeds back into `MajorityVoter` so models with consistently higher agreement get proportionally higher voting weight.

---

#### `visualizations.py`

Generates all publication-ready figures and the LaTeX reliability table.

| Function | Output | Purpose |
| --- | --- | --- |
| `plot_dimension_correlation_heatmap()` | `heatmap.png` | Spearman ρ between C1–C8 dominant scores |
| `plot_icap_distribution()` | `icap_distribution.png` | ICAP level frequency across all turns |
| `plot_engagement_trajectory()` | `trajectory.png` | Mean C4–C8 score over turn number |
| `plot_duration_vs_depth()` | `duration_scatter.png` | Duration_Sec vs. C4 score, coloured by ICAP level |
| `plot_latent_transition_heatmap()` | `lta_transitions.png` | **LTA: ICAP state transition probabilities (H2)** |
| `export_latex_table()` | `reliability_table.tex` | Publication table: Fleiss' κ, Gwet's AC2, Krippendorff's α |
| `run_full_report()` | all of the above | One-call convenience wrapper |

**Latent Transition Analysis (LTA) — `plot_latent_transition_heatmap()`:**  
Groups scored turns by session, sorts by turn number, counts consecutive-turn state transitions (Passive→Active, Active→Constructive, etc.), and row-normalises to produce a 4×4 empirical transition probability matrix. Visualised as a "Blues" heatmap. Provides direct visual evidence for/against H2 (whether students tend to move toward more engaged states as the session progresses).

---

### 4F. API Layer

**Location:** `src/api/main.py`  
**Framework:** FastAPI  
**Entry point:** `uvicorn src.api.main:app --port 8001`

This is the production-facing backend consumed by the Next.js frontend.

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/session/new` | Create new session, get `session_id` |
| `POST` | `/api/upload/{session_id}` | Upload a file (max 20 MB) |
| `GET` | `/api/session/{session_id}/history` | Retrieve past turns |
| `WS` | `/api/chat/{session_id}` | Real-time chat WebSocket |
| `GET` | `/api/profiles` | All student cognitive profiles |
| `GET` | `/api/metrics/reliability` | Fleiss' κ + Gwet's AC2 + Krippendorff's α per dimension |
| `POST` | `/api/analyze` | Trigger AnalyzerPipeline (background task) |
| `GET` | `/api/neuro/taxonomy` | C1–C8 multi-taxonomy data |
| `GET` | `/api/neuro/metrics` | Computes current LCAI, CASM, MCI, and PCRS |
| `POST` | `/api/neuro/code-task` | AST-based ICP heuristic analysis on code |

**WebSocket protocol (`/api/chat/{session_id}`):**

Client sends:
```json
{ "user_input": "...", "turn_number": 3, "duration_sec": 14.2, "file_ids": ["uuid1"] }
```

Server streams:
```json
{ "type": "token", "content": "What" }
{ "type": "token", "content": " do" }
...
{ "type": "done", "kb_used": true }
```

**Singletons** (loaded once on startup via `get_resources()`):
- `cfg` — config dict
- `searcher` — `QdrantHybridSearch`
- `graph` — NetworkX `DiGraph`
- `session_manager` — `SessionManager`
- `engine` — `TutorEngine`

**CORS:** Allows `http://localhost:3000` (the Next.js frontend). No authentication — this is a research tool.

---

### 4G. Frontend Layer

**Location:** `frontend/`  
**Framework:** Next.js 14 (App Router, TypeScript)

| Package | Purpose |
|---------|---------|
| React 18 | Component framework |
| Tailwind CSS | Utility styling |
| Framer Motion | Animations |
| Recharts | Score visualisation charts |
| React Markdown + remark-gfm | Rendering tutor markdown responses |
| Lucide React | Icons |

**Pages:**

**`app/page.tsx` — Main Chat:**  
Calls `createSession()` on mount, renders `Sidebar` + `ChatWindow`.

**`app/dashboard/page.tsx` — Research Dashboard:**  
Displays student profile cards, reliability tables, and the `NeuroTaxonomyTable`. Has a "Run Analyzer" button that triggers `POST /api/analyze`.

**`app/neuro-framework/page.tsx`:**  
Visualizes the C1-C8 framework with interactive tables and neuro-biomarker context.

**`app/code-tasks/page.tsx`:**  
Hosts the `CodeTaskAssessor` component for AST-based code complexity (ICP) analysis.

**`app/research-metrics/page.tsx`:**  
Interactive gauges and KaTeX-rendered mathematical formulas for LCAI, CASM, MCI, and PCRS.

**Key Components:**

**`ChatWindow.tsx`:**  
- Opens a WebSocket to `/api/chat/{sessionId}`
- Accumulates streaming tokens into message content
- Drag-drop file upload with progress indicators
- Scrolls to bottom on new messages
- Shows `TypingIndicator` while streaming

**`MessageBubble.tsx`:**  
- Renders markdown (supports tables, code blocks, LaTeX-adjacent formatting)
- Shows a KB indicator icon when `kb_used=true` (retrieval was used)

**`ProfileCard.tsx`:**  
- Per-student profile summary
- Blue bar = dominant score (habitual behaviour)
- Light bar = max capability (peak potential)
- Trajectory arrow: ↑ improving / → stable / ↓ declining

**`NeuroMetricsPanel.tsx` & `CodeTaskAssessor.tsx`:**
- Integrates `react-katex` to render mathematical formulas.
- `react-syntax-highlighter` for code formatting.

**`ReliabilityTable.tsx`:**

- Per-dimension Fleiss' κ, Gwet's AC2, and Krippendorff's α with interpretation labels
- Shows n_items, pct_consensus, and AC2-based overall interpretation (primary metric for H3)

---

## 5. Data Flow — End to End

### Phase 1: Ingestion (Runs Once)

```
data/raw_pdfs/*.pdf
        │
        ▼ PDFProcessor
PIL Image shards (3 pages each, 200 DPI)
        │
        ▼ VLMExtractor (Qwen3-VL-8B, QLoRA 4-bit)
Raw LLM text → JSON extraction (bracket matching)
        │
        ▼ Validation
knowledge_base.jsonl (append-only, resumable)
        │
        ├──────────────────────────┐
        ▼                          ▼
KnowledgeGraphBuilder        BGE-M3 Embedder
NetworkX DiGraph             Dense (1024-d) + Sparse vectors
.gpickle                           │
                                   ▼
                             Qdrant Collection
                             pisa_knowledge
```

### Phase 2: Student Session (Real-Time)

```
Student opens browser
        │
        ▼ /api/session/new
session_id, user_id created in SQLite sessions table
        │
        ▼ WebSocket /api/chat/{session_id}
Student sends message
        │
        ├── measure duration_sec (time since last AI message)
        ├── process any uploaded files → VLM description
        │
        ▼ TutorEngine.respond_stream()
        ├── Qdrant hybrid search (top_k=5)
        ├── Graph subgraph context
        ├── Build Socratic system prompt with context
        ├── Stream from Ollama (qwen3:8b-q8_0)
        └── Sanitise forbidden words
        │
        ▼ Stream tokens to frontend
        │
        ▼ session_manager.record_turn()
turns table ← user_input, ai_response, kb_used, duration_sec
```

### Phase 3: Offline Evaluation (Scheduled / Manual)

```
AnalyzerPipeline triggered (cron or POST /api/analyze)
        │
        ▼ Get unscored turns
turns LEFT JOIN scores WHERE scores.id IS NULL
        │
        ▼ For each turn:
        ├── Qdrant search(user_input) → PISA context
        ├── MajorityVoter.score_turn()
        │   ├── 4 ModelEvaluators in parallel (ThreadPoolExecutor)
        │   ├── Each: build rubric prompt → POST Ollama → parse JSON
        │   └── Aggregate: kappa-weighted majority vote per C1–C8
        └── Insert into scores table
        │
        ▼ UserProfiler.build_all_profiles()
Per-student dimension profiles + ICAP distributions
        │
        ▼ GET /api/metrics/reliability
Fleiss' κ + Gwet's AC2 + Krippendorff's α per dimension
        │
        ▼ visualizations.run_full_report()
heatmap, icap_distribution, trajectory, duration_scatter,
lta_transitions (LTA), reliability_table.tex
        │
        ▼ Research Dashboard (frontend/dashboard)
ProfileCards + ReliabilityTable displayed
```

---

## 6. Database Schemas

All data is stored in a single SQLite file: `data/chat_logs/sessions.db`

### `sessions` table

```sql
CREATE TABLE sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT,
    started_at   TEXT,   -- ISO 8601
    question_id  TEXT
);
```

### `turns` table

```sql
CREATE TABLE turns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT REFERENCES sessions(session_id),
    user_id          TEXT,
    turn_number      INTEGER,
    timestamp        TEXT,    -- ISO 8601
    user_input       TEXT,
    ai_response      TEXT,
    kb_context_used  INTEGER, -- 0 or 1
    duration_sec     REAL,    -- seconds since last AI message
    question_topic   TEXT DEFAULT ''
);

CREATE INDEX idx_turns_session ON turns(session_id);
CREATE INDEX idx_turns_user    ON turns(user_id);
```

### `scores` table (created by AnalyzerPipeline)

```sql
CREATE TABLE scores (
    id                    INTEGER PRIMARY KEY,
    turn_id               INTEGER REFERENCES turns(id),
    session_id            TEXT,
    user_id               TEXT,
    turn_number           INTEGER,
    duration_sec          REAL,
    dominant_icap         TEXT,
    models_responded      INTEGER,
    elapsed_sec           REAL,
    scored_at             TEXT,  -- ISO 8601

    -- Repeated for C1 through C8:
    C1_dominant           TEXT,  -- 0, 1, 2, or NA
    C1_max                TEXT,
    C1_controversial      INTEGER, -- 0 or 1
    C2_dominant           TEXT,
    C2_max                TEXT,
    C2_controversial      INTEGER,
    -- ... C3 through C8 ...

    individual_votes_json TEXT   -- JSON array of all model votes
);
```

### Qdrant Vector Store

**Collection:** `pisa_knowledge`  
**Dense:** 1024-d, Cosine distance  
**Sparse:** on-disk SPLADE index  
**Payload per point:**
```json
{
    "topic": "...",
    "question_summary": "...",
    "correct_concept": "...",
    "common_misconceptions": "...",
    "pisa_competency": "...",
    "cognitive_demand": "Low|Medium|High",
    "key_vocabulary": ["..."],
    "source_file": "filename.pdf",
    "page_range": "1-3"
}
```

---

## 7. The 8 Cognitive Dimensions (C1–C8)

Defined in `configs/icap_rubric.yaml` under the `neuro_taxonomy` block. Each is scored 0, 1, or 2 (or "NA" if not applicable).

| Dim | Cognitive Capacity | Neuro-Structural ALE | Cervical Segment | Defense Level |
|-----|--------------------|----------------------|------------------|---------------|
| **C1** | Sustained Attention | Anterior/Middle/Posterior Cingulate | Head & neck movement | Level 1 (High Adaptive) |
| **C2** | Response Inhibition | Medial Frontal Gyrus | Upper shoulders & diaphragm | Level 2 (Mental Inhibition) |
| **C3** | Speed of Information Processing | Superior Frontal Gyrus | Deltoid & biceps | Level 3 (Minor Image Distortion) |
| **C4** | Cognitive Flexibility | Cingulate Gyrus medial cluster | Wrist extension | Level 4 (Disavowal) |
| **C5** | Multiple Simultaneous Attention | Frontoparietal networks | Triceps | Level 5 (Major Image Distortion) |
| **C6** | Working Memory | Prefrontal-parietal retrieval loops | Hand & finger flexion | Level 6 (Action-oriented) |
| **C7** | Category Formation | Cingulate Gyrus medial cluster | Upper limb neuro-sensory | Level 7 (Borderline Defensive) |
| **C8** | Pattern Recognition & Inductive Thinking | Cerebellar-cortical loop networks | Cardiorespiratory autonomic | Level 8 (Psychotic/Delusional) |

**Key concepts:**
- **Dominant score** = mode (habitual behaviour, what the student usually does)
- **Max capability** = maximum achieved (what the student *can* do)
- These two can diverge: a student might be capable of Interactive reasoning (max) but default to Active behaviour (dominant)

---

## 8. Configuration Reference

All configuration lives in `configs/model_configs.yaml` and `configs/icap_rubric.yaml`.

### `configs/model_configs.yaml`

```yaml
hardware:
  vram_gb: 24
  cuda_device: "cuda:0"
  cpu_threads: 8
  vlm_device: "cuda:0"

vlm:
  model_id: "Qwen/Qwen3-VL-8B-Instruct"
  load_in_4bit: true          # QLoRA — ~5 GB VRAM
  load_in_8bit: false
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"  # Falls back to "eager"
  max_new_tokens: 4096
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  repetition_penalty: 1.0
  do_sample: true
  pdf_render_dpi: 200
  max_pages_per_shard: 3

embedding:
  model_id: "BAAI/bge-m3"
  device: "cpu"
  batch_size: 32
  dense_dim: 1024
  use_sparse: true
  use_dense: true

qdrant:
  mode: "local"
  local_path: "./data/vector_db/qdrant"
  collection_name: "pisa_knowledge"
  top_k: 8
  score_threshold: 0.55
  fusion_method: "rrf"

chatbot_llm:
  backend: "ollama"
  ollama_base_url: "http://localhost:11434"
  model_name: "qwen3:8b-q8_0"
  temperature: 0.7
  max_tokens: 1024
  top_p: 0.9

evaluator_ensemble:
  backend: "ollama"
  timeout_seconds: 120
  consensus_threshold: 0.75
  controversy_threshold: 0.50
  models:
    - name: "qwen3"
      tag: "qwen3:8b-q8_0"
      weight: 1.0
    - name: "mistral"
      tag: "mistral:7b-instruct-v0.2-q8_0"
      weight: 1.0
    - name: "llama3"
      tag: "llama3.1:8b-instruct-q8_0"
      weight: 1.0
    - name: "typhoon"
      tag: "typhoon2:8b-instruct-q4_K_M"
      weight: 1.0

session_db:
  path: "./data/chat_logs/sessions.db"
  duration_thresholds:
    passive_max: 8          # < 8s → Passive
    active_max: 25          # 8–25s → Active
    constructive_min: 25    # > 25s → Constructive/Interactive

knowledge_base:
  output_path: "./data/processed_jsonl/knowledge_base.jsonl"
  resume_from_checkpoint: true
```

---

## 9. Ensemble Voting — How It Works

This is the core of the cognitive assessment. Here is the complete flow for scoring one student turn:

```
Student turn (user_input, turn_number, duration_sec)
        │
        ├── Qdrant.search(user_input) → PISA context
        │
        ▼ MajorityVoter.score_turn()
        │
        ├── ThreadPoolExecutor.map(model.score, ...) for 4 models
        │
        │   Each ModelEvaluator:
        │   ├── Builds prompt: [rubric] + [PISA context] + [student input]
        │   ├── POST to Ollama /api/chat
        │   ├── Extract JSON from response (bracket matching)
        │   ├── Validate: all C1–C8 must be 0/1/2/"NA", confidence 0–1
        │   └── Return: {C1:1, C2:2, ..., confidence:0.85, icap:"Constructive", reasoning:"..."}
        │
        ▼ Aggregation (per dimension):
        │
        │   votes = [(score=1, weight=0.90×0.85), (score=1, weight=0.95×0.70), ...]
        │   weighted_totals = {0: 0.15, 1: 1.40, 2: 0.45}
        │   dominant = argmax → 1
        │   consensus_ratio = 1.40 / 2.00 = 0.70
        │   controversial = (0.70 < 0.50) → False
        │   max_capability = 2
        │
        ▼ Final output:
        {
          "dimension_scores": {
            "C4": {
              "dominant": 1,
              "max_capability": 2,
              "vote_distribution": {"0": 0.075, "1": 0.70, "2": 0.225},
              "controversial": false,
              "consensus_ratio": 0.70
            }
          },
          "dominant_icap": "Constructive",
          "individual_votes": [...all 4 models' raw votes...],
          "models_responded": 4,
          "elapsed_sec": 14.3
        }
```

**Why 4 models instead of 1 large model?**  
Research shows that majority voting across 4 diverse 7–8B models achieves higher accuracy on ordinal classification tasks than a single 30B model — particularly when the classification task is nuanced (like distinguishing C4 score 1 from score 2). Diversity matters: Typhoon2 brings Thai-language understanding; Mistral brings strong scientific reasoning; each model's failure modes differ.

---

## 10. Critical Design Principles

### Strict Temporal Separation (Observer Bias Prevention)

This is the most important architectural constraint in the entire system.

```
Chatbot layer:     NEVER imports src.evaluation
                   NEVER reads the scores table
                   NEVER stores or retrieves cognitive scores

Evaluation layer:  Runs AFTER sessions end
                   Cannot influence the tutor
                   Cannot be seen by students
```

**Why?** The Hawthorne Effect — if students know they're being assessed on specific dimensions, they change their behaviour to be assessed well rather than to actually learn. NECTEC requires this separation for research validity.

### Forbidden Words — Multi-Language

The tutor is a Socratic guide. It must never evaluate the student's answer (that is the evaluator's job, done offline). Any word that signals "you got it right/wrong" is detected and removed from the response at generation time.

This is enforced in two places:
1. The system prompt instructs the LLM not to use evaluative language
2. A regex pattern strips any that slip through anyway

### Hybrid Retrieval — Three Information Sources

The tutor's context comes from three complementary sources:

| Source | Captures | Limitation |
|--------|----------|-----------|
| Dense vector search | Semantic similarity (topic meaning) | Misses exact keyword matches |
| Sparse vector search | Exact keyword matches | Misses paraphrasing |
| Knowledge graph | Structural relations (misconceptions, prerequisites) | Neither captures |

All three are combined at query time. This is why students get contextually rich, misconception-aware Socratic questions.

### Resumable Ingestion

The ingestion pipeline can be interrupted at any time. On restart:
1. It reads all `source_file` fields from the existing JSONL
2. It skips any PDF that already has records in the JSONL
3. It continues from where it left off

This is important because processing a single large PISA PDF takes several minutes (VLM inference is slow), and research machines may have unreliable uptime.

### Privacy by Design

- Student file uploads: stored in-memory only, never written to disk, cleared on session end
- Session data: local SQLite database, no external calls
- No telemetry, no cloud APIs (Gemini fallback disabled by default)
- Ollama and Qdrant run entirely locally — no data leaves the machine

### Model Size Decisions

| Component | Model | Size | Rationale |
|-----------|-------|------|-----------|
| VLM (extraction) | Qwen3-VL-8B | 8B (QLoRA 4-bit, ~5 GB VRAM) | Latest-gen vision model; superior Thai/Math/multilingual extraction |
| Embedder | BGE-M3 | 570M | Outperforms larger English-only models on multilingual tasks |
| Chatbot LLM | Qwen3-8B-Instruct | 8B | Strong Thai instruction following |
| Evaluator ensemble | 4 × 7–8B | ~30B total | Majority voting beats single 30B model on ordinal classification |

---

## 11. Deployment & Environment

### Docker Compose Services

| Service | Port | Purpose | GPU Required |
|---------|------|---------|-------------|
| `qdrant` | 6333 (REST), 6334 (gRPC) | Vector database | No |
| `chatbot` | 8000 | Chainlit WebSocket UI | Yes (CUDA) |
| `api` | 8001 | FastAPI backend | No (Ollama runs separately) |
| `frontend` | 3000 | Next.js React app | No |

Ollama runs as a host service (not containerised) to handle GPU access for all LLM inference.

### Environment Variables

```bash
# Core
CONFIG_PATH=configs/model_configs.yaml
OLLAMA_HOST=http://localhost:11434

# Optional fallback (disabled by default)
GEMINI_API_KEY=your_api_key

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_WS_URL=ws://localhost:8001
```

### Python Dependencies (Key Packages)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2+ | PyTorch with CUDA 12.1 |
| `transformers` | git (latest) | Qwen3-VL model loading (`pip install git+https://github.com/huggingface/transformers`) |
| `bitsandbytes` | latest | QLoRA 4-bit quantization for Qwen3-VL |
| `FlagEmbedding` | 1.2.9+ | BGE-M3 embeddings |
| `qdrant-client` | 1.9+ | Vector database client |
| `networkx` | 3.3+ | Knowledge graph (DiGraph) |
| `aiosqlite` | 0.20+ | Async SQLite I/O |
| `fastapi` | 0.111+ | REST + WebSocket API |
| `uvicorn` | 0.30+ | ASGI server |
| `chainlit` | 1.1+ | Chat UI framework |
| `pydantic` | 2.7+ | Data validation |
| `httpx` | 0.27+ | Async HTTP client for Ollama |
| `pymupdf` | latest | PDF page rendering |
| `pandas` | latest | CSV/Excel parsing |

### Quick Start (Research Use)

**Phase 1 — Build knowledge base (once):**
```bash
cp your_pisa_files/*.pdf data/raw_pdfs/
python -m src.ingestion.run_ingestion --config configs/model_configs.yaml
```

**Phase 2 — Run sessions:**
```bash
# Option A: Chainlit (development)
chainlit run src/chatbot/app.py --port 8000

# Option B: Full stack (production)
uvicorn src.api.main:app --port 8001 &
cd frontend && npm run dev
```

**Phase 3 — Score sessions:**
```bash
python -m src.evaluation.analyzer_pipeline --config configs/model_configs.yaml
```

**Phase 4 — Export data:**
```bash
curl http://localhost:8001/api/profiles > profiles.json
curl http://localhost:8001/api/metrics/reliability > reliability.json
sqlite3 data/chat_logs/sessions.db "SELECT * FROM scores;" > scores.csv
```

---

## 12. File Manifest

```
pisa_cognitive_tutor/
│
├── src/
│   ├── ingestion/
│   │   ├── pdf_processor.py        PDF → PIL image shards (CPU)
│   │   ├── vlm_extractor.py        Qwen3-VL-8B inference + JSON extraction (QLoRA 4-bit)
│   │   ├── graph_rag_builder.py    NetworkX DiGraph construction + subgraph retrieval
│   │   └── run_ingestion.py        Pipeline orchestrator (resumable)
│   │
│   ├── retrieval/
│   │   ├── embedder.py             BGE-M3 dense + sparse encoding
│   │   └── hybrid_search.py        Qdrant hybrid search with RRF fusion
│   │
│   ├── chatbot/
│   │   ├── tutor_engine.py         Socratic response generation (no assessment imports)
│   │   ├── session_manager.py      Async SQLite session + turn logging
│   │   ├── file_processor.py       Student file uploads → VLM description
│   │   └── app.py                  Chainlit WebSocket UI (development)
│   │
│   ├── evaluation/
│   │   ├── majority_voting.py      4-model ensemble with kappa-weighted voting
│   │   ├── analyzer_pipeline.py    Offline batch assessment orchestrator
│   │   ├── user_profiler.py        Per-student cognitive profile aggregation
│   │   └── dspy_optimizer.py       (optional) Auto-tune evaluator prompts
│   │
│   ├── metrics/
│   │   ├── reliability.py          Fleiss' κ, Gwet's AC2, Krippendorff's α, kappa weight update
│   │   ├── calibration.py          Expected Calibration Error (ECE)
│   │   ├── neuro_metrics.py        LCAI, CASM, MCI, PCRS computation + ICP AST calculator
│   │   └── visualizations.py       Heatmaps, LTA transition plot, LaTeX table export
│   │
│   └── api/
│       └── main.py                 FastAPI REST + WebSocket (production)
│
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx                Main chat page
│   │   ├── dashboard/page.tsx      Research dashboard (profiles + reliability)
│   │   ├── neuro-framework/page.tsx C1-C8 Taxonomy matrix
│   │   ├── code-tasks/page.tsx      AST complexity grading
│   │   └── research-metrics/page.tsx Novel mathematical metrics
│   ├── src/components/
│   │   ├── ChatWindow.tsx          WebSocket streaming + file upload UI
│   │   ├── MessageBubble.tsx       Markdown rendering + KB indicator
│   │   ├── Sidebar.tsx             Session info + navigation
│   │   ├── ProfileCard.tsx         Per-student cognitive profile visualisation
│   │   ├── ReliabilityTable.tsx    Fleiss' κ + Gwet's AC2 + Krippendorff's α display
│   │   ├── NeuroTaxonomyTable.tsx  Interactive C1-C8 visualizer
│   │   ├── CodeTaskAssessor.tsx    AST analysis integration
│   │   └── TypingIndicator.tsx     Animated dots while streaming
│   └── src/lib/
│       ├── api.ts                  Fetch wrappers + WebSocket factory
│       └── types.ts                TypeScript interfaces (ChatMessage, CognitiveProfile, UploadedFile)
│
├── configs/
│   ├── model_configs.yaml          Hardware, models, DB, retrieval settings
│   ├── icap_rubric.yaml            8 cognitive dimensions + forbidden word list
│   └── prompts/
│       ├── socratic_tutor.txt      Chainlit + API system prompt
│       ├── vision_extractor.txt    VLM extraction prompt
│       └── evaluator_judge.txt     Rubric + scoring instructions for ensemble
│
├── data/
│   ├── raw_pdfs/                   Input PISA PDF files
│   ├── processed_jsonl/
│   │   ├── knowledge_base.jsonl    VLM extraction output (append-only)
│   │   └── model_kappa_weights.json  Per-model reliability weights
│   ├── vector_db/
│   │   ├── qdrant/                 Qdrant local storage directory
│   │   └── knowledge_graph.gpickle NetworkX DiGraph (pickled)
│   ├── chat_logs/
│   │   └── sessions.db            SQLite: sessions + turns + scores tables
│   └── reports/                   Generated by run_full_report()
│       ├── heatmap.png            Spearman correlation C1–C8
│       ├── icap_distribution.png  ICAP level frequencies
│       ├── trajectory.png         C4–C8 mean over turn number
│       ├── duration_scatter.png   Duration vs. C4 score
│       ├── lta_transitions.png    LTA ICAP state transition matrix (H2)
│       └── reliability_table.tex  LaTeX: Fleiss' κ, Gwet's AC2, Krippendorff's α
│
├── tests/
│   ├── test_socratic_prompt.py     Forbidden word detection, Fleiss' κ unit tests
│   └── test_json_extractor.py      VLM JSON extraction + validation tests
│
├── notebooks/
│   ├── 01_vision_extraction_tests.ipynb
│   ├── 02_embedding_space_analysis.ipynb
│   └── 03_cognitive_profile_eda.ipynb
│
├── requirements.txt
├── docker-compose.yml
└── README(architecture).MD         ← this file
```

---

## Module Dependency Graph

```
run_ingestion.py
    ├── pdf_processor.py
    ├── vlm_extractor.py
    ├── graph_rag_builder.py
    └── hybrid_search.py
            └── embedder.py

app.py (Chainlit)
    ├── tutor_engine.py
    │       ├── hybrid_search.py
    │       └── graph_rag_builder.py (read-only)
    ├── session_manager.py
    └── file_processor.py
            └── vlm_extractor.py (shared instance)

main.py (FastAPI)
    ├── tutor_engine.py
    ├── session_manager.py
    ├── file_processor.py
    ├── analyzer_pipeline.py         ← evaluation (API only, not chatbot)
    │       ├── majority_voting.py
    │       │       └── hybrid_search.py
    │       └── user_profiler.py
    ├── reliability.py               ← Fleiss' κ, Gwet's AC2, Krippendorff's α
    │       (compute_dimension_reliability, update_model_kappa_weights)
    └── visualizations.py            ← plots + LTA heatmap + LaTeX export
            └── reliability.py       (interpret_kappa, interpret_ac2)

# ENFORCED WALL: tutor_engine.py has ZERO imports from src/evaluation/
# ENFORCED WALL: tutor_engine.py has ZERO imports from src/metrics/
```

---

*This document describes PISA Cognitive Tutor v2.0 as built for NECTEC research. The system operationalises the ICAP Framework to quantify cognitive engagement at scale, testing H1–H4 using Gwet's AC2-validated ensemble scoring, Latent Transition Analysis of ICAP state dynamics, and Spearman correlation of response time with causal reasoning depth.*

**Key references:** Chi & Wylie (2014) ICAP Framework · Gwet (2014) AC2 / Kappa Paradox · Qwen3-VL (2025) · BGE-M3 (2024) · Typhoon 2 (2023) · DSPy (2023) · GraphRAG (2024)
