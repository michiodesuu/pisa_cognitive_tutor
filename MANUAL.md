# PISA Cognitive Tutor v2.0 — Operator & Researcher Manual

> **Platform:** NECTEC — National Electronics and Computer Technology Center  
> **Version:** 2.0 | **Compiler:** XeLaTeX (report) | **Runtime:** Ubuntu 24.04 + RTX 4090

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [One-Time Installation](#2-one-time-installation)
3. [Configuration Reference](#3-configuration-reference)
4. [Running the Full Pipeline](#4-running-the-full-pipeline)
   - [Step 1 — Ingestion](#step-1--ingestion)
   - [Step 2 — Tutoring Sessions](#step-2--tutoring-sessions)
   - [Step 3 — Offline Evaluation](#step-3--offline-evaluation)
   - [Step 4 — Generate Figures & Tables](#step-4--generate-figures--tables)
5. [Subject Category Filtering](#5-subject-category-filtering)
6. [Ablation Study Setup](#6-ablation-study-setup)
7. [API Reference](#7-api-reference)
8. [Overleaf / Report Workflow](#8-overleaf--report-workflow)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24 GB) | RTX 4090 (24 GB) |
| RAM | 32 GB | 64 GB DDR5 |
| Storage | 200 GB NVMe | 500 GB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 24.04 |
| CUDA | 12.0 | 12.1+ |
| Python | 3.10 | 3.11 |

VRAM budget at peak (all models loaded simultaneously):

| Model | VRAM |
|-------|------|
| Qwen3-VL-8B (4-bit, ingestion only) | ~5 GB |
| BGE-M3 (embedding, CPU) | 0 GB GPU |
| Qwen3:8b via Ollama (chat) | ~6 GB |
| 4× evaluator LLMs via Ollama (eval, sequential) | ~6 GB each |

Ingestion and evaluation never run at the same time as the live chat server. Total peak
during tutoring: ~6 GB (Ollama chat model). Total peak during evaluation: ~6 GB
(one evaluator at a time, sequentially via Ollama).

---

## 2. One-Time Installation

### 2-A. Python environment

```bash
cd pisa_cognitive_tutor

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install PyTorch first (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies
pip install -r requirements.txt
```

### 2-B. Ollama — LLM inference runtime

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start the daemon (keep this terminal open throughout all steps)
ollama serve

# In a new terminal — pull all 5 models (chat + 4 evaluators)
ollama pull qwen3:8b
ollama pull mistral:7b-instruct-v0.2-q8_0
ollama pull llama3.1:8b-instruct-q8_0
ollama pull scb10x/llama3.1-typhoon2-8b-instruct
```

Verify all are available:

```bash
ollama list
# Expected: qwen3:8b, mistral:7b-..., llama3.1:8b-..., typhoon2-8b-...
```

### 2-C. HuggingFace models

```bash
pip install huggingface_hub

# Vision Language Model for ingestion (~16 GB download, ~5 GB at runtime with 4-bit)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir ./models/Qwen3-VL-8B-Instruct

# BGE-M3 embedding model (~570 MB, runs on CPU)
huggingface-cli download BAAI/bge-m3 \
    --local-dir ./models/bge-m3
```

### 2-D. Frontend

```bash
cd frontend
npm install
cd ..
```

---

## 3. Configuration Reference

All runtime parameters live in `configs/model_configs.yaml`. The file is organized into
these sections:

### `vlm` — Vision Language Model (ingestion only)

| Key | Default | Notes |
|-----|---------|-------|
| `model_id` | `Qwen/Qwen3-VL-8B-Instruct` | Change only if using a different VLM |
| `load_in_4bit` | `true` | Keep `true` unless you have >16 GB VRAM free |
| `pdf_render_dpi` | `200` | Increase to `300` for scanned PDFs with small text |
| `max_pages_per_shard` | `3` | Reduce to `2` if VLM produces incomplete extractions |

### `embedding` — BGE-M3

| Key | Default | Notes |
|-----|---------|-------|
| `device` | `cpu` | Set to `cuda:0` for faster ingestion if VRAM permits |
| `batch_size` | `32` | Reduce if CPU RAM is under 16 GB |

### `chatbot_llm` — Socratic tutor

| Key | Default | Notes |
|-----|---------|-------|
| `model_name` | `qwen3:8b` | Any Ollama model tag works here |
| `temperature` | `0.7` | Lower (0.4) for more focused Socratic questions |
| `max_tokens` | `1024` | Increase only if responses are being cut off |

### `evaluator_ensemble` — offline scoring

| Key | Default | Notes |
|-----|---------|-------|
| `models` | 4 models listed | Remove a model entry to run a 3-model ensemble |
| `consensus_threshold` | `0.75` | Fraction required for non-controversial verdict |
| `timeout_seconds` | `120` | Increase on slow GPUs |

### `qdrant`

| Key | Default | Notes |
|-----|---------|-------|
| `mode` | `local` | Change to `server` and set `host`/`port` for Docker Qdrant |
| `top_k` | `8` | Number of results returned before RRF re-ranking |
| `score_threshold` | `0.55` | Lower to get more results; raise to filter noise |

### `ablation` — per-system defaults (overridable per session)

| Flag | Default | Effect when `false` |
|------|---------|---------------------|
| `use_rag` | `true` | Pure LLM baseline — no retrieval at all |
| `use_hybrid_search` | `true` | Dense-only (no sparse BM25 leg) |
| `use_knowledge_graph` | `true` | Skip 2-hop graph subgraph enrichment |
| `use_sanitizer` | `true` | Skip forbidden-word post-processing |
| `use_kappa_weighting` | `true` | Equal model weights in ensemble vote |

These are system-wide defaults. Individual sessions can override them — see
[Section 6](#6-ablation-study-setup).

### `subject_categories`

Three built-in categories. Each has an `id` (stored in Qdrant payload and session DB),
a `label`, `label_th` (Thai), and `description`. The ingestion pipeline uses the `id`
value from each knowledge base record's `subject_category` field to tag records so they
can be filtered at search time.

---

## 4. Running the Full Pipeline

You need four terminal windows. Terminal A must stay open throughout everything.

```
Terminal A  →  ollama serve                    (always running)
Terminal B  →  pipeline commands               (ingestion, evaluation, reports)
Terminal C  →  uvicorn (FastAPI backend)        (running during sessions)
Terminal D  →  npm run dev (Next.js frontend)   (running during sessions)
```

### Step 1 — Ingestion

**Prerequisites:** Ollama running. PDF files placed in `data/raw_pdfs/`.

```bash
python -m src.ingestion.run_ingestion --config configs/model_configs.yaml
```

What happens:
1. All PDFs in `data/raw_pdfs/` are discovered.
2. Each PDF is rendered to 200 DPI page images (PyMuPDF).
3. Pages are grouped into 3-page shards.
4. Each shard is sent to Qwen3-VL-8B → extracts structured JSON:
   `topic`, `question_summary`, `correct_concept`, `common_misconceptions`,
   `key_vocabulary`, `pisa_competency`, `subject_category`.
5. Records appended to `data/processed_jsonl/knowledge_base.jsonl`.
6. BGE-M3 encodes all records (dense + sparse vectors in one pass).
7. Records upserted to Qdrant at `data/vector_db/qdrant/`.
8. NetworkX knowledge graph built → `data/vector_db/knowledge_graph.gpickle`.

Expected duration: ~30–60 min per 100 PDF pages on RTX 4090.

The process is **resume-safe**: if interrupted, rerun the same command. Already-processed
PDFs are detected via `source_file` field in `knowledge_base.jsonl` and skipped.

Outputs:
```
data/processed_jsonl/knowledge_base.jsonl   (~340 records)
data/vector_db/qdrant/                      (Qdrant local DB)
data/vector_db/knowledge_graph.gpickle      (NetworkX graph)
```

### Step 2 — Tutoring Sessions

Start the backend and frontend (Terminal C and D):

```bash
# Terminal C
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal D
npm run dev --prefix frontend
```

Open `http://localhost:3000` in a browser. Each student session:
- Is limited to 12 turns.
- Records `user_input`, `ai_response`, `duration_sec`, `kb_context_used` per turn.
- Is associated with a `subject_category` and `ablation_config` (set at session creation).

To create a session programmatically (e.g., from a research script):

```bash
curl -X POST http://localhost:8001/api/session/new \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_01", "subject_category": "life_science"}'
```

The response includes `session_id`, `subject_category`, and `ablation_config` (the
resolved flags for that session).

### Step 3 — Offline Evaluation

Run **after** sessions are complete. Ollama must be running.

```bash
python -m src.evaluation.analyzer_pipeline \
    --config configs/model_configs.yaml \
    --batch 100
```

What happens:
1. All turns with no entry in the `scores` table are fetched (LEFT JOIN).
2. Qdrant retrieves top-3 context records per turn.
3. All 4 LLMs score the turn in parallel (ThreadPoolExecutor).
4. Kappa-weighted majority voting produces C1–C8 scores + dominant ICAP level.
5. Scores written to SQLite `scores` table.
6. `data/chat_logs/scores_export.csv` is updated.

Expected duration: ~18 seconds per turn.

Verify it worked:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/chat_logs/sessions.db')
turns  = conn.execute('SELECT COUNT(*) FROM turns').fetchone()[0]
scores = conn.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
print(f'Turns: {turns}  |  Scored: {scores}')
conn.close()
"
```

### Step 4 — Generate Figures & Tables

Run after evaluation. Requires a populated `scores` table.

```bash
python -c "
from pathlib import Path
from src.metrics.visualizations import run_full_report
run_full_report(
    db_path=Path('data/chat_logs/sessions.db'),
    output_dir=Path('data/reports'),
)
"
```

Outputs saved to `data/reports/`:

| File | Figure | Generator function |
|------|--------|--------------------|
| `ac2_comparison.png` | Fig 5 | `plot_ac2_comparison()` |
| `trajectory.png` | Fig 6 | `plot_engagement_trajectory()` |
| `ac2_radar.png` | Fig 7 | `plot_ac2_radar()` |
| `lta_transitions.png` | Fig 8 | `plot_latent_transition_heatmap()` |
| `duration_scatter.png` | Fig 9 | `plot_duration_vs_depth()` |
| `icap_distribution.png` | Fig 10 | `plot_icap_distribution()` |
| `heatmap.png` | Fig 11 | `plot_dimension_correlation_heatmap()` |
| `reliability_table.tex` | — | `export_latex_table()` |

Upload all files from `data/reports/` plus `data/reports/reliability_table.tex`
into the `figures/` folder of your Overleaf project.

---

## 5. Subject Category Filtering

Students sometimes receive PISA questions outside their area of competence (e.g., a
student strong in biology receives a chemistry question). Subject category filtering
lets you restrict the Qdrant knowledge base retrieval to a specific PISA domain.

### Available categories

| `id` | English label | Thai label |
|------|--------------|-----------|
| `life_science` | Life Science | ชีววิทยา |
| `physical_science` | Physical Science | วิทยาศาสตร์กายภาพ |
| `earth_space` | Earth & Space Science | โลกและอวกาศ |

### How it works

When a session is created with a `subject_category`, every Qdrant search for that session
adds a payload filter: `subject_category == <id>`. Only knowledge base records tagged with
that category are returned as context. The LLM still has full reasoning capability — only
the retrieved context is filtered.

### Setting a category via API

```bash
curl -X POST http://localhost:8001/api/session/new \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_07", "subject_category": "physical_science"}'
```

### Getting the category list for the UI

```bash
curl http://localhost:8001/api/categories
# Returns: {"categories": [{"id": "life_science", "label": "Life Science", ...}, ...]}
```

### Tagging records during ingestion

The ingestion VLM prompt instructs Qwen3-VL-8B to include a `subject_category` field
in each extracted JSON record. If the model does not produce one, the field defaults to
`""` (no filter applied). You can also manually tag records in
`data/processed_jsonl/knowledge_base.jsonl` before running ingestion, using the
`id` values above.

---

## 6. Ablation Study Setup

The ablation framework lets you compare experimental conditions on the same platform
simultaneously. Each session has an immutable `ablation_config` JSON blob stored in
the `sessions` table, so results from different groups can always be traced back to
their exact configuration.

### The five flags

| Flag | What it disables | Baseline purpose |
|------|-----------------|-----------------|
| `use_rag: false` | All Qdrant retrieval | Pure LLM baseline |
| `use_hybrid_search: false` | Sparse BM25 leg only | Dense-only retrieval |
| `use_knowledge_graph: false` | 2-hop graph enrichment | Vector-only retrieval |
| `use_sanitizer: false` | Forbidden-word filter | Measure slip rate without guard |
| `use_kappa_weighting: false` | Adaptive model weights | Equal-weight ensemble |

### Creating experimental groups

**Group A — Full system (control):**

```bash
curl -X POST http://localhost:8001/api/session/new \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_A1"}'
# ablation_config defaults to all flags true
```

**Group B — No RAG baseline:**

```bash
curl -X POST http://localhost:8001/api/session/new \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_B1", "ablation": {"use_rag": false}}'
```

**Group C — Dense-only retrieval:**

```bash
curl -X POST http://localhost:8001/api/session/new \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_C1", "ablation": {"use_hybrid_search": false}}'
```

**Group D — No graph enrichment:**

```bash
curl -X POST http://localhost:8001/api/session/new \
  -H "Content-Type: application/json" \
  -d '{"user_id": "student_D1", "ablation": {"use_knowledge_graph": false}}'
```

### Querying results by group

After evaluation, compare groups in SQLite:

```sql
-- Average C1 score per ablation group
SELECT
    json_extract(s.ablation_config, '$.use_rag')            AS use_rag,
    json_extract(s.ablation_config, '$.use_hybrid_search')  AS use_hybrid,
    COUNT(sc.id)                                             AS n_turns,
    ROUND(AVG(CASE sc.C1_dominant WHEN '2' THEN 2
                                   WHEN '1' THEN 1
                                   ELSE 0 END), 3)          AS avg_C1
FROM sessions s
JOIN turns t   ON t.session_id = s.session_id
JOIN scores sc ON sc.turn_id   = t.id
GROUP BY use_rag, use_hybrid;
```

Or export to CSV for R/Python analysis:

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("data/chat_logs/sessions.db")
df = pd.read_sql("""
    SELECT s.ablation_config, sc.*
    FROM sessions s
    JOIN turns  t  ON t.session_id = s.session_id
    JOIN scores sc ON sc.turn_id   = t.id
""", conn)
conn.close()

df["use_rag"] = df["ablation_config"].apply(
    lambda x: pd.json_normalize(pd.read_json(x, typ="series").to_dict())["use_rag"][0]
)
df.to_csv("data/chat_logs/ablation_export.csv", index=False)
```

---

## 7. API Reference

Base URL: `http://localhost:8001`

### Session management

| Method | Path | Body / Params | Description |
|--------|------|---------------|-------------|
| `POST` | `/api/session/new` | `{user_id?, question_topic?, subject_category?, ablation?}` | Create session, returns `session_id` + resolved `ablation_config` |
| `GET` | `/api/session/{id}/history` | — | All turns for a session |
| `GET` | `/api/categories` | — | Available subject categories |

### Chat

| Method | Path | Protocol | Description |
|--------|------|----------|-------------|
| `WS` | `/api/chat/{session_id}` | WebSocket | Streaming chat. Send `{"user_input": "...", "turn_number": N, "duration_sec": X, "file_ids": [...]}` |

### Research / metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/profiles` | All user cognitive profiles |
| `GET` | `/api/metrics/reliability` | Fleiss κ, AC2, Krippendorff α per dimension |
| `GET` | `/api/metrics/ece` | Expected Calibration Error |
| `POST` | `/api/analyze` | Trigger offline evaluation pipeline (admin) |

### Neuro-metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/neuro/taxonomy` | C1–C8 multi-taxonomy data |
| `GET` | `/api/neuro/metrics` | Current LCAI/CASM/MCI/PCRS values |
| `POST` | `/api/neuro/code-task` | ICP analysis of submitted code (AST grader) |

---

## 8. Overleaf / Report Workflow

### Files to upload to Overleaf `figures/` folder

| Local path | Overleaf path | Status |
|-----------|--------------|--------|
| `data/reports/ac2_comparison.png` | `figures/ac2_comparison.png` | Auto-generated |
| `data/reports/trajectory.png` | `figures/trajectory.png` | Auto-generated |
| `data/reports/ac2_radar.png` | `figures/ac2_radar.png` | Auto-generated |
| `data/reports/lta_transitions.png` | `figures/lta_transitions.png` | Auto-generated |
| `data/reports/duration_scatter.png` | `figures/duration_scatter.png` | Auto-generated |
| `data/reports/icap_distribution.png` | `figures/icap_distribution.png` | Auto-generated |
| `data/reports/heatmap.png` | `figures/heatmap.png` | Auto-generated |
| `data/reports/reliability_table.tex` | `figures/reliability_table.tex` | Auto-generated |
| *(draw manually)* | `figures/architecture.png` | Manual — 3-phase diagram |
| *(draw manually)* | `figures/retrieval_pipeline.png` | Manual — hybrid retrieval flow |
| *(screenshot)* | `figures/nextjs_screenshot.png` | Screenshot of `localhost:3000` |
| *(screenshot)* | `figures/dashboard_screenshot.png` | Screenshot of `localhost:3000/dashboard` |

### Remaining placeholders in REPORT.tex

Two `\imgplaceholderwide` / `\imgplaceholder` blocks remain (no PNG yet):

| Label | Expected file | How to create |
|-------|--------------|---------------|
| `fig:architecture` | `figures/architecture.png` | draw.io / Lucidchart — 3 boxes: Ingestion → Tutoring → Evaluation, data stores between them |
| `fig:retrieval` | `figures/retrieval_pipeline.png` | draw.io — Query → BGE-M3 → Qdrant dense + sparse → RRF → Top-5 |
| `fig:nextjs-ui` | `figures/nextjs_screenshot.png` | Screenshot of chat UI |
| `fig:nextjs-dashboard` | `figures/dashboard_screenshot.png` | Screenshot of research dashboard |

Once you have a PNG, replace its `\imgplaceholder{...}{label}` block with:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=\columnwidth]{figures/FILENAME.png}
    \caption{Your caption}
    \label{fig:LABEL}
\end{figure}
```

For `fig:architecture` only (full-width, two-column span), use `figure*`:

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/architecture.png}
    \caption{Three-Phase System Architecture: Ingestion → Tutoring → Evaluation}
    \label{fig:architecture}
\end{figure*}
```

### Compiling

REPORT.tex requires **XeLaTeX** (for Thai/Chinese font support). In Overleaf:
`Menu → Compiler → XeLaTeX`.

---

## 9. Troubleshooting

### Ollama is not responding

```bash
# Check if the daemon is running
curl http://localhost:11434/api/tags

# If not, restart
pkill ollama
ollama serve &
```

### Qdrant collection not found

The collection is created automatically on first ingest. If you deleted the collection
manually or moved the `data/vector_db/qdrant/` folder:

```bash
# Re-run ingestion (safe — will re-create and re-populate the collection)
python -m src.ingestion.run_ingestion --config configs/model_configs.yaml
```

### Sessions table is missing `subject_category` / `ablation_config` columns

The migration runs automatically on every startup of `SessionManager`. If you see
column-not-found errors in the logs, trigger the migration manually:

```bash
python -c "
import asyncio
from src.chatbot.session_manager import SessionManager
from pathlib import Path
asyncio.run(SessionManager.create(Path('configs/model_configs.yaml')))
print('Migration complete')
"
```

### Evaluation produces all-NA scores for a dimension

This usually means the LLM did not return a valid JSON score for that dimension. Check
`data/chat_logs/scores_export.csv` — if `C7_dominant` is always empty, it means
the Systems/Transfer dimension (C7) is producing NA votes from all four models. This is
a known limitation for highly abstract questions. The `reliability_table.tex` will show
`undefined` for AC2 on that dimension.

### Frontend cannot reach the backend

Verify CORS: the backend allows `http://localhost:3000` by default. If you run the
frontend on a different port, update `allow_origins` in `src/api/main.py`.

### `huggingface-cli download` times out

Use `--resume-download` to continue an interrupted download:

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir ./models/Qwen3-VL-8B-Instruct \
    --resume-download
```

### XeLaTeX fails on Thai characters in Overleaf

Ensure the compiler is set to XeLaTeX (not pdfLaTeX). The `polyglossia` package at
the top of `REPORT.tex` requires XeLaTeX. If you must use pdfLaTeX, uncomment the two
`inputenc`/`fontenc` lines in the preamble and comment out the `polyglossia` block.

---

*For architecture details, see `README(architecture).MD`.
For the step-by-step runbook, see `SCRIPT.MD`.
For the full academic report source, see `REPORT.tex`.*
