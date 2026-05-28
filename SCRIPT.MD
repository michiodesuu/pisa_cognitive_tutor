Complete Runbook — PISA Cognitive Tutor
0. Prerequisites (install once)
0-A. Python environment

cd C:\Users\Asus\Downloads\pisa_cognitive_tutor

# Install PyTorch with CUDA first (match your CUDA version — example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies
pip install -r requirements.txt

# Optional: Flash Attention for faster VLM inference (needs CUDA compiler)
pip install flash-attn --no-build-isolation
0-B. Ollama — install and pull all 4 LLM models

# Install Ollama (Windows)
# Download and run the installer from https://ollama.com/download

# Start the Ollama daemon (keep this terminal open throughout all steps)
ollama serve

# Pull all 4 evaluator models + the chat model (in a NEW terminal)
ollama pull qwen3:8b
ollama pull mistral:7b-instruct-v0.2-q8_0
ollama pull llama3.1:8b-instruct-q8_0
ollama pull scb10x/llama3.1-typhoon2-8b-instruct

# Verify all 5 are available
ollama list
0-C. Download HuggingFace models (VLM + embedder)

# Create local model directories
mkdir -p models/Qwen3-VL-8B-Instruct models/bge-m3

# Vision Language Model for ingestion (~16 GB, quantized to ~5 GB at runtime)
hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir ./models/Qwen3-VL-8B-Instruct

# BGE-M3 embedding model (~570 MB)
hf download BAAI/bge-m3 \
    --local-dir ./models/bge-m3
0-D. Node.js frontend dependencies (install once)

cd frontend
npm install
cd ..
1. Ingestion — extract PISA knowledge from PDFs
Prerequisite: Ollama must be running (ollama serve).


# Put your PISA PDF files here (Thai, English, Chinese — any mix)
ls data/raw_pdfs/
# Expected: science_th.pdf, science_en.pdf, science_zh.pdf, etc.

# Run the full ingestion pipeline
python -m src.ingestion.run_ingestion --config configs/model_configs.yaml
What this does:

Discovers all PDFs in data/raw_pdfs/
Renders each page to 200 DPI images (PyMuPDF)
Groups into 3-page shards
Sends each shard to Qwen3-VL-8B (4-bit) → extracts structured JSON (topic, competency, misconceptions, vocabulary)
Appends records to data/processed_jsonl/knowledge_base.jsonl
Builds a NetworkX knowledge graph → saves to data/vector_db/knowledge_graph.gpickle
Embeds all records via BGE-M3 (dense + sparse) → upserts into Qdrant at data/vector_db/qdrant/
Expected duration: ~30–60 min per 100 PDF pages on RTX 4090.

Resume-safe: If it crashes, rerun the same command — already-processed PDFs are skipped automatically.

Outputs:


data/processed_jsonl/knowledge_base.jsonl    ← ~340 records
data/vector_db/qdrant/                       ← Qdrant local DB
data/vector_db/knowledge_graph.gpickle      ← NetworkX graph
ingestion.log                                ← full log
2. Tutoring sessions — collect student data
Requires two terminals running simultaneously.


# Terminal B — FastAPI backend (keep running during all sessions)
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal C — Next.js frontend
npm run dev --prefix frontend
Open http://localhost:3000 in a browser. Conduct sessions with students (up to 12 turns each).

Take screenshot now for Figure 3 (Next.js chat UI). Save as figures/nextjs_screenshot.png.

What gets recorded per turn:

user_input — raw student message
ai_response — Socratic tutor response (sanitized)
duration_sec — seconds since last AI message
kb_context_used — whether Qdrant context was retrieved
Output:


data/chat_logs/sessions.db   ← sessions + turns tables populated
3. Evaluation — score all collected turns
Run this after sessions are complete. Ollama must still be running.


python -m src.evaluation.analyzer_pipeline \
    --config configs/model_configs.yaml \
    --batch 100
What this does:

Finds all turns with no entry in the scores table (LEFT JOIN query)
Retrieves PISA context from Qdrant for each turn (top-3 results)
Sends each turn to all 4 LLMs in parallel (ThreadPoolExecutor)
Applies kappa-weighted majority voting → produces C1–C8 scores + dominant ICAP level
Writes results to the scores table in SQLite
Exports data/chat_logs/scores_export.csv
Expected duration: ~18 seconds per turn × number of turns.

Outputs:


data/chat_logs/sessions.db          ← scores table now populated
data/chat_logs/scores_export.csv    ← flat CSV for R/SPSS/Stata
Verify it worked:


python -c "
import sqlite3
conn = sqlite3.connect('data/chat_logs/sessions.db')
turns  = conn.execute('SELECT COUNT(*) FROM turns').fetchone()[0]
scores = conn.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
print(f'Turns: {turns}  |  Scored: {scores}')
conn.close()
"
4. Generate all 7 figures automatically
Run this after the evaluation step has populated the scores table.


python -c "
from pathlib import Path
from src.metrics.visualizations import run_full_report
run_full_report(
    db_path=Path('data/chat_logs/sessions.db'),
    output_dir=Path('data/reports'),
)
"
What gets saved to data/reports/:

File	Figure in paper	Function
ac2_comparison.png	Fig 5	plot_ac2_comparison()
trajectory.png	Fig 6	plot_engagement_trajectory()
ac2_radar.png	Fig 7	plot_ac2_radar()
lta_transitions.png	Fig 8	plot_latent_transition_heatmap()
duration_scatter.png	Fig 9	plot_duration_vs_depth()
icap_distribution.png	Fig 10	plot_icap_distribution()
heatmap.png	Fig 11	plot_dimension_correlation_heatmap()
reliability_table.tex	—	export_latex_table()
5. Research dashboard — screenshot for Figure 4
The research dashboard is part of the Next.js frontend. Keep the FastAPI backend and Next.js frontend running (from Step 2), then open:

http://localhost:3000/dashboard

Take screenshot for Figure 4 (research dashboard). Save as figures/dashboard_screenshot.png.

6. Manual figures (Figs 1 & 2) — draw by hand
These two cannot be generated by code. Use draw.io, Lucidchart, or PowerPoint.

Figure 1 — System Architecture (Fig 1, full-width):
Use the ASCII art in REPORT.MD Section 3.3.1 as blueprint. Three boxes flowing top-to-bottom: Ingestion → Tutoring → Evaluation, with the data stores (Qdrant, SQLite, NetworkX) shown between them.

Figure 2 — Hybrid Retrieval Pipeline (Fig 2, single column):
Simple left-to-right flow: Query → BGE-M3 → two branches (Qdrant dense and Qdrant sparse) → RRF fusion → Top-5 results.

Save both as figures/architecture.png and figures/retrieval_pipeline.png.

7. Upload to Overleaf and replace placeholders

figures/
  architecture.png          ← Fig 1  (manual)
  retrieval_pipeline.png    ← Fig 2  (manual)
  nextjs_screenshot.png     ← Fig 3  (screenshot)
  dashboard_screenshot.png  ← Fig 4  (screenshot)
  ac2_comparison.png        ← Fig 5  (data/reports/)
  trajectory.png            ← Fig 6  (data/reports/)
  ac2_radar.png             ← Fig 7  (data/reports/)
  lta_transitions.png       ← Fig 8  (data/reports/)
  duration_scatter.png      ← Fig 9  (data/reports/)
  icap_distribution.png     ← Fig 10 (data/reports/)
  heatmap.png               ← Fig 11 (data/reports/)
For each placeholder in REPORT.tex, replace the \imgplaceholder{...}{label} block with:


\begin{figure}[H]
  \centering
  \includegraphics[width=\columnwidth]{figures/FILENAME.png}
  \caption{Your caption}
  \label{fig:LABEL}
\end{figure}
For Figure 1 only, use figure* (spans both columns):


\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/architecture.png}
  \caption{Three-Phase System Architecture}
  \label{fig:architecture}
\end{figure*}
Summary — execution order

Terminal A (always running):   ollama serve

Terminal B — one-time setup:
  pip install torch ...
  pip install -r requirements.txt
  huggingface-cli download Qwen/Qwen3-VL-8B-Instruct ...
  huggingface-cli download BAAI/bge-m3 ...
  ollama pull qwen3:8b
  ollama pull mistral:7b-instruct-v0.2-q8_0
  ollama pull llama3.1:8b-instruct-q8_0
  ollama pull scb10x/llama3.1-typhoon2-8b-instruct
  cd frontend && npm install && cd ..

Terminal B — pipeline:
  ① python -m src.ingestion.run_ingestion          ← fills Qdrant + graph
  ③ python -m src.evaluation.analyzer_pipeline     ← score all turns
  ④ python -c "from src.metrics.visualizations import run_full_report; ..."
     → data/reports/*.png                           ← Figs 5–11

Terminal C — FastAPI backend (keep running during sessions):
  ② uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload

Terminal D — Next.js frontend (keep running during sessions):
  ② npm run dev --prefix frontend
     [screenshot localhost:3000       → figures/nextjs_screenshot.png]
     [screenshot localhost:3000/dashboard → figures/dashboard_screenshot.png]

Manual:
  ⑤ Draw Fig 1 (architecture) + Fig 2 (retrieval) → figures/*.png

Overleaf:
  ⑥ Upload figures/ folder → replace \imgplaceholder blocks → compile
