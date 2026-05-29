"""
main.py  —  FastAPI backend
────────────────────────────
Provides REST + WebSocket endpoints consumed by the Next.js frontend.
This is separate from the Chainlit chatbot and implements the same
TutorEngine logic but exposed via HTTP/WS for the custom React UI.

Endpoints:
  POST   /api/session/new          → create session, return session_id
  WS     /api/chat/{session_id}    → streaming chat (WebSocket)
  GET    /api/session/{session_id}/history   → turn history
  GET    /api/profiles             → all user cognitive profiles (research view)
  GET    /api/metrics/reliability  → inter-rater reliability summary
  GET    /api/metrics/ece          → calibration summary
  POST   /api/analyze              → trigger offline analyzer (admin)
  GET    /api/neuro/taxonomy       → C1–C8 multi-taxonomy data
  GET    /api/neuro/metrics        → current LCAI/CASM/MCI/PCRS metric values
  POST   /api/neuro/code-task      → ICP analysis of submitted code (AST grader)

Run:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    File, UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "configs/model_configs.yaml"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api.main")

app = FastAPI(
    title="PISA Cognitive Tutor API",
    description="Backend for the PISA Cognitive Science research platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton resources ───────────────────────────────────────────────────────
_resources: Dict[str, Any] = {}


async def get_resources() -> Dict[str, Any]:
    global _resources
    if _resources:
        return _resources

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    _resources["cfg"] = cfg

    try:
        from ..retrieval.embedder import BGEEmbedder
        from ..retrieval.hybrid_search import QdrantHybridSearch
        embedder = BGEEmbedder(CONFIG_PATH)
        searcher = QdrantHybridSearch(CONFIG_PATH, embedder)
        _resources["searcher"] = searcher
        logger.info("[API] Retrieval layer loaded")
    except Exception as e:
        logger.warning(f"[API] Retrieval not available: {e}")
        _resources["searcher"] = None

    try:
        from ..ingestion.graph_rag_builder import KnowledgeGraphBuilder
        _resources["graph"] = KnowledgeGraphBuilder.load(CONFIG_PATH)
        logger.info("[API] Graph loaded")
    except Exception as e:
        _resources["graph"] = None

    from ..chatbot.session_manager import SessionManager
    _resources["session_manager"] = await SessionManager.create(CONFIG_PATH)

    from ..chatbot.tutor_engine import TutorEngine
    _resources["engine"] = TutorEngine(
        CONFIG_PATH, _resources["searcher"], _resources["graph"]
    )

    return _resources


# ── Models ────────────────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    user_id: Optional[str] = None
    question_topic: Optional[str] = ""


class NewSessionResponse(BaseModel):
    session_id: str
    user_id: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: List[Dict[str, Any]]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await get_resources()


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/session/new", response_model=NewSessionResponse)
async def new_session(req: NewSessionRequest):
    res = await get_resources()
    user_id = req.user_id or f"student_{uuid.uuid4().hex[:8]}"
    session_id = await res["session_manager"].new_session(
        user_id=user_id,
        question_id=req.question_topic or "",
    )
    logger.info(f"[API] New session: {session_id} / {user_id}")
    return NewSessionResponse(session_id=session_id, user_id=user_id)

@app.post("/api/upload/{session_id}")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Accept a file upload from the student, process it with the VLM/parser,
    and return a file_id to reference in the subsequent WebSocket message.

    Accepts: images (PNG/JPG/WEBP), PDFs, CSV, Excel, plain text.
    Max size: 20 MB (enforce in nginx/reverse proxy for production).
    """
    res = await get_resources()
    cfg = res["cfg"]

    contents = await file.read()

    if len(contents) > 20 * 1024 * 1024:  # 20 MB hard limit
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    from ..chatbot.file_processor import process_upload

    loop = asyncio.get_event_loop()
    try:
        entry = await loop.run_in_executor(
            None,
            process_upload,
            session_id,
            contents,
            file.filename or "upload",
            file.content_type or "application/octet-stream",
            cfg,
        )
    except Exception as e:
        logger.error(f"[API] File processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

    return {
        "file_id": entry["file_id"],
        "filename": entry["filename"],
        "file_type": entry["file_type"],
        # Return a short preview of the description so the UI can show it
        "description_preview": entry["description"][:300] + (
            "…" if len(entry["description"]) > 300 else ""
        ),
    }

@app.get("/api/session/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    res = await get_resources()
    try:
        turns = await res["session_manager"].get_session_turns(session_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return HistoryResponse(session_id=session_id, turns=turns)


@app.websocket("/api/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket chat endpoint.

    Client sends JSON:  {"user_input": "...", "turn_number": 3, "duration_sec": 18.4}
    Server streams:     {"type": "token", "content": "..."}
                        {"type": "done", "kb_used": true}
                        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    res = await get_resources()
    engine = res["engine"]
    session_mgr = res["session_manager"]

    try:
        turns = await session_mgr.get_session_turns(session_id)
    except Exception:
        turns = []

    # Rebuild history from DB
    history = []
    for t in turns[-12:]:
        history.append({"role": "user", "content": t["user_input"]})
        history.append({"role": "assistant", "content": t["ai_response"]})

    last_ai_time = time.perf_counter()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            user_input = data.get("user_input", "").strip()
            turn_number = int(data.get("turn_number", len(turns) + 1))
            duration_sec = float(data.get("duration_sec", time.perf_counter() - last_ai_time))
            file_ids: list = data.get("file_ids", [])
            category: str = data.get("category", "General Science")

            # Build file context to inject (add these lines right after):
            file_context = ""
            if file_ids:
                from ..chatbot.file_processor import get_file_descriptions
                file_context = get_file_descriptions(session_id, file_ids)
                if file_context:
                    logger.info(
                        f"[WS] Injecting {len(file_ids)} file(s) into context "
                        f"for session {session_id[:8]}"
                    )

            if not user_input:
                continue

            full_response = ""
            kb_used = False
            stream_error = None

            try:
                async for token, is_first_kb in engine.respond_stream(
                    user_input, history, turn_number,
                    file_context=file_context,
                    category=category,
                ):
                    full_response += token
                    if is_first_kb:
                        kb_used = True
                    await websocket.send_json({"type": "token", "content": token})
            except Exception as e:
                stream_error = e
                logger.error(f"[WS] Stream error: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "message": str(e)})

            await websocket.send_json({"type": "done", "kb_used": kb_used})

            # Always persist the turn, even if the stream partially failed
            user_id = data.get("user_id", "unknown")
            try:
                await session_mgr.record_turn(
                    session_id=session_id,
                    user_id=user_id,
                    turn_number=turn_number,
                    user_input=user_input,
                    ai_response=full_response,
                    kb_context_used=kb_used,
                    duration_sec=duration_sec,
                )
                logger.info(f"[WS] Turn {turn_number} saved for session {session_id[:8]}")
            except Exception as e:
                logger.error(f"[WS] DB write error: {e}", exc_info=True)

            if stream_error:
                break

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})
            if len(history) > 24:
                history = history[-24:]

            last_ai_time = time.perf_counter()
            turns.append({})  # increment count

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {session_id}")
        from ..chatbot.file_processor import cleanup_session_files
        cleanup_session_files(session_id)


@app.get("/api/profiles")
async def get_profiles():
    """Return all user cognitive profiles (for research dashboard)."""
    res = await get_resources()
    from ..evaluation.user_profiler import UserProfiler
    profiler = await UserProfiler.create(CONFIG_PATH)
    profiles = await profiler.build_all_profiles()
    return {"profiles": profiles}


@app.get("/api/metrics/reliability")
async def get_reliability():
    """Return inter-rater reliability for all dimensions."""
    res = await get_resources()
    db_path = Path(res["cfg"]["session_db"]["path"])
    if not db_path.exists():
        return {"error": "No scored data yet"}

    import sqlite3, json as _json
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT individual_votes_json FROM scores")
    rows = cursor.fetchall()
    conn.close()

    votes_lists = []
    for (vj,) in rows:
        try:
            votes_lists.append(_json.loads(vj))
        except Exception:
            pass

    from ..metrics.reliability import compute_dimension_reliability, interpret_kappa
    rel = compute_dimension_reliability(votes_lists)
    for dim, vals in rel.items():
        vals["interpretation"] = interpret_kappa(vals.get("fleiss_kappa"))
    return {"reliability": rel, "n_turns": len(votes_lists)}


@app.get("/api/metrics/ece")
async def get_ece():
    res = await get_resources()
    db_path = Path(res["cfg"]["session_db"]["path"])
    from ..metrics.calibration import compute_ece_from_scores_db
    result = compute_ece_from_scores_db(db_path)
    return result


@app.post("/api/analyze")
async def trigger_analyze(background_tasks: BackgroundTasks):
    """Trigger offline cognitive assessment (runs in background)."""
    async def _run():
        from ..evaluation.analyzer_pipeline import AnalyzerPipeline
        pipeline = AnalyzerPipeline(CONFIG_PATH)
        await pipeline.run(batch_size=200)

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Analyzer pipeline running in background"}


@app.get("/api/scores/export")
async def export_scores():
    """Return recent scored turns as JSON for the dashboard."""
    res = await get_resources()
    db_path = Path(res["cfg"]["session_db"]["path"])
    if not db_path.exists():
        return {"scores": []}
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT * FROM scores ORDER BY id DESC LIMIT 500")
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    conn.close()
    return {"scores": [dict(zip(cols, r)) for r in rows]}


# ── Neuro-Software Engineering Endpoints ─────────────────────────────────────

@app.get("/api/neuro/taxonomy")
async def get_neuro_taxonomy():
    """
    Return the full C1–C8 multi-taxonomy dataset.
    Maps each cognitive dimension to its neuro-structural ALE cluster,
    somatosensory cervical segment, PSYDEFCONV defense level,
    clinical criteria, and activity reserve classification.
    """
    from ..metrics.neuro_metrics import NEURO_TAXONOMY
    return {"taxonomy": NEURO_TAXONOMY, "n_dimensions": len(NEURO_TAXONOMY)}


class NeuroMetricsRequest(BaseModel):
    """Optional override parameters for metric computation."""
    theta_a: float = 0.7         # Age-bracketed white-matter integration
    kede: float = 20.0           # Knowledge Discovery Efficiency
    tau: float = 8.5             # Fixation-to-completion latency (seconds)
    icp_sum: float = 3.0         # Total ICP across active AST nodes
    lf_hf_ratio: float = 2.1     # HRV spectral ratio
    rmssd: float = 42.0          # Root mean square successive differences (ms)
    emg_cervical_delta: float = 0.15  # Normalized trapezius EMG
    icp_active: float = 3.0      # Active code block ICP
    adaptive_freqs: List[float] = [0.6, 0.5, 0.4, 0.3]
    defensive_freqs: List[float] = [0.2, 0.1, 0.1, 0.05]
    dmrs_weights: Optional[List[float]] = None
    perplexity: float = 0.3
    c1_inhibitor: float = 1.25   # Normalized CSF C1-esterase inhibitor (µg/mL)
    stie1: float = 0.85          # sTie-1 concentration (µg/mL)
    kede_observed: float = 20.0
    kede_baseline: float = 20.0


@app.get("/api/neuro/metrics")
async def get_neuro_metrics():
    """
    Return current computed values of the four research metrics:
    LCAI, CASM, MCI, PCRS — using default/simulated parameters.
    """
    from ..metrics.neuro_metrics import (
        compute_lcai, compute_casm, compute_mci, compute_pcrs
    )
    lcai = compute_lcai(theta_a=0.7, kede=20.0, tau=8.5, icp_sum=3.0)
    casm = compute_casm(lf_hf_ratio=2.1, rmssd=42.0, emg_cervical_delta=0.15, icp_active=3.0)
    mci  = compute_mci(
        adaptive_freqs=[0.6, 0.5, 0.4, 0.3],
        defensive_freqs=[0.2, 0.1, 0.1, 0.05],
        perplexity=0.3
    )
    pcrs = compute_pcrs(c1_inhibitor=1.25, stie1=0.85, kede_observed=20.0, casm=casm)

    return {
        "LCAI": round(lcai, 4),
        "CASM": round(casm, 4),
        "MCI":  round(mci, 4),
        "PCRS": round(pcrs, 4),
        "note": "Computed using default/simulated telemetry parameters"
    }


class CodeTaskRequest(BaseModel):
    task_id: int         # 1, 2, or 3
    code: str            # User-submitted code to analyze
    language: str = "python"


@app.post("/api/neuro/code-task")
async def analyze_code_task(req: CodeTaskRequest):
    """
    Accept user-submitted code for one of the three cognitive assessment tasks.
    Performs:
      1. ICP heuristic analysis (AST-based complexity scoring)
      2. Comparison against expected ICP reduction targets
      3. Returns feedback and cognitive dimension assessment
    """
    from ..metrics.neuro_metrics import estimate_icp_from_code

    task_targets = {
        1: {"name": "Recursive State Refactoring", "target_icp": 1, "dims": ["C6", "C4"]},
        2: {"name": "Distractor-Heavy Debugging",   "target_icp": 0, "dims": ["C1", "C2"]},
        3: {"name": "Algorithmic Pattern Synthesis","target_icp": 1, "dims": ["C8", "C7"]},
    }

    if req.task_id not in task_targets:
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")

    target = task_targets[req.task_id]
    icp_result = estimate_icp_from_code(req.code)
    total_icp = icp_result["total_icp"]
    target_icp = target["target_icp"]

    achieved = total_icp <= target_icp
    feedback = (
        f"✓ ICP = {total_icp} meets target (≤{target_icp}). Working memory load reduced."
        if achieved else
        f"⚠ ICP = {total_icp} exceeds target ({target_icp}). "
        f"Consider flattening control flow and reducing nesting."
    )

    return {
        "task_id": req.task_id,
        "task_name": target["name"],
        "target_dimensions": target["dims"],
        "icp_analysis": icp_result,
        "target_icp": target_icp,
        "achieved": achieved,
        "feedback": feedback,
    }
