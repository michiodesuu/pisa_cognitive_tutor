"""
app.py  ─  Chainlit chatbot UI
───────────────────────────────
The PISA Cognitive Tutor interactive interface.

Architecture:
  - Chainlit handles WebSocket streaming UI
  - TutorEngine retrieves context and generates Socratic responses
  - SessionManager persists all turns with timestamps and duration
  - NO cognitive assessment happens here (strict separation per NECTEC requirement)

Run:
  chainlit run src/chatbot/app.py --port 8000

Environment variables:
  CONFIG_PATH  (default: configs/model_configs.yaml)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import chainlit as cl
import httpx
import yaml

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "configs/model_configs.yaml"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("chatbot.app")

# ── Global singletons ─────────────────────────────────────────────────────────
_searcher = None
_graph = None
_session_manager = None
_cfg = None
# Sentinel: True = init attempted (success or failure). Prevents retrying the
# slow BGE-M3 load on every session when initialization previously failed.
_heavy_init_done = False


@cl.on_chat_start
async def on_chat_start():
    """Initialise resources for this chat session."""
    global _searcher, _graph, _session_manager, _cfg, _heavy_init_done

    if _cfg is None:
        _cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    # ── Session manager (fast — just SQLite) ──────────────────────────────
    if _session_manager is None:
        from .session_manager import SessionManager
        _session_manager = await SessionManager.create(CONFIG_PATH)
        logger.info("[App] Session manager initialized")

    user_id = f"student_{uuid.uuid4().hex[:8]}"
    session_id = await _session_manager.new_session(user_id=user_id)

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("turn_number", 0)
    cl.user_session.set("last_ai_time", time.perf_counter())
    cl.user_session.set("history", [])

    # ── Welcome message — shown immediately, before any slow init ─────────
    res = await cl.AskActionMessage(
        content="Please select a subject category you want to explore:",
        actions=[
            cl.Action(name="General Science", value="General Science", label="General Science"),
            cl.Action(name="Biology", value="Biology", label="Biology"),
            cl.Action(name="Chemistry", value="Chemistry", label="Chemistry"),
            cl.Action(name="Physics", value="Physics", label="Physics"),
            cl.Action(name="Earth Science", value="Earth Science", label="Earth Science"),
        ],
        timeout=120
    ).send()

    category = "General Science"
    if res and res.get("value"):
        category = res.get("value")
    cl.user_session.set("category", category)

    await cl.Message(
        content=(
            f"👋 **Welcome to the PISA Science Tutor ({category})**\n\n"
            "I am here to help you think through science questions step by step.\n"
            "You can type in **Thai, English, or Chinese** — whichever is most "
            "comfortable for you.\n\n"
            "Whenever you are ready, describe the science question or topic "
            "you would like to explore, and we will work through it together. 🔬"
        ),
        author="Tutor",
    ).send()

    # ── Ollama health check (fast) ────────────────────────────────────────
    ollama_url = _cfg["chatbot_llm"]["ollama_base_url"]
    ollama_model = _cfg["chatbot_llm"]["model_name"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as _hc:
            tags = (await _hc.get(f"{ollama_url}/api/tags")).json()
            available = [m["name"] for m in tags.get("models", [])]
            if not any(ollama_model in name for name in available):
                logger.warning(
                    f"[App] Model '{ollama_model}' not found in Ollama. "
                    f"Available: {available}. Run: ollama pull {ollama_model}"
                )
                await cl.Message(
                    content=(
                        f"⚠️ **Model not loaded**: `{ollama_model}` is not available in Ollama.\n\n"
                        f"Run this in a terminal, then refresh:\n"
                        f"```\nollama pull {ollama_model}\n```"
                    ),
                    author="System",
                ).send()
    except Exception as e:
        logger.error(f"[App] Ollama not reachable at {ollama_url}: {e}")
        await cl.Message(
            content=(
                "⚠️ **AI backend unavailable**: Ollama is not running.\n\n"
                "Start it in a terminal, then refresh this page:\n"
                "```\nollama serve\n```"
            ),
            author="System",
        ).send()

    # ── Heavy init: BGE-M3 + Qdrant + graph — only once, in a thread ─────
    # run_in_executor keeps the event loop unblocked so other sessions can
    # start while this one loads the ~570 MB embedding model.
    if not _heavy_init_done:
        _heavy_init_done = True  # set before await — blocks concurrent inits
        loop = asyncio.get_event_loop()

        try:
            from ..retrieval.embedder import BGEEmbedder
            from ..retrieval.hybrid_search import QdrantHybridSearch
            embedder = await loop.run_in_executor(
                None, lambda: BGEEmbedder(CONFIG_PATH)
            )
            _searcher = QdrantHybridSearch(CONFIG_PATH, embedder)
            logger.info("[App] Qdrant searcher initialized")
        except Exception as e:
            logger.error(f"[App] Could not initialize searcher: {e}")

        try:
            from ..ingestion.graph_rag_builder import KnowledgeGraphBuilder
            _graph = await loop.run_in_executor(
                None, lambda: KnowledgeGraphBuilder.load(CONFIG_PATH)
            )
            logger.info("[App] Knowledge graph loaded")
        except Exception as e:
            logger.warning(f"[App] Could not load graph: {e}")

    # ── Create engine (after init so it gets the real _searcher/_graph) ───
    from .tutor_engine import TutorEngine
    engine = TutorEngine(CONFIG_PATH, _searcher, _graph)
    cl.user_session.set("engine", engine)

    logger.info(f"[App] New session: {session_id} | user: {user_id}")


@cl.on_message
async def on_message(message: cl.Message):
    """Handle an incoming student message."""
    user_id: str = cl.user_session.get("user_id")
    session_id: str = cl.user_session.get("session_id")
    turn_number: int = cl.user_session.get("turn_number") + 1
    engine = cl.user_session.get("engine")
    last_ai_time: float = cl.user_session.get("last_ai_time", time.perf_counter())
    history: list = cl.user_session.get("history", [])

    # ── Measure student typing duration ───────────────────────────────────
    duration_sec = round(time.perf_counter() - last_ai_time, 2)
    user_text = message.content.strip()

    if not user_text:
        return

    cl.user_session.set("turn_number", turn_number)

    # ── Stream tutor response ─────────────────────────────────────────────
    response_msg = cl.Message(content="", author="Tutor")
    await response_msg.send()

    full_response = ""
    kb_used = False
    category = cl.user_session.get("category", "General Science")
    try:
        async for token, is_first_kb in engine.respond_stream(
            user_text, history, turn_number, category=category
        ):
            full_response += token
            await response_msg.stream_token(token)
            if is_first_kb:
                kb_used = True
    except Exception as e:
        logger.error(f"[App] Error during response generation: {e}")
        full_response = (
            "I encountered a technical difficulty. "
            "Please try rephrasing your question."
        )
        await response_msg.update()

    await response_msg.update()

    # ── Update session state ──────────────────────────────────────────────
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": full_response})
    if len(history) > 24:
        history = history[-24:]
    cl.user_session.set("history", history)
    cl.user_session.set("last_ai_time", time.perf_counter())

    # ── Persist turn to SQLite + CSV ──────────────────────────────────────
    if _session_manager:
        try:
            await _session_manager.record_turn(
                session_id=session_id,
                user_id=user_id,
                turn_number=turn_number,
                user_input=user_text,
                ai_response=full_response,
                kb_context_used=kb_used,
                duration_sec=duration_sec,
            )
        except Exception as e:
            logger.error(f"[App] Failed to record turn: {e}")

    # ── Graceful conversation close after 14 turns ────────────────────────
    if turn_number >= 14:
        await cl.Message(
            content=(
                "We have had a rich conversation! 🎉\n"
                "To start a new exploration, refresh the page. "
                "Your responses have been saved for analysis."
            ),
            author="System",
        ).send()

    logger.info(
        f"[App] {session_id} | turn={turn_number} | "
        f"dur={duration_sec}s | kb={kb_used} | "
        f"input_len={len(user_text)}"
    )


@cl.on_chat_end
async def on_chat_end():
    logger.info(
        f"[App] Session ended: {cl.user_session.get('session_id', 'unknown')}"
    )
