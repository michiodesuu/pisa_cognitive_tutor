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

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import chainlit as cl
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

# ── Global singletons (initialised on startup) ────────────────────────────────
_searcher = None
_graph = None
_session_manager = None
_cfg = None


@cl.on_chat_start
async def on_chat_start():
    """Initialise resources for this chat session."""
    global _searcher, _graph, _session_manager, _cfg

    # ── Lazy-load singletons ──────────────────────────────────────────────
    if _cfg is None:
        _cfg = yaml.safe_load(CONFIG_PATH.read_text())

    if _searcher is None:
        try:
            from ..retrieval.embedder import BGEEmbedder
            from ..retrieval.hybrid_search import QdrantHybridSearch

            embedder = BGEEmbedder(CONFIG_PATH)
            _searcher = QdrantHybridSearch(CONFIG_PATH, embedder)
            logger.info("[App] Qdrant searcher initialized")
        except Exception as e:
            logger.error(f"[App] Could not initialize searcher: {e}")

    if _graph is None:
        try:
            from ..ingestion.graph_rag_builder import KnowledgeGraphBuilder
            _graph = KnowledgeGraphBuilder.load(CONFIG_PATH)
            logger.info("[App] Knowledge graph loaded")
        except Exception as e:
            logger.warning(f"[App] Could not load graph: {e}")

    if _session_manager is None:
        from .session_manager import SessionManager
        _session_manager = await SessionManager.create(CONFIG_PATH)
        logger.info("[App] Session manager initialized")

    # ── Per-session state (stored in cl.user_session) ─────────────────────
    user_id = f"student_{uuid.uuid4().hex[:8]}"
    session_id = await _session_manager.new_session(user_id=user_id)

    from .tutor_engine import TutorEngine
    engine = TutorEngine(CONFIG_PATH, _searcher, _graph)

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("turn_number", 0)
    cl.user_session.set("engine", engine)
    cl.user_session.set("last_ai_time", time.perf_counter())
    cl.user_session.set("history", [])

    # ── Welcome message ───────────────────────────────────────────────────
    await cl.Message(
        content=(
            "👋 **Welcome to the PISA Science Tutor**\n\n"
            "I am here to help you think through science questions step by step.\n"
            "You can type in **Thai, English, or Chinese** — whichever is most "
            "comfortable for you.\n\n"
            "Whenever you are ready, describe the science question or topic "
            "you would like to explore, and we will work through it together. 🔬"
        ),
        author="Tutor",
    ).send()

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
    try:
        async for token, is_first_kb in engine.respond_stream(
            user_text, history, turn_number
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
    # Keep last 12 turns in memory context
    if len(history) > 24:
        history = history[-24:]
    cl.user_session.set("history", history)
    cl.user_session.set("last_ai_time", time.perf_counter())

    # ── Persist turn to SQLite ────────────────────────────────────────────
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

    # ── Graceful conversation close after 12 turns ────────────────────────
    if turn_number >= 12:
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
