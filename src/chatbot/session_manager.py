"""
session_manager.py
───────────────────
Async SQLite session management.  Records every student interaction with:
  - Timestamp (ISO 8601)
  - User_ID
  - User_Input (verbatim student text)
  - AI_Response
  - KB_Context_Used (bool)
  - Duration_Sec (seconds between previous AI message and student's reply)
  - Turn_Number
  - Session_ID

Duration_Sec is a cognitive depth proxy:
  < 8s   → likely Passive (copy-paste or surface recall)
  8–25s  → likely Active (reading and selecting)
  > 25s  → likely Constructive/Interactive (reasoning and composing)
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
import yaml

logger = logging.getLogger(__name__)

CREATE_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    question_id  TEXT DEFAULT ''
);
"""

CREATE_TURNS_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    turn_number     INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    user_input      TEXT NOT NULL,
    ai_response     TEXT NOT NULL,
    kb_context_used INTEGER NOT NULL DEFAULT 0,
    duration_sec    REAL NOT NULL DEFAULT 0.0,
    question_topic  TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_unscored ON turns(session_id) WHERE duration_sec >= 0;
"""


class SessionManager:
    """
    Async SQLite session manager.

    Usage (in Chainlit):
    --------------------
    manager = await SessionManager.create(config_path)
    session_id = await manager.new_session(user_id="student_42")
    await manager.record_turn(session_id, user_id, turn_number=1,
                              user_input="...", ai_response="...",
                              kb_context_used=True, duration_sec=18.3)
    """

    CSV_COLUMNS = [
        "timestamp", "session_id", "user_id", "turn_number",
        "user_input", "ai_response", "kb_context_used",
        "duration_sec", "question_topic",
    ]

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.csv_path = db_path.parent / "conversations.csv"
        self._db: Optional[aiosqlite.Connection] = None
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.CSV_COLUMNS).writeheader()

    @classmethod
    async def create(cls, config_path: Path) -> "SessionManager":
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        db_path = Path(cfg["session_db"]["path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mgr = cls(db_path)
        await mgr._init_db()
        return mgr

    async def _init_db(self):
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._db.execute(CREATE_SESSIONS_SQL)
        await self._db.execute(CREATE_TURNS_SQL)
        for stmt in CREATE_INDEX_SQL.strip().split(";"):
            if stmt.strip():
                try:
                    await self._db.execute(stmt)
                except Exception:
                    pass  # index may already exist
        await self._db.commit()
        logger.info(f"[SessionManager] DB ready at {self.db_path}")

    async def new_session(
        self, user_id: str, question_id: str = ""
    ) -> str:
        """Create a new session and return its UUID."""
        session_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO sessions (session_id, user_id, started_at, question_id) "
            "VALUES (?, ?, ?, ?)",
            (session_id, user_id, started_at, question_id),
        )
        await self._db.commit()
        return session_id

    async def record_turn(
        self,
        session_id: str,
        user_id: str,
        turn_number: int,
        user_input: str,
        ai_response: str,
        kb_context_used: bool,
        duration_sec: float,
        question_topic: str = "",
    ):
        """Persist one conversation turn."""
        timestamp = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO turns
               (session_id, user_id, turn_number, timestamp,
                user_input, ai_response, kb_context_used,
                duration_sec, question_topic)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, user_id, turn_number, timestamp,
                user_input, ai_response, int(kb_context_used),
                round(duration_sec, 2), question_topic,
            ),
        )
        await self._db.commit()

        # Mirror to CSV
        row = {
            "timestamp": timestamp,
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_number,
            "user_input": user_input,
            "ai_response": ai_response,
            "kb_context_used": int(kb_context_used),
            "duration_sec": round(duration_sec, 2),
            "question_topic": question_topic,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.CSV_COLUMNS).writerow(row)

    async def get_session_turns(
        self, session_id: str
    ) -> List[Dict[str, Any]]:
        """Return all turns for a session, ordered by turn_number."""
        cursor = await self._db.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY turn_number ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    async def get_unscored_turns(
        self, batch_size: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Return turns not yet processed by the offline analyzer.
        We detect 'unscored' via a LEFT JOIN on the scores table (if it exists).
        Falls back to returning all turns if scores table does not exist.
        """
        try:
            cursor = await self._db.execute(
                """
                SELECT t.* FROM turns t
                LEFT JOIN scores s ON s.turn_id = t.id
                WHERE s.id IS NULL
                ORDER BY t.id ASC
                LIMIT ?
                """,
                (batch_size,),
            )
        except Exception:
            # scores table doesn't exist yet
            cursor = await self._db.execute(
                "SELECT * FROM turns ORDER BY id ASC LIMIT ?",
                (batch_size,),
            )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    async def get_session_history_for_prompt(
        self, session_id: str, max_turns: int = 6
    ) -> List[Dict[str, str]]:
        """
        Return last N turns formatted as chat history for LLM context.
        """
        turns = await self.get_session_turns(session_id)
        recent = turns[-max_turns:]
        history = []
        for t in recent:
            history.append({"role": "user", "content": t["user_input"]})
            history.append({"role": "assistant", "content": t["ai_response"]})
        return history

    async def close(self):
        if self._db:
            await self._db.close()


class TurnTimer:
    """Context manager to measure typing duration for one student turn."""

    def __init__(self):
        self._start: Optional[float] = None
        self.duration_sec: float = 0.0

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is not None:
            self.duration_sec = round(time.perf_counter() - self._start, 2)
        return self.duration_sec
