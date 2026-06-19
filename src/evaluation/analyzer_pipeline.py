"""
analyzer_pipeline.py
─────────────────────
Offline cognitive assessment pipeline.
Runs as a scheduled job AFTER student interactions are complete.
Strict separation from the chatbot layer (no imports from src.chatbot).

Pipeline:
  1. Fetch unscored turns from SQLite
  2. Retrieve PISA context from Qdrant for each turn
  3. Run MajorityVoter ensemble
  4. Write scores to SQLite (scores table)
  5. Export CSV for analysis

Schedule:
  Run manually:  python -m src.evaluation.analyzer_pipeline
  Or cron:       */15 * * * * cd /path/to/project && python -m src.evaluation.analyzer_pipeline
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import aiosqlite
import yaml

from .majority_voting import DIMENSIONS, MajorityVoter

logger = logging.getLogger(__name__)

CREATE_SCORES_TABLE = """
CREATE TABLE IF NOT EXISTS scores (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id           INTEGER NOT NULL,
    session_id        TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    turn_number       INTEGER,
    duration_sec      REAL,
    dominant_icap     TEXT,
    models_responded  INTEGER,
    elapsed_sec       REAL,
    scored_at         TEXT,
    C1_dominant       TEXT,
    C1_max            TEXT,
    C1_controversial  INTEGER,
    C2_dominant       TEXT,
    C2_max            TEXT,
    C2_controversial  INTEGER,
    C3_dominant       TEXT,
    C3_max            TEXT,
    C3_controversial  INTEGER,
    C4_dominant       TEXT,
    C4_max            TEXT,
    C4_controversial  INTEGER,
    C5_dominant       TEXT,
    C5_max            TEXT,
    C5_controversial  INTEGER,
    C6_dominant       TEXT,
    C6_max            TEXT,
    C6_controversial  INTEGER,
    C7_dominant       TEXT,
    C7_max            TEXT,
    C7_controversial  INTEGER,
    C8_dominant       TEXT,
    C8_max            TEXT,
    C8_controversial  INTEGER,
    individual_votes_json TEXT,
    FOREIGN KEY (turn_id) REFERENCES turns(id)
);
"""


class AnalyzerPipeline:
    """
    Runs the offline cognitive assessment on unscored chat turns.

    Parameters
    ----------
    config_path : Path
    db_path     : Path (SQLite)
    """

    def __init__(self, config_path: Path):
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.config_path = config_path
        self.db_path = Path(self.cfg["session_db"]["path"])
        self.voter: MajorityVoter = MajorityVoter(config_path)
        self._db: aiosqlite.Connection = None

    async def _init_db(self):
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute(CREATE_SCORES_TABLE)
        await self._db.commit()

    async def _get_unscored_turns(self, batch_size: int = 100) -> List[Dict]:
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
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    def _get_pisa_context(self, user_input: str) -> str:
        """
        Retrieve PISA context in a subprocess to isolate BGE/Qdrant segfaults.
        Falls back to empty string on any failure.
        """
        script = (
            "import sys, json\n"
            "from pathlib import Path\n"
            "from src.retrieval.embedder import BGEEmbedder\n"
            "from src.retrieval.hybrid_search import QdrantHybridSearch\n"
            "cfg = Path(sys.argv[1])\n"
            "query = sys.argv[2]\n"
            "e = BGEEmbedder(cfg)\n"
            "s = QdrantHybridSearch(cfg, e)\n"
            "results = s.search(query, top_k=3)\n"
            "print(s.format_context_for_prompt(results))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, str(self.config_path), user_input],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
            if proc.returncode != 0:
                logger.debug(f"[Analyzer] Retrieval subprocess failed (rc={proc.returncode})")
        except subprocess.TimeoutExpired:
            logger.debug("[Analyzer] Retrieval subprocess timed out")
        except Exception as e:
            logger.debug(f"[Analyzer] Retrieval subprocess error: {e}")
        return ""

    async def _save_score(self, turn: Dict, result: Dict[str, Any]):
        """Persist scoring result to the scores table."""
        scored_at = datetime.now(timezone.utc).isoformat()
        dim_scores = result["dimension_scores"]

        row = {
            "turn_id": turn["id"],
            "session_id": turn["session_id"],
            "user_id": turn["user_id"],
            "turn_number": turn["turn_number"],
            "duration_sec": turn["duration_sec"],
            "dominant_icap": result["dominant_icap"],
            "models_responded": result["models_responded"],
            "elapsed_sec": result["elapsed_sec"],
            "scored_at": scored_at,
            "individual_votes_json": json.dumps(result["individual_votes"], ensure_ascii=False),
        }
        for dim in DIMENSIONS:
            ds = dim_scores.get(dim, {})
            if isinstance(ds, dict):
                row[f"{dim}_dominant"] = str(ds.get("dominant", "NA"))
                row[f"{dim}_max"] = str(ds.get("max_capability", "NA"))
                row[f"{dim}_controversial"] = int(ds.get("controversial", False))
            else:
                row[f"{dim}_dominant"] = str(ds)
                row[f"{dim}_max"] = str(ds)
                row[f"{dim}_controversial"] = 0

        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        await self._db.execute(
            f"INSERT INTO scores ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
        await self._db.commit()

    async def run(self, batch_size: int = 100):
        """Main async entry point."""
        await self._init_db()

        turns = await self._get_unscored_turns(batch_size)
        logger.info(f"[Analyzer] Found {len(turns)} unscored turn(s)")

        if not turns:
            logger.info("[Analyzer] Nothing to score — exiting")
            await self._db.close()
            return

        loop = asyncio.get_event_loop()
        processed = 0
        t_start = time.perf_counter()

        for turn in turns:
            user_input = turn.get("user_input", "")
            if not user_input.strip():
                continue

            pisa_context = self._get_pisa_context(user_input)

            # Run blocking voter in thread pool
            result = await loop.run_in_executor(
                None,
                self.voter.score_turn,
                user_input,
                pisa_context,
                turn.get("turn_number", 1),
                turn.get("duration_sec", 0.0),
            )

            await self._save_score(turn, result)
            processed += 1
            logger.info(
                f"[Analyzer] Scored turn {turn['id']} "
                f"({turn['user_id']}, turn {turn['turn_number']}) "
                f"→ ICAP={result['dominant_icap']}"
            )

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[Analyzer] Processed {processed} turns in {elapsed:.1f}s "
            f"({elapsed/max(processed,1):.1f}s/turn)"
        )

        await self._export_csv()
        await self._db.close()

    async def _export_csv(self):
        """Export scores table to CSV for analysis."""
        export_path = Path("data/chat_logs/scores_export.csv")
        cursor = await self._db.execute("SELECT * FROM scores ORDER BY id")
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        with export_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        logger.info(f"[Analyzer] Exported {len(rows)} scored rows to {export_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()
    pipeline = AnalyzerPipeline(Path(args.config))
    asyncio.run(pipeline.run(batch_size=args.batch))


if __name__ == "__main__":
    main()
