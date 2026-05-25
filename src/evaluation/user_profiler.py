"""
user_profiler.py
─────────────────
Builds per-student cognitive profiles from the scored turns database.

Output per student:
  - Dominant Score per dimension (most frequent score, reflecting habitual behaviour)
  - Max Capability per dimension (highest score achieved, reflecting peak potential)
  - ICAP Level profile (fraction of turns at each level)
  - Engagement Trajectory (does cognitive depth increase over turns?)
  - Duration correlation (does longer thinking → higher C4-C8?)
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
import yaml

from .majority_voting import DIMENSIONS

logger = logging.getLogger(__name__)

ICAP_ORDER = {"Passive": 0, "Active": 1, "Constructive": 2, "Interactive": 3}


def _numeric_scores(scores: List[Any]) -> List[int]:
    return [s for s in scores if isinstance(s, int)]


class UserProfiler:
    """
    Reads the scores table and builds per-user cognitive profiles.

    Usage
    -----
    profiler = await UserProfiler.create(config_path)
    profiles = await profiler.build_all_profiles()
    summary = profiler.format_profile(profiles["student_abc123"])
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path

    @classmethod
    async def create(cls, config_path: Path) -> "UserProfiler":
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls(Path(cfg["session_db"]["path"]))

    async def _fetch_scores(self) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(str(self.db_path)) as db:
            try:
                cursor = await db.execute(
                    "SELECT * FROM scores ORDER BY user_id, turn_number"
                )
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                return [dict(zip(cols, row)) for row in rows]
            except Exception as e:
                logger.error(f"[Profiler] Could not fetch scores: {e}")
                return []

    def _build_user_profile(self, user_id: str, rows: List[Dict]) -> Dict[str, Any]:
        """Compute the full cognitive profile for one user from their score rows."""
        n = len(rows)
        if n == 0:
            return {}

        # ── Per-dimension aggregation ─────────────────────────────────────
        dim_dominant_scores: Dict[str, List[Any]] = defaultdict(list)
        dim_max_scores: Dict[str, List[Any]] = defaultdict(list)

        icap_levels: List[str] = []
        durations: List[float] = []
        turn_c48_scores: List[tuple] = []  # (turn_number, avg C4-C8 score)

        for row in rows:
            for dim in DIMENSIONS:
                dom = row.get(f"{dim}_dominant", "NA")
                mx = row.get(f"{dim}_max", "NA")
                try:
                    dom = int(dom)
                except (ValueError, TypeError):
                    dom = "NA"
                try:
                    mx = int(mx)
                except (ValueError, TypeError):
                    mx = "NA"
                dim_dominant_scores[dim].append(dom)
                dim_max_scores[dim].append(mx)

            lvl = row.get("dominant_icap", "Unknown")
            icap_levels.append(lvl)
            dur = row.get("duration_sec", 0.0) or 0.0
            durations.append(float(dur))

            # C4–C8 average for trajectory
            high_scores = []
            for dim in ["C4", "C5", "C6", "C7", "C8"]:
                v = row.get(f"{dim}_dominant", "NA")
                try:
                    high_scores.append(int(v))
                except (ValueError, TypeError):
                    pass
            if high_scores:
                turn_c48_scores.append(
                    (row.get("turn_number", 0), sum(high_scores) / len(high_scores))
                )

        # ── Dominant and Max per dimension ────────────────────────────────
        dimension_profile = {}
        for dim in DIMENSIONS:
            numeric_dom = _numeric_scores(dim_dominant_scores[dim])
            numeric_max = _numeric_scores(dim_max_scores[dim])
            if numeric_dom:
                counter = Counter(numeric_dom)
                dominant = counter.most_common(1)[0][0]
                max_cap = max(numeric_max) if numeric_max else dominant
            else:
                dominant = "NA"
                max_cap = "NA"
            dimension_profile[dim] = {
                "dominant_score": dominant,
                "max_capability": max_cap,
                "n_responses": len(numeric_dom),
                "score_distribution": dict(Counter(numeric_dom)),
            }

        # ── ICAP level distribution ───────────────────────────────────────
        icap_counter = Counter(l for l in icap_levels if l != "Unknown")
        total_icap = sum(icap_counter.values()) or 1
        icap_profile = {
            level: round(count / total_icap, 3)
            for level, count in icap_counter.items()
        }

        # Dominant ICAP level (the most common one)
        dominant_icap = (
            max(icap_counter, key=lambda k: icap_counter[k])
            if icap_counter else "Unknown"
        )

        # ── Max ICAP level achieved ───────────────────────────────────────
        unique_levels = set(icap_levels)
        max_icap = max(
            (l for l in unique_levels if l in ICAP_ORDER),
            key=lambda k: ICAP_ORDER[k],
            default="Unknown",
        )

        # ── Engagement trajectory ─────────────────────────────────────────
        trajectory = "unknown"
        if len(turn_c48_scores) >= 3:
            sorted_by_turn = sorted(turn_c48_scores)
            first_half = [s for _, s in sorted_by_turn[: len(sorted_by_turn) // 2]]
            second_half = [s for _, s in sorted_by_turn[len(sorted_by_turn) // 2 :]]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            delta = avg_second - avg_first
            if delta > 0.2:
                trajectory = "improving"
            elif delta < -0.2:
                trajectory = "declining"
            else:
                trajectory = "stable"

        # ── Duration–depth correlation (Spearman rank, simple) ────────────
        dur_c4_pairs = []
        for row in rows:
            dur = row.get("duration_sec", 0.0) or 0.0
            c4 = row.get("C4_dominant", "NA")
            try:
                dur_c4_pairs.append((float(dur), int(c4)))
            except (ValueError, TypeError):
                pass

        dur_corr = None
        if len(dur_c4_pairs) >= 5:
            from scipy.stats import spearmanr
            durations_arr = [d for d, _ in dur_c4_pairs]
            c4_arr = [c for _, c in dur_c4_pairs]
            try:
                corr, _ = spearmanr(durations_arr, c4_arr)
                dur_corr = round(float(corr), 3)
            except Exception:
                dur_corr = None

        return {
            "user_id": user_id,
            "n_turns_scored": n,
            "dimension_profile": dimension_profile,
            "icap_level_distribution": icap_profile,
            "dominant_icap": dominant_icap,
            "max_icap_achieved": max_icap,
            "engagement_trajectory": trajectory,
            "avg_duration_sec": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "duration_causal_correlation": dur_corr,
            "controversy_rate": round(
                sum(
                    1
                    for row in rows
                    for dim in DIMENSIONS
                    if row.get(f"{dim}_controversial", 0)
                ) / max(n * len(DIMENSIONS), 1),
                3,
            ),
        }

    async def build_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Return per-user profile dict keyed by user_id."""
        all_rows = await self._fetch_scores()
        if not all_rows:
            return {}

        # Group by user
        by_user: Dict[str, List[Dict]] = defaultdict(list)
        for row in all_rows:
            by_user[row["user_id"]].append(row)

        profiles = {}
        for uid, rows in by_user.items():
            profiles[uid] = self._build_user_profile(uid, rows)
            logger.info(
                f"[Profiler] {uid}: {len(rows)} turns, "
                f"ICAP={profiles[uid]['dominant_icap']}, "
                f"max={profiles[uid]['max_icap_achieved']}, "
                f"traj={profiles[uid]['engagement_trajectory']}"
            )
        return profiles

    @staticmethod
    def format_profile(profile: Dict[str, Any]) -> str:
        """Human-readable profile summary."""
        lines = [
            f"User: {profile['user_id']}",
            f"Turns scored: {profile['n_turns_scored']}",
            f"Dominant ICAP: {profile['dominant_icap']} | Max achieved: {profile['max_icap_achieved']}",
            f"Engagement trajectory: {profile['engagement_trajectory']}",
            f"Avg response time: {profile['avg_duration_sec']}s",
            "",
            "Dimension scores (dominant → max capability):",
        ]
        for dim, dp in profile["dimension_profile"].items():
            lines.append(
                f"  {dim}: {dp['dominant_score']} → {dp['max_capability']}  "
                f"(n={dp['n_responses']}, dist={dp['score_distribution']})"
            )
        return "\n".join(lines)
