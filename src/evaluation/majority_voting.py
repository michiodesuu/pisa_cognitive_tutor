"""
majority_voting.py
───────────────────
Runs an ensemble of 4 local LLMs to score student responses on C1-C8.
Implements confidence-weighted voting and controversy detection.

Voting algorithm:
  1. Each model returns {C1..C8: score, confidence, icap_level, reasoning}
  2. Scores weighted by model's historical Fleiss' Kappa (loaded from disk)
  3. Weighted plurality vote for each dimension
  4. If max_vote / total_weight < controversy_threshold → mark as controversial
  5. User profile uses "Dominant Score" (mode) AND "Max Capability" (max) per dimension
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

logger = logging.getLogger(__name__)

DIMENSIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
VALID_SCORES = {0, 1, 2}  # NA handled separately
KAPPA_WEIGHTS_PATH = Path("data/processed_jsonl/model_kappa_weights.json")


def _load_kappa_weights(cfg: dict) -> Dict[str, float]:
    """Load per-model Fleiss' Kappa weights.  Defaults to 1.0 if not yet computed."""
    if KAPPA_WEIGHTS_PATH.exists():
        return json.loads(KAPPA_WEIGHTS_PATH.read_text(encoding="utf-8"))
    # Equal weights by default
    return {m["name"]: 1.0 for m in cfg["evaluator_ensemble"]["models"]}


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from model output."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find first {...}
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise score dict from one model."""
    validated = {}
    for dim in DIMENSIONS:
        val = scores.get(dim, "NA")
        if val == "NA" or val is None:
            validated[dim] = "NA"
        else:
            try:
                v = int(val)
                validated[dim] = v if v in VALID_SCORES else "NA"
            except (ValueError, TypeError):
                validated[dim] = "NA"
    validated["confidence"] = float(scores.get("confidence", 0.5))
    validated["icap_level"] = scores.get("icap_level", "Unknown")
    validated["reasoning"] = scores.get("reasoning", "")
    return validated


class ModelEvaluator:
    """Wrapper for a single Ollama model in the ensemble."""

    def __init__(self, model_cfg: dict, ensemble_cfg: dict):
        self.name: str = model_cfg["name"]
        self.tag: str = model_cfg["tag"]
        self.weight: float = model_cfg.get("weight", 1.0)
        self.base_url: str = ensemble_cfg["ollama_base_url"]
        self.timeout: int = ensemble_cfg["timeout_seconds"]

    def score(
        self,
        student_input: str,
        pisa_context: str,
        rubric_text: str,
        turn_number: int,
        duration_sec: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Call this model synchronously (designed to run in thread pool).
        Returns validated score dict or None on failure.
        """
        prompt = (
            f"{rubric_text}\n\n"
            f"<PISA_CONTEXT>\n{pisa_context}\n</PISA_CONTEXT>\n\n"
            f"TURN_NUMBER: {turn_number}\n"
            f"DURATION_SEC: {duration_sec}\n\n"
            f"<STUDENT_RESPONSE>\n{student_input}\n</STUDENT_RESPONSE>\n\n"
            "Return ONLY valid JSON with scores for C1-C8."
        )
        payload = {
            "model": self.tag,
            "messages": [
                {"role": "system", "content": "You are a PISA cognitive assessment expert. "
                 "Return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }
        try:
            resp = httpx.post(
                self.base_url.rstrip("/") + "/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
            parsed = _extract_json(raw)
            if parsed is None:
                logger.warning(f"[{self.name}] Could not parse JSON from response")
                return None
            validated = _validate_scores(parsed)
            validated["model_name"] = self.name
            return validated
        except Exception as e:
            logger.warning(f"[{self.name}] Scoring failed: {e}")
            return None


class MajorityVoter:
    """
    Ensemble scorer with confidence-weighted majority voting.

    Usage
    -----
    voter = MajorityVoter(config_path)
    result = voter.score_turn(
        student_input="...",
        pisa_context="...",
        turn_number=3,
        duration_sec=22.4
    )
    """

    def __init__(self, config_path: Path):
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        ens_cfg = self.cfg["evaluator_ensemble"]
        self.kappa_weights = _load_kappa_weights(self.cfg)
        self.models = [
            ModelEvaluator(m, ens_cfg) for m in ens_cfg["models"]
        ]
        # Update weights from Kappa
        for m in self.models:
            m.weight = self.kappa_weights.get(m.name, 1.0)

        self.rubric_text = Path("configs/prompts/evaluator_judge.txt").read_text(encoding="utf-8").strip()
        self.consensus_threshold: float = ens_cfg["consensus_threshold"]
        self.controversy_threshold: float = ens_cfg["controversy_threshold"]

    def score_turn(
        self,
        student_input: str,
        pisa_context: str,
        turn_number: int,
        duration_sec: float,
    ) -> Dict[str, Any]:
        """
        Run all ensemble models in parallel and aggregate results.
        Returns a comprehensive scoring dict.
        """
        t0 = time.perf_counter()

        # Run models in parallel (each is a blocking HTTP call)
        model_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = {
                executor.submit(
                    m.score, student_input, pisa_context,
                    self.rubric_text, turn_number, duration_sec
                ): m.name
                for m in self.models
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        model_results.append(result)
                except Exception as e:
                    logger.warning(f"[Voter] Model {name} raised exception: {e}")

        elapsed = time.perf_counter() - t0

        if not model_results:
            logger.error("[Voter] All models failed — returning NA scores")
            return self._empty_result(student_input, turn_number, duration_sec)

        # ── Aggregate per dimension ───────────────────────────────────────
        dimension_scores = {}
        dimension_meta = {}
        for dim in DIMENSIONS:
            votes: Dict[Any, float] = {}
            for res in model_results:
                score = res.get(dim, "NA")
                weight = self.kappa_weights.get(res.get("model_name", ""), 1.0)
                # Factor in the model's self-reported confidence
                eff_weight = weight * res.get("confidence", 0.5)
                votes[score] = votes.get(score, 0.0) + eff_weight

            total_w = sum(votes.values())
            if total_w == 0:
                dimension_scores[dim] = "NA"
                dimension_meta[dim] = {"controversial": False, "vote_dist": {}}
                continue

            best_score = max(votes, key=lambda s: votes[s])
            best_frac = votes[best_score] / total_w
            is_controversial = best_frac < self.controversy_threshold

            # Max Capability: highest numeric score across all models
            numeric_scores = [
                v for v in votes if isinstance(v, int)
            ]
            max_capability = max(numeric_scores) if numeric_scores else "NA"

            # Dominant score (plurality winner)
            dominant = best_score if best_score != "NA" else (
                numeric_scores[0] if numeric_scores else "NA"
            )

            dimension_scores[dim] = {
                "dominant": dominant,
                "max_capability": max_capability,
                "vote_distribution": {str(k): round(v, 3) for k, v in votes.items()},
                "controversial": is_controversial,
                "consensus_ratio": round(best_frac, 3),
            }

        # ── ICAP level ────────────────────────────────────────────────────
        icap_votes: Dict[str, int] = {}
        for res in model_results:
            lvl = res.get("icap_level", "Unknown")
            icap_votes[lvl] = icap_votes.get(lvl, 0) + 1
        dominant_icap = max(icap_votes, key=lambda k: icap_votes[k]) if icap_votes else "Unknown"

        # ── Individual model logs (for transparency / Kappa computation) ──
        individual_votes = [
            {
                "model": res.get("model_name"),
                "confidence": res.get("confidence"),
                "icap_level": res.get("icap_level"),
                "reasoning": res.get("reasoning", ""),
                "scores": {d: res.get(d, "NA") for d in DIMENSIONS},
            }
            for res in model_results
        ]

        return {
            "student_input": student_input,
            "turn_number": turn_number,
            "duration_sec": duration_sec,
            "dimension_scores": dimension_scores,
            "dominant_icap": dominant_icap,
            "individual_votes": individual_votes,
            "models_responded": len(model_results),
            "models_total": len(self.models),
            "elapsed_sec": round(elapsed, 2),
        }

    def _empty_result(self, student_input, turn_number, duration_sec):
        return {
            "student_input": student_input,
            "turn_number": turn_number,
            "duration_sec": duration_sec,
            "dimension_scores": {d: "NA" for d in DIMENSIONS},
            "dominant_icap": "Unknown",
            "individual_votes": [],
            "models_responded": 0,
            "models_total": len(self.models),
            "elapsed_sec": 0.0,
        }
