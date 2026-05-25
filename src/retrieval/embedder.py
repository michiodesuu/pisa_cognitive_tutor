"""
embedder.py
────────────
BGE-M3 multilingual embedding model.
Supports dense (1024-d) + sparse (SPLADE-style) vectors simultaneously.
Handles Thai, Chinese, English code-switching natively.

BGE-M3 paper: https://arxiv.org/abs/2309.07597
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_model_instance = None


def _get_or_load(cfg: dict):
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    from FlagEmbedding import BGEM3FlagModel
    source = cfg["embedding"].get("local_path", cfg["embedding"]["model_id"])
    if not Path(source).exists():
        source = cfg["embedding"]["model_id"]
    _model_instance = BGEM3FlagModel(
        source,
        use_fp16=True,      # saves memory, negligible accuracy drop
        device=cfg["embedding"]["device"],
    )
    logger.info(f"[Embedder] Loaded BGE-M3 from {source}")
    return _model_instance


class BGEEmbedder:
    """
    Wraps BGEM3FlagModel to provide:
    - encode_dense(texts) → List[List[float]]  (1024-d dense vectors)
    - encode_sparse(texts) → List[Dict[int, float]]  (sparse token weights)
    - encode_both(texts) → List of (dense, sparse) tuples
    """

    def __init__(self, config_path: Path):
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self._model = None
        self.batch_size: int = self.cfg["embedding"]["batch_size"]
        self.max_length: int = self.cfg["embedding"]["max_length"]
        self.use_sparse: bool = self.cfg["embedding"]["use_sparse"]

    def _ensure_loaded(self):
        if self._model is None:
            self._model = _get_or_load(self.cfg)

    def encode_dense(self, texts: List[str]) -> List[List[float]]:
        """Return normalised dense vectors."""
        self._ensure_loaded()
        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"].tolist()

    def encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """Return sparse lexical weights (token_id → weight)."""
        self._ensure_loaded()
        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return output["lexical_weights"]  # list of dicts

    def encode_both(
        self, texts: List[str]
    ) -> List[Tuple[List[float], Dict[int, float]]]:
        """Return (dense_vec, sparse_weights) for each text."""
        self._ensure_loaded()
        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=self.use_sparse,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"].tolist()
        sparse = output.get("lexical_weights", [{} for _ in texts])
        return list(zip(dense, sparse))

    @staticmethod
    def record_to_text(record: dict) -> str:
        """
        Convert a JSONL knowledge record to a single embedding text.
        Combines topic, question summary, concept, and misconceptions.
        This is the canonical text representation for indexing.
        """
        parts = [
            record.get("topic", ""),
            record.get("question_summary", "")[:800],   # truncate long stimuli
            "Correct concept: " + record.get("correct_concept", ""),
            "Misconceptions: " + record.get("common_misconceptions", ""),
            "Vocabulary: " + ", ".join(record.get("key_vocabulary", [])),
        ]
        return " | ".join(p for p in parts if p.strip())
