"""
hybrid_search.py
─────────────────
Qdrant-backed hybrid search combining:
  1. Dense vector search (BGE-M3 dense embeddings, cosine similarity)
  2. Sparse vector search (BGE-M3 lexical weights, BM25-style)
  3. Graph subgraph augmentation (NetworkX neighbourhood)

Fusion: Reciprocal Rank Fusion (RRF) over dense + sparse result lists.

Uses Qdrant's native hybrid search API (qdrant_client >= 1.7.0).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class QdrantHybridSearch:
    """
    Hybrid retrieval over the PISA knowledge base.

    Usage
    -----
    searcher = QdrantHybridSearch(config_path, embedder)
    searcher.ingest_jsonl(jsonl_path)
    results = searcher.search("photosynthesis light reaction", top_k=5)
    """

    def __init__(self, config_path: Path, embedder):
        self.cfg = yaml.safe_load(config_path.read_text())
        self.embedder = embedder
        self._client = None
        self._collection_name: str = self.cfg["qdrant"]["collection_name"]
        self._dense_dim: int = self.cfg["qdrant"]["dense_dim"]
        self._dense_name: str = self.cfg["qdrant"]["dense_vector_name"]
        self._sparse_name: str = self.cfg["qdrant"]["sparse_vector_name"]
        self._top_k: int = self.cfg["qdrant"]["top_k"]
        self._score_threshold: float = self.cfg["qdrant"]["score_threshold"]

    def _ensure_client(self):
        if self._client is not None:
            return
        from qdrant_client import QdrantClient
        if self.cfg["qdrant"]["mode"] == "local":
            path = self.cfg["qdrant"]["local_path"]
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(
                host=self.cfg["qdrant"]["host"],
                port=self.cfg["qdrant"]["port"],
            )
        logger.info(f"[Qdrant] Connected (mode={self.cfg['qdrant']['mode']})")

    def _ensure_collection(self):
        from qdrant_client.http.models import (
            Distance,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
            VectorsConfig,
        )

        self._ensure_client()
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection_name in existing:
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                self._dense_name: VectorParams(
                    size=self._dense_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self._sparse_name: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                ),
            },
        )
        logger.info(f"[Qdrant] Created collection '{self._collection_name}'")

    def ingest_jsonl(self, jsonl_path: Path):
        """Encode all records and upsert into Qdrant."""
        from qdrant_client.http.models import PointStruct, SparseVector

        self._ensure_collection()

        records: List[Dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        if not records:
            logger.warning("[Qdrant] No records to ingest")
            return

        logger.info(f"[Qdrant] Encoding {len(records)} records with BGE-M3...")
        texts = [self.embedder.record_to_text(r) for r in records]
        embeddings = self.embedder.encode_both(texts)

        points = []
        for idx, (record, (dense_vec, sparse_weights)) in enumerate(
            zip(records, embeddings)
        ):
            # Convert sparse weights: {token_id: weight} → SparseVector
            sparse_indices = [int(k) for k in sparse_weights.keys()]
            sparse_values = [float(v) for v in sparse_weights.values()]

            point = PointStruct(
                id=idx,
                vector={
                    self._dense_name: dense_vec,
                    self._sparse_name: SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                },
                payload={
                    "topic": record.get("topic", ""),
                    "question_summary": record.get("question_summary", ""),
                    "correct_concept": record.get("correct_concept", ""),
                    "common_misconceptions": record.get("common_misconceptions", ""),
                    "pisa_competency": record.get("pisa_competency", ""),
                    "cognitive_demand": record.get("cognitive_demand", ""),
                    "key_vocabulary": record.get("key_vocabulary", []),
                    "source_file": record.get("source_file", ""),
                    "page_range": record.get("page_range", ""),
                },
            )
            points.append(point)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points[i : i + batch_size],
            )
        logger.info(f"[Qdrant] Ingested {len(points)} points")

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: dense + sparse, fused with RRF.
        Returns list of payload dicts sorted by relevance.
        """
        from qdrant_client.http.models import (
            FusionQuery,
            NearestQuery,
            Prefetch,
            SparseVector,
        )

        self._ensure_collection()
        k = top_k or self._top_k

        (dense_vec, sparse_weights) = self.embedder.encode_both([query])[0]
        sparse_indices = [int(k2) for k2 in sparse_weights.keys()]
        sparse_values = [float(v) for v in sparse_weights.values()]

        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using=self._dense_name,
                        limit=k * 2,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                        using=self._sparse_name,
                        limit=k * 2,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=k,
                score_threshold=self._score_threshold,
                with_payload=True,
            )
            payloads = [r.payload for r in results.points if r.payload]
        except Exception as e:
            logger.warning(f"[Qdrant] Hybrid search failed, falling back to dense: {e}")
            # Fallback to dense-only
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=(self._dense_name, dense_vec),
                limit=k,
                score_threshold=self._score_threshold,
                with_payload=True,
            )
            payloads = [r.payload for r in results if r.payload]

        return payloads

    def format_context_for_prompt(
        self, results: List[Dict[str, Any]]
    ) -> str:
        """Format retrieved documents into a context block for the tutor prompt."""
        if not results:
            return "No relevant knowledge base context found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[KB-{i}] Topic: {r.get('topic','')}\n"
                f"  Concept: {r.get('correct_concept','')}\n"
                f"  Misconceptions: {r.get('common_misconceptions','')}\n"
                f"  PISA Competency: {r.get('pisa_competency','')}\n"
                f"  Question stimulus (excerpt): "
                f"{r.get('question_summary','')[:300]}..."
            )
        return "\n\n".join(parts)
