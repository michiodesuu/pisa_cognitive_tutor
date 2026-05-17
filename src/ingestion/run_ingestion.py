"""
run_ingestion.py
─────────────────
Entry point for the full ingestion pipeline.

Orchestrates:
  1. PDF discovery
  2. Async Producer-Consumer: PDFProcessor → VLMExtractor
  3. JSONL write (with smart resume)
  4. GraphRAG build from completed JSONL
  5. Vector DB ingestion (calls retrieval/embedder)

Usage:
  python -m src.ingestion.run_ingestion --config configs/model_configs.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from .graph_rag_builder import KnowledgeGraphBuilder
from .pdf_processor import PDFProcessor
from .vlm_extractor import VLMExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ingestion.runner")


def _load_processed_files(jsonl_path: Path) -> Set[str]:
    """Return set of source_file values already in the JSONL."""
    seen = set()
    if not jsonl_path.exists():
        return seen
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add(rec.get("source_file", ""))
            except Exception:
                pass
    return seen


async def run_pipeline(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    kb_cfg = cfg["knowledge_base"]

    raw_dir = Path("data/raw_pdfs")
    output_path = Path(kb_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor = PDFProcessor(
        dpi=cfg["vlm"]["pdf_render_dpi"],
        max_pages_per_shard=cfg["vlm"]["max_pages_per_shard"],
    )
    extractor = VLMExtractor(
        config_path=config_path,
        prompt_path=Path("configs/prompts/vision_extractor.txt"),
    )

    pdfs = PDFProcessor.discover_pdfs(raw_dir)
    if not pdfs:
        logger.error("No PDFs found in data/raw_pdfs/. Please add PISA PDF files.")
        return

    # Smart resume
    processed: Set[str] = set()
    if kb_cfg.get("resume_from_checkpoint", True):
        processed = _load_processed_files(output_path)
        if processed:
            logger.info(f"Resume: skipping {len(processed)} already-processed files")

    loop = asyncio.get_event_loop()
    total_records = 0
    t_start = time.perf_counter()

    with output_path.open("a", encoding="utf-8") as out_f:
        for pdf_path in pdfs:
            if pdf_path.name in processed:
                logger.info(f"  [SKIP] {pdf_path.name}")
                continue

            logger.info(f"[PIPELINE] Processing {pdf_path.name}")
            file_records: List[Dict[str, Any]] = []

            async for shard in processor.shard_pdf(pdf_path):
                # Run GPU inference in executor to keep event loop responsive
                records = await loop.run_in_executor(
                    None,
                    extractor.extract_shard,
                    shard.images,
                    pdf_path.name,
                    shard.page_range_str,
                )
                file_records.extend(records)
                logger.info(
                    f"  Shard {shard.shard_index} "
                    f"(pages {shard.page_range_str}): {len(records)} records"
                )

            # Write all records for this file
            for rec in file_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            total_records += len(file_records)
            logger.info(
                f"[PIPELINE] {pdf_path.name}: {len(file_records)} records written"
            )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"\n[PIPELINE] COMPLETE: {total_records} total records in {elapsed:.1f}s"
    )

    # ── Build knowledge graph ─────────────────────────────────────────────────
    logger.info("[PIPELINE] Building knowledge graph...")
    builder = KnowledgeGraphBuilder(config_path)
    builder.ingest_jsonl(output_path)
    builder.save()

    # ── Ingest into Qdrant ────────────────────────────────────────────────────
    logger.info("[PIPELINE] Ingesting into Qdrant vector DB...")
    from ..retrieval.embedder import BGEEmbedder
    from ..retrieval.hybrid_search import QdrantHybridSearch

    embedder = BGEEmbedder(config_path)
    searcher = QdrantHybridSearch(config_path, embedder)
    searcher.ingest_jsonl(output_path)
    logger.info("[PIPELINE] Vector DB ingestion complete.")


def main():
    parser = argparse.ArgumentParser(description="Run PISA ingestion pipeline")
    parser.add_argument("--config", default="configs/model_configs.yaml")
    args = parser.parse_args()
    asyncio.run(run_pipeline(Path(args.config)))


if __name__ == "__main__":
    main()
