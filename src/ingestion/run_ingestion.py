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
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml
from tqdm import tqdm


class _TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


from .graph_rag_builder import KnowledgeGraphBuilder
from .pdf_processor import PDFProcessor
from .vlm_extractor import VLMExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        _TqdmLoggingHandler(),
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
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
            tqdm.write(f"[RESUME] Skipping {len(processed)} already-processed file(s)")

    pending = [p for p in pdfs if p.name not in processed]
    if not pending:
        tqdm.write("[PIPELINE] All PDFs already processed. Nothing to do.")
    else:
        tqdm.write(f"[PIPELINE] {len(pending)} PDF(s) to process, {len(processed)} skipped")

    loop = asyncio.get_event_loop()
    total_records = 0
    t_start = time.perf_counter()

    with output_path.open("a", encoding="utf-8") as out_f:
        pdf_bar = tqdm(pending, desc="PDFs", unit="pdf", position=0)
        for pdf_path in pdf_bar:
            pdf_bar.set_description(f"PDF: {pdf_path.name[:38]}")
            file_records: List[Dict[str, Any]] = []

            page_count = await processor.get_page_count(pdf_path)
            n_shards = math.ceil(page_count / cfg["vlm"]["max_pages_per_shard"])

            shard_bar = tqdm(
                total=n_shards,
                desc="  shards",
                unit="shard",
                position=1,
                leave=False,
            )
            async for shard in processor.shard_pdf(pdf_path):
                records = await loop.run_in_executor(
                    None,
                    extractor.extract_shard,
                    shard.images,
                    pdf_path.name,
                    shard.page_range_str,
                )
                file_records.extend(records)
                shard_bar.update(1)
                shard_bar.set_postfix(records=len(file_records), pages=shard.page_range_str)
            shard_bar.close()

            for rec in file_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            total_records += len(file_records)
            pdf_bar.set_postfix(total_records=total_records)
            tqdm.write(f"  [DONE] {pdf_path.name}: {len(file_records)} records")

    elapsed = time.perf_counter() - t_start
    tqdm.write(f"\n[PIPELINE] Extraction complete: {total_records} records in {elapsed:.1f}s")
    logger.info(f"[PIPELINE] COMPLETE: {total_records} total records in {elapsed:.1f}s")

    # ── Build knowledge graph ─────────────────────────────────────────────────
    tqdm.write("[PIPELINE] Building knowledge graph...")
    with tqdm(total=1, desc="Knowledge graph", unit="step") as bar:
        builder = KnowledgeGraphBuilder(config_path)
        builder.ingest_jsonl(output_path)
        builder.save()
        bar.update(1)

    # ── Ingest into Qdrant ────────────────────────────────────────────────────
    tqdm.write("[PIPELINE] Ingesting into Qdrant vector DB...")
    from ..retrieval.embedder import BGEEmbedder
    from ..retrieval.hybrid_search import QdrantHybridSearch

    embedder = BGEEmbedder(config_path)
    searcher = QdrantHybridSearch(config_path, embedder)
    searcher.ingest_jsonl(output_path)
    tqdm.write("[PIPELINE] Done.")


def main():
    parser = argparse.ArgumentParser(description="Run PISA ingestion pipeline")
    parser.add_argument("--config", default="configs/model_configs.yaml")
    args = parser.parse_args()
    asyncio.run(run_pipeline(Path(args.config)))


if __name__ == "__main__":
    main()
