"""
pdf_processor.py
─────────────────
Async PDF-to-image pipeline.  Converts each page to a PIL Image using PyMuPDF
and yields shards of N pages at a time.  Runs entirely on CPU so VLM VRAM
remains free.

Producer in the ingestion pipeline:
  PDFProcessor ──(page images)──> VLMExtractor ──(JSON)──> GraphRAGBuilder
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PageShard:
    """A shard of consecutive pages from one PDF."""
    pdf_path: Path
    shard_index: int
    page_range: Tuple[int, int]   # (start_page, end_page) inclusive, 0-indexed
    images: List[Image.Image] = field(default_factory=list)

    @property
    def page_range_str(self) -> str:
        return f"{self.page_range[0]+1}-{self.page_range[1]+1}"


class PDFProcessor:
    """
    Async PDF processor.

    Parameters
    ----------
    dpi : int
        Rendering resolution.  200 DPI produces ~1654×2339 px for A4 — enough
        for Qwen3-VL to read math and tables without blurring.
    max_pages_per_shard : int
        Number of pages per shard.  Larger = more VRAM per forward pass.
        Keep at 3 for 24 GB VRAM with Qwen3-VL-8B.
    """

    def __init__(self, dpi: int = 200, max_pages_per_shard: int = 3):
        self.dpi = dpi
        self.max_pages_per_shard = max_pages_per_shard
        # Pre-compute the zoom matrix once
        zoom = dpi / 72.0
        self._mat = fitz.Matrix(zoom, zoom)

    # ── internal ──────────────────────────────────────────────────────────────

    def _render_page(self, page: fitz.Page) -> Image.Image:
        """Render a single fitz.Page to a PIL Image (CPU-bound)."""
        pix = page.get_pixmap(matrix=self._mat, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    async def _render_page_async(
        self, loop: asyncio.AbstractEventLoop, page: fitz.Page
    ) -> Image.Image:
        return await loop.run_in_executor(None, self._render_page, page)

    # ── public API ────────────────────────────────────────────────────────────

    async def shard_pdf(
        self, pdf_path: Path
    ) -> AsyncGenerator[PageShard, None]:
        """
        Async generator.  Yields PageShard objects, each containing
        up to `max_pages_per_shard` rendered PIL Images.

        Usage
        -----
        async for shard in processor.shard_pdf(pdf_path):
            results = await vlm.extract_shard(shard)
        """
        doc: Optional[fitz.Document] = None
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            logger.info(
                f"[PDFProcessor] {pdf_path.name}: {total_pages} pages, "
                f"sharding at {self.max_pages_per_shard} pages/shard"
            )
            loop = asyncio.get_event_loop()
            shard_idx = 0
            n_shards = math.ceil(total_pages / self.max_pages_per_shard)

            page_bar = tqdm(
                total=total_pages,
                desc=f"    rendering {pdf_path.name[:30]}",
                unit="page",
                position=2,
                leave=False,
            )
            for start in range(0, total_pages, self.max_pages_per_shard):
                end = min(start + self.max_pages_per_shard - 1, total_pages - 1)

                tasks = [
                    self._render_page_async(loop, doc[p])
                    for p in range(start, end + 1)
                ]
                images = await asyncio.gather(*tasks)
                page_bar.update(end - start + 1)

                shard = PageShard(
                    pdf_path=pdf_path,
                    shard_index=shard_idx,
                    page_range=(start, end),
                    images=list(images),
                )
                yield shard
                shard_idx += 1

                await asyncio.sleep(0)
            page_bar.close()

        except Exception as exc:
            logger.error(f"[PDFProcessor] Failed to process {pdf_path}: {exc}")
            raise
        finally:
            if doc:
                doc.close()

    async def get_page_count(self, pdf_path: Path) -> int:
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count

    @staticmethod
    def discover_pdfs(raw_pdf_dir: Path) -> List[Path]:
        """Return sorted list of all PDFs in directory."""
        pdfs = sorted(raw_pdf_dir.glob("*.pdf"))
        logger.info(f"[PDFProcessor] Discovered {len(pdfs)} PDF(s) in {raw_pdf_dir}")
        return pdfs
