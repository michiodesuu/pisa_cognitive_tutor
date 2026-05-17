"""
vlm_extractor.py
─────────────────
Runs Qwen2.5-VL-7B locally to extract structured JSON knowledge records
from PISA PDF page images.  Bypasses OCR entirely — the VLM reads math,
tables, Thai/Chinese/English text natively from the raw image.

Consumer in the ingestion pipeline:
  PDFProcessor ──(PageShard)──> VLMExtractor ──(List[dict])──> GraphRAGBuilder
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

# ── lazy imports to avoid loading at module level ─────────────────────────────
_qwen_model = None
_qwen_processor = None


def _load_model(cfg: dict) -> tuple:
    """Load Qwen2.5-VL model and processor (singleton)."""
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return _qwen_model, _qwen_processor

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = cfg["vlm"]["model_id"]
    local_path = Path(cfg["vlm"]["local_path"])
    source = str(local_path) if local_path.exists() else model_id

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg["vlm"]["torch_dtype"], torch.bfloat16)

    attn = cfg["vlm"]["attn_implementation"]
    try:
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            attn_implementation=attn,
            device_map=cfg["vlm"]["vlm_device"],
        )
    except Exception:
        logger.warning(
            f"[VLM] {attn} not available, falling back to {cfg['vlm']['attn_fallback']}"
        )
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            attn_implementation=cfg["vlm"]["attn_fallback"],
            device_map=cfg["vlm"]["vlm_device"],
        )

    _qwen_processor = AutoProcessor.from_pretrained(
        source,
        min_pixels=cfg["vlm"]["min_pixels"] * 28 * 28,
        max_pixels=cfg["vlm"]["max_pixels"] * 28 * 28,
    )

    _qwen_model.eval()
    logger.info(f"[VLM] Loaded {model_id} on {cfg['vlm']['vlm_device']}")
    return _qwen_model, _qwen_processor


def _extract_json_list(raw: str) -> List[Dict[str, Any]]:
    """
    Surgical JSON extractor.  Finds the outermost [...] block even if the
    model prepends/appends stray text or markdown fences.
    """
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Find outermost [...] via bracket matching
    start = raw.find("[")
    if start == -1:
        return []
    depth = 0
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass
                break
    return []


def _validate_record(record: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
    """Validate and normalise a single extracted record."""
    required = [
        "topic", "pisa_competency", "question_summary",
        "correct_concept", "common_misconceptions",
    ]
    for key in required:
        if key not in record or not record[key]:
            logger.warning(f"[VLM] Record from {source_file} missing field '{key}' — skipping")
            return None

    # Ensure correct types
    if isinstance(record.get("common_misconceptions"), list):
        record["common_misconceptions"] = "; ".join(record["common_misconceptions"])
    if not isinstance(record.get("key_vocabulary"), list):
        record["key_vocabulary"] = []
    if not isinstance(record.get("sub_questions"), list):
        record["sub_questions"] = []

    record.setdefault("source_file", source_file)
    record.setdefault("pisa_knowledge_type", "Content")
    record.setdefault("cognitive_demand", "Medium")
    record.setdefault("visual_description", "")
    return record


class VLMExtractor:
    """
    Extracts structured PISA knowledge records from page images using
    Qwen2.5-VL-7B running locally.

    Parameters
    ----------
    config_path : Path
        Path to model_configs.yaml
    prompt_path : Path
        Path to vision_extractor.txt
    """

    def __init__(self, config_path: Path, prompt_path: Path):
        self.cfg = yaml.safe_load(config_path.read_text())
        self.system_prompt = prompt_path.read_text().strip()
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._processor = _load_model(self.cfg)

    def _build_messages(self, images: List[Image.Image], source_file: str, page_range_str: str):
        """Build the Qwen2.5-VL chat message with image content."""
        content = []
        for img in images:
            content.append({"type": "image", "image": img})

        content.append({
            "type": "text",
            "text": (
                f"Source file: {source_file}, pages: {page_range_str}\n"
                "Extract ALL PISA question units from these pages. "
                "Return ONLY the JSON array, nothing else."
            ),
        })

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    def extract_shard(
        self, images: List[Image.Image], source_file: str, page_range_str: str
    ) -> List[Dict[str, Any]]:
        """
        Run VLM inference on a shard of page images.
        Returns a list of validated knowledge records.
        Runs synchronously (call from async context via run_in_executor).
        """
        self._ensure_loaded()

        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(images, source_file, page_range_str)

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.cfg["vlm"]["vlm_device"])

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg["vlm"]["max_new_tokens"],
                temperature=self.cfg["vlm"]["temperature"],
                do_sample=self.cfg["vlm"]["do_sample"],
            )

        # Decode only the generated tokens (strip the input)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        elapsed = time.perf_counter() - t0

        logger.info(
            f"[VLM] {source_file} pages {page_range_str}: "
            f"{len(raw)} chars in {elapsed:.1f}s"
        )

        records = _extract_json_list(raw)
        validated = []
        for rec in records:
            v = _validate_record(rec, source_file)
            if v:
                validated.append(v)

        logger.info(
            f"[VLM] {source_file} pages {page_range_str}: "
            f"{len(validated)}/{len(records)} valid records"
        )
        return validated
