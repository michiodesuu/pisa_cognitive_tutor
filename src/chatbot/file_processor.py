"""
file_processor.py
──────────────────
Processes student-uploaded files for the tutoring context.

Supported inputs:
  - Images (PNG, JPG, WEBP, GIF) → described by Qwen3-VL-8B
  - PDF pages → rendered then described by VLM
  - CSV / Excel tables → parsed to Markdown table
  - Plain text / code → returned as-is with truncation

The output is always a clean text description injected into the
tutor's context block — the LLM never receives raw bytes.

Storage: files are held in an in-memory dict keyed by file_id.
file_id is a UUID tied to the session. No files are written to disk
(privacy) unless explicitly configured.
"""

from __future__ import annotations

import io
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# ── In-memory store: {file_id: {meta + description}} ─────────────────────────
# Keyed by session_id → list of file entries so they're scoped per student
_file_store: Dict[str, list] = {}

SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg",
                         "image/webp", "image/gif"}
SUPPORTED_TABLE_TYPES = {"text/csv", "application/vnd.ms-excel",
                         "application/vnd.openxmlformats-officedocument"
                         ".spreadsheetml.sheet"}
MAX_TEXT_CHARS = 3000   # truncation limit for plain text uploads
MAX_TABLE_ROWS = 50     # max rows shown in Markdown table


# ── VLM singleton (reuses the same instance as ingestion if loaded) ───────────
_vlm_model = None
_vlm_processor = None


def _ensure_vlm(cfg: dict):
    """Load Qwen3-VL if not already in memory."""
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_id = cfg["vlm"]["model_id"]
    local_path = Path(cfg["vlm"]["local_path"])
    source = str(local_path) if local_path.exists() else model_id

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg["vlm"]["torch_dtype"], torch.bfloat16)

    quantization_config = None
    if cfg["vlm"].get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg["vlm"].get("load_in_8bit"):
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        attn_implementation=cfg["vlm"]["attn_implementation"],
        device_map=cfg["hardware"]["vlm_device"],
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    try:
        _vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(source, **load_kwargs)
    except Exception:
        load_kwargs["attn_implementation"] = cfg["vlm"]["attn_fallback"]
        _vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(source, **load_kwargs)

    _vlm_processor = AutoProcessor.from_pretrained(
        source,
        min_pixels=cfg["vlm"]["min_pixels"] * 28 * 28,
        max_pixels=cfg["vlm"]["max_pixels"] * 28 * 28,
    )
    _vlm_model.eval()
    logger.info(f"[FileProcessor] VLM loaded: {model_id}")
    return _vlm_model, _vlm_processor


# ── Image description via VLM ─────────────────────────────────────────────────

def _describe_image_with_vlm(image: Image.Image, cfg: dict,
                              filename: str = "") -> str:
    """Run Qwen2.5-VL on a PIL Image and return a descriptive text."""
    import torch

    model, processor = _ensure_vlm(cfg)

    prompt = (
        "You are a precision scientific document parser assisting a Socratic science tutor. "
        "Your output will be injected directly into a tutoring system — be exhaustive and structured. "
        "Analyze every visual element present and report it with maximum fidelity. "
        "Do NOT summarize, interpret, or simplify — preserve all raw information exactly as shown.\n\n"

        "Follow this analysis protocol based on what you detect:\n\n"

        "## IF THE IMAGE CONTAINS A GRAPH OR CHART:\n"
        "- State the chart type (line, bar, scatter, pie, histogram, box plot, etc.)\n"
        "- X-axis: exact label, unit, scale type (linear/log), min value, max value, tick intervals\n"
        "- Y-axis: exact label, unit, scale type (linear/log), min value, max value, tick intervals\n"
        "- If a second Y-axis exists, describe it separately\n"
        "- List ALL data series by name, color, and marker style\n"
        "- Report key data points numerically (peaks, troughs, intercepts, plateaus, inflection points)\n"
        "- Describe the overall trend per series (increasing, decreasing, periodic, asymptotic, exponential, etc.)\n"
        "- Note any error bars, confidence intervals, or shaded regions with their values\n"
        "- State the title, legend text, footnotes, and any annotations exactly as written\n"
        "- Identify any outliers and their approximate coordinates\n\n"

        "## IF THE IMAGE CONTAINS A DATA TABLE:\n"
        "- Reproduce the ENTIRE table as a Markdown table — do not skip any rows or columns\n"
        "- Preserve exact column headers including units (e.g., 'Pressure (kPa)', 'Time (s)')\n"
        "- Preserve all numeric values exactly, including significant figures and decimal places\n"
        "- Note any merged cells, sub-headers, or hierarchical column structure\n"
        "- If the table has a title or caption, include it above the Markdown table\n"
        "- After the table, list any footnotes or symbols explained below it (e.g., *, †, N/A)\n\n"

        "## IF THE IMAGE CONTAINS A SCIENTIFIC DIAGRAM OR ILLUSTRATION:\n"
        "- Identify the type of diagram (experimental setup, biological system, circuit, process flow, "
        "structural model, chemical apparatus, anatomical cross-section, etc.)\n"
        "- List every labeled component with its exact label text\n"
        "- Describe spatial relationships: which component connects to which, in what direction, "
        "and using what medium (wire, pipe, arrow, membrane, etc.)\n"
        "- For directional diagrams (flow charts, reaction pathways, energy diagrams): state the "
        "sequence of steps and the direction of flow or transformation\n"
        "- For structural diagrams (molecules, cells, ecosystems): describe hierarchy and containment "
        "(e.g., 'mitochondria is enclosed within the cell membrane')\n"
        "- Note any numerical values attached to components (e.g., resistances, concentrations, "
        "distances, angles, voltages)\n"
        "- Describe any color coding and what each color represents\n\n"

        "## IF THE IMAGE CONTAINS EQUATIONS OR MATHEMATICAL EXPRESSIONS:\n"
        "- Reproduce every equation in LaTeX notation enclosed in $...$ for inline or $$...$$ for display\n"
        "- Define every symbol and variable that appears, including subscripts and superscripts\n"
        "- State the physical meaning of each term if labels or context suggest it\n"
        "- Note the equation number or label if present (e.g., 'Equation 3', 'Eq. 2.4')\n"
        "- If units are shown, include them\n\n"

        "## IF THE IMAGE CONTAINS TEXT PASSAGES OR QUESTION STEMS:\n"
        "- Transcribe the full text verbatim, preserving line breaks and paragraph structure\n"
        "- Reproduce any bold, italic, or underlined emphasis using Markdown (**bold**, *italic*, __underline__)\n"
        "- If numbered or lettered sub-questions are present, list each one separately\n\n"

        "## IF THE IMAGE CONTAINS MULTIPLE ELEMENTS (mixed graph + table + diagram, etc.):\n"
        "- Analyze each element in its own clearly labeled section using ## headers\n"
        "- After analyzing each element separately, add a final section:\n"
        "  '## RELATIONSHIPS BETWEEN ELEMENTS' describing how the components relate to each other "
        "(e.g., 'the table in Figure 2 provides the data plotted in Graph 1')\n\n"

        "## GENERAL REQUIREMENTS (apply to all images):\n"
        "- Languages: if text appears in Thai, Chinese, or any language other than English, "
        "transcribe it in the original script AND provide an English translation in brackets\n"
        "- Uncertainty: if a value is partially obscured or unclear, write it as '[unclear: ~42]' "
        "rather than guessing silently\n"
        "- Scale bars or rulers: if present, state the scale (e.g., '1 cm = 50 μm')\n"
        "- Do NOT write phrases like 'the image shows' or 'I can see' — "
        "output the content directly as structured facts\n"
        "- Do NOT add interpretation, conclusions, or tutoring commentary — "
        "that is the tutor's job, not yours\n"
        "- End your output with a one-line tag listing detected content types, e.g.:\n"
        "  [CONTENT: line_graph, data_table, labeled_diagram]"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    vlm_cfg = cfg["vlm"]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=vlm_cfg["max_new_tokens"],
            do_sample=vlm_cfg["do_sample"],
            temperature=vlm_cfg["temperature"],
            top_p=vlm_cfg["top_p"],
            top_k=vlm_cfg["top_k"],
            repetition_penalty=vlm_cfg["repetition_penalty"],
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
    ]
    description = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    prefix = f"[Image: {filename}]\n" if filename else "[Uploaded Image]\n"
    return prefix + description


# ── Table parsing ─────────────────────────────────────────────────────────────

def _parse_table(file_bytes: bytes, mime_type: str, filename: str) -> str:
    """Convert CSV or Excel to a Markdown table string."""
    try:
        import pandas as pd
        if mime_type == "text/csv" or filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes), nrows=MAX_TABLE_ROWS)
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), nrows=MAX_TABLE_ROWS)

        if df.empty:
            return f"[Table: {filename}] — empty file"

        md = df.to_markdown(index=False)
        header = f"[Table uploaded: {filename}] ({len(df)} rows shown)\n\n"
        return header + md
    except Exception as e:
        logger.warning(f"[FileProcessor] Table parse failed: {e}")
        return f"[Table: {filename}] — could not parse: {e}"


# ── PDF → image then VLM ──────────────────────────────────────────────────────

def _process_pdf(file_bytes: bytes, cfg: dict, filename: str) -> str:
    """Render first 3 pages of PDF → PIL Images → VLM description."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return f"[PDF: {filename}] — PyMuPDF not installed"

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_to_read = min(3, len(doc))
    descriptions = []

    zoom = cfg["vlm"]["pdf_render_dpi"] / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in range(pages_to_read):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        desc = _describe_image_with_vlm(img, cfg, filename=f"{filename} p.{i+1}")
        descriptions.append(desc)

    doc.close()
    return "\n\n---\n\n".join(descriptions)


# ── Plain text ────────────────────────────────────────────────────────────────

def _process_text(file_bytes: bytes, filename: str) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        return f"[File: {filename}] — could not decode as text"
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n\n[… truncated at {MAX_TEXT_CHARS} chars]"
    return f"[Text file: {filename}]\n\n{text}"


# ── Public API ────────────────────────────────────────────────────────────────

def process_upload(
    session_id: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    cfg: dict,
) -> Dict[str, Any]:
    """
    Process an uploaded file and store its description.

    Returns
    -------
    dict with keys:
        file_id   : str  (UUID to reference in WS message)
        filename  : str
        mime_type : str
        description : str  (text injected into tutor context)
        file_type : str   ("image" | "table" | "pdf" | "text")
    """
    file_id = str(uuid.uuid4())
    mime_lower = mime_type.lower()

    if mime_lower in SUPPORTED_IMAGE_TYPES or any(
        filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")
    ):
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # Resize if too large (> 2000px on longest side) to save VRAM
        max_side = 1800
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        description = _describe_image_with_vlm(image, cfg, filename)
        file_type = "image"

    elif mime_lower in SUPPORTED_TABLE_TYPES or any(
        filename.lower().endswith(ext) for ext in (".csv", ".xls", ".xlsx")
    ):
        description = _parse_table(file_bytes, mime_lower, filename)
        file_type = "table"

    elif mime_lower == "application/pdf" or filename.lower().endswith(".pdf"):
        description = _process_pdf(file_bytes, cfg, filename)
        file_type = "pdf"

    else:
        description = _process_text(file_bytes, filename)
        file_type = "text"

    entry = {
        "file_id": file_id,
        "filename": filename,
        "mime_type": mime_type,
        "description": description,
        "file_type": file_type,
    }

    if session_id not in _file_store:
        _file_store[session_id] = []
    _file_store[session_id].append(entry)

    logger.info(
        f"[FileProcessor] Processed {file_type} '{filename}' "
        f"for session {session_id[:8]} → {len(description)} chars"
    )
    return entry


def get_file_descriptions(session_id: str, file_ids: list) -> str:
    """
    Retrieve descriptions for a list of file_ids belonging to a session.
    Returns a formatted context block ready for injection into the prompt.
    """
    entries = _file_store.get(session_id, [])
    id_map = {e["file_id"]: e for e in entries}

    blocks = []
    for fid in file_ids:
        entry = id_map.get(fid)
        if entry:
            blocks.append(
                f"<STUDENT_UPLOADED_{entry['file_type'].upper()}>\n"
                f"{entry['description']}\n"
                f"</STUDENT_UPLOADED_{entry['file_type'].upper()}>"
            )
        else:
            logger.warning(f"[FileProcessor] file_id {fid} not found for session {session_id[:8]}")

    return "\n\n".join(blocks)


def cleanup_session_files(session_id: str):
    """Call on session end to free memory."""
    _file_store.pop(session_id, None)