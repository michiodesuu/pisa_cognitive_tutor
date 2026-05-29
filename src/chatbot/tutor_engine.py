"""
tutor_engine.py
────────────────
The Neutral Socratic Tutoring Engine.

Responsibilities:
  1. Retrieve context from Qdrant (hybrid search) + graph subgraph
  2. Build the chat prompt (system + history + context + user message)
  3. Call the LLM (Ollama or HuggingFace backend)
  4. Stream tokens back to the UI
  5. NEVER assess cognitive dimensions (that is the offline evaluator's job)
  6. NEVER use forbidden judgment words

Strict separation: this module has NO import from src.evaluation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)

_FORBIDDEN_PATTERN = re.compile(
    r"\b(correct|incorrect|wrong|right answer|well done|good job|try again|"
    r"exactly|perfect|ถูกต้อง|ผิด|เก่งมาก|ดีมาก|ถูกแล้ว|ลองใหม่|正确|不对|很好|对了)\b",
    re.IGNORECASE | re.UNICODE,
)


def _sanitize_response(text: str) -> str:
    """Safety net: replace any forbidden judgment words that slipped through."""
    def replace_match(m):
        logger.warning(f"[Tutor] Forbidden word caught and removed: '{m.group()}'")
        return "..."
    return _FORBIDDEN_PATTERN.sub(replace_match, text)


class TutorEngine:
    """
    Neutral Socratic tutor.

    Parameters
    ----------
    config_path : Path
    searcher    : QdrantHybridSearch instance
    graph       : NetworkX DiGraph (may be None)
    """

    def __init__(self, config_path: Path, searcher, graph=None):
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.searcher = searcher
        self.graph = graph
        self._system_prompt = (
            Path("configs/prompts/socratic_tutor.txt").read_text(encoding="utf-8").strip()
        )
        self._llm_cfg = self.cfg["chatbot_llm"]

    # ── context retrieval ────────────────────────────────────────────────────

    async def _retrieve_context(self, user_message: str, category: Optional[str] = None) -> tuple[str, bool]:
        """
        Returns (context_block, kb_used_flag).
        Runs BGE-M3 search in a thread so it doesn't block the event loop.
        """
        import asyncio
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, lambda: self.searcher.search(user_message, top_k=5, category=category)
            )
            vector_ctx = self.searcher.format_context_for_prompt(results)
        except Exception as e:
            logger.warning(f"[Tutor] Retrieval failed: {e}")
            vector_ctx = ""
            results = []

        graph_ctx = ""
        if self.graph is not None and results:
            from ..ingestion.graph_rag_builder import KnowledgeGraphBuilder
            seed = results[0].get("correct_concept", user_message)
            gctx = await loop.run_in_executor(
                None,
                lambda: KnowledgeGraphBuilder.get_subgraph_context(
                    self.graph, seed,
                    max_hops=self.cfg["graph"]["max_hops"],
                    max_nodes=self.cfg["graph"]["max_subgraph_nodes"],
                ),
            )
            graph_ctx = f"\nGraph context: {gctx['graph_summary']}"

        kb_used = bool(results)
        full_ctx = (vector_ctx + graph_ctx).strip()
        return full_ctx, kb_used

    # ── prompt construction ──────────────────────────────────────────────────

    def _build_messages(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        context: str,
        turn_number: int,
        file_context: str = "",
    ) -> List[Dict[str, str]]:
        """Build the full message list for the LLM."""
        # Inject context into system prompt
        system = self._system_prompt
        if context:
            system += (
                f"\n\n<KNOWLEDGE_BASE_CONTEXT>\n{context}\n</KNOWLEDGE_BASE_CONTEXT>"
                f"\n\nCurrent turn: {turn_number}"
            )
        if file_context:                            # ← ADD THIS BLOCK
            system += (
                f"\n\n<STUDENT_UPLOADED_MATERIALS>\n"
                f"The student has uploaded the following material(s) with their message. "
                f"Use these to formulate your Socratic questions — ask about what they "
                f"observe in the uploaded content rather than explaining it yourself.\n\n"
                f"{file_context}\n"
                f"</STUDENT_UPLOADED_MATERIALS>"
            )
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    # ── LLM backends ─────────────────────────────────────────────────────────

    async def _stream_ollama(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream from local Ollama server."""
        import json as _json
        url = self._llm_cfg["ollama_base_url"].rstrip("/") + "/api/chat"
        payload = {
            "model": self._llm_cfg["model_name"],
            "messages": messages,
            "stream": True,
            "think": False,  # disable Qwen3 extended thinking — prevents multi-minute waits
            "options": {
                "temperature": self._llm_cfg["temperature"],
                "num_predict": self._llm_cfg["max_tokens"],
                "top_p": self._llm_cfg["top_p"],
                "num_ctx": 8192,  # ensure enough context for full history + system prompt
            },
        }
        # connect=10s, read=300s — connect must be fast; reading a full streamed
        # response can legitimately take minutes on a slow GPU
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = _json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except Exception:
                        continue

    async def _stream_hf(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream from a HuggingFace Transformers model loaded locally."""
        # Lazy import to avoid loading when using Ollama
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        import threading, torch
        model_id = self._llm_cfg.get("hf_model_id", "Qwen/Qwen2.5-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self._llm_cfg["max_tokens"],
            temperature=self._llm_cfg["temperature"],
            do_sample=True,
        )
        thread = threading.Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()

    # ── public API ────────────────────────────────────────────────────────────

    async def respond_stream(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        turn_number: int = 1,
        file_context: str = "",
        category: Optional[str] = None,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Async generator that yields (token, kb_used_flag) tuples.
        kb_used_flag is True only on the first token if context was retrieved.

        Usage in Chainlit:
          full_response = ""
          async for token, kb_used in engine.respond_stream(msg, history, turn):
              full_response += token
              cl_msg.stream_token(token)
        """
        context, kb_used = await self._retrieve_context(user_message, category=category)
        messages = self._build_messages(
            user_message, history, context, turn_number,
            file_context=file_context,     # ← ADD THIS ARGUMENT
        )

        backend = self._llm_cfg.get("backend", "ollama")
        first = True
        full_response = ""

        try:
            if backend == "ollama":
                gen = self._stream_ollama(messages)
            else:
                gen = self._stream_hf(messages)

            async for token in gen:
                full_response += token
                yield token, (kb_used and first)
                first = False

        except Exception as e:
            logger.error(
                f"[Tutor] LLM streaming failed ({type(e).__name__}): {e}",
                exc_info=True,
            )
            fallback = (
                "I seem to be having trouble thinking right now. "
                "Could you rephrase what you were exploring?"
            )
            yield fallback, False
            full_response = fallback

        # Final sanitization pass
        safe = _sanitize_response(full_response)
        if safe != full_response:
            # Log but do not re-stream (already streamed)
            logger.warning("[Tutor] Response contained forbidden words — sanitized in log")
