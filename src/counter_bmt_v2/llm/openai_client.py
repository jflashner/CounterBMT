"""Minimal OpenAI chat wrapper for multimodal/text calls.

This module isolates external API calls so perception and DAG stages can share
one client abstraction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OpenAIChatClient:
    model: str = "gpt-5-mini"
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError("openai package is required for OpenAIChatClient") from exc

        self._client = OpenAI(api_key=key, timeout=self.timeout_s)

    def _is_gpt5_family(self) -> bool:
        return str(self.model).strip().lower().startswith("gpt-5")

    @staticmethod
    def _safe_dump_response(resp: Any) -> str:
        try:
            if hasattr(resp, "model_dump_json"):
                return str(resp.model_dump_json(indent=2))
        except Exception:
            pass
        try:
            if hasattr(resp, "model_dump"):
                return json.dumps(resp.model_dump(), indent=2, default=str)
        except Exception:
            pass
        return repr(resp)

    @staticmethod
    def _extract_text_content(message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue
                text = None
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                    if text is None and getattr(item, "type", None) == "output_text":
                        text = getattr(item, "content", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip()
        return ""

    def complete(
        self,
        *,
        prompt: str,
        images_base64: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        content: List[Dict[str, Any]] = []
        for img in images_base64 or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}",
                        "detail": "high",
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            # GPT-5-family chat requests expect `max_completion_tokens`.
            # Older model families generally tolerate it as well, but keep a
            # fallback for older SDK/server combinations below.
            "max_completion_tokens": max_tokens,
        }
        if not self._is_gpt5_family():
            request["temperature"] = temperature
        if response_format is not None:
            request["response_format"] = response_format
        try:
            resp = self._client.chat.completions.create(**request)
        except TypeError:
            request.pop("max_completion_tokens", None)
            request["max_tokens"] = max_tokens
            resp = self._client.chat.completions.create(**request)

        choice0 = resp.choices[0]
        message = choice0.message
        text = self._extract_text_content(message)
        if text:
            return text

        request_id = getattr(resp, "_request_id", None)
        finish_reason = getattr(choice0, "finish_reason", None)
        refusal = getattr(message, "refusal", None)
        raw_response = self._safe_dump_response(resp)
        exc = ValueError(
            "OpenAI returned empty assistant content. "
            f"model={self.model!r} finish_reason={finish_reason!r} "
            f"refusal={refusal!r} request_id={request_id!r}"
        )
        setattr(exc, "raw_response", raw_response)
        setattr(exc, "raw_excerpt", raw_response[:2000])
        raise exc
