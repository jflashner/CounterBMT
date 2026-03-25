"""Minimal OpenAI chat wrapper for multimodal/text calls.

This module isolates external API calls so perception and DAG stages can share
one client abstraction.
"""

from __future__ import annotations

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

    def complete(
        self,
        *,
        prompt: str,
        images_base64: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
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
            "temperature": temperature,
            # GPT-5-family chat requests expect `max_completion_tokens`.
            # Older model families generally tolerate it as well, but keep a
            # fallback for older SDK/server combinations below.
            "max_completion_tokens": max_tokens,
        }
        try:
            resp = self._client.chat.completions.create(**request)
        except TypeError:
            request.pop("max_completion_tokens", None)
            request["max_tokens"] = max_tokens
            resp = self._client.chat.completions.create(**request)
        return (resp.choices[0].message.content or "").strip()
