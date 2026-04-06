from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contract import semantic_contract_json_schema


def _load_api_key_from_dotenv(dotenv_path: Optional[str | Path]) -> Optional[str]:
    if not dotenv_path:
        return None
    path = Path(dotenv_path).expanduser()
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            return value.strip().strip('"').strip("'")
    return None


def api_key_available(*, dotenv_path: Optional[str | Path] = None) -> bool:
    return resolve_openai_api_key(dotenv_path=dotenv_path) is not None


def resolve_openai_api_key(*, api_key: Optional[str] = None, dotenv_path: Optional[str | Path] = None) -> Optional[str]:
    if api_key:
        return str(api_key).strip()
    try:
        import os

        env_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
        if env_key:
            return env_key
    except Exception:
        pass
    return _load_api_key_from_dotenv(dotenv_path)


def _extract_text_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _encode_image_path(path: str | Path) -> Tuple[str, str]:
    image_path = Path(path).expanduser()
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return mime, data


class OpenAIVLMSemanticClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        dotenv_path: Optional[str | Path] = None,
        timeout_s: float = 180.0,
    ) -> None:
        resolved_key = resolve_openai_api_key(api_key=api_key, dotenv_path=dotenv_path)
        self.api_key = resolved_key
        self.timeout_s = float(timeout_s)
        self._client = None
        if resolved_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=resolved_key, timeout=self.timeout_s)

    @property
    def available(self) -> bool:
        return self._client is not None

    def build_chat_completion_request(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path],
        model_name: str,
        image_detail: str,
        max_completion_tokens: int = 1800,
        json_schema: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for path in image_paths:
            mime, encoded = _encode_image_path(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}",
                        "detail": str(image_detail),
                    },
                }
            )
        content.append({"type": "text", "text": str(prompt)})
        return {
            "model": str(model_name),
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": int(max_completion_tokens),
            "reasoning_effort": "low",
            "response_format": {
                "type": "json_schema",
                "json_schema": dict(json_schema or semantic_contract_json_schema()),
            },
        }

    def label_contract(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str | Path],
        model_name: str,
        image_detail: str,
        max_completion_tokens: int = 1800,
        json_schema: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("OpenAI API key is not available")
        request = self.build_chat_completion_request(
            prompt=prompt,
            image_paths=image_paths,
            model_name=model_name,
            image_detail=image_detail,
            max_completion_tokens=max_completion_tokens,
            json_schema=json_schema,
        )
        response = self._client.chat.completions.create(**request)
        text = _extract_text_content(response.choices[0].message)
        if not text:
            raise ValueError(f"OpenAI returned empty content for model={model_name!r}")
        return json.loads(text)

    def write_batch_requests(
        self,
        *,
        requests: Sequence[Mapping[str, Any]],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in requests:
                f.write(json.dumps(dict(row), sort_keys=True) + "\n")
        return path

    def submit_batch(
        self,
        *,
        request_manifest_path: str | Path,
        completion_window: str = "24h",
        endpoint: str = "/v1/chat/completions",
    ) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("OpenAI API key is not available")
        manifest_path = Path(request_manifest_path).expanduser()
        with manifest_path.open("rb") as f:
            uploaded = self._client.files.create(file=f, purpose="batch")
        batch = self._client.batches.create(
            input_file_id=str(uploaded.id),
            endpoint=str(endpoint),
            completion_window=str(completion_window),
        )
        return {
            "input_file_id": str(uploaded.id),
            "batch_id": str(batch.id),
            "status": str(batch.status),
        }

    def wait_for_batch(
        self,
        *,
        batch_id: str,
        poll_interval_s: float = 20.0,
        timeout_s: float = 3600.0,
    ) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("OpenAI API key is not available")
        start = time.time()
        while True:
            batch = self._client.batches.retrieve(str(batch_id))
            status = str(batch.status)
            if status in {"completed", "failed", "expired", "cancelled"}:
                return {
                    "batch_id": str(batch.id),
                    "status": status,
                    "output_file_id": None if getattr(batch, "output_file_id", None) is None else str(batch.output_file_id),
                    "error_file_id": None if getattr(batch, "error_file_id", None) is None else str(batch.error_file_id),
                }
            if (time.time() - start) > float(timeout_s):
                raise TimeoutError(f"Timed out waiting for batch {batch_id}")
            time.sleep(float(poll_interval_s))

    def download_batch_output(self, *, file_id: str, output_path: str | Path) -> Path:
        if not self.available:
            raise RuntimeError("OpenAI API key is not available")
        response = self._client.files.content(str(file_id))
        content = response.text if hasattr(response, "text") else response.read().decode("utf-8")
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


def parse_batch_output_line(row: Mapping[str, Any]) -> Dict[str, Any]:
    response = dict(row.get("response") or {})
    body = dict(response.get("body") or {})
    choices = list(body.get("choices") or [])
    content_text = ""
    if choices:
        message = dict(choices[0].get("message") or {})
        content = message.get("content")
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(str(item["text"]))
            content_text = "\n".join(text_parts).strip()
    return {
        "custom_id": str(row.get("custom_id") or ""),
        "status_code": int(response.get("status_code") or 0),
        "body": body,
        "content_text": content_text,
    }
