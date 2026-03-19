"""Runtime guard helpers for real-run vs debug-fallback behavior."""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence, Tuple

_GPT4O_ALIAS_WARNED = False


def normalize_openai_backend(value: str, *, field_name: str) -> str:
    """Normalize OpenAI backend aliases to the canonical backend name."""
    global _GPT4O_ALIAS_WARNED

    backend = str(value).strip().lower()
    if backend == "gpt4o":
        if not _GPT4O_ALIAS_WARNED:
            warnings.warn(
                f"`{field_name}=gpt4o` is deprecated; use `{field_name}=openai`.",
                DeprecationWarning,
                stacklevel=2,
            )
            _GPT4O_ALIAS_WARNED = True
        return "openai"
    return backend


def require_debug_fallbacks(
    *,
    allow_debug_fallbacks: bool,
    violations: Sequence[Tuple[str, str]],
) -> None:
    if allow_debug_fallbacks or not violations:
        return

    parts = [f"{name}={value}" for name, value in violations]
    joined = ", ".join(parts)
    raise ValueError(
        "Debug-only setting(s) require `--allow-debug-fallbacks`: "
        f"{joined}"
    )


def coalesce_debug_fallbacks(
    *,
    allow_debug_fallbacks: bool,
    legacy_fallback_flag: bool | None = None,
) -> bool:
    """Preserve explicit legacy fallback flags while preferring the shared guard."""
    if legacy_fallback_flag is not None:
        return bool(legacy_fallback_flag)
    return bool(allow_debug_fallbacks)


def collect_debug_violations(
    items: Iterable[Tuple[str, str, bool]],
) -> list[Tuple[str, str]]:
    violations: list[Tuple[str, str]] = []
    for name, value, enabled in items:
        if enabled:
            violations.append((str(name), str(value)))
    return violations
