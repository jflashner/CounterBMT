from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from scripts.counterfactual.label_waymax_sdc_path_semantics import (  # type: ignore[attr-defined]
    _aggregate_scene_ambiguity,
    _slot_path_row,
    write_json,
    write_jsonl,
)
from bmt.counterfactual.vlm_semantics.client import OpenAIVLMSemanticClient, parse_batch_output_line
from bmt.counterfactual.vlm_semantics.sdc_path_contract import (
    SLOT_IDS,
    make_empty_sdc_path_contract,
    normalize_sdc_path_contract,
)


DEFAULT_BATCH_MAX_BYTES = 180_000_000
DEFAULT_BATCH_MAX_REQUESTS = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM labeling on an already-rendered postsplit semantics bundle.")
    parser.add_argument("--bundle-root", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--image-detail", type=str, default="")
    parser.add_argument("--dotenv", type=str, default="")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-s", type=float, default=3.0)
    parser.add_argument("--batch-mode", action="store_true")
    parser.add_argument("--submit-batch", action="store_true")
    parser.add_argument("--wait-for-batch", action="store_true")
    parser.add_argument("--completion-window", type=str, default="24h")
    parser.add_argument("--batch-max-bytes", type=int, default=DEFAULT_BATCH_MAX_BYTES)
    parser.add_argument("--batch-max-requests", type=int, default=DEFAULT_BATCH_MAX_REQUESTS)
    parser.add_argument("--batch-shard-indices", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _example_dir_from_request_json(path: Path) -> Path:
    return path.expanduser().resolve().parent


def _contract_available(example_dir: Path) -> bool:
    return (example_dir / "contract_normalized.json").is_file()


def _slot_contract_paths(example_dir: Path, slot_id: str) -> Tuple[Path, Path]:
    return example_dir / f"contract_raw_{slot_id}.json", example_dir / f"contract_normalized_{slot_id}.json"


def _slot_contract_available(example_dir: Path, slot_id: str) -> bool:
    raw_path, normalized_path = _slot_contract_paths(example_dir, slot_id)
    return raw_path.is_file() and normalized_path.is_file()


def _load_slot_rows(render_metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in list(render_metadata.get("slot_metadata") or []) if str(row.get("slot_id") or "") in SLOT_IDS]


def _collect_completed_bundle_rows(
    *,
    all_examples: List[tuple[str, List[Dict[str, Any]]]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    for example_id, slot_requests in all_examples:
        example_dir = _example_dir_from_request_json(Path(slot_requests[0]["request_json"]))
        aggregated_raw_path = example_dir / "contract_raw.json"
        aggregated_norm_path = example_dir / "contract_normalized.json"
        render_metadata_path = example_dir / "render_metadata.json"
        if (not aggregated_raw_path.is_file()) or (not aggregated_norm_path.is_file()) or (not render_metadata_path.is_file()):
            continue
        render_metadata = _read_json(render_metadata_path)
        normalized_contract = _read_json(aggregated_norm_path)
        raw_rows.append(
            {
                "example_id": str(example_id),
                "scenario_id": str(render_metadata.get("scenario_id") or ""),
                "sdc_id": str(render_metadata.get("sdc_id") or ""),
                "raw_contract_path": str(aggregated_raw_path.resolve()),
                "normalized_contract_path": str(aggregated_norm_path.resolve()),
                "contract": normalized_contract,
            }
        )
        aggregate_rows.append(
            {
                "example_id": str(example_id),
                "scenario_id": str(render_metadata.get("scenario_id") or ""),
                "sdc_id": str(render_metadata.get("sdc_id") or ""),
                "current_time_index": int(render_metadata.get("current_time_index") or 0),
                "slot_metadata": render_metadata.get("slot_metadata"),
                "images": render_metadata.get("images"),
                "prompt_paths": {
                    slot_id: str((example_dir / f"prompt_{slot_id}.txt").resolve())
                    for slot_id in SLOT_IDS
                    if (example_dir / f"prompt_{slot_id}.txt").is_file()
                },
                "request_jsons": {
                    slot_id: str((example_dir / f"request_{slot_id}.json").resolve())
                    for slot_id in SLOT_IDS
                    if (example_dir / f"request_{slot_id}.json").is_file()
                },
                "contract": normalized_contract,
            }
        )
    return raw_rows, aggregate_rows


def _build_custom_id(*, example_id: str, slot_id: str) -> str:
    return f"{example_id}::{slot_id}"


def _load_request_payload(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(spec["request_payload"])


def _resolved_request_settings(
    spec: Mapping[str, Any],
    *,
    model_override: str,
    image_detail_override: str,
) -> Tuple[str, str, str, List[str], Dict[str, Any]]:
    request_payload = _load_request_payload(spec)
    model_name = str(model_override).strip() or str(request_payload.get("model") or "")
    image_detail = str(image_detail_override).strip() or str(request_payload.get("image_detail") or "")
    prompt = str(request_payload.get("prompt") or "")
    image_paths = [str(path) for path in list(request_payload.get("image_paths") or []) if str(path).strip()]
    json_schema = dict(request_payload.get("json_schema") or {})
    return model_name, image_detail, prompt, image_paths, json_schema


def _build_slot_specs(
    *,
    all_examples: Sequence[tuple[str, List[Dict[str, Any]]]],
    skip_existing: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    specs: List[Dict[str, Any]] = []
    example_contexts: Dict[str, Dict[str, Any]] = {}
    for example_id, slot_requests in all_examples:
        example_dir = _example_dir_from_request_json(Path(slot_requests[0]["request_json"]))
        render_metadata_path = example_dir / "render_metadata.json"
        if not render_metadata_path.is_file():
            continue
        render_metadata = _read_json(render_metadata_path)
        example_contexts[str(example_id)] = {
            "example_dir": example_dir,
            "render_metadata": render_metadata,
            "slot_rows": _load_slot_rows(render_metadata),
            "scenario_id": str(render_metadata.get("scenario_id") or ""),
            "sdc_id": str(render_metadata.get("sdc_id") or ""),
            "current_time_index": int(render_metadata.get("current_time_index") or 0),
        }
        for request_row in sorted(slot_requests, key=lambda row: SLOT_IDS.index(str(row.get("slot_id") or "gt"))):
            slot_id = str(request_row.get("slot_id") or "")
            if slot_id not in SLOT_IDS:
                continue
            if bool(skip_existing) and _slot_contract_available(example_dir, slot_id):
                continue
            request_json_path = Path(str(request_row["request_json"])).expanduser().resolve()
            specs.append(
                {
                    "custom_id": _build_custom_id(example_id=str(example_id), slot_id=slot_id),
                    "example_id": str(example_id),
                    "slot_id": slot_id,
                    "scenario_id": str(request_row.get("scenario_id") or render_metadata.get("scenario_id") or ""),
                    "sdc_id": str(request_row.get("sdc_id") or render_metadata.get("sdc_id") or ""),
                    "request_json_path": str(request_json_path),
                    "request_payload": _read_json(request_json_path),
                }
            )
    return specs, example_contexts


def _build_batch_item(
    *,
    spec: Mapping[str, Any],
    client: OpenAIVLMSemanticClient,
    model_override: str,
    image_detail_override: str,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    model_name, image_detail, prompt, image_paths, json_schema = _resolved_request_settings(
        spec,
        model_override=model_override,
        image_detail_override=image_detail_override,
    )
    body = client.build_chat_completion_request(
        prompt=prompt,
        image_paths=image_paths,
        model_name=model_name,
        image_detail=image_detail,
        max_completion_tokens=int(max_completion_tokens),
        json_schema=json_schema,
    )
    row = {
        "custom_id": str(spec["custom_id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return {
        "custom_id": str(spec["custom_id"]),
        "request_row": row,
        "approx_bytes": len(json.dumps(row, sort_keys=True)),
        "example_id": str(spec["example_id"]),
        "slot_id": str(spec["slot_id"]),
    }


def _shard_batch_items(
    *,
    batch_items: Sequence[Mapping[str, Any]],
    max_requests: int,
    max_bytes: int,
) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_bytes = 0
    max_requests = max(1, int(max_requests))
    max_bytes = max(1, int(max_bytes))
    for item in batch_items:
        item_dict = dict(item)
        item_bytes = int(item_dict.get("approx_bytes") or 0)
        would_overflow = bool(current) and (
            (len(current) >= max_requests) or ((current_bytes + item_bytes) > max_bytes)
        )
        if would_overflow:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(item_dict)
        current_bytes += item_bytes
    if current:
        shards.append(current)
    return shards


def _write_batch_shards(
    *,
    client: OpenAIVLMSemanticClient,
    batch_items: Sequence[Mapping[str, Any]],
    batch_root: Path,
    max_requests: int,
    max_bytes: int,
) -> List[Dict[str, Any]]:
    manifest_dir = batch_root / "request_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shards = _shard_batch_items(batch_items=batch_items, max_requests=max_requests, max_bytes=max_bytes)
    job_rows: List[Dict[str, Any]] = []
    for shard_idx, shard in enumerate(shards):
        manifest_path = manifest_dir / f"batch_shard_{shard_idx:04d}.jsonl"
        client.write_batch_requests(
            requests=[dict(item["request_row"]) for item in shard],
            output_path=manifest_path,
        )
        job_rows.append(
            {
                "shard_index": int(shard_idx),
                "request_manifest_path": str(manifest_path.resolve()),
                "num_requests": int(len(shard)),
                "approx_bytes": int(sum(int(item["approx_bytes"]) for item in shard)),
                "custom_ids": [str(item["custom_id"]) for item in shard],
                "status": "prepared",
                "input_file_id": "",
                "batch_id": "",
                "output_file_id": "",
                "error_file_id": "",
                "batch_output_file": "",
            }
        )
    return job_rows


def _parse_batch_shard_indices(text: str) -> Optional[set[int]]:
    cleaned = str(text).strip()
    if not cleaned:
        return None
    shard_indices: set[int] = set()
    for part in cleaned.split(","):
        token = str(part).strip()
        if not token:
            continue
        shard_indices.add(int(token))
    return shard_indices


def _filter_batch_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    shard_indices: Optional[set[int]],
) -> List[Dict[str, Any]]:
    if not shard_indices:
        return [dict(job) for job in jobs]
    return [
        dict(job)
        for job in jobs
        if int(job.get("shard_index") or -1) in shard_indices
    ]


def _custom_ids_for_jobs(jobs: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for job in jobs:
        for custom_id in list(job.get("custom_ids") or []):
            token = str(custom_id).strip()
            if token:
                out.add(token)
    return out


def _merge_batch_jobs(
    *,
    base_jobs: Sequence[Mapping[str, Any]],
    updated_jobs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    merged = {int(job.get("shard_index") or -1): dict(job) for job in base_jobs}
    for job in updated_jobs:
        merged[int(job.get("shard_index") or -1)] = dict(job)
    return [merged[idx] for idx in sorted(merged)]


def _submit_batch_jobs(
    *,
    client: OpenAIVLMSemanticClient,
    jobs: Sequence[Mapping[str, Any]],
    completion_window: str,
) -> List[Dict[str, Any]]:
    submitted: List[Dict[str, Any]] = []
    for job in jobs:
        job_row = dict(job)
        payload = client.submit_batch(
            request_manifest_path=job_row["request_manifest_path"],
            completion_window=str(completion_window),
        )
        job_row.update(payload)
        job_row["status"] = str(payload.get("status") or "submitted")
        submitted.append(job_row)
    return submitted


def _wait_and_download_batch_jobs(
    *,
    client: OpenAIVLMSemanticClient,
    jobs: Sequence[Mapping[str, Any]],
    batch_root: Path,
) -> List[Dict[str, Any]]:
    output_dir = batch_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: List[Dict[str, Any]] = []
    for job in jobs:
        job_row = dict(job)
        batch_id = str(job_row.get("batch_id") or "")
        if not batch_id:
            completed.append(job_row)
            continue
        status = client.wait_for_batch(batch_id=batch_id)
        job_row.update(status)
        output_file_id = str(status.get("output_file_id") or "")
        if output_file_id:
            output_path = output_dir / f"batch_shard_{int(job_row['shard_index']):04d}_output.jsonl"
            client.download_batch_output(file_id=output_file_id, output_path=output_path)
            job_row["batch_output_file"] = str(output_path.resolve())
        completed.append(job_row)
    return completed


def _parse_batch_output_files(
    *,
    jobs: Sequence[Mapping[str, Any]],
    spec_by_custom_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    parsed_contracts: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    for job in jobs:
        output_path = Path(str(job.get("batch_output_file") or "")).expanduser()
        if not output_path.is_file():
            continue
        with output_path.open("rt", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                parsed = parse_batch_output_line(json.loads(text))
                custom_id = str(parsed.get("custom_id") or "")
                spec = spec_by_custom_id.get(custom_id)
                if spec is None:
                    continue
                if int(parsed.get("status_code") or 0) != 200:
                    failures.append(
                        {
                            "example_id": str(spec["example_id"]),
                            "slot_id": str(spec["slot_id"]),
                            "custom_id": custom_id,
                            "error": f"batch_status_code={parsed.get('status_code')}",
                        }
                    )
                    continue
                content_text = str(parsed.get("content_text") or "").strip()
                if not content_text:
                    failures.append(
                        {
                            "example_id": str(spec["example_id"]),
                            "slot_id": str(spec["slot_id"]),
                            "custom_id": custom_id,
                            "error": "empty_batch_content",
                        }
                    )
                    continue
                try:
                    parsed_contracts[custom_id] = dict(json.loads(content_text))
                except Exception as exc:
                    failures.append(
                        {
                            "example_id": str(spec["example_id"]),
                            "slot_id": str(spec["slot_id"]),
                            "custom_id": custom_id,
                            "error": f"invalid_json: {exc}",
                        }
                    )
    return parsed_contracts, failures


def _run_sync_label_request(
    *,
    spec: Mapping[str, Any],
    client: OpenAIVLMSemanticClient,
    model_override: str,
    image_detail_override: str,
    max_completion_tokens: int,
    max_retries: int,
    retry_sleep_s: float,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], int, float]:
    model_name, image_detail, prompt, image_paths, json_schema = _resolved_request_settings(
        spec,
        model_override=model_override,
        image_detail_override=image_detail_override,
    )
    last_error: Optional[Exception] = None
    start_t = time.time()
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            payload = client.label_contract(
                prompt=prompt,
                image_paths=image_paths,
                model_name=model_name,
                image_detail=image_detail,
                max_completion_tokens=int(max_completion_tokens),
                json_schema=json_schema,
            )
            return str(spec["custom_id"]), dict(payload), None, int(attempt), float(time.time() - start_t)
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, int(max_retries)):
                break
            time.sleep(float(retry_sleep_s))
    return str(spec["custom_id"]), None, ("" if last_error is None else str(last_error)), max(1, int(max_retries)), float(time.time() - start_t)


def _run_sync_labeling(
    *,
    specs: Sequence[Mapping[str, Any]],
    client: OpenAIVLMSemanticClient,
    model_override: str,
    image_detail_override: str,
    max_completion_tokens: int,
    max_retries: int,
    retry_sleep_s: float,
    num_workers: int,
    progress_every: int,
    on_slot_success: Optional[Callable[[Mapping[str, Any], Mapping[str, Any]], None]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if not client.available:
        raise RuntimeError("OpenAI API key is not available")
    resolved: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    max_workers = max(1, int(num_workers))
    progress_every = max(1, int(progress_every))
    total = int(len(specs))
    completed = 0
    num_success = 0
    num_failed = 0
    first_success_logged = False
    first_failure_logged = False

    print(
        json.dumps(
            {
                "event": "sync_label_start",
                "total_slots": total,
                "num_workers": max_workers,
                "api_key_available": bool(client.available),
                "model_override": str(model_override),
                "image_detail_override": str(image_detail_override),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def _log_progress(*, custom_id: str, status: str, attempts: int, elapsed_s: float, error: Optional[str] = None) -> None:
        nonlocal completed, num_success, num_failed, first_success_logged, first_failure_logged
        completed += 1
        if status == "ok":
            num_success += 1
            if not first_success_logged:
                print(
                    json.dumps(
                        {
                            "event": "api_first_success",
                            "custom_id": str(custom_id),
                            "completed": int(completed),
                            "total_slots": total,
                            "attempts": int(attempts),
                            "elapsed_s": round(float(elapsed_s), 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                first_success_logged = True
        else:
            num_failed += 1
            if not first_failure_logged:
                print(
                    json.dumps(
                        {
                            "event": "api_first_failure",
                            "custom_id": str(custom_id),
                            "completed": int(completed),
                            "total_slots": total,
                            "attempts": int(attempts),
                            "elapsed_s": round(float(elapsed_s), 3),
                            "error": str(error or ""),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                first_failure_logged = True
        if completed == total or (completed % progress_every) == 0:
            print(
                json.dumps(
                    {
                        "event": "sync_label_progress",
                        "completed": int(completed),
                        "total_slots": total,
                        "succeeded": int(num_success),
                        "failed": int(num_failed),
                        "last_custom_id": str(custom_id),
                        "last_status": str(status),
                        "last_attempts": int(attempts),
                        "last_elapsed_s": round(float(elapsed_s), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if max_workers <= 1 or len(specs) <= 1:
        for spec in specs:
            custom_id, raw_contract, error, attempts, elapsed_s = _run_sync_label_request(
                spec=spec,
                client=client,
                model_override=model_override,
                image_detail_override=image_detail_override,
                max_completion_tokens=max_completion_tokens,
                max_retries=max_retries,
                retry_sleep_s=retry_sleep_s,
            )
            if raw_contract is not None:
                resolved[custom_id] = raw_contract
                if on_slot_success is not None:
                    on_slot_success(spec, raw_contract)
                _log_progress(custom_id=custom_id, status="ok", attempts=attempts, elapsed_s=elapsed_s)
            else:
                failures.append(
                    {
                        "example_id": str(spec["example_id"]),
                        "slot_id": str(spec["slot_id"]),
                        "custom_id": custom_id,
                        "error": str(error or ""),
                    }
                )
                _log_progress(custom_id=custom_id, status="error", attempts=attempts, elapsed_s=elapsed_s, error=error)
        return resolved, failures

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _run_sync_label_request,
                spec=spec,
                client=client,
                model_override=model_override,
                image_detail_override=image_detail_override,
                max_completion_tokens=max_completion_tokens,
                max_retries=max_retries,
                retry_sleep_s=retry_sleep_s,
            ): dict(spec)
            for spec in specs
        }
        for future in as_completed(future_map):
            spec = future_map[future]
            custom_id, raw_contract, error, attempts, elapsed_s = future.result()
            if raw_contract is not None:
                resolved[custom_id] = raw_contract
                if on_slot_success is not None:
                    on_slot_success(spec, raw_contract)
                _log_progress(custom_id=custom_id, status="ok", attempts=attempts, elapsed_s=elapsed_s)
            else:
                failures.append(
                    {
                        "example_id": str(spec["example_id"]),
                        "slot_id": str(spec["slot_id"]),
                        "custom_id": custom_id,
                        "error": str(error or ""),
                    }
                )
                _log_progress(custom_id=custom_id, status="error", attempts=attempts, elapsed_s=elapsed_s, error=error)
    return resolved, failures


def _write_slot_contract_files(
    *,
    spec: Mapping[str, Any],
    raw_contract: Mapping[str, Any],
    example_contexts: Mapping[str, Mapping[str, Any]],
    model_override: str,
    image_detail_override: str,
) -> None:
    example_context = dict(example_contexts[str(spec["example_id"])])
    example_dir = Path(str(example_context["example_dir"])).expanduser().resolve()
    slot_id = str(spec["slot_id"])
    raw_path, normalized_path = _slot_contract_paths(example_dir, slot_id)
    model_name, _image_detail, _prompt, _image_paths, _json_schema = _resolved_request_settings(
        spec,
        model_override=model_override,
        image_detail_override=image_detail_override,
    )
    write_json(raw_path, dict(raw_contract))
    normalized_contract = normalize_sdc_path_contract(
        raw_contract,
        example_id=str(spec["example_id"]),
        scenario_id=str(example_context.get("scenario_id") or ""),
        sdc_id=str(example_context.get("sdc_id") or ""),
        current_time_index=int(example_context.get("current_time_index") or 0),
        model_name=model_name,
    )
    write_json(normalized_path, normalized_contract)


def _finalize_example_contract(
    *,
    example_id: str,
    slot_requests: Sequence[Mapping[str, Any]],
    example_contexts: Mapping[str, Mapping[str, Any]],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    example_context = dict(example_contexts[str(example_id)])
    example_dir = Path(str(example_context["example_dir"])).expanduser().resolve()
    render_metadata = dict(example_context["render_metadata"])
    slot_rows = [dict(row) for row in list(example_context.get("slot_rows") or [])]
    slot_row_lookup = {str(row["slot_id"]): dict(row) for row in slot_rows}

    slot_raw_contracts: Dict[str, Dict[str, Any]] = {}
    slot_normalized_contracts: Dict[str, Dict[str, Any]] = {}
    for slot_id in SLOT_IDS:
        raw_path, normalized_path = _slot_contract_paths(example_dir, slot_id)
        if raw_path.is_file() and normalized_path.is_file():
            slot_raw_contracts[slot_id] = _read_json(raw_path)
            slot_normalized_contracts[slot_id] = _read_json(normalized_path)

    if not slot_normalized_contracts or len(slot_normalized_contracts) != len(slot_rows):
        return None

    model_name = str(next(iter(slot_normalized_contracts.values())).get("model_name") or "")
    aggregated_payload = make_empty_sdc_path_contract(
        example_id=example_id,
        scenario_id=str(example_context.get("scenario_id") or ""),
        sdc_id=str(example_context.get("sdc_id") or ""),
        current_time_index=int(example_context.get("current_time_index") or 0),
        model_name=model_name,
    )
    aggregated_payload["scene_ambiguity"] = _aggregate_scene_ambiguity(list(slot_normalized_contracts.values()))
    aggregated_payload["highlighted_paths"] = [
        _slot_path_row(slot_normalized_contracts[slot_id], slot_row=slot_row_lookup[slot_id])
        for slot_id in SLOT_IDS
        if slot_id in slot_normalized_contracts and slot_id in slot_row_lookup
    ]
    aggregated_payload["use_for_training"] = bool(
        aggregated_payload["highlighted_paths"]
        and all(bool(row.get("is_valid_target")) for row in aggregated_payload["highlighted_paths"])
        and all(bool(contract.get("use_for_training")) for contract in slot_normalized_contracts.values())
    )
    aggregated_payload["notes"] = [
        f"per_slot_requests={len(slot_normalized_contracts)}",
        "source_bundle=postsplit_existing_render",
    ]
    normalized_contract = normalize_sdc_path_contract(
        aggregated_payload,
        example_id=example_id,
        scenario_id=str(example_context.get("scenario_id") or ""),
        sdc_id=str(example_context.get("sdc_id") or ""),
        current_time_index=int(example_context.get("current_time_index") or 0),
        model_name=model_name,
    )
    aggregated_raw = {
        "example_id": example_id,
        "scenario_id": str(example_context.get("scenario_id") or ""),
        "sdc_id": str(example_context.get("sdc_id") or ""),
        "current_time_index": int(example_context.get("current_time_index") or 0),
        "mode": "per_slot_requests",
        "slot_raw_contracts": slot_raw_contracts,
    }
    write_json(example_dir / "contract_raw.json", aggregated_raw)
    write_json(example_dir / "contract_normalized.json", normalized_contract)
    raw_row = {
        "example_id": example_id,
        "scenario_id": str(example_context.get("scenario_id") or ""),
        "sdc_id": str(example_context.get("sdc_id") or ""),
        "raw_contract_path": str((example_dir / "contract_raw.json").resolve()),
        "normalized_contract_path": str((example_dir / "contract_normalized.json").resolve()),
        "contract": normalized_contract,
    }
    aggregate_row = {
        "example_id": example_id,
        "scenario_id": str(example_context.get("scenario_id") or ""),
        "sdc_id": str(example_context.get("sdc_id") or ""),
        "current_time_index": int(example_context.get("current_time_index") or 0),
        "slot_metadata": render_metadata.get("slot_metadata"),
        "images": render_metadata.get("images"),
        "prompt_paths": {
            slot_id: str((example_dir / f"prompt_{slot_id}.txt").resolve())
            for slot_id in SLOT_IDS
            if (example_dir / f"prompt_{slot_id}.txt").is_file()
        },
        "request_jsons": {
            slot_id: str((example_dir / f"request_{slot_id}.json").resolve())
            for slot_id in SLOT_IDS
            if (example_dir / f"request_{slot_id}.json").is_file()
        },
        "contract": normalized_contract,
    }
    return raw_row, aggregate_row


def _count_completed_slot_contracts(all_examples: Sequence[tuple[str, List[Dict[str, Any]]]]) -> int:
    total = 0
    for _example_id, slot_requests in all_examples:
        example_dir = _example_dir_from_request_json(Path(slot_requests[0]["request_json"]))
        for slot_id in SLOT_IDS:
            if _slot_contract_available(example_dir, slot_id):
                total += 1
    return total


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    request_manifest_path = bundle_root / "postsplit_request_manifest.jsonl"
    if not request_manifest_path.is_file():
        raise SystemExit(f"Missing request manifest: {request_manifest_path}")

    client = OpenAIVLMSemanticClient(dotenv_path=(None if not str(args.dotenv).strip() else str(args.dotenv)))

    request_rows = []
    for line in request_manifest_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        request_rows.append(dict(json.loads(text)))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in request_rows:
        grouped.setdefault(str(row.get("example_id") or ""), []).append(dict(row))

    all_examples = sorted(grouped.items(), key=lambda item: item[0])
    if int(args.max_examples) > 0:
        all_examples = all_examples[: int(args.max_examples)]

    slot_specs, example_contexts = _build_slot_specs(all_examples=all_examples, skip_existing=bool(args.skip_existing))
    spec_by_custom_id = {str(spec["custom_id"]): dict(spec) for spec in slot_specs}
    raw_contracts_written: Dict[str, Dict[str, Any]] = {}
    failed_slots: List[Dict[str, Any]] = []
    all_examples_map = {str(example_id): list(slot_requests) for example_id, slot_requests in all_examples}

    def _handle_slot_success(spec: Mapping[str, Any], raw_contract: Mapping[str, Any]) -> None:
        custom_id = str(spec["custom_id"])
        _write_slot_contract_files(
            spec=spec,
            raw_contract=raw_contract,
            example_contexts=example_contexts,
            model_override=str(args.model),
            image_detail_override=str(args.image_detail),
        )
        raw_contracts_written[custom_id] = dict(raw_contract)
        finalized = _finalize_example_contract(
            example_id=str(spec["example_id"]),
            slot_requests=all_examples_map[str(spec["example_id"])],
            example_contexts=example_contexts,
        )
        if finalized is not None:
            print(
                json.dumps(
                    {
                        "event": "example_contract_finalized",
                        "example_id": str(spec["example_id"]),
                        "custom_id": custom_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    batch_root = bundle_root / "postsplit_batch"
    batch_jobs_path = batch_root / "batch_jobs.json"
    prepared_jobs: List[Dict[str, Any]] = []
    parsed_batch_custom_ids: List[str] = []
    requested_batch_shard_indices = _parse_batch_shard_indices(str(args.batch_shard_indices))

    if bool(args.batch_mode):
        batch_items = [
            _build_batch_item(
                spec=spec,
                client=client,
                model_override=str(args.model),
                image_detail_override=str(args.image_detail),
                max_completion_tokens=int(args.max_completion_tokens),
            )
            for spec in slot_specs
        ]
        prepared_jobs = _write_batch_shards(
            client=client,
            batch_items=batch_items,
            batch_root=batch_root,
            max_requests=int(args.batch_max_requests),
            max_bytes=int(args.batch_max_bytes),
        )
        batch_jobs: List[Dict[str, Any]]
        if bool(args.submit_batch):
            if not client.available:
                raise SystemExit("OpenAI API key is not available for batch submission")
            jobs_to_submit = _filter_batch_jobs(
                prepared_jobs,
                shard_indices=requested_batch_shard_indices,
            )
            submitted_jobs = _submit_batch_jobs(
                client=client,
                jobs=jobs_to_submit,
                completion_window=str(args.completion_window),
            )
            batch_jobs = _merge_batch_jobs(
                base_jobs=prepared_jobs,
                updated_jobs=submitted_jobs,
            )
            write_json(batch_jobs_path, {"jobs": batch_jobs})
        elif batch_jobs_path.is_file():
            batch_jobs = [dict(row) for row in list(_read_json(batch_jobs_path).get("jobs") or [])]
        else:
            batch_jobs = [dict(row) for row in prepared_jobs]
            write_json(batch_jobs_path, {"jobs": batch_jobs})

        if bool(args.wait_for_batch):
            if not client.available:
                raise SystemExit("OpenAI API key is not available for batch polling")
            jobs_to_wait = _filter_batch_jobs(
                batch_jobs,
                shard_indices=requested_batch_shard_indices,
            )
            missing_batch_ids = [
                int(job.get("shard_index") or -1)
                for job in jobs_to_wait
                if not str(job.get("batch_id") or "").strip()
            ]
            if missing_batch_ids:
                raise SystemExit(
                    "Requested --wait-for-batch for shard(s) without submitted batch_id: "
                    + ",".join(str(idx) for idx in sorted(missing_batch_ids))
                    + ". Run with --submit-batch first."
                )
            waited_jobs = _wait_and_download_batch_jobs(
                client=client,
                jobs=jobs_to_wait,
                batch_root=batch_root,
            )
            batch_jobs = _merge_batch_jobs(
                base_jobs=batch_jobs,
                updated_jobs=waited_jobs,
            )
            write_json(batch_jobs_path, {"jobs": batch_jobs})

        parsed_batch_rows, batch_failures = _parse_batch_output_files(
            jobs=batch_jobs,
            spec_by_custom_id=spec_by_custom_id,
        )
        for custom_id, raw_contract in parsed_batch_rows.items():
            _write_slot_contract_files(
                spec=spec_by_custom_id[custom_id],
                raw_contract=raw_contract,
                example_contexts=example_contexts,
                model_override=str(args.model),
                image_detail_override=str(args.image_detail),
            )
            raw_contracts_written[custom_id] = raw_contract
        parsed_batch_custom_ids = sorted(parsed_batch_rows.keys())
        failed_slots.extend(batch_failures)

        fallback_target_custom_ids = (
            _custom_ids_for_jobs(_filter_batch_jobs(batch_jobs, shard_indices=requested_batch_shard_indices))
            if bool(args.wait_for_batch)
            else set(str(spec["custom_id"]) for spec in slot_specs)
        )
        unresolved_specs = [
            spec
            for spec in slot_specs
            if str(spec["custom_id"]) in fallback_target_custom_ids
            and str(spec["custom_id"]) not in raw_contracts_written
        ]
        if bool(args.wait_for_batch) and unresolved_specs:
            sync_rows, sync_failures = _run_sync_labeling(
                specs=unresolved_specs,
                client=client,
                model_override=str(args.model),
                image_detail_override=str(args.image_detail),
                max_completion_tokens=int(args.max_completion_tokens),
                max_retries=int(args.max_retries),
                retry_sleep_s=float(args.retry_sleep_s),
                num_workers=int(args.num_workers),
                progress_every=int(args.progress_every),
                on_slot_success=_handle_slot_success,
            )
            for custom_id, raw_contract in sync_rows.items():
                if custom_id not in raw_contracts_written:
                    _handle_slot_success(spec_by_custom_id[custom_id], raw_contract)
            failed_slots.extend(sync_failures)
            failed_slots = [
                dict(row)
                for row in failed_slots
                if str(row.get("custom_id") or "") not in raw_contracts_written
            ]
    else:
        if slot_specs:
            sync_rows, sync_failures = _run_sync_labeling(
                specs=slot_specs,
                client=client,
                model_override=str(args.model),
                image_detail_override=str(args.image_detail),
                max_completion_tokens=int(args.max_completion_tokens),
                max_retries=int(args.max_retries),
                retry_sleep_s=float(args.retry_sleep_s),
                num_workers=int(args.num_workers),
                progress_every=int(args.progress_every),
                on_slot_success=_handle_slot_success,
            )
            for custom_id, raw_contract in sync_rows.items():
                if custom_id not in raw_contracts_written:
                    _handle_slot_success(spec_by_custom_id[custom_id], raw_contract)
            failed_slots.extend(sync_failures)

    raw_contract_rows_this_run: List[Dict[str, Any]] = []
    aggregate_index_rows_this_run: List[Dict[str, Any]] = []
    num_examples_labeled = 0
    for example_id, slot_requests in all_examples:
        finalized = _finalize_example_contract(
            example_id=str(example_id),
            slot_requests=slot_requests,
            example_contexts=example_contexts,
        )
        if finalized is None:
            continue
        raw_row, aggregate_row = finalized
        raw_contract_rows_this_run.append(raw_row)
        aggregate_index_rows_this_run.append(aggregate_row)
        num_examples_labeled += 1

    raw_contract_rows, aggregate_index_rows = _collect_completed_bundle_rows(all_examples=all_examples)
    total_slot_contracts = _count_completed_slot_contracts(all_examples)
    raw_contracts_path = bundle_root / "postsplit_contracts_raw.jsonl"
    write_jsonl(raw_contracts_path, raw_contract_rows)
    semantics_index_path = bundle_root / "postsplit_semantics_index.jsonl"
    write_jsonl(semantics_index_path, aggregate_index_rows)

    batch_info = {
        "batch_mode": bool(args.batch_mode),
        "batch_root": str(batch_root.resolve()),
        "batch_jobs_json": str(batch_jobs_path.resolve()),
        "num_prepared_shards": int(len(prepared_jobs)),
        "requested_batch_shard_indices": (
            None if requested_batch_shard_indices is None else sorted(int(idx) for idx in requested_batch_shard_indices)
        ),
        "prepared_request_manifests": [str(job["request_manifest_path"]) for job in prepared_jobs],
        "parsed_batch_custom_ids": parsed_batch_custom_ids,
        "batch_max_bytes": int(args.batch_max_bytes),
        "batch_max_requests": int(args.batch_max_requests),
        "submit_batch": bool(args.submit_batch),
        "wait_for_batch": bool(args.wait_for_batch),
        "completion_window": str(args.completion_window),
    }
    summary = {
        "bundle_root": str(bundle_root),
        "request_manifest_jsonl": str(request_manifest_path),
        "num_examples_requested": int(len(all_examples)),
        "num_slot_specs_considered": int(len(slot_specs)),
        "num_examples_labeled_this_run": int(num_examples_labeled),
        "num_slot_requests_completed_this_run": int(len(raw_contracts_written)),
        "num_examples_completed_total": int(len(raw_contract_rows)),
        "num_slot_contracts_completed_total": int(total_slot_contracts),
        "num_failed_slots": int(len(failed_slots)),
        "failed_slots": failed_slots,
        "model_override": str(args.model),
        "image_detail_override": str(args.image_detail),
        "skip_existing": bool(args.skip_existing),
        "num_workers": int(args.num_workers),
        "api_key_available": bool(client.available),
        "raw_contracts_jsonl": str(raw_contracts_path.resolve()),
        "semantics_index_jsonl": str(semantics_index_path.resolve()),
        "batch_info": batch_info,
    }
    write_json(bundle_root / "postsplit_vlm_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
