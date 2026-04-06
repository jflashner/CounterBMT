from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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
from bmt.counterfactual.vlm_semantics.client import OpenAIVLMSemanticClient
from bmt.counterfactual.vlm_semantics.sdc_path_contract import (
    SLOT_IDS,
    make_empty_sdc_path_contract,
    normalize_sdc_path_contract,
)


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
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _example_dir_from_request_json(path: Path) -> Path:
    return path.expanduser().resolve().parent


def _contract_available(example_dir: Path) -> bool:
    return (example_dir / "contract_normalized.json").is_file()


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


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    request_manifest_path = bundle_root / "postsplit_request_manifest.jsonl"
    if not request_manifest_path.is_file():
        raise SystemExit(f"Missing request manifest: {request_manifest_path}")

    client = OpenAIVLMSemanticClient(dotenv_path=(None if not str(args.dotenv).strip() else str(args.dotenv)))
    if not client.available:
        raise SystemExit("OpenAI API key is not available")

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

    raw_contract_rows_this_run: List[Dict[str, Any]] = []
    aggregate_index_rows_this_run: List[Dict[str, Any]] = []
    num_examples_labeled = 0
    num_slot_requests = 0
    failed_slots: List[Dict[str, Any]] = []

    for example_idx, (example_id, slot_requests) in enumerate(all_examples, start=1):
        example_dir = _example_dir_from_request_json(Path(slot_requests[0]["request_json"]))
        if bool(args.skip_existing) and _contract_available(example_dir):
            print(f"[skip-existing] {example_id}")
            continue

        render_metadata_path = example_dir / "render_metadata.json"
        if not render_metadata_path.is_file():
            print(f"[skip-missing-render-metadata] {example_id}")
            continue
        render_metadata = _read_json(render_metadata_path)
        slot_rows = _load_slot_rows(render_metadata)
        slot_row_lookup = {str(row["slot_id"]): dict(row) for row in slot_rows}

        scenario_id = str(render_metadata.get("scenario_id") or "")
        sdc_id = str(render_metadata.get("sdc_id") or "")
        current_time_index = int(render_metadata.get("current_time_index") or 0)

        print(f"[{example_idx}/{len(all_examples)}] labeling {example_id}")
        slot_raw_contracts: Dict[str, Dict[str, Any]] = {}
        slot_normalized_contracts: Dict[str, Dict[str, Any]] = {}

        for request_row in sorted(slot_requests, key=lambda row: SLOT_IDS.index(str(row.get("slot_id") or "gt"))):
            slot_id = str(request_row.get("slot_id") or "")
            raw_path = example_dir / f"contract_raw_{slot_id}.json"
            normalized_path = example_dir / f"contract_normalized_{slot_id}.json"
            if bool(args.skip_existing) and raw_path.is_file() and normalized_path.is_file():
                slot_raw_contracts[slot_id] = _read_json(raw_path)
                slot_normalized_contracts[slot_id] = _read_json(normalized_path)
                continue
            request_json = _read_json(Path(request_row["request_json"]))
            model_name = str(args.model).strip() or str(request_json.get("model") or "")
            image_detail = str(args.image_detail).strip() or str(request_json.get("image_detail") or "")
            image_paths = [str(path) for path in list(request_json.get("image_paths") or []) if str(path).strip()]
            prompt = str(request_json.get("prompt") or "")
            json_schema = dict(request_json.get("json_schema") or {})
            raw_contract = None
            last_error: Optional[Exception] = None
            for attempt in range(1, max(1, int(args.max_retries)) + 1):
                try:
                    raw_contract = client.label_contract(
                        prompt=prompt,
                        image_paths=image_paths,
                        model_name=model_name,
                        image_detail=image_detail,
                        max_completion_tokens=int(args.max_completion_tokens),
                        json_schema=json_schema,
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max(1, int(args.max_retries)):
                        break
                    print(f"[retry {attempt}/{int(args.max_retries)}] {example_id} {slot_id}: {exc}")
                    time.sleep(float(args.retry_sleep_s))
            if raw_contract is None:
                failed_slots.append(
                    {
                        "example_id": example_id,
                        "slot_id": slot_id,
                        "error": "" if last_error is None else str(last_error),
                    }
                )
                print(f"[slot-failed] {example_id} {slot_id}: {last_error}")
                continue
            num_slot_requests += 1
            write_json(raw_path, raw_contract)
            normalized_contract = normalize_sdc_path_contract(
                raw_contract,
                example_id=example_id,
                scenario_id=scenario_id,
                sdc_id=sdc_id,
                current_time_index=current_time_index,
                model_name=model_name,
            )
            write_json(normalized_path, normalized_contract)
            slot_raw_contracts[slot_id] = raw_contract
            slot_normalized_contracts[slot_id] = normalized_contract

        if not slot_normalized_contracts:
            continue
        if len(slot_normalized_contracts) != len(slot_rows):
            print(f"[example-incomplete] {example_id}: have {len(slot_normalized_contracts)}/{len(slot_rows)} slot contracts")
            continue

        model_name = str(args.model).strip() or str(next(iter(slot_normalized_contracts.values())).get("model_name") or "")
        aggregated_payload = make_empty_sdc_path_contract(
            example_id=example_id,
            scenario_id=scenario_id,
            sdc_id=sdc_id,
            current_time_index=current_time_index,
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
            scenario_id=scenario_id,
            sdc_id=sdc_id,
            current_time_index=current_time_index,
            model_name=model_name,
        )
        aggregated_raw = {
            "example_id": example_id,
            "scenario_id": scenario_id,
            "sdc_id": sdc_id,
            "current_time_index": int(current_time_index),
            "mode": "per_slot_requests",
            "slot_raw_contracts": slot_raw_contracts,
        }
        write_json(example_dir / "contract_raw.json", aggregated_raw)
        write_json(example_dir / "contract_normalized.json", normalized_contract)
        raw_contract_rows_this_run.append(
            {
                "example_id": example_id,
                "scenario_id": scenario_id,
                "sdc_id": sdc_id,
                "raw_contract_path": str((example_dir / "contract_raw.json").resolve()),
                "normalized_contract_path": str((example_dir / "contract_normalized.json").resolve()),
                "contract": normalized_contract,
            }
        )
        aggregate_index_rows_this_run.append(
            {
                "example_id": example_id,
                "scenario_id": scenario_id,
                "sdc_id": sdc_id,
                "current_time_index": int(current_time_index),
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
        num_examples_labeled += 1

    raw_contract_rows, aggregate_index_rows = _collect_completed_bundle_rows(all_examples=all_examples)
    total_slot_contracts = sum(
        1
        for example_id, slot_requests in all_examples
        for slot_id in SLOT_IDS
        if (_example_dir_from_request_json(Path(slot_requests[0]["request_json"])) / f"contract_normalized_{slot_id}.json").is_file()
    )
    raw_contracts_path = bundle_root / "postsplit_contracts_raw.jsonl"
    write_jsonl(raw_contracts_path, raw_contract_rows)
    semantics_index_path = bundle_root / "postsplit_semantics_index.jsonl"
    write_jsonl(semantics_index_path, aggregate_index_rows)
    summary = {
        "bundle_root": str(bundle_root),
        "request_manifest_jsonl": str(request_manifest_path),
        "num_examples_requested": int(len(all_examples)),
        "num_examples_labeled_this_run": int(num_examples_labeled),
        "num_slot_requests_completed_this_run": int(num_slot_requests),
        "num_examples_completed_total": int(len(raw_contract_rows)),
        "num_slot_contracts_completed_total": int(total_slot_contracts),
        "num_failed_slots": int(len(failed_slots)),
        "failed_slots": failed_slots,
        "model_override": str(args.model),
        "image_detail_override": str(args.image_detail),
        "skip_existing": bool(args.skip_existing),
        "raw_contracts_jsonl": str(raw_contracts_path.resolve()),
        "semantics_index_jsonl": str(semantics_index_path.resolve()),
    }
    write_json(bundle_root / "postsplit_vlm_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
