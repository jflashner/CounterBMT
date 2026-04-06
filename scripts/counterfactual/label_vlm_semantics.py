from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.path_eval_bundle import load_json, write_json, write_jsonl
from bmt.counterfactual.vlm_semantics.client import OpenAIVLMSemanticClient, parse_batch_output_line
from bmt.counterfactual.vlm_semantics.contract import normalize_contract, should_escalate_contract
from bmt.counterfactual.vlm_semantics.fuse import merge_pass_contracts
from bmt.counterfactual.vlm_semantics.prompt import build_all_prompts


PASS_ORDER = ("context_only", "context_plus_gt", "context_plus_anchor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label VLM semantic contracts from rendered context views.")
    parser.add_argument("--render-manifest", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model-default", type=str, default="gpt-5.4-mini")
    parser.add_argument("--model-escalate", type=str, default="gpt-5.4")
    parser.add_argument("--image-detail-default", type=str, default="low", choices=("low", "high", "original", "auto"))
    parser.add_argument("--image-detail-escalate", type=str, default="high", choices=("low", "high", "original", "auto"))
    parser.add_argument("--escalate-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--batch-mode", action="store_true")
    parser.add_argument("--submit-batch", action="store_true")
    parser.add_argument("--wait-for-batch", action="store_true")
    parser.add_argument("--completion-window", type=str, default="24h")
    parser.add_argument("--batch-output-file", type=str, default="")
    return parser.parse_args()


def _load_render_rows(path: str | Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"]]
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    raise ValueError(f"Unexpected render manifest format: {path}")


def _build_request_rows(
    *,
    render_rows: Sequence[Mapping[str, Any]],
    client: OpenAIVLMSemanticClient,
    model_name: str,
    image_detail: str,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for row in render_rows:
        prompts = build_all_prompts(row)
        images = dict(row.get("images") or {})
        for pass_name in PASS_ORDER:
            request = client.build_chat_completion_request(
                prompt=prompts[pass_name],
                image_paths=[images[pass_name]],
                model_name=model_name,
                image_detail=image_detail,
            )
            requests.append(
                {
                    "custom_id": f"{row['example_id']}::{pass_name}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request,
                }
            )
    return requests


def _merge_contract_row(
    *,
    row: Mapping[str, Any],
    pass_payloads: Mapping[str, Mapping[str, Any]],
    model_name: str,
    escalated: bool,
) -> Dict[str, Any]:
    contract = merge_pass_contracts(
        example_id=str(row["example_id"]),
        scenario_id=str(row["scenario_id"]),
        agent_id=str(row["agent_id"]),
        image_set_ids={
            "context_only": [str(row["images"]["context_only"])],
            "context_plus_gt": [str(row["images"]["context_plus_gt"])],
            "context_plus_anchor": [str(row["images"]["context_plus_anchor"])],
        },
        model_name=str(model_name),
        context_only_contract=pass_payloads.get("context_only"),
        context_plus_gt_contract=pass_payloads.get("context_plus_gt"),
        context_plus_anchor_contract=pass_payloads.get("context_plus_anchor"),
    )
    return {
        "example_id": str(row["example_id"]),
        "scenario_id": str(row["scenario_id"]),
        "agent_id": str(row["agent_id"]),
        "decision_time_idx": int(row["decision_time_idx"]),
        "selected_mode": str(row.get("selected_mode") or "factual"),
        "requested_branch_label": str(row.get("requested_branch_label") or ""),
        "geometry_branch_label": str(row.get("geometry_branch_label") or ""),
        "geometry_branch_id": str(row.get("geometry_branch_id") or ""),
        "geometry_light_group_id": row.get("geometry_light_group_id"),
        "geometry_primary_light_id": row.get("geometry_primary_light_id"),
        "selected_candidate_id": row.get("selected_candidate_id"),
        "selected_candidate_geometry_branch_id": row.get("selected_candidate_geometry_branch_id"),
        "selected_candidate_geometry_label": row.get("selected_candidate_geometry_label"),
        "candidate_id_map": list(row.get("candidate_id_map") or []),
        "images": dict(row.get("images") or {}),
        "metadata_json": row.get("metadata_json"),
        "pass_contracts": {name: pass_payloads.get(name) for name in PASS_ORDER},
        "contract": contract,
        "escalated": bool(escalated),
        "model_name_used": str(model_name),
    }


def _run_sync_labeling(
    *,
    render_rows: Sequence[Mapping[str, Any]],
    client: OpenAIVLMSemanticClient,
    model_default: str,
    model_escalate: str,
    image_detail_default: str,
    image_detail_escalate: str,
    escalate_confidence_threshold: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row in render_rows:
        prompts = build_all_prompts(row)
        image_paths = dict(row.get("images") or {})
        pass_payloads: Dict[str, Dict[str, Any]] = {}
        for pass_name in PASS_ORDER:
            payload = client.label_contract(
                prompt=prompts[pass_name],
                image_paths=[image_paths[pass_name]],
                model_name=model_default,
                image_detail=image_detail_default,
            )
            pass_payloads[pass_name] = normalize_contract(
                payload,
                example_id=str(row["example_id"]),
                scenario_id=str(row["scenario_id"]),
                agent_id=str(row["agent_id"]),
                image_set_ids={pass_name: [image_paths[pass_name]]},
                model_name=model_default,
            )
        merged_default = _merge_contract_row(
            row=row,
            pass_payloads=pass_payloads,
            model_name=model_default,
            escalated=False,
        )
        if should_escalate_contract(merged_default["contract"], confidence_threshold=float(escalate_confidence_threshold)):
            pass_payloads = {}
            for pass_name in PASS_ORDER:
                payload = client.label_contract(
                    prompt=prompts[pass_name],
                    image_paths=[image_paths[pass_name]],
                    model_name=model_escalate,
                    image_detail=image_detail_escalate,
                )
                pass_payloads[pass_name] = normalize_contract(
                    payload,
                    example_id=str(row["example_id"]),
                    scenario_id=str(row["scenario_id"]),
                    agent_id=str(row["agent_id"]),
                    image_set_ids={pass_name: [image_paths[pass_name]]},
                    model_name=model_escalate,
                )
            results.append(
                _merge_contract_row(
                    row=row,
                    pass_payloads=pass_payloads,
                    model_name=model_escalate,
                    escalated=True,
                )
            )
        else:
            results.append(merged_default)
    return results


def _parse_batch_output_to_rows(
    *,
    render_rows: Sequence[Mapping[str, Any]],
    batch_output_path: str | Path,
    default_model_name: str,
    escalate_model_name: str,
    client: OpenAIVLMSemanticClient,
    image_detail_escalate: str,
    escalate_confidence_threshold: float,
) -> List[Dict[str, Any]]:
    by_example: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    with Path(batch_output_path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parsed = parse_batch_output_line(json.loads(text))
            if parsed["status_code"] != 200 or not parsed["content_text"]:
                continue
            custom_id = str(parsed["custom_id"])
            if "::" not in custom_id:
                continue
            example_id, pass_name = custom_id.split("::", 1)
            if pass_name not in PASS_ORDER:
                continue
            by_example[example_id][pass_name] = json.loads(parsed["content_text"])

    render_by_example = {str(row["example_id"]): dict(row) for row in render_rows}
    merged_rows: List[Dict[str, Any]] = []
    for example_id, row in render_by_example.items():
        pass_payloads = {
            pass_name: normalize_contract(
                by_example.get(example_id, {}).get(pass_name),
                example_id=example_id,
                scenario_id=str(row["scenario_id"]),
                agent_id=str(row["agent_id"]),
                image_set_ids={pass_name: [str(row["images"][pass_name])]},
                model_name=default_model_name,
            )
            for pass_name in PASS_ORDER
        }
        merged = _merge_contract_row(row=row, pass_payloads=pass_payloads, model_name=default_model_name, escalated=False)
        if should_escalate_contract(merged["contract"], confidence_threshold=float(escalate_confidence_threshold)) and client.available:
            prompts = build_all_prompts(row)
            escalated_payloads: Dict[str, Dict[str, Any]] = {}
            for pass_name in PASS_ORDER:
                payload = client.label_contract(
                    prompt=prompts[pass_name],
                    image_paths=[row["images"][pass_name]],
                    model_name=escalate_model_name,
                    image_detail=image_detail_escalate,
                )
                escalated_payloads[pass_name] = normalize_contract(
                    payload,
                    example_id=example_id,
                    scenario_id=str(row["scenario_id"]),
                    agent_id=str(row["agent_id"]),
                    image_set_ids={pass_name: [str(row["images"][pass_name])]},
                    model_name=escalate_model_name,
                )
            merged = _merge_contract_row(row=row, pass_payloads=escalated_payloads, model_name=escalate_model_name, escalated=True)
        merged_rows.append(merged)
    return merged_rows


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    render_rows = _load_render_rows(args.render_manifest)
    client = OpenAIVLMSemanticClient(dotenv_path=args.dotenv)

    request_rows = _build_request_rows(
        render_rows=render_rows,
        client=client,
        model_name=str(args.model_default),
        image_detail=str(args.image_detail_default),
    )
    request_manifest_path = outdir / "vlm_request_manifest.jsonl"
    client.write_batch_requests(requests=request_rows, output_path=request_manifest_path)

    raw_contract_rows: List[Dict[str, Any]] = []
    batch_info: Dict[str, Any] = {
        "api_key_available": bool(client.available),
        "batch_mode": bool(args.batch_mode),
        "submitted_batch": False,
        "waited_for_batch": False,
        "batch_output_file": None,
    }
    batch_output_path = Path(args.batch_output_file).expanduser() if args.batch_output_file else None

    if args.batch_mode and batch_output_path is not None and batch_output_path.is_file():
        batch_info["batch_output_file"] = str(batch_output_path)
        raw_contract_rows = _parse_batch_output_to_rows(
            render_rows=render_rows,
            batch_output_path=batch_output_path,
            default_model_name=str(args.model_default),
            escalate_model_name=str(args.model_escalate),
            client=client,
            image_detail_escalate=str(args.image_detail_escalate),
            escalate_confidence_threshold=float(args.escalate_confidence_threshold),
        )
    elif not args.skip_api and client.available:
        if args.batch_mode:
            if args.submit_batch:
                batch_info.update(
                    client.submit_batch(
                        request_manifest_path=request_manifest_path,
                        completion_window=str(args.completion_window),
                    )
                )
                batch_info["submitted_batch"] = True
                if args.wait_for_batch:
                    status = client.wait_for_batch(batch_id=batch_info["batch_id"])
                    batch_info.update(status)
                    batch_info["waited_for_batch"] = True
                    output_file_id = status.get("output_file_id")
                    if output_file_id:
                        batch_output_path = batch_output_path or (outdir / "vlm_batch_output.jsonl")
                        client.download_batch_output(file_id=str(output_file_id), output_path=batch_output_path)
                        batch_info["batch_output_file"] = str(batch_output_path)
                        raw_contract_rows = _parse_batch_output_to_rows(
                            render_rows=render_rows,
                            batch_output_path=batch_output_path,
                            default_model_name=str(args.model_default),
                            escalate_model_name=str(args.model_escalate),
                            client=client,
                            image_detail_escalate=str(args.image_detail_escalate),
                            escalate_confidence_threshold=float(args.escalate_confidence_threshold),
                        )
        else:
            raw_contract_rows = _run_sync_labeling(
                render_rows=render_rows,
                client=client,
                model_default=str(args.model_default),
                model_escalate=str(args.model_escalate),
                image_detail_default=str(args.image_detail_default),
                image_detail_escalate=str(args.image_detail_escalate),
                escalate_confidence_threshold=float(args.escalate_confidence_threshold),
            )

    write_jsonl(outdir / "vlm_contracts_raw.jsonl", raw_contract_rows)
    summary = {
        "num_examples": int(len(render_rows)),
        "num_requests": int(len(request_rows)),
        "num_contract_rows": int(len(raw_contract_rows)),
        "api_key_available": bool(client.available),
        "api_labeling_ran": bool(len(raw_contract_rows) > 0),
        "request_manifest_jsonl": str(request_manifest_path.resolve()),
        "vlm_contracts_raw_jsonl": str((outdir / "vlm_contracts_raw.jsonl").resolve()),
        "model_default": str(args.model_default),
        "model_escalate": str(args.model_escalate),
        "image_detail_default": str(args.image_detail_default),
        "image_detail_escalate": str(args.image_detail_escalate),
        "escalate_confidence_threshold": float(args.escalate_confidence_threshold),
        "completion_window": str(args.completion_window),
        "batch_info": batch_info,
    }
    write_json(outdir / "vlm_label_summary.json", summary)
    print(summary["vlm_contracts_raw_jsonl"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
