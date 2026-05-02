from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from easydict import EasyDict
import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import omegaconf
import torch
from PIL import Image, ImageDraw

try:
    import seaborn  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from matplotlib import cm

    seaborn_stub = types.ModuleType("seaborn")

    def _fallback_color_palette(name: str = "colorblind", n_colors: int = 10):
        cmap = cm.get_cmap("tab20", int(max(1, n_colors)))
        return [tuple(float(v) for v in cmap(i)[:3]) for i in range(int(max(1, n_colors)))]

    seaborn_stub.color_palette = _fallback_color_palette  # type: ignore[attr-defined]
    sys.modules["seaborn"] = seaborn_stub

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.dataset.dataset import InfgenDataset
from bmt.eval.evaluate_scenario_metrics import EvaluationLightningModule, _load_eval_model
from bmt.eval.scenario_evaluator import Evaluator
from bmt.gradio_ui.plot import plot_gt, plot_pred
from bmt.tokenization import get_tokenizer
from bmt.utils.config import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render scene-context trajectory plots for SDC semantic-control validation examples "
            "using a trained checkpoint."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml",
    )
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sampling-method", type=str, default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--grid-columns", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=960)
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _to_builtin(value: Any) -> Any:
    if isinstance(value, EasyDict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    return value


def _load_config(args: argparse.Namespace):
    data_dir = str(Path(args.data_dir).expanduser().resolve())
    control_index = str(Path(args.control_index).expanduser().resolve())
    teacher_ckpt = str(args.teacher_ckpt or args.ckpt).strip()
    return omegaconf.OmegaConf.create(
        {
            "DATA": {
                "TRAINING_DATA_DIR": data_dir,
                "TEST_DATA_DIR": data_dir,
                "COUNTERFACTUAL_CONTROL_INDEX_TRAIN": control_index,
                "COUNTERFACTUAL_CONTROL_INDEX_VAL": control_index,
                "COUNTERFACTUAL_CONTROL_INDEX": "",
                "COUNTERFACTUAL_CONTROL_CODE_DIR": "",
                "COUNTERFACTUAL_MODE": "sdc_semantic_only",
                "COUNTERFACTUAL_WEIGHTED_SAMPLER": False,
            },
            "SAMPLING": {
                "SAMPLING_METHOD": str(args.sampling_method),
                "TEMPERATURE": float(args.temperature),
                "TOPP": float(args.topp),
            },
            "MODEL": {
                "LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT": teacher_ckpt,
            },
        }
    )


def _resolve_device(device_name: str) -> torch.device:
    text = str(device_name).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _scenario_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    source_kind = str(row.get("source_kind") or "")
    slot_id = str(row.get("slot_id") or "")
    factual_rank = 0 if source_kind == "factual_gt" else 1
    return (factual_rank, slot_id)


def _to_numpy_output(output_dict: Dict[str, Any]) -> Dict[str, Any]:
    output_np: Dict[str, Any] = {}
    for key, value in output_dict.items():
        if torch.is_tensor(value):
            if value.ndim >= 1 and int(value.shape[0]) == 1:
                value = value[0]
            output_np[key] = value.detach().cpu().numpy()
        else:
            output_np[key] = value
    return output_np


def _ensure_plot_fields(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    if "vis/map_feature" not in data_dict and "encoder/map_feature" in data_dict:
        data_dict["vis/map_feature"] = data_dict["encoder/map_feature"]
    return data_dict


def _annotate_and_resize(image: Image.Image, title: str, *, tile_size: int, title_height: int = 88) -> Image.Image:
    image = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size + title_height), color=(255, 255, 255))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), title, fill=(0, 0, 0))
    return canvas


def _save_grid(panels: Sequence[Tuple[str, Image.Image]], out_path: Path, *, columns: int, tile_size: int) -> None:
    if not panels:
        return

    prepared = [_annotate_and_resize(img, title, tile_size=tile_size) for title, img in panels]
    cell_w, cell_h = prepared[0].size
    cols = max(1, int(columns))
    rows = int(math.ceil(len(prepared) / cols))
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(245, 245, 245))

    for idx, panel in enumerate(prepared):
        row = idx // cols
        col = idx % cols
        grid.paste(panel, (col * cell_w, row * cell_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    control_index = Path(args.control_index).expanduser().resolve()
    rows = _read_jsonl(control_index)
    selected_sids: List[str] = []
    for row in rows:
        sid = str(row.get("scenario_id") or "")
        if sid and sid not in selected_sids:
            selected_sids.append(sid)
        if len(selected_sids) >= int(args.num_scenes):
            break
    selected_sid_set = set(selected_sids)

    runtime_config = _load_config(args)
    model = _load_eval_model(runtime_config, str(Path(args.ckpt).expanduser().resolve()))
    config = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.create(_to_builtin(model.config)), runtime_config)
    model.config = config
    tokenizer = get_tokenizer(config)
    dataset = InfgenDataset(config, "test", backward_prediction=False)
    module = EvaluationLightningModule(
        model=model,
        evaluator=Evaluator(key_metrics_only=True),
        tokenizer=tokenizer,
        config=config,
        dataset=dataset,
        eval_mode="GPTmodel",
        multi_mode=False,
        num_modes=1,
        save_path=str(outdir / "unused_metrics"),
    )
    module.eval()
    module.model.eval()
    device = _resolve_device(args.device)
    module.model.to(device)

    scenario_to_rows: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {sid: [] for sid in selected_sids}
    for idx, row in enumerate(rows):
        sid = str(row.get("scenario_id") or "")
        if sid in selected_sid_set:
            scenario_to_rows[sid].append((idx, row))

    manifest: List[Dict[str, Any]] = []
    for scenario_id in selected_sids:
        row_entries = sorted(scenario_to_rows[scenario_id], key=lambda item: _scenario_sort_key(item[1]))
        if not row_entries:
            continue

        scenario_dir = outdir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        gt_idx = next((idx for idx, row in row_entries if str(row.get("source_kind") or "") == "factual_gt"), row_entries[0][0])
        gt_raw = dataset[gt_idx]
        gt_vis = module.preprocess_GPTmodel(copy.deepcopy(gt_raw), backward_prediction=False)
        gt_vis_np = _ensure_plot_fields(_to_numpy_output(gt_vis))
        gt_png = scenario_dir / f"{scenario_id}__gt_context.png"
        gt_img = plot_gt(gt_vis_np, save_path=str(gt_png))

        panels: List[Tuple[str, Image.Image]] = [("GT context", gt_img)]
        row_manifest: List[Dict[str, Any]] = []

        for row_idx, row in row_entries:
            raw_data = dataset[row_idx]
            input_data = module.preprocess_GPTmodel(copy.deepcopy(raw_data), backward_prediction=False)
            with torch.no_grad():
                output_data = module.GPT_AR(input_data, backward_prediction=False, teacher_forcing=False)
            output_data = tokenizer.detokenize(
                output_data,
                detokenizing_gt=False,
                backward_prediction=False,
                teacher_forcing=False,
            )
            output_np = _ensure_plot_fields(_to_numpy_output(output_data))

            slot_id = str(row.get("slot_id") or ("factual_gt" if str(row.get("source_kind") or "") == "factual_gt" else f"row_{row_idx:04d}"))
            source_kind = str(row.get("source_kind") or "")
            label = str(row.get("requested_semantic_label") or "")
            stem = f"{scenario_id}__{slot_id}"

            pred_png = scenario_dir / f"{stem}__pred.png"
            pred_img = plot_pred(output_np, save_path=str(pred_png))
            title = f"{slot_id} | {source_kind} | {label}"
            panels.append((title, pred_img))

            row_manifest.append(
                {
                    "row_index": int(row_idx),
                    "slot_id": slot_id,
                    "source_kind": source_kind,
                    "requested_semantic_label": label,
                    "pred_png": str(pred_png),
                }
            )

        grid_png = scenario_dir / f"{scenario_id}__grid.png"
        _save_grid(panels, grid_png, columns=int(args.grid_columns), tile_size=int(args.tile_size))

        manifest.append(
            {
                "scenario_id": scenario_id,
                "grid_png": str(grid_png),
                "gt_context_png": str(gt_png),
                "rows": row_manifest,
            }
        )

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"num_scenes": len(manifest), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
