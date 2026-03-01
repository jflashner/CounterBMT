"""VLM-based DAG-conformance alignment verifier (sampled, cached, bounded)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from counter_bmt_v2.config import VLMAlignmentConfig
from counter_bmt_v2.contracts import BayesianDAG, Intervention, ScenarioInput, TrajectoryRollout
from counter_bmt_v2.llm import OpenAIChatClient

from .vlm_alignment_evidence import build_alignment_evidence_bundle
from .vlm_alignment_prompt import build_alignment_prompt, parse_alignment_response

logger = logging.getLogger(__name__)


@dataclass
class AlignmentBatchResult:
    scores: np.ndarray
    scored_mask: np.ndarray
    diagnostics: Dict[str, float]


class VLMAlignmentVerifier:
    """Scores rollout DAG-conformance with GPT-4o under a strict cost budget."""

    def __init__(
        self,
        *,
        cfg: VLMAlignmentConfig,
        output_dir: str | Path,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(str(cfg.cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[OpenAIChatClient] = None
        if str(cfg.backend) == "gpt4o":
            try:
                self._client = OpenAIChatClient(
                    model=str(cfg.model),
                    api_key=cfg.api_key,
                    timeout_s=float(cfg.per_call_timeout_s),
                )
            except Exception as exc:
                logger.warning("VLM alignment verifier disabled (OpenAI init failed): %s", exc)
                self._client = None

    def _hash_u64(self, text: str) -> int:
        h = hashlib.sha256(text.encode("utf-8")).digest()[:8]
        return int.from_bytes(h, byteorder="big", signed=False)

    def _deterministic_unit(self, text: str) -> float:
        u = self._hash_u64(text)
        return float((u % 10_000_000) / 10_000_000.0)

    def _rollout_hash(self, rollout: TrajectoryRollout) -> str:
        arr = np.asarray(rollout.trajectory_xy, dtype=np.float32)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _cache_key(
        self,
        *,
        scenario_id: str,
        rollout: TrajectoryRollout,
        dag_text: str,
        intervention_text: str,
    ) -> str:
        payload = {
            "scenario_id": str(scenario_id),
            "prompt_version": str(self.cfg.prompt_version),
            "model": str(self.cfg.model),
            "rollout_hash": self._rollout_hash(rollout),
            "dag_hash": hashlib.sha256(dag_text.encode("utf-8")).hexdigest(),
            "intervention_text": str(intervention_text),
        }
        src = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cached_score(self, key: str) -> Optional[float]:
        p = self._cache_path(key)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            score = float(data.get("score"))
            if np.isfinite(score):
                return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return None
        return None

    def _save_cache(
        self,
        *,
        key: str,
        score: float,
        payload: Dict[str, object],
    ) -> None:
        out = dict(payload)
        out["score"] = float(np.clip(float(score), 0.0, 1.0))
        out["cached_at_unix"] = time.time()
        self._cache_path(key).write_text(json.dumps(out, indent=2), encoding="utf-8")

    def _select_indices(self, *, step: int, scenario_id: str, total: int) -> List[int]:
        if total <= 0:
            return []
        every = max(1, int(self.cfg.every_n_steps))
        if step % every != 0:
            return []
        sample_rate = float(np.clip(self.cfg.sample_rate, 0.0, 1.0))
        cand: List[Tuple[int, float]] = []
        for i in range(total):
            key = f"{scenario_id}|{step}|{i}|{self.cfg.prompt_version}"
            u = self._deterministic_unit(key)
            if u <= sample_rate:
                cand.append((i, u))
        cand.sort(key=lambda x: x[1])
        m = max(0, int(self.cfg.max_calls_per_step))
        if m > 0:
            cand = cand[:m]
        return [i for i, _ in cand]

    def _encode_images(self, image_paths: Sequence[str]) -> List[str]:
        out: List[str] = []
        for p in image_paths:
            path = Path(p)
            if not path.is_file():
                continue
            out.append(base64.b64encode(path.read_bytes()).decode("utf-8"))
        return out

    def _mock_score(self, rollout: TrajectoryRollout, scenario: ScenarioInput) -> float:
        pred = np.asarray(rollout.trajectory_xy, dtype=np.float32)
        obs = np.asarray(scenario.ego_trajectory_xy, dtype=np.float32) if scenario.ego_trajectory_xy is not None else np.zeros((0, 2), dtype=np.float32)
        if pred.ndim != 2 or pred.shape[0] < 2:
            return 0.0
        if obs.ndim == 2 and obs.shape[0] >= 2:
            n = min(obs.shape[0], pred.shape[0])
            err = np.linalg.norm(pred[:n, :2] - obs[:n, :2], axis=-1)
            e = float(np.mean(err))
            return float(np.clip(np.exp(-0.15 * e), 0.0, 1.0))
        disp = float(np.linalg.norm(pred[-1, :2] - pred[0, :2]))
        return float(np.clip(disp / 20.0, 0.0, 1.0))

    def _score_one(
        self,
        *,
        scenario: ScenarioInput,
        dag: BayesianDAG,
        intervention: Intervention,
        rollout: TrajectoryRollout,
        rollout_index: int,
        step: int,
        evidence_dir: Path,
    ) -> Tuple[Optional[float], Dict[str, object]]:
        t0 = time.perf_counter()
        bundle = build_alignment_evidence_bundle(
            scene=scenario,
            dag=dag,
            intervention=intervention,
            rollout=rollout,
            out_dir=evidence_dir,
            num_frames=int(self.cfg.num_frames),
            max_agents_render=int(self.cfg.max_agents_render),
        )
        frame_paths = [f.path for f in bundle.frames_for_vlm]
        prompt = build_alignment_prompt(
            scenario_id=str(scenario.scenario_id),
            intervention_text=bundle.intervention_text,
            dag_text=bundle.dag_text,
            prompt_version=str(self.cfg.prompt_version),
            num_frames=len(frame_paths),
        )
        if bool(self.cfg.save_evidence_artifacts):
            (evidence_dir / "dag_compact.txt").write_text(bundle.dag_text, encoding="utf-8")
            (evidence_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        if str(self.cfg.backend) == "mock" or self._client is None:
            score = self._mock_score(rollout, scenario)
            return score, {
                "backend": "mock",
                "rollout_index": int(rollout_index),
                "latency_ms": 1000.0 * (time.perf_counter() - t0),
                "conformance_score": float(score),
                "confidence": 0.5,
                "matched_factors": [],
                "violations": [],
                "reason": "mock_alignment_score",
            }

        raw = self._client.complete(
            prompt=prompt,
            images_base64=self._encode_images(frame_paths),
            temperature=0.1,
            max_tokens=800,
        )
        parsed = parse_alignment_response(raw)
        if bool(self.cfg.save_evidence_artifacts):
            (evidence_dir / "response_raw.txt").write_text(str(raw), encoding="utf-8")
        if parsed is None:
            return None, {
                "backend": "gpt4o",
                "rollout_index": int(rollout_index),
                "latency_ms": 1000.0 * (time.perf_counter() - t0),
                "error": "invalid_response",
            }
        payload = {
            "backend": "gpt4o",
            "rollout_index": int(rollout_index),
            "latency_ms": 1000.0 * (time.perf_counter() - t0),
            "conformance_score": float(parsed.score),
            "confidence": float(parsed.confidence),
            "matched_factors": parsed.matched_factors,
            "violations": parsed.violations,
            "reason": parsed.reason,
            "raw": parsed.raw,
        }
        if bool(self.cfg.save_evidence_artifacts):
            (evidence_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return float(parsed.score), payload

    def score_rollouts(
        self,
        *,
        step: int,
        scenario: ScenarioInput,
        dag: BayesianDAG,
        intervention: Intervention,
        rollouts: Sequence[TrajectoryRollout],
    ) -> AlignmentBatchResult:
        n = int(len(rollouts))
        neutral = float(np.clip(float(self.cfg.neutral_score), 0.0, 1.0))
        scores = np.full((n,), neutral, dtype=np.float32)
        scored_mask = np.zeros((n,), dtype=bool)

        diagnostics: Dict[str, float] = {
            "calls_attempted": 0.0,
            "calls_success": 0.0,
            "cache_hits": 0.0,
            "timeouts": 0.0,
            "errors": 0.0,
            "scored_fraction": 0.0,
            "latency_ms_mean": 0.0,
            "step_skipped": 0.0,
        }
        if not bool(self.cfg.enabled) or str(self.cfg.source_mode) != "vlm_replace":
            diagnostics["step_skipped"] = 1.0
            return AlignmentBatchResult(scores=scores, scored_mask=scored_mask, diagnostics=diagnostics)

        selected = self._select_indices(step=int(step), scenario_id=str(scenario.scenario_id), total=n)
        if not selected:
            diagnostics["step_skipped"] = 1.0
            return AlignmentBatchResult(scores=scores, scored_mask=scored_mask, diagnostics=diagnostics)

        if bool(self.cfg.save_evidence_artifacts):
            step_dir = self.output_dir / str(self.cfg.evidence_subdir) / f"step_{int(step):06d}" / f"scenario_{scenario.scenario_id}"
        else:
            step_dir = self.output_dir / ".tmp_vlm_alignment" / f"scenario_{scenario.scenario_id}"
        step_dir.mkdir(parents=True, exist_ok=True)
        intervention_text = f"{intervention.variable}={intervention.value}"
        dag_text = "\n".join(
            [
                f"nodes={len(dag.nodes)} edges={len(dag.edges)}",
                *[f"{e.parent_id}->{e.child_id}" for e in dag.edges[:50]],
            ]
        )

        pending: List[int] = []
        for i in selected:
            key = self._cache_key(
                scenario_id=str(scenario.scenario_id),
                rollout=rollouts[i],
                dag_text=dag_text,
                intervention_text=intervention_text,
            )
            cached = self._load_cached_score(key)
            if cached is not None:
                scores[i] = float(cached)
                scored_mask[i] = True
                diagnostics["cache_hits"] += 1.0
                continue
            pending.append(i)

        if pending:
            diagnostics["calls_attempted"] = float(len(pending))
            future_to_idx: Dict[object, int] = {}
            latencies: List[float] = []
            with ThreadPoolExecutor(max_workers=max(1, int(self.cfg.max_concurrency))) as ex:
                for i in pending:
                    evidence_dir = step_dir / f"rollout_{i:03d}"
                    fut = ex.submit(
                        self._score_one,
                        scenario=scenario,
                        dag=dag,
                        intervention=intervention,
                        rollout=rollouts[i],
                        rollout_index=int(i),
                        step=int(step),
                        evidence_dir=evidence_dir,
                    )
                    future_to_idx[fut] = int(i)

                start = time.perf_counter()
                remaining = float(max(0.0, float(self.cfg.step_wait_budget_s)))
                unresolved = set(future_to_idx.keys())
                while unresolved and remaining > 0.0:
                    done, not_done = wait(unresolved, timeout=remaining, return_when=FIRST_COMPLETED)
                    if not done:
                        break
                    for fut in done:
                        unresolved.discard(fut)
                        i = future_to_idx[fut]
                        try:
                            score, payload = fut.result(timeout=max(0.0, float(self.cfg.per_call_timeout_s)))
                        except Exception as exc:
                            diagnostics["errors"] += 1.0
                            logger.debug("VLM alignment call failed for rollout %d: %s", i, exc)
                            continue
                        latency = float(payload.get("latency_ms", 0.0))
                        if np.isfinite(latency) and latency > 0.0:
                            latencies.append(latency)
                        if score is None:
                            diagnostics["errors"] += 1.0
                            continue
                        score = float(np.clip(float(score), 0.0, 1.0))
                        scores[i] = score
                        scored_mask[i] = True
                        diagnostics["calls_success"] += 1.0
                        key = self._cache_key(
                            scenario_id=str(scenario.scenario_id),
                            rollout=rollouts[i],
                            dag_text=dag_text,
                            intervention_text=intervention_text,
                        )
                        self._save_cache(
                            key=key,
                            score=score,
                            payload={
                                "scenario_id": str(scenario.scenario_id),
                                "step": int(step),
                                "rollout_index": int(i),
                                "payload": payload,
                            },
                        )
                    elapsed = time.perf_counter() - start
                    remaining = float(max(0.0, float(self.cfg.step_wait_budget_s) - elapsed))

                if unresolved:
                    diagnostics["timeouts"] += float(len(unresolved))
                    for fut in unresolved:
                        fut.cancel()
                if latencies:
                    diagnostics["latency_ms_mean"] = float(np.mean(np.asarray(latencies, dtype=np.float32)))

        if np.any(scored_mask):
            mean_score = float(np.mean(scores[scored_mask]))
            if str(self.cfg.unscored_policy) == "step_mean_fill":
                scores[~scored_mask] = mean_score
        else:
            scores[:] = neutral

        diagnostics["scored_fraction"] = float(np.mean(scored_mask.astype(np.float32))) if n > 0 else 0.0
        return AlignmentBatchResult(scores=scores, scored_mask=scored_mask, diagnostics=diagnostics)
