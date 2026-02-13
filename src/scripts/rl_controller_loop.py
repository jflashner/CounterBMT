"""
rl_controller_loop.py

Minimal RL loop skeleton:
    local LLM controller (Qwen3 8B) -> BMT rollout -> VLM judge -> reward

This is intentionally lightweight and hackable. It does NOT implement PPO/GRPO
updates yet; it logs rollouts so you can plug in your favorite trainer.
"""

import argparse
import json
import inspect
import logging
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np

from counter_bmt.bmt_generator import CounterBMTGenerator, BiasedTokenSampler
from counter_bmt.scenario_export import export_trajectory_only, create_dataset_summary
from counter_bmt.scenarionet_visualizer import ScenarioNetVisualizer, ScenarioNetDatabase
from counter_bmt.llm_intervention_planner import LLMInterventionPlanner, build_trajectory_context
from counter_bmt.vlm_extractor import (
    VLMSafetyCriticalExtractor,
    TimestampedImage,
    GPT4oClient,
    MockGPT4oClient,
)

# Reuse the scenario loader from the main pipeline
from scripts.run_full_pipeline import (
    load_scenario_for_bmt_input,
    stage_1_load_and_visualize,
    resolve_scenario_index,
    _prepare_bmt_input,
    _run_bmt_generation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rl_controller_loop")


# =============================================================================
# Controller (Qwen3 8B)
# =============================================================================

@dataclass
class InterventionPlan:
    target_type: str  # "maneuver" or "decision"
    target_value: str
    aggressiveness: str = "normal"
    timestamp: Optional[float] = None
    description: str = ""


class QwenController:
    """
    Minimal local LLM controller using Qwen3 8B (via transformers).
    Outputs a JSON plan for the desired intervention.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        model=None,
        tokenizer=None,
        quantize_4bit: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise ImportError(
                "transformers + torch required for local Qwen controller. "
                "Install with: pip install torch transformers"
            ) from e

        self.torch = torch
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            quant_config = None
            if quantize_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except Exception as e:
                    raise ImportError(
                        "bitsandbytes required for 4-bit quantization. "
                        "Install with: pip install bitsandbytes"
                    ) from e
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    bnb_4bit_quant_type="nf4",
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
                quantization_config=quant_config,
            )
            self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        allowed_maneuvers = [
            "straight", "accelerate", "decelerate", "stop",
            "lane_change_left", "lane_change_right", "left_turn", "right_turn",
        ]
        allowed_decisions = ["proceed", "yield", "slow_down", "speed_up", "merge"]

        return (
            "You are a driving intervention planner. "
            "Output a JSON object only (no extra text).\n\n"
            "Context:\n"
            f"- scenario_id: {context.get('scenario_id')}\n"
            f"- iteration: {context.get('iteration')}\n"
            "\n"
            "Choose a target intervention.\n"
            f"Allowed maneuvers: {allowed_maneuvers}\n"
            f"Allowed decisions: {allowed_decisions}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "target_type": "maneuver" | "decision",\n'
            '  "target_value": "<value>",\n'
            '  "aggressiveness": "passive" | "normal" | "aggressive",\n'
            '  "timestamp": <float or null>,\n'
            '  "description": "<short description>"\n'
            "}\n"
        )

    def _extract_json(self, text: str) -> Optional[Dict]:
        # Grab the first JSON object by brace matching
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        return None
        return None

    def generate_plan(
        self,
        context: Dict[str, Any]
    ) -> Tuple[Optional[InterventionPlan], str, str, str, Optional[Any], Optional[Any]]:
        prompt = self._build_prompt(context)

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt")

        # Handle BatchEncoding vs Tensor
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        input_len = input_ids.shape[-1]

        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = output_ids[0, input_len:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        parsed = self._extract_json(response_text) or self._extract_json(decoded)
        if not parsed:
            return None, decoded, prompt, response_text, input_ids[0], response_ids

        plan = InterventionPlan(
            target_type=parsed.get("target_type", "maneuver"),
            target_value=parsed.get("target_value", "straight"),
            aggressiveness=parsed.get("aggressiveness", "normal"),
            timestamp=parsed.get("timestamp", None),
            description=parsed.get("description", ""),
        )
        return plan, decoded, prompt, response_text, input_ids[0], response_ids


class QwenPlannerClient:
    """Adapter to use local Qwen model as LLMInterventionPlanner client."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 1500):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def complete(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        max_new = max(max_tokens or 0, self.max_new_tokens)
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        input_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = output_ids[0, input_len:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text


# =============================================================================
# Judge
# =============================================================================

def _normalize_choice(s: str) -> str:
    return s.lower().strip().replace(" ", "_").replace("-", "_")


def score_features(features, target_type: str, target_value: str) -> Tuple[float, Dict]:
    details = {"target_type": target_type, "target_value": target_value}

    if target_type == "maneuver":
        best = 0.0
        best_match = None
        for m in features.maneuver_sequence:
            if m.maneuver_type.value == target_value:
                conf = float(getattr(m, "confidence", 1.0) or 1.0)
                if conf > best:
                    best = conf
                    best_match = m
        details["match"] = asdict(best_match) if best_match else None
        return best, details

    if target_type == "decision":
        best = 0.0
        best_match = None
        for d in features.critical_decisions:
            choice = _normalize_choice(d.ground_truth_choice)
            if choice == target_value:
                conf = float(getattr(d, "confidence", 1.0) or 1.0)
                if conf > best:
                    best = conf
                    best_match = d
        details["match"] = asdict(best_match) if best_match else None
        return best, details

    details["match"] = None
    return 0.0, details


def run_vlm_judge(
    replay_dir: Path,
    target_type: str,
    target_value: str,
    num_frames: int,
    client,
    debug_dir: Path,
) -> Tuple[float, Dict, Any, List[Tuple[str, float]], Optional[Path]]:
    visualizer = ScenarioNetVisualizer(data_dir=str(replay_dir))
    extractor = VLMSafetyCriticalExtractor(
        client=client,
        debug=True,
        debug_output_dir=str(debug_dir),
    )

    try:
        saved_images, trajectory, scenario_id = visualizer.render_scenario(
            scenario_index=0,
            num_frames=num_frames,
            output_dir=str(debug_dir / "frames"),
        )
        images = [TimestampedImage(path=p, timestamp=t) for p, t in saved_images]
        features = extractor.extract(images, scenario_id=scenario_id, trajectory=trajectory)
        reward, details = score_features(features, target_type, target_value)
        # Find latest VLM debug log if available
        debug_log_path = None
        debug_log_dir = Path(debug_dir) / "vlm_debug"
        if debug_log_dir.exists():
            logs = list(debug_log_dir.glob("*.json"))
            if logs:
                debug_log_path = max(logs, key=lambda p: p.stat().st_mtime)
        return reward, details, features, saved_images, debug_log_path
    finally:
        visualizer.close()


# =============================================================================
# Loop
# =============================================================================

def build_intervention(plan: InterventionPlan) -> Dict[str, Any]:
    variable = "maneuver_0" if plan.target_type == "maneuver" else "decision_0"
    return {
        "variable": variable,
        "value": plan.target_value,
        "original_value": None,
        "aggressiveness": plan.aggressiveness,
        "timestamp": plan.timestamp,
        "description": plan.description,
    }


def extract_ego_positions(output_dict) -> np.ndarray:
    """Extract ego positions [T, 2] from BMT output."""
    for key in ["decoder/reconstructed_position", "decoder/agent_position"]:
        if key in output_dict:
            positions = output_dict[key]
            break
    else:
        raise ValueError("No position data in BMT output")

    if hasattr(positions, "cpu"):
        positions = positions.cpu().numpy()

    if positions.ndim == 4:  # [B, T, N, D]
        positions = positions[0]
    if positions.ndim == 3:  # [T, N, D]
        return positions[:, 0, :2]
    raise ValueError(f"Unexpected position shape: {positions.shape}")


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses/enums/arrays to JSON-friendly structures."""
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def manual_grpo_update(
    model,
    optimizer,
    queries,
    responses,
    rewards,
    pad_token_id: int,
) -> Dict[str, float]:
    """Fallback GRPO-style update when trainer.step is unavailable."""
    import torch

    device = next(model.parameters()).device
    batch_size = len(queries)
    if batch_size == 0:
        return {"loss": 0.0, "reward_mean": 0.0, "reward_std": 0.0}

    prompt_lens = [q.numel() for q in queries]
    response_lens = [r.numel() for r in responses]
    max_prompt = max(prompt_lens)
    max_response = max(response_lens)

    prompt_ids = torch.full(
        (batch_size, max_prompt),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    prompt_mask = torch.zeros((batch_size, max_prompt), dtype=torch.long, device=device)
    completion_ids = torch.full(
        (batch_size, max_response),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    completion_mask = torch.zeros((batch_size, max_response), dtype=torch.long, device=device)

    for i, (q, r) in enumerate(zip(queries, responses)):
        q = q.to(device)
        r = r.to(device)
        q_len = q.numel()
        r_len = r.numel()
        if q_len:
            prompt_ids[i, :q_len] = q
            prompt_mask[i, :q_len] = 1
        if r_len:
            completion_ids[i, :r_len] = r
            completion_mask[i, :r_len] = 1

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1)
    target_ids = input_ids[:, 1:]
    token_logprobs = logprobs[:, :-1, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Mask for completion tokens in the shifted target space
    target_mask = torch.zeros_like(token_logprobs)
    for i, (q_len, r_len) in enumerate(zip(prompt_lens, response_lens)):
        if r_len == 0:
            continue
        start = max(q_len - 1, 0)
        end = start + r_len
        target_mask[i, start:end] = 1

    seq_logprob = (token_logprobs * target_mask).sum(dim=1)
    rewards_t = torch.tensor(rewards, device=device, dtype=seq_logprob.dtype)
    advantages = rewards_t - rewards_t.mean()

    loss = -(advantages * seq_logprob).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "reward_mean": float(rewards_t.mean().item()),
        "reward_std": float(rewards_t.std().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RL loop skeleton for CounterBMT")
    parser.add_argument("--data-dir", required=True, help="ScenarioNet data directory")
    parser.add_argument("--scenario-index", type=int, default=0, help="Scenario index")
    parser.add_argument("--scenario-id", type=str, default=None, help="Scenario ID (preferred over index)")
    parser.add_argument("--scenario-file", type=str, default=None, help="Direct scenario .pkl path")
    parser.add_argument(
        "--scenario-list",
        type=str,
        default=None,
        help="Path to text file with scenario IDs/indices/paths (one per line)",
    )
    parser.add_argument(
        "--scenario-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive index range of scenarios to run (e.g., 0 49)",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="Number of scenarios to run from the dataset start",
    )
    parser.add_argument("--bmt-checkpoint", required=True, help="BMT checkpoint path")
    parser.add_argument("--num-iters", type=int, default=3, help="Number of RL iterations")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames for VLM judge")
    parser.add_argument("--n-samples", type=int, default=1, help="BMT samples per iteration")
    parser.add_argument(
        "--rollouts-per-prompt",
        type=int,
        default=1,
        help="Number of controller rollouts per iteration (GRPO group size)",
    )
    parser.add_argument("--temperature", type=float, default=None, help="BMT sampling temperature")
    parser.add_argument("--bias-strength", type=float, default=8.0, help="Token bias strength")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Random exploration rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--mock-vlm", action="store_true", help="Use mock VLM (no API)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--vlm-model", type=str, default="gpt-4o", help="VLM model name")
    parser.add_argument("--controller-model", type=str, default="Qwen/Qwen3-8B", help="Qwen model name")
    parser.add_argument("--llm-max-new-tokens", type=int, default=512, help="Controller max new tokens")
    parser.add_argument(
        "--planner-client",
        type=str,
        default="gpt4o",
        choices=["gpt4o", "qwen", "none"],
        help="Client for LLMInterventionPlanner",
    )
    parser.add_argument("--planner-max-new-tokens", type=int, default=2048, help="Planner max new tokens")
    parser.add_argument("--quantize-4bit", action="store_true", help="Load controller in 4-bit (bitsandbytes)")
    parser.add_argument("--use-grpo", action="store_true", help="Enable TRL GRPO updates")
    parser.add_argument("--use-trl", action="store_true", help="(deprecated) Alias for --use-grpo")
    parser.add_argument("--trl-lr", type=float, default=1e-5, help="TRL learning rate")
    parser.add_argument("--trl-epochs", type=int, default=1, help="TRL GRPO epochs")
    parser.add_argument("--trl-kl-coef", type=float, default=0.05, help="TRL KL/Beta coefficient")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "rl_loop" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # VLM client
    if args.mock_vlm:
        vlm_client = MockGPT4oClient()
    else:
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
        vlm_client = GPT4oClient(model=args.vlm_model)

    # Load BMT model
    logger.info("RL: Loading BMT model...")
    generator = CounterBMTGenerator.from_checkpoint(args.bmt_checkpoint)
    generator.compiler.DEFAULT_ENCOURAGE_BIAS = args.bias_strength

    def parse_scenario_list(path: str) -> List[Dict[str, Any]]:
        targets = []
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if p.suffix == ".pkl" and p.exists():
                targets.append({"scenario_file": str(p)})
            elif line.isdigit():
                targets.append({"scenario_index": int(line)})
            else:
                targets.append({"scenario_id": line})
        return targets

    def summary_list_length(data_dir: str) -> int:
        from metadrive.scenario.utils import read_dataset_summary
        _, summary_list, _ = read_dataset_summary(str(Path(data_dir)))
        return len(summary_list)

    scenario_targets: List[Dict[str, Any]]
    if args.scenario_list:
        scenario_targets = parse_scenario_list(args.scenario_list)
    elif args.scenario_range:
        start, end = args.scenario_range
        if start < 0 or end < start:
            raise ValueError(f"Invalid scenario range: {start} {end}")
        scenario_targets = [{"scenario_index": i} for i in range(start, end + 1)]
    elif args.num_scenarios is not None:
        total = summary_list_length(args.data_dir)
        n = max(0, min(args.num_scenarios, total))
        scenario_targets = [{"scenario_index": i} for i in range(n)]
    else:
        scenario_targets = [{
            "scenario_id": args.scenario_id,
            "scenario_index": args.scenario_index,
            "scenario_file": args.scenario_file,
        }]

    # Controller (optional TRL GRPO)
    grpo_trainer = None
    grpo_optimizer = None
    use_grpo = args.use_grpo or args.use_trl
    if use_grpo:
        logger.info("RL: Initializing TRL GRPO...")
        try:
            from transformers import AutoTokenizer
            from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
        except Exception as e:
            raise ImportError(
                "TRL required for --use-grpo. Install with: pip install trl"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(args.controller_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        if args.quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise ImportError(
                    "bitsandbytes required for 4-bit quantization. "
                    "Install with: pip install bitsandbytes"
                ) from e
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.controller_model,
            trust_remote_code=True,
            torch_dtype=None,
            device_map="auto",
            quantization_config=quant_config,
        )
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        if not hasattr(model, "add_model_tags"):
            model.add_model_tags = lambda *_args, **_kwargs: None
        grpo_optimizer = torch.optim.AdamW(model.parameters(), lr=args.trl_lr)

        # Build GRPO config in a version-tolerant way
        config_kwargs = {
            "model_name": args.controller_model,
            "learning_rate": args.trl_lr,
            "batch_size": 1,
            "mini_batch_size": 1,
            "ppo_epochs": args.trl_epochs,
            "grpo_epochs": args.trl_epochs,
            "num_train_epochs": args.trl_epochs,
            "init_kl_coef": args.trl_kl_coef,
            "kl_coef": args.trl_kl_coef,
            "beta": args.trl_kl_coef,
        }
        cfg_sig = inspect.signature(GRPOConfig)
        filtered_cfg = {k: v for k, v in config_kwargs.items() if k in cfg_sig.parameters}
        grpo_config = GRPOConfig(**filtered_cfg)

        # Build GRPO trainer with best-effort kwargs
        def _constant_reward_fn(*_args, **_kwargs) -> float:
            return 0.0

        trainer_kwargs = {
            "config": grpo_config,
            "model": model,
            "tokenizer": tokenizer,
            "reward_funcs": [_constant_reward_fn],
        }
        tr_sig = inspect.signature(GRPOTrainer)
        filtered_tr = {k: v for k, v in trainer_kwargs.items() if k in tr_sig.parameters}
        grpo_trainer = GRPOTrainer(**filtered_tr)

        controller = QwenController(
            model_name=args.controller_model,
            model=grpo_trainer.model,
            tokenizer=tokenizer,
            quantize_4bit=args.quantize_4bit,
            max_new_tokens=args.llm_max_new_tokens,
        )
        logger.info("TRL GRPO enabled for controller updates")
    else:
        logger.info("RL: Initializing controller (no GRPO)...")
        controller = QwenController(
            model_name=args.controller_model,
            quantize_4bit=args.quantize_4bit,
            max_new_tokens=args.llm_max_new_tokens,
        )

    # LLM planner client selection
    if args.planner_client == "none":
        planner_client = None
    elif args.planner_client == "qwen":
        planner_client = QwenPlannerClient(
            controller.model,
            controller.tokenizer,
            max_new_tokens=args.planner_max_new_tokens,
        )
    else:
        planner_client = None if args.mock_vlm else vlm_client

    # Enable LLM-based intervention planning (as in run_full_pipeline)
    generator.compiler.llm_planner = LLMInterventionPlanner(llm_client=planner_client)

    for target in scenario_targets:
        last_reward = 0.0
        scenario_out_dir = output_dir
        scenario_id = target.get("scenario_id")
        scenario_index = target.get("scenario_index")
        scenario_file_path = target.get("scenario_file")

        # Resolve scenario
        logger.info("RL: Resolving scenario...")
        if scenario_id and not scenario_file_path:
            scenario_index = resolve_scenario_index(Path(args.data_dir), scenario_id)
        stage1 = None
        if not scenario_file_path:
            # Stage 1 uses ScenarioEnv ordering and returns the actual file path
            logger.info("RL: Running stage_1_load_and_visualize...")
            stage1 = stage_1_load_and_visualize(
                data_dir=Path(args.data_dir),
                scenario_index=scenario_index,
                output_dir=output_dir / "stage1",
                num_frames=args.num_frames,
            )
            scenario_file_path = stage1.get("scenario_file_path")
            if not scenario_file_path:
                raise ValueError("stage_1_load_and_visualize did not return scenario_file_path")
            if scenario_id and stage1.get("scenario_id") != scenario_id:
                logger.warning(
                    f"Requested scenario_id={scenario_id} but stage1 loaded {stage1.get('scenario_id')}"
                )
            scenario_id = stage1.get("scenario_id", scenario_id)

        if scenario_id:
            scenario_out_dir = output_dir / scenario_id
        else:
            scenario_out_dir = output_dir / f"idx_{scenario_index}"
        scenario_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RL: Loading scenario file: {scenario_file_path}")
        scenario_data = load_scenario_for_bmt_input(
            data_dir=args.data_dir,
            scenario_index=scenario_index,
            config=generator.config,
            tokenizer=generator.tokenizer,
            file_path=scenario_file_path,
        )

        # Map center for exporting
        map_center = None
        preprocessed = scenario_data.get("preprocessed", {})
        if "metadata/map_center" in preprocessed:
            map_center = preprocessed["metadata/map_center"]

        rollout_log = scenario_out_dir / "rollouts.jsonl"

        for it in range(args.num_iters):
            logger.info(f"RL: Iteration {it} start")
            iter_dir = scenario_out_dir / f"iter_{it:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            grpo_queries = []
            grpo_responses = []
            grpo_rewards = []

            for r in range(max(1, args.rollouts_per_prompt)):
                logger.info(f"RL: Rollout {r} start")
                context = {
                    "scenario_id": scenario_data.get("scenario_id"),
                    "iteration": it,
                    "rollout": r,
                    "last_reward": last_reward,
                }

                # Exploration: random plan occasionally
                if random.random() < args.epsilon:
                    plan = InterventionPlan(
                        target_type="maneuver",
                        target_value=random.choice([
                            "lane_change_left", "lane_change_right", "left_turn",
                            "right_turn", "accelerate", "decelerate", "straight"
                        ]),
                        aggressiveness=random.choice(["passive", "normal", "aggressive"]),
                        timestamp=None,
                        description="random exploration",
                    )
                    llm_raw = "<random>"
                    prompt = ""
                    response_text = ""
                    query_ids = None
                    response_ids = None
                else:
                    logger.info("RL: Generating plan from controller...")
                    plan, llm_raw, prompt, response_text, query_ids, response_ids = controller.generate_plan(context)
                    if plan is None:
                        plan = InterventionPlan(target_type="maneuver", target_value="straight")
                        prompt = ""
                        response_text = ""
                        query_ids = None
                        response_ids = None

                intervention = build_intervention(plan)
                logger.info(f"RL: Intervention = {intervention}")

                rollout_dir = iter_dir / f"rollout_{r:03d}"
                replay_dir = rollout_dir / "replay_scenarios"
                replay_dir.mkdir(parents=True, exist_ok=True)

                # BMT rollout (reuse run_full_pipeline generation path to avoid re-tokenize)
                logger.info("RL: Preparing BMT input...")
                input_data = scenario_data.get("preprocessed", scenario_data["raw_data"])
                input_dict = _prepare_bmt_input(input_data, generator.device, generator.config)
                logger.info("RL: Compiling token biases...")
                # Build trajectory context for LLM planner (if we have stage1 trajectory)
                traj_context = None
                if stage1 and stage1.get("trajectory") is not None:
                    traj_context = build_trajectory_context(
                        trajectory=stage1["trajectory"][:, :2],
                        intervention_time_s=plan.timestamp,
                        current_maneuver=plan.target_value if plan.target_type == "maneuver" else None,
                        aggressiveness=plan.aggressiveness,
                    )

                token_biases = generator.compiler.compile_from_dag_intervention(
                    intervention=intervention,
                    encourage_bias=args.bias_strength,
                    use_llm_planning=True,
                    trajectory_context=traj_context,
                    debug_output_dir=str(rollout_dir / "planner_debug"),
                    intervention_idx=it,
                )
                logger.info(f"RL: Bias groups = {len(token_biases)}")
                sampler = BiasedTokenSampler(token_biases, generator.compiler.token_space)

                logger.info("RL: Running BMT generation...")
                output_dict = _run_bmt_generation(
                    generator.model,
                    generator.config,
                    generator.tokenizer,
                    input_dict,
                    generator.device,
                    use_bias=True,
                    sampler=sampler,
                    temperature=args.temperature,
                )
                logger.info("RL: Extracting ego positions...")
                ego_positions = extract_ego_positions(output_dict)

                # Export for replay/judge
                logger.info("RL: Exporting replay scenario...")
                export_path = replay_dir / (
                    f"sd_counterfactual_1.0_{scenario_data['scenario_id']}_iter{it:03d}_r{r:03d}.pkl"
                )
                export_trajectory_only(
                    trajectory=ego_positions,
                    original_scenario=scenario_data["raw_data"],
                    output_path=export_path,
                    intervention_name=f"{plan.target_type}:{plan.target_value}",
                    original_file_path=scenario_data.get("file_path"),
                    map_center=map_center,
                )
                create_dataset_summary([export_path], replay_dir)

                # VLM judge
                logger.info("RL: Running VLM judge...")
                reward, details, features, saved_images, vlm_debug_log = run_vlm_judge(
                    replay_dir=replay_dir,
                    target_type=plan.target_type,
                    target_value=plan.target_value,
                    num_frames=args.num_frames,
                    client=vlm_client,
                    debug_dir=rollout_dir / "vlm_debug",
                )
                last_reward = reward
                logger.info(f"RL: Reward = {reward}")

                log_entry = {
                    "iteration": it,
                    "rollout": r,
                    "plan": asdict(plan),
                    "intervention": intervention,
                    "reward": reward,
                    "details": to_jsonable(details),
                    "llm_raw": llm_raw,
                    "prompt": prompt,
                    "response_text": response_text,
                    "export_path": str(export_path),
                }
                with open(rollout_log, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                # Human-readable debug outputs
                with open(rollout_dir / "llm_output.txt", "w") as f:
                    f.write("=== Controller Prompt ===\n")
                    f.write(prompt or "<empty>")
                    f.write("\n\n=== Controller Response (raw) ===\n")
                    f.write(llm_raw or "<empty>")
                    f.write("\n\n=== Parsed Plan ===\n")
                    f.write(json.dumps(asdict(plan), indent=2))

                with open(rollout_dir / "vlm_output.txt", "w") as f:
                    f.write(features.summary())
                    f.write("\n\n=== VLM Raw Responses ===\n")
                    raw = getattr(features, "vlm_raw_responses", {})
                    f.write(json.dumps(raw, indent=2))

                # Save reward context with images + VLM prompt/response log path
                reward_report = {
                    "scenario_id": scenario_data.get("scenario_id"),
                    "iteration": it,
                    "rollout": r,
                    "target_type": plan.target_type,
                    "target_value": plan.target_value,
                    "reward": reward,
                    "image_paths": [p for p, _ in saved_images],
                    "image_timestamps": [t for _, t in saved_images],
                    "vlm_debug_log": str(vlm_debug_log) if vlm_debug_log else None,
                    "vlm_raw_responses": to_jsonable(getattr(features, "vlm_raw_responses", {})),
                }
                with open(rollout_dir / "reward_report.json", "w") as f:
                    json.dump(reward_report, f, indent=2)

                logger.info(
                    f"[iter {it} rollout {r}] reward={reward:.3f} "
                    f"target={plan.target_type}:{plan.target_value}"
                )

                if query_ids is not None and response_ids is not None:
                    grpo_queries.append(query_ids)
                    grpo_responses.append(response_ids)
                    grpo_rewards.append(reward)

            # TRL GRPO update (grouped rollouts)
            if grpo_trainer is not None and grpo_queries and grpo_responses:
                logger.info("RL: Running GRPO step...")
                if hasattr(grpo_trainer, "step"):
                    grpo_stats = grpo_trainer.step(
                        grpo_queries,
                        grpo_responses,
                        grpo_rewards,
                    )
                elif callable(getattr(grpo_trainer, "_step", None)):
                    grpo_stats = grpo_trainer._step(
                        grpo_queries,
                        grpo_responses,
                        grpo_rewards,
                    )
                else:
                    pad_id = tokenizer.pad_token_id
                    if pad_id is None:
                        pad_id = tokenizer.eos_token_id
                    grpo_stats = manual_grpo_update(
                        grpo_trainer.model,
                        grpo_optimizer,
                        grpo_queries,
                        grpo_responses,
                        grpo_rewards,
                        pad_id,
                    )
                with open(output_dir / "trl_grpo_stats.jsonl", "a") as f:
                    f.write(json.dumps(grpo_stats, default=str) + "\n")

    summary = {
        "num_iters": args.num_iters,
        "output_dir": str(output_dir),
        "scenario_count": len(scenario_targets),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. Logs in {output_dir}")


if __name__ == "__main__":
    main()
