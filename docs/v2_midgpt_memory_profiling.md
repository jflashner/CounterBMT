# v2 MidGPT Memory Profiling

Use [tools/profile_v2_midgpt_memory.py](/Users/joshuaflashner/Projects/CounterBMT/tools/profile_v2_midgpt_memory.py) to measure the v2 supervised parity stack without the extra orchestration from the learning probe.

The script:
- builds the real v2 supervised model, tokenizer, loader, optimizer, and train step
- runs a short train-only loop
- records per-batch collate shapes and token counts
- polls `nvidia-smi` during each batch to estimate GPU peak memory

Recommended first run on the H200 box:

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv-v2/bin/python tools/profile_v2_midgpt_memory.py \
  --train-data-dir /path/to/scenarionet/waymo/training \
  --val-data-dir /path/to/scenarionet/waymo/validation \
  --output-dir outputs/v2_midgpt_memory_profile \
  --runtime-preset legacy_midgpt_recipe \
  --batch-size 10 \
  --limit-train-batches 5 \
  --profile-batches 5 \
  --precision bf16-mixed
```

Important defaults:
- `--runtime-preset legacy_midgpt_recipe`
  keeps the profiler aligned with the forward-only parity recipe
- `--collate-padding-mode` defaults from that preset to `batch_local`
- `--xla-preallocate false`
  is the default for profiling so the measured peaks reflect actual demand more than reserved allocator pool size

Key outputs:
- `run_meta.json`
- `fit_start.json`
- `memory_profile.json`

Fields worth checking in `memory_profile.json`:
- `profiled_batches[*].batch_summary.collate_shape`
- `profiled_batches[*].batch_summary.active_agents_per_sample`
- `profiled_batches[*].batch_summary.valid_map_tokens_per_sample`
- `profiled_batches[*].batch_summary.active_traffic_lights_per_sample`
- `profiled_batches[*].peak_during_step.total_used_bytes`

Notes:
- The memory numbers come from `nvidia-smi`, so they are GPU-level peaks rather than framework allocator internals.
- For the cleanest comparison against legacy, run on one visible GPU with `CUDA_VISIBLE_DEVICES=0`.
- If you want to see compile-heavy first-step cost separately from steady-state cost, use `--warmup-batches 0` and inspect batch `0`, then rerun with `--warmup-batches 1`.
