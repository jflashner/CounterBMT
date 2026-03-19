# Legacy MidGPT Memory Profiling

Use this when you want a concrete answer to:

- how much GPU memory the released legacy Adv-BMT MidGPT trainer actually uses
- what batch-local shapes produced that memory footprint

The profiler script is:

- [profile_legacy_midgpt_memory.py](/Users/joshuaflashner/Projects/CounterBMT/tools/profile_legacy_midgpt_memory.py)

## What It Records

For a short training run, it writes:

- `run_meta.json`
- `fit_start.json`
- `memory_profile.json`
- optionally `peak_memory_summary.txt`

The key measurements are:

- trainable and total parameter counts
- memory after the model is moved to GPU
- memory after each profiled batch is transferred to GPU
- peak memory after backward
- peak memory at train-batch end
- batch-local shapes:
  - padded agents
  - active agents per sample
  - modeled agents
  - decoder token steps
  - valid map tokens
- active traffic lights

The profiler also applies a small profiling-only collate sanitizer for legacy
metadata fields:

- NumPy scalars are converted to plain Python scalars
- unsupported non-array metadata objects are converted to `repr(...)` strings

This does not change the model inputs. It only prevents the legacy collate
assertion from tripping on metadata-like fields the model never reads.

## Recommended First Run

Run this inside the dedicated legacy environment on the H200 box:

```bash
.venv-legacy-adv-bmt/bin/python tools/profile_legacy_midgpt_memory.py \
  --legacy-root src/Adv-BMT \
  --config-name 0202_midgpt \
  --train-data-dir /path/to/scenarionet/waymo/training \
  --val-data-dir /path/to/scenarionet/waymo/validation \
  --output-dir outputs/legacy_midgpt_memory_profile \
  --batch-size 10 \
  --limit-train-batches 5 \
  --profile-batches 5
```

This matches the released MidGPT batch size and records five real train batches.

## Useful Variants

Profile the legacy stack with a lower agent cap:

```bash
.venv-legacy-adv-bmt/bin/python tools/profile_legacy_midgpt_memory.py \
  --legacy-root src/Adv-BMT \
  --config-name 0202_midgpt \
  --train-data-dir /path/to/scenarionet/waymo/training \
  --val-data-dir /path/to/scenarionet/waymo/validation \
  --output-dir outputs/legacy_midgpt_memory_profile_max64 \
  --batch-size 10 \
  --max-agents 64 \
  --limit-train-batches 5
```

Write a full CUDA allocator summary for the peak batch:

```bash
.venv-legacy-adv-bmt/bin/python tools/profile_legacy_midgpt_memory.py \
  --legacy-root src/Adv-BMT \
  --config-name 0202_midgpt \
  --train-data-dir /path/to/scenarionet/waymo/training \
  --val-data-dir /path/to/scenarionet/waymo/validation \
  --output-dir outputs/legacy_midgpt_memory_profile \
  --write-memory-summary
```

## How To Read The Output

`fit_start.json` tells you how much memory the model itself occupies once it is
on GPU.

`memory_profile.json` tells you how much memory the actual training batches and
their activations consume. That is the number to compare against the v2 parity
path.

The most important fields are:

- `profiled_batches[*].batch_summary.padded_agents`
- `profiled_batches[*].batch_summary.active_agents_per_sample`
- `profiled_batches[*].batch_summary.decoder_token_steps`
- `profiled_batches[*].batch_end_peak.max_allocated_bytes`
- `profiled_batches[*].batch_end_peak.max_reserved_bytes`

That combination lets us distinguish:

- model size
- batch padding effects
- activation growth from relation-heavy attention
