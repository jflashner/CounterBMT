# P2 Decoder Parity (MidGPT)

This document tracks P2 parity checks for decoder input assembly and relation-aware attention semantics in `counter_bmt_v2`.

## Scope
- MidGPT parity target (`0202_midgpt.yaml`) only.
- Dense masked relation attention path in JAX/NNX.
- No runtime TensorFlow/Waymo imports in v2 training.
- Cache parity is out of scope for P2.

## Core Runtime Flags
- `--model-preset midgpt_parity`
- `--tokenizer-mode adv_bmt_parity`

## Commands

1. Decoder mask parity smoke
```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/inspect_decoder_masks.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --n 20 --batch-size 4 --skip-steps 5
```

2. Decoder input parity smoke
```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_decoder_inputs.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --n 20 --batch-size 4 --skip-steps 5
```

3. Legacy side-by-side decoder input gate
```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_decoder_inputs.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --n 50 --batch-size 4 --skip-steps 5 \
  --legacy-check --legacy-root src/Adv-BMT \
  --max-embedding-diff 2e-4 --min-mask-match 0.9995
```

4. Export one decoder parity batch
```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/export_decoder_batch.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --index 0 \
  --skip-steps 5 \
  --out outputs/parity/decoder_batch_0
```

5. Training smoke with P2 path enabled
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p2_smoke_decoder_parity \
  --model-preset midgpt_parity \
  --tokenizer-mode adv_bmt_parity \
  --max-steps 80 --batch-size 2 \
  --mode mixed --reverse-prob 0.5 --skip-steps 5 \
  --eval-every 40 --eval-batches 2 --log-every 10
```

## Expected Gates
- `inspect_decoder_masks.py`: `a2t_causal_valid_match_rate == 1.0`, no NaNs.
- `compare_decoder_inputs.py --legacy-check`:
  - `input_mask_match_rate >= 0.9995`
  - `decoder_embedding_abs_max_common_valid <= 2e-4`
  - no NaNs.
- P2 training smoke:
  - train/eval runs to completion,
  - no CE index/range failures,
  - checkpoint written.
