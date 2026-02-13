# CounterBMT v2 Day 1 Implementation Notes

This document summarizes the Day 1 refactor work and explains the new data-loading and NNX trajectory stack behavior.

## 1) Files changed and what each one does

1. `src/counter_bmt_v2/data/__init__.py`
- Exposes the new data API:
  - `NNXBMTSceneSample`
  - `ScenarioNetNNXLoader`
  - `collate_nnx_scene_samples`

2. `src/counter_bmt_v2/data/scenarionet.py`
- New minimal ScenarioNet loader for NNX training.
- Implements a clean extraction path (no legacy Adv-BMT trainer coupling) for:
  - agent trajectories / headings / velocities / masks / IDs / type IDs / shapes
  - map vectors in Adv-BMT-style 27D map feature format
  - traffic-light feature tensors
- Adds `collate_nnx_scene_samples` for padded batched tensors.

3. `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`
- Extended with explicit scene encoding support:
  - added `NNXSceneEncoderConfig`
  - added `NNXSceneTokenEncoder`
- Updated A2S attention to use scene token masks.
- Updated `NNXBidirectionalMotionTransformer` so it can consume either:
  - precomputed scene tokens, or
  - raw map + traffic-light tensors and encode internally.
- Updated `autoregressive_token_rollout` to pass optional scene tensor inputs.
- Paper-alignment comments were added throughout.

4. `src/counter_bmt_v2/trajectory_jax/presets.py`
- Added two explicit architecture presets:
  - `paper_like_small_config()`: faster local iteration
  - `paper_like_full_config()`: closer to Adv-BMT scale intent

5. `src/counter_bmt_v2/trajectory_jax/__init__.py`
- Exported new scene encoder config/class and presets.

6. `src/counter_bmt_v2/__init__.py`
- Exported new data loader/collate symbols at package root.

7. `src/counter_bmt_v2/ROADMAP.md`
- Added Day 1 progress section.

## 2) Data loading flow (new `ScenarioNetNNXLoader`)

Paper intent reference:
- Adv-BMT scene context is built from agents + map + traffic signals.
- We keep that factorization explicit in the emitted tensors.

### 2.1 Scenario discovery
- Loader scans:
  - `data_dir/sd_*.pkl`
  - `data_dir/_*/sd_*.pkl`
- Files are sorted deterministically for reproducible splits.

### 2.2 Agent extraction
- Keeps agents with at least one valid step.
- Sorts by valid-length (most informative first).
- Forces SDC as index `0` when available.
- Emits:
  - `agent_position_xy [T,N,2]`
  - `agent_heading [T,N]`
  - `agent_velocity_xy [T,N,2]`
  - `agent_valid_mask [T,N]`
  - `agent_ids [N]`, `agent_type_ids [N]`, `agent_shape [N,3]`

### 2.3 Map feature extraction (27D)
- Converts map feature polyline/polygon/position into segment vectors.
- Packs map vectors into Adv-BMT-style 27D structure:
  - geometric fields (start/end/direction/heading/sin/cos/length)
  - categorical type flags (slots 13..24)
  - cumulative distance
  - validity channel
- Splits long polylines into chunks of `max_vectors_per_map_feature`.
- Truncates to nearest `max_map_features` to SDC current XY (distance-based).

### 2.4 Traffic-light extraction
- Emits `traffic_light_feature [T,L,7]` with:
  - stop-point xyz
  - one-hot-like state channels: green/yellow/red/unknown
- Emits masks and static positions:
  - `traffic_light_valid_mask [T,L]`
  - `traffic_light_position [L,3]`

### 2.5 Batch collation
- `collate_nnx_scene_samples` pads batch to max dims in that batch:
  - `T_max`, `N_max`, `M_max`, `V_max`, `L_max`
- Returns pure NumPy batch dict; JAX conversion happens later in train/infer steps.

## 3) Scene encoding flow (`NNXSceneTokenEncoder`)

Paper intent reference:
- Scene representation should preserve map and signal structure for A2S conditioning.

### 3.1 Map token encoding
- Input: `map_feature [B,M,V,27]`, `map_feature_valid_mask [B,M,V]`, `map_position [B,M,3]`
- Steps:
  1. per-vector projection: `27 -> d_model`
  2. masked mean pool over vectors (`V`) to one token per map chunk
  3. add projected map-position embedding
  4. mask invalid map tokens

### 3.2 Traffic-light token encoding
- Input can be temporal (`[B,T,L,7]`) or compact (`[B,L,7]`).
- If temporal:
  - masked mean pool over time (`T`) to one token per light.
- Then:
  - project `7 -> d_model`
  - add projected traffic-light position embedding
  - mask invalid light tokens

### 3.3 Scene token assembly
- Concatenates map tokens and traffic-light tokens into `[B,S,d_model]`.
- Concatenates masks into `[B,S]`.
- Caps token count with `max_scene_tokens`.
- Applies final RMSNorm.

## 4) NNX model flow (`NNXBidirectionalMotionTransformer`)

Paper intent reference:
- Shared token space for motion actions.
- Relation-aware decoding with A2A / A2T / A2S.

### 4.1 Input embeddings
- Motion token embedding from previous action tokens.
- Agent type embedding.
- Agent ID embedding.
- Agent shape projection.
- Continuous motion projection.
- Reverse indicator embedding.
- Summed into hidden tensor `h [B,T,N,d_model]`.

### 4.2 Scene context source
- Priority order:
  1. provided `scene_tokens` (+ optional `scene_token_mask`)
  2. encode from raw `scene_map_*` + `scene_tl_*`
  3. fallback pooled tokens from hidden state (mean/max)

### 4.3 Decoder blocks
- Each block applies:
  - A2A attention (agent-to-agent at same time)
  - A2T attention (agent-to-time)
  - A2S attention (agent-to-scene tokens; now mask-aware)
  - feed-forward update

### 4.4 Output
- Final RMSNorm + linear token head -> logits `[B,T,N,1089]`
- `1089 = 33 * 33` from Adv-BMT token-space settings.

## 5) Why synthetic vs real loader tensor shapes did not match

Short answer: yes, this is expected.

### 5.1 What happened
- Synthetic smoke test intentionally used manual tiny dimensions, e.g.:
  - `B=2, T=4, N=3, M=5, V=6, L=2`
- Real loader test used actual scenario data and loader limits, e.g.:
  - `B=1, T=5, N=8` in that forward smoke
  - map/traffic dimensions from real sample and configured caps.

### 5.2 Why this is correct
- The model is designed for variable scene sizes in:
  - number of agents `N`
  - number of map tokens `M`
  - vectors per map token `V`
  - number of traffic lights `L`
  - temporal horizon `T`
- Only the vocab dimension is fixed (`1089`) by token-space config.

### 5.3 One subtle collation detail
- In mixed batches, even if one sample has `L=0`, batch `L_max` can be >0 if another sample has lights.
- That is why you can see shapes like `[B, T, 8, 7]` after collating two samples where one had no lights.

## 6) Expected invariants you can rely on

1. Final token logits always have last dim `1089` with current preset.
2. Scene/channel dims are allowed to vary before batching.
3. After `collate_nnx_scene_samples`, dimensions are batch-padded to batch max.
4. A2S will now use real scene context whenever provided (instead of fallback pooling).

