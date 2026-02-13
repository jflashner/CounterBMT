# Literature Review Notes (PromptBN + Adv-BMT)

This note captures implementation constraints extracted from:
- `PromptBN.pdf` ("Bayesian Network Structure Discovery Using Large Language Models")
- `Adv-BMT.pdf` ("Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation")

## 1) PromptBN principles applied to v2 DAG builder

Paper takeaways (PromptBN phase):
1. Single-step structure induction from variable metadata.
2. Dual graph representation in output:
   - node-centric (parents per node)
   - edge-centric (directed edge list)
3. Mandatory post-generation validation:
   - structural consistency across both representations
   - DAG acyclicity check
4. Retry loop with bounded attempts.

How this maps to code:
- `PromptBNDAGBuilder` (`src/counter_bmt_v2/causal/promptbn.py`) follows one-shot LLM query + validation + bounded retries.
- It enforces:
  - only known node IDs
  - node-parent / edge consistency
  - acyclicity via topological sort check
- It also asks for CPTs in the same one-shot response, then normalizes rows.

## 2) Adv-BMT principles applied to NNX trajectory implementation

Paper constraints used as design targets:
1. **Bidirectional tokenization** over acceleration and yaw-rate control space.
2. Shared token grid for forward and reverse prediction.
3. Quantization ranges:
   - acceleration in `[-10, 10]` m/s²
   - yaw rate in `[-pi/2, pi/2]` rad/s
   - `K=33` bins per axis (=> 1089 tokens)
4. Midpoint integration with `dt=0.5s` for trajectory reconstruction.
5. Decoder architecture intent:
   - GPT-style autoregressive decoding
   - relation-aware components (agent-to-agent, agent-to-time, agent-to-scene)
6. Training objective intent:
   - cross-entropy over discrete motion tokens
   - top-p sampling at inference for diversity.

How this maps to code starter:
- `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py` will keep these as first-class configs and tokenizer behavior.
- Initial scaffolding focuses on tokenization/reconstruction correctness first; full relation-attention blocks are staged next.

## 3) Implementation sequencing (principled + fast)

1. VLM extraction contract + real GPT-4o adapter.
2. PromptBN-style DAG induction with strict validation.
3. NNX BMT tokenizer + reversible dynamics implementation.
4. NNX scene encoder/decoder blocks mirroring Adv-BMT components.
5. GRPO training loop integration on top of these modules.

