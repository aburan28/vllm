# Partial Weight Reload

Partial weight reload updates a subset of a running engine's parameters in
place — from a delta checkpoint or a name-filtered full checkpoint — without a
full model weight reload and without restarting the process. It builds on
[layerwise reloading](layerwise.md): copies land in the original tensor
storage, so CUDA graphs and compiled artifacts remain valid, and only the
modules actually touched by the update are re-processed (re-quantized,
repacked).

Use it when only part of the model changed: a fine-tuned `lm_head` or
embedding, a handful of transformer layers, a router, or an RL policy delta.
For whole-checkpoint refits, use a [full reload](layerwise.md) or a
[weight transfer engine](weight_transfer/README.md).

## How it works

1. **Select** — the update is described by `weight_name_prefixes`: only
   checkpoint tensors whose name starts with one of the prefixes are loaded.
   With a delta checkpoint (a safetensors file holding just the changed
   tensors), pass `[""]` to select everything in it.
2. **Resolve** — each selected checkpoint name is mapped through the model's
   `WeightsMapper` (so `q_proj` resolves to the fused `qkv_proj` module) and
   matched to its deepest enclosing module; that module's subtree becomes part
   of the *touched set*. A name that cannot be resolved fails the reload
   before the model is touched.
3. **Bracket** — only the touched set is bracketed with
   `initialize_layerwise_reload(model, modules=...)`. Untouched modules are
   never disturbed: no meta swap, no re-processing, no warnings.
4. **Load and commit** — `model.load_weights(...)` streams the selected
   tensors through the normal weight loaders (checkpoint-name mapping, TP
   sharding, and fused/packed params all work as at startup). When a layer
   has received all of its weights, it is processed
   (`process_weights_after_loading`) and the result is copied into the
   original tensor storage.
5. **Finalize** — `finalize_layerwise_reload(..., partial=True)` restores any
   bracketed-but-unloaded layers silently and *rolls back* layers that
   received only some of their weights (see below).

### Layers are the atomic unit

A bracketed layer either receives its complete checkpoint weight set and is
committed, or it is rolled back to its previous weights and reported in the
resulting error. Processing a half-loaded layer would commit partially
uninitialized data, so it is never done in partial mode.

Supply every checkpoint weight of a layer you update:

- all of `q_proj`/`k_proj`/`v_proj` for a fused QKV layer, and both of
  `gate_proj`/`up_proj` for a fused MLP layer,
- every expert tensor of a fused MoE layer (with expert parallelism, ranks
  holding none of the updated experts are unaffected),
- for a single-tensor layer like `lm_head` or a norm, just that tensor.

Tied weights (e.g. `lm_head.weight` tied to `model.embed_tokens.weight`) share
storage, so updating the canonical name updates both; supplying only the
alias fails loudly.

## Usage

Weights must not change while a forward pass reads them. In the online server
pause first, exactly as for [weight transfer](async_rl.md); offline, call
between `generate` calls. `reload_weights` resets the prefix cache after the
update, so KV computed with the old weights is never reused.

### Online (`VLLM_SERVER_DEV_MODE=1`)

```bash
curl -X POST localhost:8000/pause -d ''
curl -X POST localhost:8000/reload_weights \
    -H "Content-Type: application/json" \
    -d '{
        "weights_path": "/checkpoints/step_100_delta",
        "weight_name_prefixes": [""],
        "weight_version": "step-100"
    }'
curl -X POST localhost:8000/resume -d ''
```

Omitting `weight_name_prefixes` performs a full reload of `weights_path`.
`/pause` clears the prefix cache by default, which is required for
correctness after any weight change. Weight-version spans of in-flight
requests follow the [weight version](weight_transfer/README.md) mechanism.

### Offline

```python
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# update only the LM head and the last two decoder layers from a
# full checkpoint
llm.reload_weights(
    weights_path="/checkpoints/step_100",
    weight_name_prefixes=[
        "lm_head.",
        "model.layers.30.",
        "model.layers.31.",
    ],
)

# or apply a delta checkpoint containing only changed tensors
llm.reload_weights(
    weights_path="/checkpoints/step_100_delta",
    weight_name_prefixes=[""],
    weight_version="step-100",
)
```

### Producing a delta checkpoint

A delta checkpoint is an ordinary safetensors file/folder restricted to the
changed tensors, using checkpoint-format names and full (unsharded) tensors:

```python
import torch
from safetensors.torch import save_file

changed = {
    name: param.detach().cpu()
    for name, param in trainer_model.state_dict().items()
    if name.startswith(("lm_head.", "model.layers.31."))
}
save_file(changed, "/checkpoints/step_100_delta/model.safetensors")
```

## Guarantees and limitations

- **In-place**: graph-visible tensor storage is preserved; CUDA graphs are
  not re-captured.
- **Scoped**: untouched modules are not re-processed, and the engine's model
  path is not repointed at `weights_path` (unlike a full reload).
- **Failure semantics**: an unresolvable name fails before the model is
  touched. If loading fails midway, complete layers are committed and
  incomplete ones are rolled back — the model stays serveable, but the caller
  must retry or full-reload to reach the intended state.
- Quantized layers are supported through their normal
  `process_weights_after_loading`, but require high-precision
  checkpoint-format inputs for the touched layers, like a full reload.
- Only loaders exposing `get_all_weights` (the default loader) support
  reloading from disk; any iterator of `(name, tensor)` works via the Python
  API.
- The layer-completeness accounting follows the
  [layerwise limitations](layerwise.md). Padding a loader writes itself (e.g.
  zeroed vocab padding) counts toward completeness; a quant scheme that pads
  weights without writing the padding may report a complete layer as
  incomplete and roll it back (the reload then fails loudly rather than
  corrupting).
