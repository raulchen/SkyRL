# Batch Size Investigation Results

**Model**: Qwen3-4B-Instruct
**Hardware**: 8× 40GB GPUs (TP=8)
**Sequence Length**: 8192 tokens

---

## Training (Forward-Backward)

### Results

| Batch Size | Status | Peak Memory (GPU 0) |
|------------|--------|---------------------|
| 2 | PASS | 25.7 GiB |
| 64 | PASS | 25.7 GiB |
| 128 | PASS | 25.7 GiB |
| 256 | PASS | 25.7 GiB |
| 512 | PASS | 25.7 GiB |
| **1024** | **PASS** | **25.7 GiB** |

### Key Finding

**Memory is constant regardless of batch size!**

This is due to the `fori_loop` optimization with gradient checkpointing:
1. Only ONE layer's activations are in memory at any time
2. XLA compiles a single body function and reuses it for all 36 layers
3. cuDNN workspace is shared across layers (~770 MiB instead of 27.7 GiB)

### Maximum Training Batch Size

**batch=1024+ at seq=8192** (limited by throughput, not memory)

---

## Sampling (Inference)

### Results by Sequence Length (n=1)

| Max Tokens | Status | Notes |
|------------|--------|-------|
| 8192 | FAIL | OOM: 54.07 GiB allocation |
| 4096 | FAIL | OOM: 27.07 GiB allocation |
| 3840 | PASS | |
| 3072 | PASS | |
| 2048 | PASS | |

### Results by Batch Size (max_tokens=2048)

| n (samples) | Status |
|-------------|--------|
| 1 | PASS |
| 4 | PASS |
| 8 | PASS |

### Key Finding

**Sampling suffers from the same 36-closure cuDNN workspace issue.**

The inference path creates 36 different layer closures, causing XLA to allocate
36 separate cuDNN workspaces (~770 MiB each × 2 for 8k seq ≈ 54 GiB).

This is the same root cause we fixed in training with `fori_loop`. The inference
path would need a similar optimization to support 8k+ sequences.

### Maximum Sampling Configuration

- **n=1, max_tokens ≤ 3840**
- **n=8, max_tokens ≤ 2048**

---

## Root Cause: 36-Closure cuDNN Workspace Problem

Both training (before fix) and sampling suffer from the same issue:

```
Per-layer Python loop:
for layer in self.layers:       # 36 iterations
    output = layer(input)       # Each creates a different traced closure
```

XLA sees 36 different attention operations and allocates separate cuDNN
workspaces for each:
- Per workspace: ~770 MiB (scales with seq_len)
- Total: 36 × 770 MiB = **27.7 GiB** (for 4k seq) or **54 GiB** (for 8k seq)

### Training Fix (Implemented)

```python
# fori_loop: XLA compiles ONE body function
stacked_state = stack_all_layer_states()
jax.lax.fori_loop(0, 36, layer_body, carry)  # ONE workspace shared
```

### Sampling (Investigation Results)

We attempted to apply fori_loop to the inference path but encountered challenges:

1. **Prefill path**: fori_loop is now applied to prefill (no KV cache). However, this alone
   doesn't fix the OOM because the decode path is also compiled.

2. **Decode path**: Attempts to apply fori_loop to decode increased memory usage:
   - Standard loop: 54 GiB OOM
   - fori_loop with stacked KV cache: 83-137 GiB OOM
   - The additional memory comes from carrying large stacked KV cache through fori_loop

3. **Root cause**: The `_prefill_and_decode` JIT function compiles BOTH prefill and decode
   together. Even if prefill uses fori_loop (1 workspace), the decode loop creates 36
   closures that XLA allocates workspaces for.

4. **Why training works**: Training only compiles the forward pass with fori_loop. The
   backward pass uses gradient checkpointing to recompute, sharing the same workspace.
   Inference has no backward pass, and the decode loop adds separate operations.

**Potential fixes (not yet implemented):**
- Separate prefill and decode into different JIT functions
- Implement a more sophisticated stacked-parameter model architecture
- Use a different attention implementation with smaller workspace requirements

---

## Recommendations

### For Training
- Use batch sizes up to 1024+ at seq=8192 without memory concerns
- Memory is constant due to fori_loop optimization

### For Sampling
- Limit max_tokens to ~3840 for single sequences
- For parallel sampling (n>1), reduce max_tokens proportionally
- **Future work**: Apply fori_loop to inference path for 8k+ support

---

## Configuration

```json
{
  "tensor_parallel_size": 8,
  "shard_attention_heads": false,
  "gradient_checkpointing": true,
  "loss_chunk_size": 1024
}
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
TF_GPU_ALLOCATOR=cuda_malloc_async
```
