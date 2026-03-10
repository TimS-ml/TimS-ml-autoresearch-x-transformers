# x-transformers Adjustable Parameters

Comprehensive reference of all adjustable parameters for the `Decoder` and
`TransformerWrapper` classes used in `train.py`. Extracted from the
[x-transformers README](https://github.com/lucidrains/x-transformers).

**How to use this:** When editing `train.py`, you configure these parameters in
the `build_model()` function, in the `TransformerWrapper(...)` and
`Decoder(...)` constructors.

---

## Table of Contents

- [Core Architecture](#core-architecture)
- [Normalization](#normalization)
- [Feedforward Network](#feedforward-network)
- [Attention](#attention)
- [Positional Encoding](#positional-encoding)
- [Regularization & Dropout](#regularization--dropout)
- [Residual Connections](#residual-connections)
- [Memory & Recurrence](#memory--recurrence)
- [Layer Structure](#layer-structure)
- [TransformerWrapper-level Options](#transformerwrapper-level-options)
- [AutoregressiveWrapper Options](#autoregressivewrapper-options)
- [Recommended Combinations](#recommended-combinations)
- [Model Sizing Guide](#model-sizing-guide)

---

## Core Architecture

These are the fundamental parameters that determine model size and capacity.

| Parameter | Type | Default | Where | Description |
|-----------|------|---------|-------|-------------|
| `dim` | int | — | Decoder | Model hidden dimension. Primary knob for model capacity. |
| `depth` | int | — | Decoder | Number of transformer layers. More depth = more capacity but slower. |
| `heads` | int | 8 | Decoder | Number of attention heads. Usually `dim // 64` or `dim // 128`. |
| `num_tokens` | int | — | TransformerWrapper | Vocabulary size (256 for byte-level enwik8). |
| `max_seq_len` | int | — | TransformerWrapper | Maximum sequence length. |

---

## Normalization

Different normalization strategies affect training stability and convergence speed.

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `use_rmsnorm` | bool | False | RMSNorm instead of LayerNorm. Simpler, no mean centering. Found to be best variant. Used in Retro, Gopher, LLaMA. | [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) |
| `use_simple_rmsnorm` | bool | False | Even simpler: `l2norm(x) * sqrt(dim)` with no learned gamma. No performance loss per TransNormer paper. | [Qin et al. 2023](https://arxiv.org/abs/2307.14995) |
| `use_scalenorm` | bool | False | ScaleNorm — simpler alternative to LayerNorm. Faster convergence reported. | [Nguyen & Salazar 2019](https://arxiv.org/abs/1910.05895) |
| `sandwich_norm` | bool | False | Extra layernorm on branch outputs (pre-norm + post-norm on each sublayer). Stabilizes training. From CogView. | [Ding et al. 2021](https://arxiv.org/abs/2105.13290) |
| `resi_dual` | bool | False | Hybrid pre+post layernorm. Reduces representation collapse while maintaining stability. | [Microsoft 2023](https://arxiv.org/abs/2304.14802) |
| `resi_dual_scale` | float | 0.1 | Scale factor for prenorm residual in resi_dual (prevents fp16 overflow). | |
| `pre_norm` | bool | True | Use pre-layernorm (default). Set False for post-layernorm. | |
| `attn_head_scale` | bool | False | Per-head scaling after attention aggregation (Normformer). Slight convergence improvement. | [Normformer 2022](https://openreview.net/forum?id=GMYWzWztDx5) |
| `ff_post_act_ln` | bool | False | Extra layernorm after feedforward activation (Normformer). | [Normformer 2022](https://openreview.net/forum?id=GMYWzWztDx5) |

---

## Feedforward Network

The feedforward (MLP) block processes each position independently after attention.

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `ff_glu` | bool | False | **Gated Linear Unit** in feedforward. Usually helps. "You should always turn this on." | [Shazeer 2020](https://arxiv.org/abs/2002.05202) |
| `ff_swish` | bool | False | Use Swish activation. Combine with `ff_glu=True` for **SwiGLU** (used in PaLM, LLaMA). | [PaLM 2022](https://arxiv.org/abs/2204.02311) |
| `ff_relu_squared` | bool | False | ReLU^2 activation (from Primer NAS). Simpler and better than GELU in autoregressive setting. **Note:** if using GLU, GELU still better. | [So et al. 2021](https://arxiv.org/abs/2109.08668) |
| `ff_mult` | int | 4 | FFN expansion factor. Inner dim = `dim * ff_mult`. Try 2 (smaller/faster) or 8 (larger). With GLU, effective expansion is `ff_mult * 2/3`. | |
| `ff_no_bias` | bool | False | Remove bias from feedforward layers. Increases throughput, no accuracy loss. Trend started with PaLM. | [PaLM 2022](https://arxiv.org/abs/2204.02311) |
| `ff_dropout` | float | 0.0 | Dropout in feedforward sublayer. | |

---

## Attention

Parameters controlling the self-attention mechanism.

### Core Attention

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `attn_flash` | bool | False | Use PyTorch `scaled_dot_product_attention` (Flash Attention via SDP). Faster + less memory. **Always turn this on.** | [Dao et al. 2022](https://arxiv.org/abs/2205.14135) |
| `attn_qk_norm` | bool | False | L2-normalize queries and keys (cosine similarity attention). Prevents overflow, removes numerical stability issues. Proven at 22B scale by Google Brain. | [Henry et al. 2020](https://arxiv.org/abs/2010.04245), [Dehghani et al. 2023](https://arxiv.org/abs/2302.05442) |
| `attn_qk_norm_groups` | int | 1 | Number of groups for grouped QK normalization. Bounds similarity to `[-groups, groups]`. 8 or 16 recommended. | |
| `attn_qk_norm_scale` | float | None | Fixed scale for cosine sim attention (e.g. 10). Alternative to groups. | |
| `attn_qk_norm_dim_scale` | bool | False | Learned scale per feature dimension (as in Google Brain 22B paper). | |
| `attn_dropout` | float | 0.0 | Dropout on attention weights. | |

### Multi-Query / Grouped-Query Attention

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `attn_one_kv_head` | bool | False | Multi-Query Attention: single KV head, multi-headed queries. Memory-efficient for inference. | [Shazeer 2019](https://arxiv.org/abs/1911.02150) |
| `attn_kv_heads` | int | None | **Grouped-Query Attention (GQA)**: number of KV heads. E.g. `heads=8, attn_kv_heads=2` means 4 query heads share 1 KV head. Saves memory. | [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) |

### Persistent Memory KV

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `attn_num_mem_kv` | int | 0 | Number of learned persistent memory key/value pairs prepended to attention. "Keeping the feedforwards and adding memory key/values leads to even better performance." | [Sukhbaatar et al. 2019](https://arxiv.org/abs/1907.01470) |

### Sparse Attention

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `attn_sparse_topk` | int | None | Keep only top-k attention values before softmax. Paper recommends k=8. | [Zhao et al. 2019](https://arxiv.org/abs/1912.11637) |
| `attn_sparse_topk_straight_through` | bool | False | Straight-through gradients for sparse topk. | |
| `attn_hard` | bool | False | Extreme case: only propagate single argmax value. | |

### Attention Variants

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `attn_on_attn` | bool | False | Gate attention output with queries. Found to perform worse with concatenation, better without. | [Huang et al. 2019](https://arxiv.org/abs/1908.06954) |
| `attn_gate_values` | bool | False | Gate aggregated values with input (AlphaFold2-style). Small but noticeable improvement. | [AlphaFold2](https://github.com/deepmind/alphafold) |
| `attn_pre_talking_heads` | bool | False | Linear mixing across heads pre-softmax (Talking Heads). Extra memory/compute. | [Shazeer et al. 2020](https://arxiv.org/abs/2003.02436) |
| `attn_post_talking_heads` | bool | False | Linear mixing across heads post-softmax. | |
| `residual_attn` | bool | False | Residualize pre-attention scores across layers. Best with post-norm. Allows higher LR. | [He et al. 2020](https://arxiv.org/abs/2012.11747) |

---

## Positional Encoding

How the model understands token ordering.

| Parameter | Type | Default | Where | Description | Paper |
|-----------|------|---------|-------|-------------|-------|
| `rotary_pos_emb` | bool | False | Decoder | **RoPE (Rotary Positional Embeddings)**. Standard for modern transformers. Relative positions via rotations. "Highly recommend when working on ordered sequences." Used in PaLM, LLaMA, etc. | [Su et al. 2021](https://arxiv.org/abs/2104.09864) |
| `rotary_xpos` | bool | False | Decoder | Modified RoPE for length extrapolation (adds ALiBi-like decay). | [Sun et al. 2022](https://arxiv.org/abs/2212.10554) |
| `rotary_xpos_scale_base` | int | 512 | Decoder | Receptive field scale for rotary_xpos. | |
| `rel_pos_bias` | bool | False | Decoder | T5-style learned relative position bias added to attention matrix. Cheap relative positional encoding. | [Raffel et al. 2020](https://arxiv.org/abs/1910.10683) |
| `alibi_pos_bias` | bool | False | Decoder | ALiBi: static linear bias on attention. Length extrapolation. May hinder global attention. | [Press et al. 2021](https://ofir.io/train_short_test_long.pdf) |
| `alibi_num_heads` | int | heads | Decoder | Only apply ALiBi to this many heads (others can attend far distances). | |
| `dynamic_pos_bias` | bool | False | Decoder | Learned position bias that generalizes to longer sequences. First place in RNA folding competition. | [CrossFormer](https://arxiv.org/abs/2108.00154), [SwinV2](https://arxiv.org/abs/2111.09883) |
| `dynamic_pos_bias_log_distance` | bool | False | Decoder | Use log distances for dynamic pos bias (linear is better for language). | |
| `use_abs_pos_emb` | bool | True | TransformerWrapper | Absolute positional embeddings. Can be turned off when using RoPE/ALiBi. Causal models can learn positions implicitly. | [Haviv et al. 2022](https://arxiv.org/abs/2203.16634) |

---

## Regularization & Dropout

| Parameter | Type | Default | Where | Description |
|-----------|------|---------|-------|-------------|
| `layer_dropout` | float | 0.0 | Decoder | **Stochastic depth**: randomly drop entire layers during training. Needs careful tuning. |
| `attn_dropout` | float | 0.0 | Decoder | Dropout on attention weights post-softmax. |
| `ff_dropout` | float | 0.0 | Decoder | Dropout in feedforward block. |
| `emb_dropout` | float | 0.0 | TransformerWrapper | Dropout after embedding layer. |

---

## Residual Connections

How sublayer outputs are added back to the residual stream.

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `gate_residual` | bool | False | Gated residual connections. Shown to increase stability and performance in RL tasks. | [Parisotto et al. 2019](https://arxiv.org/abs/1910.06764) |
| `scale_residual` | bool | False | Learned residual scaling (Normformer). Slight improvements but occasional instability. | [Normformer 2022](https://openreview.net/forum?id=GMYWzWztDx5) |

---

## Memory & Recurrence

For extending context or adding persistent learned memory.

| Parameter | Type | Default | Where | Description | Paper |
|-----------|------|---------|-------|-------------|-------|
| `num_memory_tokens` | int | 0 | TransformerWrapper | Learned tokens (like CLS) passed through all layers alongside input. Also known as "register tokens" — alleviates attention outliers. | [Burtsev 2020](https://arxiv.org/abs/2006.11527), [Darcet et al. 2023](https://arxiv.org/abs/2309.16588) |
| `max_mem_len` | int | 0 | TransformerWrapper | Enable Transformer-XL recurrence with this memory length. Requires `rel_pos_bias=True` or `rotary_pos_emb=True`. | |
| `shift_mem_down` | int | 0 | TransformerWrapper | Enhanced recurrence: route memory of layer N to layer N-1 on next step. | [Ding et al. 2021](https://arxiv.org/abs/2012.15688) |

---

## Layer Structure

Control the arrangement and sharing of attention/feedforward blocks.

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `macaron` | bool | False | **Macaron configuration**: place attention between two half-step feedforward layers (FFN-Attn-FFN). Based on dynamical systems POV. Used in Conformer. | [Lu et al. 2019](https://arxiv.org/abs/1906.02762), [Conformer](https://arxiv.org/abs/2005.08100) |
| `sandwich_coef` | int | None | Sandwich layer reordering: blocks of attention followed by blocks of feedforward. Optimal at 6. | [Press et al. 2020](https://arxiv.org/abs/1911.03864) |
| `weight_tie_layers` | bool | False | Tie weights across all layers (ALBERT-style). Dramatically reduces parameters. | [Lan et al. 2019](https://arxiv.org/abs/1909.11942) |
| `custom_layers` | tuple | None | Custom layer sequence, e.g. `('a', 'f', 'a', 'f')`. | |
| `layers_execute_order` | tuple | None | Custom execution order of layers (0-indexed). Allows weight-sharing patterns. | |
| `shift_tokens` | int/tuple | 0 | Shift a subset of feature dimensions by 1 token. Helps convergence for **character-level** training. May not help with BPE + RoPE. | [PENG Bo 2021](https://zhuanlan.zhihu.com/p/191393788) |

---

## TransformerWrapper-level Options

These are set on the `TransformerWrapper` constructor (not `Decoder`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_tokens` | int | — | Vocabulary size. |
| `max_seq_len` | int | — | Maximum sequence length. |
| `use_abs_pos_emb` | bool | True | Whether to use absolute positional embeddings. |
| `l2norm_embed` | bool | False | L2-normalize embeddings + small init (fixnorm). Improves convergence. |
| `post_emb_norm` | bool | False | LayerNorm right after embeddings (BLOOM/YaLM-style). Use either this or `l2norm_embed`, not both. |
| `emb_dropout` | float | 0.0 | Dropout after embedding. |
| `num_memory_tokens` | int | 0 | Number of learned memory/register tokens. |
| `max_mem_len` | int | 0 | Transformer-XL memory length. |
| `shift_mem_down` | int | 0 | Enhanced recurrence shift. |

---

## AutoregressiveWrapper Options

These are set on the `AutoregressiveWrapper` wrapping the model.

| Parameter | Type | Default | Description | Paper |
|-----------|------|---------|-------------|-------|
| `mask_prob` | float | 0.0 | **Forgetful Causal Masking**: randomly mask tokens during autoregressive training (like combining MLM with AR). Paper uses 0.15. Significantly better zero-shot performance. | [Liu et al. 2022](https://arxiv.org/abs/2210.13432) |

---

## Recommended Combinations

Based on the x-transformers README and common practice in modern LLMs:

### Baseline (modern defaults)
```python
Decoder(
    dim=512, depth=6, heads=8,
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_relu_squared=True,
)
```

### SwiGLU (PaLM/LLaMA-style)
```python
Decoder(
    dim=512, depth=6, heads=8,
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_glu=True,        # enable gating
    ff_swish=True,       # Swish activation
    ff_no_bias=True,     # no bias (PaLM style)
)
```

### Memory-efficient (GQA)
```python
Decoder(
    dim=768, depth=8, heads=12,
    attn_kv_heads=4,     # grouped-query attention
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_glu=True,
    ff_swish=True,
)
```

### Kitchen sink (many features)
```python
Decoder(
    dim=640, depth=8, heads=10,
    rotary_pos_emb=True,
    attn_flash=True,
    attn_qk_norm=True,
    use_rmsnorm=True,
    ff_glu=True,
    ff_swish=True,
    macaron=True,            # sandwich FFN
    attn_num_mem_kv=16,      # persistent memory
    shift_tokens=1,          # good for char-level
    sandwich_norm=True,      # extra stability
)
```

---

## Model Sizing Guide

Phil Wang's guideline: **1:5 model-to-data ratio** (tokens seen = 5 x params).

For a **5-minute training budget**, token throughput depends on your GPU. Rough estimates:

| Params | dim | depth | heads | Tokens needed (5x) | Notes |
|--------|-----|-------|-------|--------------------:|-------|
| ~5M | 256 | 6 | 4 | 25M | Very small, fast iteration |
| ~19M | 512 | 6 | 8 | 95M | Good starting point for 6-8 GB VRAM |
| ~57M | 768 | 8 | 12 | 285M | Needs 12+ GB VRAM, fewer tokens in 5 min |
| ~64M | 640 | 12 | 10 | 320M | Deeper, narrower variant |

**Rule of thumb:** Larger models see fewer tokens in the same time. There's a sweet
spot between model capacity and training tokens that depends on your specific GPU.
Start small (19M params), establish a baseline, then try scaling up.

**VRAM usage** is primarily driven by `dim`, `depth`, `batch_size`, and `max_seq_len`.
Using `attn_kv_heads` (GQA) can significantly reduce attention memory for larger models.
