# Quantization scheme used in the QAT sweep

Methods note for the paper and for Giuseppe (hls4ml). Documents exactly what
fixed-point format the `(w_bits, a_bits)` sweep used, since the integer/fractional
split is a real FPGA design decision, not just a bit count.

Source: `QATfinal_notebook.ipynb` (`build_qdense_cnn` / `build_qdense_rcnn`); identical
in `train_quantized_qdense.ipynb` and `train_quantized.ipynb`.

## What was passed

```python
w_q = quantized_bits(w_bits, 1)              # kernel_quantizer and bias_quantizer
a_q = quantized_relu(a_bits)                 # QActivation between layers
```

- Weights: `integer=1` was set **explicitly** (not a default).
- Activations: `quantized_relu` with only the bit count → `integer=0` (its default).
- `w_bits = 32` is **not quantized**: that branch falls back to plain float32 `Dense`,
  serving as the in-sweep full-precision baseline.

## The sign-bit subtlety

QKeras `quantized_bits(bits, integer, keep_negative=True)` defaults `keep_negative=True`,
and the `integer` argument counts integer bits **above the binary point, excluding the
sign**. So the real layout is:

```
total bits = 1 (sign) + integer + fractional
fractional = bits - integer - 1
```

`quantized_bits(4, 1)` is therefore **1 sign + 1 integer + 2 fractional**, NOT
1 integer + 3 fractional. (Naive readings that ignore `keep_negative` get this wrong
by one bit.) Verified empirically from the quantized grid's step size.

## Weight quantizer — measured grid (signed)

| call | sign | integer | fractional | step | range |
|---|---|---|---|---|---|
| `quantized_bits(2, 1)` | 1 | 1 | 0 | 1.0   | [-2, +1]      |
| `quantized_bits(4, 1)` | 1 | 1 | 2 | 0.25  | [-2, +1.75]   |
| `quantized_bits(8, 1)` | 1 | 1 | 6 | 2^-6  | [-2, +1.984]  |
| `w=32`                 | — | — | — | float32 (unquantized) | — |

At 2-bit, weights have **zero fractional bits** — literally integers {-2,-1,0,1}.

## Activation quantizer — measured grid (unsigned, ReLU >= 0)

| call | integer | fractional | step | range |
|---|---|---|---|---|
| `quantized_relu(2)` | 0 | 2 | 0.25   | [0, 0.75]   |
| `quantized_relu(4)` | 0 | 4 | 0.0625 | [0, 0.9375] |
| `quantized_relu(8)` | 0 | 8 | 2^-8   | [0, 0.996]  |

## hls4ml / ap_fixed equivalent (Giuseppe's language)

ap_fixed's integer field **includes** the sign bit, so it is `integer + 1`:

- Weights:    `quantized_bits(B, 1, keep_negative=1)`  ↔  `ap_fixed<B, 2>`   (range ≈ [-2, +2))
- Activations:`quantized_relu(A)` (unsigned, integer=0) ↔  `ap_ufixed<A, 0>` (range [0, 1))

## Caveat to state explicitly in the paper

The integer/fractional split was **fixed across the entire sweep** — weights pinned to
1 integer bit, activations to 0 — and only the total width `B`/`A` was swept. So:

- Weights span ≈[-2, +2) at **every** width; only resolution changes. Any trained weight
  beyond ±2 saturates.
- Activations are capped at **[0, 1)** at every width; values >= 1 saturate.

The `(w_bits, a_bits)` Pareto results therefore hold **under this specific fixed-point
assumption**, not a profiled one. For an actual FPGA deployment the integer bits should be
chosen from the measured weight/activation ranges (hls4ml's profiling tools do this) rather
than fixed at 1/0. Naming this pre-empts the critique and is a clean "deployment / future
work" line.
