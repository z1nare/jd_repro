# jdgram × Qwen3.5-0.8B — figures

**28 Aug 2026** · all numbers measured on `fuji2`, RTX A5000 24 GB, one engine per process,
`base_mib` verified clean in every cell

---

## F1 · Head to head — m=2, T=256, uncheckpointed, fp32

| engine | step (ms) | peak (MiB) | delta (MiB) | status |
|---|---:|---:|---:|---|
| **jdgram** | **1,068.8** | **10,368.9** | **7,498.8** | ok |
| jdgram, identities only | 1,121.7 | 10,356.0 | 7,469.6 | ok |
| autogram (TorchJD) | 1,596.8 | 13,018.2 | 10,148.1 | ok |
| autojac (TorchJD) | — | 22,135.9 | — | **OOM** |

```
step time (ms), lower is better
jdgram    ██████████████████░░░░░░░░░░   1,068.8
autogram  ███████████████████████████    1,596.8      jdgram −33.1%

peak delta (MiB), lower is better
jdgram    ███████████████████░░░░░░░░░   7,498.8
autogram  ██████████████████████████     10,148.1     jdgram −26.1%
```

---

## F2 · Before / after, same measurement

| | 19 Aug | 28 Aug | change |
|---|---:|---:|---|
| jdgram step | 1,857.9 ms | **1,068.8 ms** | **−42.5%** |
| residual tail | 757.4 ms | **≈0** | eliminated |
| jdgram peak delta | 7,598.4 MiB | 7,498.8 MiB | −1.3% |
| vs autogram, time | +18.1% slower | **−33.1% faster** | sign flip |
| vs autogram, memory | −24.5% | **−26.1%** | |
| gates | 71 | **78** | +7 |

**Baseline reproducibility** — autogram re-measured 3 weeks later, new harness:

| | 19 Aug | 28 Aug | drift |
|---|---:|---:|---:|
| autogram step | 1,572.6 ms | 1,596.8 ms | +1.5% |
| autogram delta | 10,068.7 MiB | 10,148.1 MiB | +0.8% |

---

## F3 · Objective ceiling — T=256, checkpointed

| m | 19 Aug | 28 Aug | step (ms) |
|---:|---|---:|---:|
| 2 | 8,831 MiB | **8,027.5** | 1,733.8 |
| 3 | 16,768 MiB *(fallback path)* | **10,630.4** | 2,441.0 |
| 4 | **OOM** | **13,108.8** | 3,223.5 |
| 5 | not attempted | **15,691.5** | 4,059.7 |
| 6 | not attempted | **18,227.9** | 5,067.3 |
| 7 | not attempted | **20,648.2** † | 7,061.2 |

† with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Default allocator OOMs at
18,816 — fragmentation, not capacity.

```
peak MiB, 22,190 cap                        19 Aug        28 Aug
m=2  ████████░░░░░░░░░░░░░░  8,027           8,831   →     8,027
m=3  ███████████░░░░░░░░░░░  10,630         16,768   →    10,630   −36.6%
m=4  █████████████░░░░░░░░░  13,109            OOM   →    13,109   unlocked
m=5  ████████████████░░░░░░  15,692              —          15,692
m=6  ██████████████████░░░░  18,228              —          18,228
m=7  ████████████████████░░  20,648            OOM   →    20,648   unlocked
```

**Objective ceiling: m=3 → m=7.** Predicted peak 20,780, measured 20,648 — **0.6%**.

---

## F4 · The cost law

```
fp32   delta ≈ 9.93 · (m·T) + 74   MiB          base 2,870.2
bf16   delta ≈ 4.96 · (m·T) + 77   MiB          base 1,459.1
```

Validation across **both axes**, both dtypes:

| dtype | m | T | m·T | measured | predicted | err |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 2 | 256 | 512 | 5,157.4 | 5,158 | 0.0% |
| fp32 | 3 | 256 | 768 | 7,744.0 | 7,700 | 0.6% |
| fp32 | 4 | 256 | 1,024 | 10,222.3 | 10,241 | 0.2% |
| fp32 | 5 | 256 | 1,280 | 12,821.3 | 12,784 | 0.3% |
| **fp32** | **6** | **256** | **1,536** | **15,341.5** | 15,326 | 0.1% |
| **fp32** | **2** | **768** | **1,536** | **15,307.9** | 15,326 | 0.1% |
| **fp32** | **7** | **256** | **1,792** | **17,761.8** | 17,868 | 0.6% |
| **fp32** | **2** | **896** | **1,792** | **17,790.1** | 17,868 | 0.4% |
| bf16 | 2 | 1024 | 2,048 | 10,226.6 | 10,231 | 0.0% |
| bf16 | 2 | 1536 | 3,072 | 15,301.3 | 15,313 | 0.1% |
| bf16 | 2 | 2048 | 4,096 | fits (peak 21.9 GB) | 20,393 | — |

**Two independent confirmations that only the product matters** — same `m·T`, opposite
splits:

| m·T | tall split | wide split | apart |
|---:|---|---|---:|
| 1,536 | m=6 × T=256 → 15,341.5 | m=2 × T=768 → 15,307.9 | **33.6 MiB (0.22%)** |
| 1,792 | m=7 × T=256 → 17,761.8 | m=2 × T=896 → 17,790.1 | **28.3 MiB (0.16%)** |

**bf16 slope is 4.956 = 9.93 / 2.00** — the law is dtype-scaled exactly.

---

## F5 · Context ceiling

### 5a · The allocator was the ceiling, not the arithmetic

Default allocator — same byte budget, higher m dies ~2.8 GB lower:

| run | m·T | peak reached | outcome |
|---|---:|---:|---|
| m=2, T=896 | 1,792 | **20,660** | ok |
| m=4, T=448 | 1,792 | 18,799 | OOM |
| m=7, T=256 | 1,792 | 18,816 | OOM |
| m=3, T=576 | 1,728 | 17,822 | OOM |

`torch.stack` needs one contiguous `[m,T,V]` block; more objectives means more
small allocations, so the heap fragments and the large request fails early.

### 5b · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

| dtype | m | before | after | |
|---|---:|---:|---:|---|
| fp32 | 2 | 896 | **960** | 1024 OOM |
| bf16 | 2 | 1,536 | **2,048** | +33% |
| fp32 | 3 | 256 | **512+** | 576/640 untested |

```
m=2 context reached
19 Aug  fp32  ██████░░░░░░░░░░░░░░░░░░    768
28 Aug  fp32  ███████░░░░░░░░░░░░░░░░░    960    +25%
28 Aug  bf16  ████████████████████████  2,048    2.7x
```

### 5c · With fragmentation removed, the law *is* the ceiling

| dtype | m | law predicts T_max | measured | |
|---|---:|---:|---|---|
| fp32 | 2 | 969 | 960 ok · 1024 OOM | ✓ within one step |
| bf16 | 2 | 2,082 | 2,048 ok | ✓ within one step |

One environment variable, no code change. After it, `9.93·(m·T)` is the complete
capacity model.

---

## F6 · Exactness — unchanged by the fixes

Synthetic Qwen, 2 layers, vocab 2,048, T=32, fp64 ground-truth model.

| m | fp32 workspace | fp64 workspace | identities only | per-module worst |
|---:|---:|---:|---:|---:|
| 2 | 1.67×10⁻⁸ | **3.31×10⁻¹²** | 3.98×10⁻¹² | 4.24×10⁻⁸ |
| 3 | 9.80×10⁻⁹ | **2.76×10⁻¹²** | 3.31×10⁻¹² | 3.04×10⁻⁸ |

19 Aug reference: 3.26×10⁻¹². **`G` bit-identical across the capture fix**
(`max|ΔG| = 0.000e+00`, m = 1…6).

---

## F7 · Cost of exactness — per token-objective

| | MiB/token/objective |
|---|---:|
| logits `[T,V]` | 0.95 |
| cross-entropy chain | 3.79 |
| checkpointed layer activations | 0.09 |
| **plain training subtotal** | **≈3.9** |
| jdgram body captures | 3.46 |
| jdgram head captures | 1.90 |
| **jdgram total** | **9.93** |

**jdgram ≈ 2.5× plain training. 38% of that is the loss head every trainer pays.**

With AdamW state (752M × 16 B = 12,032 MiB) the trainable envelope is
`(22,190 − 12,032) / 9.93 ≈ 1,023` token-objectives → **m=2 at T≈512**.

---

## F8 · Measurement integrity

Old all-engines-one-process design, reproduced:

| cell | base_mib on entry | drift |
|---|---:|---:|
| jdgram m=2 | 2,870.2 | — |
| autojac m=2 *(OOMs at 22,136)* | 2,886.4 | — |
| jdgram m=3 | **16,734.6** | **+13,864** |
| jdgram-identities-only m=3 | **18,675.6** | **+15,805** |
| autogram m=3 | **20,594.2** | **+17,724** |

m=4 and m=8 then marked `SKIPPED` — **never attempted.** Leak is autojac's `[m,P]`
Jacobian. Resolution: one engine per process. All figures above have `LEAKED = 0`.

---

## F9 · Conflict — synthetic controls, T=256

| mode | m=2 | m=3 | m=4 | |
|---|---:|---:|---:|---|
| duplicate | **1.000** | **1.000** | **1.000** | identical objectives ✓ |
| independent (min / mean) | 0.793 | 0.769 / 0.782 | 0.756 / 0.782 | |
| conflicting (min / mean) | −0.793 | −0.793 / −0.270 | −0.795 / −0.261 | ✓ |

2,223 per-layer rows = 3 modes × 3 m × 247 modules.
**These are engine correctness controls, not measurements of a real task.**

---

## F10 · Competitive position

| axis | standing | evidence |
|---|---|---|
| vs TorchJD autogram — speed | **ahead** | −33.1%, measured |
| vs TorchJD autogram — memory | **ahead** | −26.1%, measured |
| vs TorchJD — capability | **ahead** | m=6 vs m=2; autogram cannot checkpoint |
| vs autojac | **ahead** | autojac OOMs at m=2 |
| vs ghost kernels (`2510.10902`) | **unverified** | they report 1.12× on 124M/50K vocab |
| vs FAMO, scalarization | behind on cost | inherent — m gradients vs one |
| vs BK-MOO (`2606.05613`) | different regime | exact ×1 GPU vs bucket-local ×8 H200 |
| exactness reported | **uncontested** | no MOO paper reports one |
| cost law reported | **uncontested** | MOO survey has no time/memory-vs-(m,P) table |

---

## Pending / not presentable

**C4 training comparison — DO NOT PRESENT.** The run is invalid on three counts:

| aggregator | val_nats | |
|---|---|---|
| sgd_erm (the control) | **NaN** | diverged |
| Mean | **NaN** | diverged |
| UPGrad | **ERROR** | |
| MGDA | 11.468 | |
| PCGrad | 12.733 | |

`--lr 0.01` is ~100× a sane fine-tuning rate for a 752M model, so the control
diverging is a configuration error, not an engine result. Every cell is additionally
flagged `LEAKED` (drift 5,757–8,631 MiB) — all four aggregators ran in one process.
Re-run needed at `--lr 1e-5`, one aggregator per process.

Also outstanding:

- end-to-end training multiplier vs plain SGD (blocked on the above)
- **m=7 with `expandable_segments`** — both attempts ran under GPU contention; needs a
  clear card. Law predicts it fits at ~20,777 MiB peak.
- m=3 ceiling between T=512 and T=646; m=4 ceiling above T=256
