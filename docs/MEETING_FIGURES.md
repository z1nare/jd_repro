# jdgram × Qwen3.5-0.8B — figures

**28 Aug 2026** · measured on `fuji2`, RTX A5000 24 GB, one engine per process,
`base_mib` verified clean and `LEAKED = 0` in every cell below

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

## F2 · Before / after

| | 19 Aug | 28 Aug | change |
|---|---:|---:|---|
| jdgram step | 1,857.9 ms | **1,068.8 ms** | **−42.5%** |
| residual tail | 757.4 ms | **≈0** | eliminated |
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

† with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The default allocator OOMs at
18,816 — fragmentation, not capacity.

```
peak MiB, 22,190 cap                 19 Aug          28 Aug
m=2  ████████░░░░░░░░░░░░░░           8,831    →     8,027
m=3  ███████████░░░░░░░░░░░          16,768    →    10,630    −36.6%
m=4  █████████████░░░░░░░░░             OOM    →    13,109    unlocked
m=5  ████████████████░░░░░░               —          15,692
m=6  ██████████████████░░░░               —          18,228
m=7  ████████████████████░░             OOM    →    20,648 †  unlocked
```

**⭐ Objective ceiling: m=3 → m=7.** The law predicted 20,780 MiB; measured 20,648 — **0.6%**.

---

## F4 · The cost law

```
fp32   delta ≈ 9.93 · (m·T) + 74   MiB          base 2,870.2
bf16   delta ≈ 4.96 · (m·T) + 77   MiB          base 1,459.1
```

| dtype | m | T | m·T | measured | predicted | err |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 2 | 256 | 512 | 5,157.4 | 5,158 | 0.0% |
| fp32 | 3 | 256 | 768 | 7,744.0 | 7,700 | 0.6% |
| fp32 | 4 | 256 | 1,024 | 10,222.3 | 10,241 | 0.2% |
| fp32 | 5 | 256 | 1,280 | 12,821.3 | 12,784 | 0.3% |
| **fp32** | **6** | **256** | **1,536** | **15,341.5** | 15,326 | 0.1% |
| **fp32** | **2** | **768** | **1,536** | **15,307.9** | 15,326 | 0.1% |
| fp32 | 2 | 896 | 1,792 | 17,790.1 | 17,868 | 0.4% |
| bf16 | 2 | 1024 | 2,048 | 10,226.6 | 10,236 | 0.1% |
| bf16 | 2 | 1536 | 3,072 | 15,301.3 | 15,314 | 0.1% |

**Rows 5 and 6:** same `m·T`, opposite splits — 6 objectives × 256 tokens versus
2 × 768. **15,341.5 against 15,307.9: 33 MiB apart.** Memory depends on the product,
not the split.

**bf16 slope 4.956 = 9.93 / 2.00** — the law is dtype-scaled exactly.

---

## F5 · Context ceiling

### 5a · The allocator was the ceiling, not the arithmetic

Default allocator, same byte budget — higher m dies ~2.8 GB lower:

| run | m·T | peak reached | outcome |
|---|---:|---:|---|
| m=2, T=896 | 1,792 | **20,660** | ok |
| m=4, T=448 | 1,792 | 18,799 | OOM |
| m=3, T=576 | 1,728 | 17,822 | OOM |

`torch.stack` needs one contiguous `[m,T,V]` block; more objectives means more small
allocations, so the heap fragments and the large request fails early.

### 5b · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

| dtype | m | before | after | |
|---|---:|---:|---:|---|
| fp32 | 2 | 896 | **960** | 1,024 OOM |
| bf16 | 2 | 1,536 | **2,048** | +33% |
| fp32 | 3 | 256 | **512+** | 576 / 640 untested |

```
m=2 context reached
19 Aug  fp32  ██████░░░░░░░░░░░░░░░░░░    768
28 Aug  fp32  ███████░░░░░░░░░░░░░░░░░    960    +25%
28 Aug  bf16  ████████████████████████  2,048    2.7x
```

### 5c · With fragmentation removed, the law *is* the ceiling

| dtype | m | law predicts T_max | measured | |
|---|---:|---:|---|---|
| fp32 | 2 | 969 | 960 ok · 1,024 OOM | ✓ within one step |
| bf16 | 2 | 2,082 | 2,048 ok | ✓ within one step |

One environment variable, no code change.

---

## F6 · Exactness

Synthetic Qwen, 2 layers, vocab 2,048, T=32, fp64 ground-truth model.

| m | fp32 workspace | fp64 workspace | identities only | per-module worst |
|---:|---:|---:|---:|---:|
| 2 | 1.67×10⁻⁸ | **3.31×10⁻¹²** | 3.98×10⁻¹² | 4.24×10⁻⁸ |
| 3 | 9.80×10⁻⁹ | **2.76×10⁻¹²** | 3.31×10⁻¹² | 3.04×10⁻⁸ |

**Like-for-like against 19 Aug** — mean over all 22 `rel_err` checks:

| | 19 Aug | 28 Aug |
|---|---:|---:|
| mean, all checks | ~1.2×10⁻⁸ | **1.24×10⁻⁸** |
| mean, fp64 workspace only | 3.26×10⁻¹² | **3.04×10⁻¹²** |

The all-checks mean is dominated by the ten fp32-workspace per-module blocks
(mean 2.45×10⁻⁸), so it measures **workspace precision**, not the method. The
method's own floor is the fp64 row at ~3×10⁻¹². **Neither moved.**

Two further checks:
- **`G` bit-identical across the capture fix** — `max|ΔG| = 0.000e+00`, m = 1…6
- **Duplicate objectives give cosine exactly 1.000** at m = 2, 3, 4 — independent
  confirmation the off-diagonals are computed correctly, not just plausibly
- **78 gates pass** on two different machines (A5000 and a laptop 5070 Ti)

---

## F7 · Cost of exactness — per token × objective

| | MiB | note |
|---|---:|---|
| cross-entropy chain | 3.79 | 4 vocab-width tensors × 0.947 each |
| checkpointed layer activations | 0.09 | |
| **plain training subtotal** | **3.88** | |
| jdgram body captures | 3.46 | `2·Σd_out·4` |
| jdgram head captures | 1.89 | `2·V·4` |
| workspaces, kernels, remainder | 0.70 | |
| **jdgram total (measured)** | **9.93** | |

**jdgram ≈ 2.6× plain training. 38% of jdgram's cost is the loss head every trainer pays.**

With AdamW state (752M × 16 B = 12,032 MiB) the trainable envelope is
`(22,190 − 12,032) / 9.93 ≈ 1,023` token-objectives → **m=2 at T ≈ 512**.

---

## F8 · Measurement integrity

The 19 Aug m=4 OOM was an artefact. Reproduced with the old all-engines-one-process design:

- autojac runs first at m=2, OOMs, and **does not release its `[m,P]` Jacobian**
- baseline climbs **2,870 → 20,594 MiB** across the next three cells
- every later cell starts on a near-full card, OOMs, and the harness's don't-retry
  rule marks m=4 and m=8 `SKIPPED` — **never attempted**

Fix: a guard that flags any cell whose baseline has drifted, plus one engine per
process. Every figure above has `LEAKED = 0`.

---

## F9 · Competitive position

| axis | standing | evidence |
|---|---|---|
| vs autogram — speed | **ahead** | −33.1%, measured |
| vs autogram — memory | **ahead** | −26.1%, measured |
| vs autogram — capability | **ahead** | m=7 vs m=2; autogram cannot checkpoint |
| vs autojac | **ahead** | autojac OOMs at m=2 |
| vs ghost kernels (`2510.10902`) | **same identity, tied-weight gap** | their §5.3 sums per-layer kernels — drops tied cross terms |
| vs Phantom Clipping (`2405.18194`) | **diagonal only** | App. B Claim B.1 has 3 terms (i=j); off-diagonal needs 4 |
| vs FAMO, scalarization | behind on cost | inherent — m gradients vs one |
| vs BK-MOO (`2606.05613`) | different trade-off | m GPUs × 1 backward vs 1 GPU × m backwards |
| exactness reported | **uncontested** | no MOO paper reports one |
| cost law reported | **uncontested** | MOO survey has no time/memory-vs-(m,P) table |

---

## Not yet measured

- end-to-end training multiplier vs plain SGD (the C4 run diverged at `--lr 0.01`; needs
  a re-run at `1e-5`, one aggregator per process)
- m=8 — the law puts it at ~23,240 MiB against a 22,190 cap, so it should *not* fit
- m=3 ceiling between T=512 and T=646; m=4 ceiling above T=256
