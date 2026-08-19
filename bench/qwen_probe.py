"""Structural probe: what does jdgram already cover on Qwen3.5-0.8B, and what is missing?

Step one of the port. Answers, from the real model rather than from the model card:

1. Which parameterized modules exist, and does :func:`registry.dispatch` resolve each?
2. Where does :func:`registry.collect_hookable_modules` stop early?  It returns as soon
   as a module owns direct parameters, so a block holding bare ``nn.Parameter``\\ s
   collapses to one opaque unit and its children are never reached.
3. Which parameters end up owned by no hookable module at all -- those contribute
   Gramian terms that would be silently missing rather than loudly refused.
4. Does a real forward + Gramian run under ``driver="loop"``?

Nothing here trains, and ``--structural-only`` needs no weights and no GPU.

    python bench/qwen_probe.py --structural-only
    python bench/qwen_probe.py --model Qwen/Qwen3.5-0.8B --m 2 --T 128
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import torch
from torch import nn

from jdgram.engine import registry


def _fmt(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000.0
    return f"{n:.1f}T"


def walk(model: nn.Module) -> None:
    """Every module owning direct parameters, and whether a handler resolves."""
    rows, unresolved = [], Counter()
    for name, mod in model.named_modules():
        direct = [p for p in mod.parameters(recurse=False) if p.requires_grad]
        if not direct:
            continue
        n_params = sum(p.numel() for p in direct)
        try:
            registry.dispatch(mod)
            ok, why = True, ""
        except KeyError as exc:
            ok, why = False, str(exc).split(".")[0]
            unresolved[type(mod).__name__] += n_params
        rows.append((name, type(mod).__name__, n_params, ok, why))

    print(f"\n{'='*100}\nPARAMETERIZED MODULES ({len(rows)} own direct params)\n{'='*100}")
    print(f"{'handler':>8}  {'params':>9}  {'type':<34} example")
    seen: dict[tuple[str, bool], list] = {}
    for name, tname, n, ok, _ in rows:
        seen.setdefault((tname, ok), [0, 0, name])
        seen[(tname, ok)][0] += 1
        seen[(tname, ok)][1] += n
    for (tname, ok), (count, tot, example) in sorted(
        seen.items(), key=lambda kv: -kv[1][1]
    ):
        mark = "  OK  " if ok else " MISS "
        print(f"{mark:>8}  {_fmt(tot):>9}  {tname:<34} x{count:<4} e.g. {example}")

    if unresolved:
        print(f"\n{'-'*100}\nNO IDENTITY REGISTERED -- these need a handler:")
        for tname, n in unresolved.most_common():
            print(f"    {tname:<40} {_fmt(n):>10} params")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    missed = sum(unresolved.values())
    print(f"\n  covered {_fmt(total - missed)} / {_fmt(total)} trainable params "
          f"({100.0 * (total - missed) / max(total, 1):.1f}%)")


def collapse(model: nn.Module) -> list[str]:
    """What the walk reaches, and what would have to be excluded to run at all.

    Returns the qualified names whose direct parameters have no identity -- the
    set a partial run would have to omit deliberately.
    """
    print(f"\n{'='*100}\nHOOKABLE-MODULE COLLECTION\n{'='*100}")
    try:
        collected = registry.collect_hookable_modules(model)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  collect_hookable_modules RAISED: {type(exc).__name__}: {exc}")
        return []

    # Deduplicate on tensor identity: tied weights (GPT-2's wte/lm_head, Qwen's
    # embed_tokens/lm_head) appear as a direct parameter of two modules, and
    # summing per-module double-counts them past 100%.
    def _numel(mods) -> int:
        seen: dict[int, int] = {}
        for mod in mods:
            for p in mod.parameters(recurse=False):
                if p.requires_grad:
                    seen[id(p)] = p.numel()
        return sum(seen.values())

    reached = _numel(collected.values())
    total = _numel([m for _, m in model.named_modules()])
    print(f"  collected {len(collected)} module(s), {_fmt(reached)} of {_fmt(total)} "
          f"direct params reached ({100.0*reached/max(total,1):.1f}%)")

    unhandled = registry.unhandled_direct_params(model)
    if not unhandled:
        print("  every collected module dispatches -- nothing to exclude")
        return []

    by_type: dict[str, list] = {}
    for name, n in unhandled.items():
        by_type.setdefault(type(model.get_submodule(name)).__name__, [0, 0, name])
        e = by_type[type(model.get_submodule(name)).__name__]
        e[0] += 1
        e[1] += n
    print(f"\n  {len(unhandled)} module(s) own direct params with NO identity."
          f"  Each is a hard KeyError until handled:\n")
    print(f"    {'type':<32} {'count':>6} {'params':>10}  example")
    for tname, (count, tot, example) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        print(f"    {tname:<32} {count:>6} {_fmt(tot):>10}  {example}")
    print(f"\n  total unhandled: {_fmt(sum(unhandled.values()))} of {_fmt(total)} "
          f"({100.0*sum(unhandled.values())/max(total,1):.3f}%)")
    return sorted(unhandled)


def live(model, tok, m: int, T: int, driver: str, device: str,
         exclude: list[str] | None = None) -> None:
    """A real forward + Gramian, to see which guard fires first."""
    from jdgram.engine.hooks import compute_gramian
    from jdgram.identities.tied import tied_gramian

    print(f"\n{'='*100}\nLIVE GRAMIAN  (m={m}, T={T}, driver={driver!r})\n{'='*100}")
    V = model.config.vocab_size if hasattr(model.config, "vocab_size") else 1000
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, min(V, 30000), (m, T), generator=g).to(device)

    def losses_fn():
        out = model(input_ids=idx)
        logits = out.logits if hasattr(out, "logits") else out[0]
        per_tok = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            idx[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(m, T - 1)
        return per_tok.mean(dim=1)

    modules = registry.collect_hookable_modules(model, exclude=set(exclude or []))

    # Tied parameters are only additive across modules when the cross terms are
    # supplied via shared_handlers. Summing per-module Gramians over a tied pair
    # drops them -- the exact defect jdgram measures autogram committing on
    # GPT-2. Detect and say so rather than print a confident wrong number.
    owners: dict[int, list[str]] = {}
    for nm, mod in modules.items():
        for p in mod.parameters(recurse=False):
            if p.requires_grad:
                owners.setdefault(id(p), []).append(nm)
    tied = {k: v for k, v in owners.items() if len(v) > 1}
    shared: dict = {}
    for names in tied.values():
        head = [n for n in names if isinstance(modules[n], nn.Linear)]
        emb = [n for n in names if isinstance(modules[n], nn.Embedding)]
        if len(names) == 2 and len(head) == 1 and len(emb) == 1:
            h, e = head[0], emb[0]
            print(f"  TIED: {e} == {h}  -> four-term identity (II.4)")
            shared[frozenset(names)] = (
                lambda caps, h=h, e=e: tied_gramian(
                    caps[h].A, caps[h].X, caps[e].A, caps[e].X
                )
            )
        else:
            print(f"  TIED group {sorted(names)} is not a (Linear, Embedding) pair --")
            print("  no handler wired, so its cross terms would be MISSING.")

    if exclude:
        print(f"  excluding {len(exclude)} module(s) with no identity --")
        print("  this Gramian is PARTIAL, not exact. Diagnostic only.")
    try:
        res = compute_gramian(model, losses_fn, modules=modules,
                              shared_handlers=shared or None, driver=driver)
        G = res.total if hasattr(res, "total") else res
        print(f"  SUCCESS  G={tuple(G.shape)} dtype={G.dtype}")
        d = G.diagonal().clamp_min(1e-30).sqrt()
        cos = G / d.unsqueeze(0) / d.unsqueeze(1)
        print(f"  diagonal      {[f'{v:.4e}' for v in G.diagonal().tolist()]}")
        print(f"  off-diag cos  {[f'{v:+.4f}' for v in cos[~torch.eye(m, dtype=bool)].tolist()]}")
        print(f"  peak memory   {torch.cuda.max_memory_allocated()/2**20:.0f} MiB"
              if device.startswith("cuda") else "")
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        print(f"  {type(exc).__name__}: {exc}\n")
        tb = traceback.format_exc().splitlines()
        # The frames inside jdgram are the informative ones; transformers'
        # internals just add noise.
        keep = [ln for ln in tb if "jdgram" in ln or "qwen_probe" in ln
                or ln.strip().startswith(("File", "raise", "assert"))]
        print("\n".join(f"  {ln}" for ln in keep[-14:]))
        print("\n  ^ this is the next thing to fix.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--driver", default="loop", choices=("loop", "squashed", "batched"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--allow-partial", action="store_true",
                   help="exclude modules with no identity and run anyway; the "
                        "resulting Gramian is PARTIAL, for diagnosis only")
    p.add_argument("--structural-only", action="store_true",
                   help="config only: no weights, no GPU, no forward")
    args = p.parse_args()

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        print("transformers not installed in this env", file=sys.stderr)
        return 2
    print(f"transformers {transformers.__version__}   torch {torch.__version__}")

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    print(f"model_type={getattr(cfg, 'model_type', '?')}  "
          f"layers={getattr(text_cfg, 'num_hidden_layers', '?')}  "
          f"hidden={getattr(text_cfg, 'hidden_size', '?')}  "
          f"vocab={getattr(text_cfg, 'vocab_size', '?')}  "
          f"tied={getattr(cfg, 'tie_word_embeddings', '?')}")
    lt = getattr(text_cfg, "layer_types", None)
    if lt:
        print(f"layer_types: {dict(Counter(lt))}")

    if args.structural_only:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        walk(model)
        collapse(model)
        print("\nstructural pass only -- rerun without --structural-only for a live Gramian")
        return 0

    # transformers 5.x takes `dtype`; 4.x spells the same thing `torch_dtype`.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, trust_remote_code=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True
        )
    model = model.to(args.device)
    model.train()
    walk(model)
    skip = collapse(model)
    live(model, None, args.m, args.T, args.driver, args.device,
         exclude=skip if args.allow_partial else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
