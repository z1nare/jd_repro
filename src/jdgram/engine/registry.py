"""Module type -> Gramian identity dispatch.

The table that makes the nanochat port "re-registration, not new math": RMSNorm
resolves to II.5, GQA/Flash-Attention projections to II.1, and the attention
math itself to nothing (II.6 -- parameter-free ops contribute no Gramian terms,
so they are never hooked in the first place).

Module collection and the compatibility checks are ported from TorchJD 0.17.0
``torchjd/autogram/_engine.py`` (``_hook_module_recursively``,
``_check_module_is_compatible``), MIT licence, (c) Valerian Rey, Pierre Quinton.

Handler contract, uniform across layer families::

    handler(module, A, X) -> Tensor  # [m, m] float64

    A: stacked upstream gradient, one slice per objective
    X: stacked forward capture -- activations for Linear/Norm,
       token indices for Embedding

Two lookup mechanisms:

* **type registry** -- exact ``type(module)``, then MRO, so subclasses inherit.
* **predicate registry** -- the escape hatch for architecture-specific classes
  that share no base with a torch built-in.  nanoGPT defines its own
  ``LayerNorm`` (to get an optional bias) which subclasses ``nn.Module``
  directly, so no MRO walk can find it; nanochat's ``RMSNorm`` is the same
  situation.  A predicate matches both without this package importing either.

Keep in sync with ``docs/operator_table.md`` -- that table is the
human-readable view of this dispatch, including each entry's gate status.
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch
from torch import nn

from jdgram.identities import embedding as embedding_id
from jdgram.identities import linear as linear_id
from jdgram.identities import norm as norm_id


class IdentityHandler(Protocol):
    def __call__(self, module: nn.Module, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor: ...


TYPE_REGISTRY: dict[type, IdentityHandler] = {}
PREDICATE_REGISTRY: list[tuple[Callable[[nn.Module], bool], IdentityHandler]] = []

# Cross-instance coupling breaks the per-objective decomposition entirely, so
# these are rejected rather than silently producing a wrong Gramian. Irrelevant
# for the transformer target (LayerNorm/RMSNorm), listed for completeness.
_UNSUPPORTED_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
    nn.RNNBase,
)


def register(module_type: type) -> Callable[[IdentityHandler], IdentityHandler]:
    def decorator(handler: IdentityHandler) -> IdentityHandler:
        TYPE_REGISTRY[module_type] = handler
        return handler

    return decorator


def register_predicate(
    predicate: Callable[[nn.Module], bool],
) -> Callable[[IdentityHandler], IdentityHandler]:
    def decorator(handler: IdentityHandler) -> IdentityHandler:
        PREDICATE_REGISTRY.append((predicate, handler))
        return handler

    return decorator


def dispatch(module: nn.Module) -> IdentityHandler:
    """Resolve a module to its identity handler: exact type, then MRO, then predicates."""
    handler = TYPE_REGISTRY.get(type(module))
    if handler is not None:
        return handler

    for base in type(module).__mro__:
        if base in TYPE_REGISTRY:
            return TYPE_REGISTRY[base]

    for predicate, predicate_handler in PREDICATE_REGISTRY:
        if predicate(module):
            return predicate_handler

    raise KeyError(
        f"no Gramian identity registered for {type(module).__name__}. Register one with "
        f"@register({type(module).__name__}) or @register_predicate(...), and add the row to "
        f"docs/operator_table.md."
    )


def has_handler(module: nn.Module) -> bool:
    try:
        dispatch(module)
    except KeyError:
        return False
    return True


@register(nn.Linear)
def linear_handler(module: nn.Linear, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return linear_id.sequence_gramian(A, X, module.bias is not None)


@register(nn.Embedding)
def token_embedding_handler(module: nn.Embedding, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # X is the index tensor the forward hook captured, not an activation.
    return embedding_id.token_embedding_gramian(A, X)


def positional_embedding_handler(
    module: nn.Embedding, A: torch.Tensor, X: torch.Tensor
) -> torch.Tensor:
    """II.3 handler for a position embedding.

    Not registered by type: ``wte`` and ``wpe`` are both ``nn.Embedding``, so
    type dispatch cannot separate them. Pass this in via the driver's
    ``handler_overrides`` for the position-embedding module.
    """
    return embedding_id.positional_embedding_gramian(A)


@register(nn.LayerNorm)
def layernorm_handler(module: nn.Module, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    eps = getattr(module, "eps", 1e-5)
    return norm_id.norm_gramian(A, X, getattr(module, "bias", None) is not None, eps=eps)


def _is_norm_like(module: nn.Module) -> bool:
    """Match custom LayerNorm/RMSNorm classes that subclass nn.Module directly.

    Deliberately narrow: a 1-D ``weight``, an optional 1-D ``bias``, no other
    direct parameters, and a class name that says what it is. Anything broader
    would start silently claiming modules whose identity is not II.5.
    """
    if not type(module).__name__.endswith(("LayerNorm", "RMSNorm")):
        return False
    direct = dict(module.named_parameters(recurse=False))
    weight = direct.pop("weight", None)
    bias = direct.pop("bias", None)
    if direct or weight is None or weight.ndim != 1:
        return False
    return bias is None or bias.ndim == 1


register_predicate(_is_norm_like)(layernorm_handler)


def check_module_supported(name: str, module: nn.Module) -> None:
    if isinstance(module, _UNSUPPORTED_MODULE_TYPES):
        raise ValueError(
            f"{name} is a {type(module).__name__}, which couples batch elements and so has no "
            f"per-objective Gramian decomposition. Replace it (e.g. GroupNorm / InstanceNorm) "
            f"or exclude it from the hooked set."
        )


def collect_hookable_modules(
    model: nn.Module,
    prefix: str = "",
) -> dict[str, nn.Module]:
    """Modules owning trainable parameters *directly*, keyed by qualified name.

    Recurses only into modules with no direct trainable parameters, which is what
    keeps a parent and its child from both claiming the same parameter. For
    nanoGPT this yields ``wte``, ``wpe``, each block's norms and Linears, ``ln_f``
    and ``lm_head`` -- and skips ``Block`` / ``MLP`` / ``CausalSelfAttention``,
    whose only parameters live in children.

    Parameter-free ops (SDPA, GELU, residual adds, Dropout) never appear: they
    hold nothing, so II.6 applies and autograd propagates ``A`` through them
    with no identity needed.
    """
    collected: dict[str, nn.Module] = {}

    if any(p.requires_grad for p in model.parameters(recurse=False)):
        check_module_supported(prefix or type(model).__name__, model)
        collected[prefix or type(model).__name__] = model
        return collected

    for child_name, child in model.named_children():
        qualified = f"{prefix}.{child_name}" if prefix else child_name
        collected.update(collect_hookable_modules(child, qualified))
    return collected


def find_shared_parameters(modules: dict[str, nn.Module]) -> dict[str, list[str]]:
    """Group hooked module names by any trainable parameter they share.

    Weight tying (nanoGPT's ``wte.weight is lm_head.weight``) shows up here as a
    group of two. It matters because per-module Gramians are only additive when
    the modules own disjoint parameters: for a shared parameter the true
    per-objective gradient is the *sum* over sites, and the Frobenius product of
    two sums carries cross terms that summing per-module Gramians drops. The
    driver refuses to guess -- see :mod:`jdgram.engine.hooks`.
    """
    by_param: dict[int, tuple[str, list[str]]] = {}
    for name, module in modules.items():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            # Key on the qualified name of the first owner seen, so two distinct
            # shared parameters that happen to share a local name stay separate.
            qualified = f"{name}.{param_name}"
            entry = by_param.setdefault(id(param), (qualified, []))
            entry[1].append(name)

    return {
        qualified: sorted(owners)
        for qualified, owners in by_param.values()
        if len(owners) > 1
    }
