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

from typing import Callable, Container, Protocol

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
    from jdgram.engine.materialize import materialized_gramian
    from jdgram.engine.router import route

    m, T, _ = A.shape
    d_out, d_in = module.out_features, module.in_features
    P_layer = d_out * d_in
    if route(m, T, P_layer, d_out, d_in) == "dfirst":
        return materialized_gramian(A, X, module.bias is not None)
    return linear_id.sequence_gramian(A, X, module.bias is not None)


@register(nn.Embedding)
def token_embedding_handler(module: nn.Embedding, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # X is the index tensor the forward hook captured, not an activation.
    return embedding_id.sequence_gramian(
        A, X, num_embeddings=module.num_embeddings
    )


def positional_embedding_handler(
    module: nn.Embedding, A: torch.Tensor, X: torch.Tensor
) -> torch.Tensor:
    """II.3 handler for a position embedding.

    Not registered by type: ``wte`` and ``wpe`` are both ``nn.Embedding``, so
    type dispatch cannot separate them. Pass this in via the driver's
    ``handler_overrides`` for the position-embedding module.
    """
    return embedding_id.positional_embedding_gramian(A)


def _norm_eps(module: nn.Module, *, default: float | None) -> float:
    """Read a norm module's epsilon, whatever the author decided to call it.

    ``nn.LayerNorm`` and ``Qwen3_5RMSNorm`` store ``eps``; ``Qwen3RMSNorm`` and
    ``Qwen3_5RMSNormGated`` store ``variance_epsilon`` -- the two conventions
    coexist inside a single HuggingFace model file. ``torch.nn.RMSNorm`` stores
    ``eps = None`` unless one is passed, so a plain ``getattr`` is not enough.

    ``default`` is a float only where the value is genuinely known from the
    module's source (nanoGPT hardcodes 1e-5 in ``forward`` and exposes nothing);
    it is ``None`` for RMSNorm, where guessing 1e-5 against a config that says
    1e-6 is exactly the silent-wrong-number failure this exists to prevent.
    """
    for attr in ("eps", "variance_epsilon"):
        value = getattr(module, attr, None)
        if value is not None:
            return float(value)
    if default is not None:
        return default
    raise ValueError(
        f"{type(module).__name__} exposes neither .eps nor .variance_epsilon. Refusing to "
        f"assume one: the Gramian is wrong by the ratio of the true and assumed epsilon, "
        f"silently. Pass the module through handler_overrides with an explicit eps."
    )


def _norm_handler(center: bool, *, eps_default: float | None):
    def handler(module: nn.Module, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return norm_id.norm_gramian(
            A, X, getattr(module, "bias", None) is not None,
            eps=_norm_eps(module, default=eps_default), center=center,
        )

    return handler


# center=True subtracts the mean, center=False does not. Dispatching both
# families to one handler silently hands RMSNorm the LayerNorm xhat, and since
# xhat *is* d(output)/d(gamma), every per-objective gradient is then wrong --
# symmetric, PSD and correctly shaped, so nothing downstream can catch it.
layernorm_handler = _norm_handler(center=True, eps_default=1e-5)
rmsnorm_handler = _norm_handler(center=False, eps_default=None)

register(nn.LayerNorm)(layernorm_handler)
if hasattr(nn, "RMSNorm"):  # torch >= 2.4; its MRO is (RMSNorm, Module) so the
    register(nn.RMSNorm)(rmsnorm_handler)  # MRO walk below would never find it


def _is_norm_like(module: nn.Module, suffix: str) -> bool:
    """Match custom LayerNorm/RMSNorm classes that subclass nn.Module directly.

    Deliberately narrow: a 1-D ``weight``, an optional 1-D ``bias``, no other
    direct parameters, and a class name that says what it is. Anything broader
    would start silently claiming modules whose identity is not II.5.

    In particular this must NOT be widened to match ``*RMSNormGated``. A gated
    norm computes ``w * xhat * silu(gate)``, so d(out)/d(w) carries the gate
    factor -- and the forward hook only captures ``args[0]``, so the gate is not
    even available. Those classes currently raise KeyError from ``dispatch``,
    which is the correct outcome; matching them would produce a wrong number.
    """
    if not type(module).__name__.endswith(suffix):
        return False
    direct = dict(module.named_parameters(recurse=False))
    weight = direct.pop("weight", None)
    bias = direct.pop("bias", None)
    if direct or weight is None or weight.ndim != 1:
        return False
    return bias is None or bias.ndim == 1


# Two predicates, not one tuple: the suffixes are disjoint, and keeping them
# separate is what makes the center= choice follow from the class name.
register_predicate(lambda m: _is_norm_like(m, "RMSNorm"))(rmsnorm_handler)
register_predicate(lambda m: _is_norm_like(m, "LayerNorm"))(layernorm_handler)


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
    *,
    exclude: Container[str] | None = None,
) -> dict[str, nn.Module]:
    """Modules owning trainable parameters *directly*, keyed by qualified name.

    A module is collected when it owns direct trainable parameters. The walk then
    continues into its children regardless, because a module can be both a
    parameter holder and a container.

    That distinction matters. An earlier version returned as soon as a module
    owned direct parameters, on the reasoning that a parent and child must not
    both claim the same parameter -- true, but the premise does not follow.
    ``nn.Linear``/``nn.Embedding``/norms have no parameterized children, so for
    those the two behaviours are identical. For a block that holds a couple of
    bare ``nn.Parameter``\\ s *and* several parameterized submodules, stopping
    early made the whole block one opaque unit and its children unreachable.
    Qwen3.5's ``Qwen3_5GatedDeltaNet`` is exactly that shape: 32 direct
    parameters (``A_log``, ``dt_bias``) in front of five ``nn.Linear``
    submodules, so 18 blocks hid ~190M parameters -- a quarter of the model --
    behind handlers that already existed.

    Each collected module's handler is responsible for that module's *direct*
    parameters only. That is automatic for every identity registered today,
    since none of them has parameterized children.

    ``exclude`` drops qualified names from the result. Use it to knowingly leave
    out a module whose direct parameters have no identity, when the caller
    accepts the Gramian is then partial -- ``dispatch`` still raises for anything
    collected and unhandled, so omission has to be deliberate rather than silent.

    Parameter-free ops (SDPA, GELU, residual adds, Dropout) never appear: they
    hold nothing, so II.6 applies and autograd propagates ``A`` through them
    with no identity needed.
    """
    collected: dict[str, nn.Module] = {}
    name = prefix or type(model).__name__

    if any(p.requires_grad for p in model.parameters(recurse=False)):
        if exclude is None or name not in exclude:
            check_module_supported(name, model)
            collected[name] = model

    for child_name, child in model.named_children():
        qualified = f"{prefix}.{child_name}" if prefix else child_name
        collected.update(collect_hookable_modules(child, qualified, exclude=exclude))
    return collected


def unhandled_direct_params(model: nn.Module) -> dict[str, int]:
    """Qualified name -> direct trainable parameter count, for modules with no identity.

    What a partial run would silently omit. Reported so the caller can decide
    whether the omission is acceptable before passing those names to
    :func:`collect_hookable_modules`'s ``exclude``.
    """
    out: dict[str, int] = {}
    for name, mod in model.named_modules():
        direct = [p for p in mod.parameters(recurse=False) if p.requires_grad]
        if not direct or has_handler(mod):
            continue
        out[name or type(mod).__name__] = sum(p.numel() for p in direct)
    return out


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
