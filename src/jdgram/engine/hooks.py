"""Hook plumbing: capture (A, X) per layer, then fire each layer's identity.

Hook injection, the forward-phase flag and the pytree handling of module outputs
are ported from TorchJD 0.17.0 ``torchjd/autogram/_module_hook_manager.py``
(``ModuleHookManager``, ``BoolRef``, ``Hook``), MIT licence, (c) Valerian Rey,
Pierre Quinton.

Orchestration (decided and verified before this module was written, and the one
part that deliberately differs from upstream):

* ONE shared forward pass. ``X`` is identical across objectives, so each
  module's forward hook fires once and stores its input once.
* m SEPARATE reverse passes, one per objective. A module that treats batch
  elements independently sees an upstream gradient that is nonzero only at batch
  row ``i`` on pass ``i``, so the driver keeps that row and stacks across passes.
* Identities fire ONCE per module, after the loop -- not incrementally.

TorchJD instead vmaps a single reverse pass and accumulates inside
``backward``, which is strictly better at scale but couples the identity to the
autograd internals. The loop here mirrors ``gates/brute_force.py``'s own
structure, which is exactly what lets a gate diff the two cleanly: same loop
shape, different inner computation. Upstream's streaming
``remaining_counter`` is a memory optimisation for real shapes (step 6+), not a
correctness requirement at gate sizes.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.overrides import is_tensor_like
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from jdgram.engine.edges import EdgeRegistry
from jdgram.engine.node import GramianNode
from jdgram.engine.registry import (
    IdentityHandler,
    collect_hookable_modules,
    dispatch,
    find_shared_parameters,
)


class BoolRef:
    """A mutable boolean, so hooks can see a phase change made from outside."""

    def __init__(self, value: bool) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return self.value


@dataclass
class ModuleCapture:
    """Everything the engine records for one hooked module."""

    name: str
    module: nn.Module
    forward_calls: int = 0
    inputs: list[PyTree] = field(default_factory=list)
    output_shapes: list[torch.Size] = field(default_factory=list)
    grads: list[tuple[Tensor, ...]] = field(default_factory=list)

    def record_forward(self, args: tuple[PyTree, ...], outputs: list[Tensor]) -> None:
        self.forward_calls += 1
        self.inputs.append(args[0] if args else None)
        self.output_shapes.append(outputs[0].shape)

    def record_backward(self, grad_outputs: tuple[Tensor, ...]) -> None:
        self.grads.append(grad_outputs)

    def clear_grads(self) -> None:
        self.grads.clear()


@dataclass
class LayerCapture:
    """Per-module captures assembled across the m reverse passes.

    Handed to identity handlers, and to the shared-parameter handlers that gate
    5e supplies for weight tying.
    """

    name: str
    module: nn.Module
    A: Tensor
    X: PyTree


@dataclass
class GramianResult:
    total: Tensor
    per_module: dict[str, Tensor]
    per_shared_group: dict[frozenset[str], Tensor]


class _Hook:
    def __init__(
        self,
        capture: ModuleCapture,
        phase: BoolRef,
        register_edge: Callable[[object], None],
    ) -> None:
        self.capture = capture
        self.phase = phase
        self.register_edge = register_edge

    def __call__(
        self,
        _module: nn.Module,
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        outputs: PyTree,
    ) -> PyTree:
        if self.phase:
            return outputs

        flat_outputs, output_spec = tree_flatten(outputs)
        rg_indices = [
            i
            for i, out in enumerate(flat_outputs)
            if is_tensor_like(out) and out.requires_grad
        ]
        if not rg_indices:
            # Reachable when a module owns a trainable parameter but returns
            # nothing differentiable; there is no Gramian contribution to make.
            return outputs

        rg_outputs = [flat_outputs[i] for i in rg_indices]
        self.capture.record_forward(args, rg_outputs)

        # Register a child edge of the node so the driver can force the reverse
        # pass through it without asking for parameter gradients. The smallest
        # output is the cheapest edge to hold.
        smallest = min(rg_outputs, key=lambda t: t.numel())
        self.register_edge(get_gradient_edge(smallest))

        wrapped = GramianNode.apply(self.capture, *rg_outputs)
        for i, out in zip(rg_indices, wrapped, strict=True):
            flat_outputs[i] = out
        return tree_unflatten(flat_outputs, output_spec)


class ModuleHookManager:
    """Installs and owns the forward hooks for a set of modules.

    Hooks are removed via ``weakref.finalize``: a live hook keeps the graph
    alive through the nodes that reference it, and those form a reference cycle
    the collector will not break on its own.
    """

    def __init__(self, target_edges: EdgeRegistry) -> None:
        self._target_edges = target_edges
        self.phase = BoolRef(False)
        self.captures: dict[str, ModuleCapture] = {}
        self._handles: list[RemovableHandle] = []
        self._finalizer = weakref.finalize(
            self, ModuleHookManager._remove_handles, self._handles
        )

    def hook_module(self, name: str, module: nn.Module) -> ModuleCapture:
        capture = ModuleCapture(name=name, module=module)
        hook = _Hook(capture, self.phase, self._target_edges.register)
        self.captures[name] = capture
        self._handles.append(module.register_forward_hook(hook, with_kwargs=True))
        return capture

    @staticmethod
    def _remove_handles(handles: list[RemovableHandle]) -> None:
        for handle in handles:
            handle.remove()
        handles.clear()

    def remove_hooks(self) -> None:
        ModuleHookManager._remove_handles(self._handles)

    def __enter__(self) -> ModuleHookManager:
        return self

    def __exit__(self, *exc) -> None:
        self.remove_hooks()


def _is_batched(capture: ModuleCapture, m: int) -> bool:
    """Whether this module's captures carry a leading objective dimension.

    Most modules are applied per batch element, so both their input and output
    lead with a dimension of size m and objective ``i``'s gradient lives at row
    ``i``.

    A position embedding is the exception: it is called with a bare ``[T]``
    position vector and its ``[T, d]`` output broadcasts over the batch.
    Autograd's broadcast-backward has already summed over the batch by the time
    the hook sees the gradient -- and since only row ``i`` was seeded on pass
    ``i``, that sum *is* objective ``i``'s gradient. So the capture is taken
    whole rather than indexed.

    Requiring both input and output to lead with m keeps that case out. The rule
    is only ambiguous when the sequence length equals m; pass
    ``batched_overrides`` if you hit that.
    """
    output_shape = capture.output_shapes[0]
    input_value = capture.inputs[0]
    if not output_shape or output_shape[0] != m:
        return False
    if not is_tensor_like(input_value) or input_value.ndim == 0:
        return False
    return input_value.shape[0] == m


def _objective_slice(grad: Tensor, i: int, batched: bool) -> Tensor:
    return grad[i] if batched else grad


def compute_gramian(
    model: nn.Module,
    compute_losses: Callable[[], Tensor],
    *,
    modules: dict[str, nn.Module] | None = None,
    handler_overrides: dict[str, IdentityHandler] | None = None,
    shared_handlers: dict[frozenset[str], Callable[[dict[str, LayerCapture]], Tensor]] | None = None,
    batched_overrides: dict[str, bool] | None = None,
    use_leaf_edges: bool = True,
) -> GramianResult:
    """Exact Gramian of the m objectives, accumulated per hooked module.

    :param compute_losses: runs the forward pass and returns the ``[m]`` vector
        of per-objective losses. Passed as a callable so this package stays
        independent of any particular model or loss construction.
    :param modules: hooked set; defaults to every module owning trainable
        parameters directly.
    :param handler_overrides: per-module-name identity, for cases type dispatch
        cannot resolve -- notably a position embedding, which is an
        ``nn.Embedding`` like the token embedding but needs II.3.
    :param shared_handlers: identity for a group of modules sharing a parameter,
        keyed by the frozenset of their names. Required when tying is present:
        per-module Gramians are only additive for disjoint parameters, so summing
        them across a tied pair drops the cross terms.
    :param batched_overrides: force the objective-slicing rule for a module.

    ``per_module`` is what per-layer gates diff against brute force's
    per-parameter blocks, so a failure names its layer instead of only reporting
    that the total is wrong.
    """
    handler_overrides = handler_overrides or {}
    shared_handlers = shared_handlers or {}
    batched_overrides = batched_overrides or {}

    if modules is None:
        modules = collect_hookable_modules(model)
    if not modules:
        raise ValueError("no hookable modules: nothing owns trainable parameters")

    shared = find_shared_parameters(modules)
    grouped_names: dict[str, frozenset[str]] = {}
    for param_name, owners in shared.items():
        group = frozenset(owners)
        if group not in shared_handlers:
            raise ValueError(
                f"{param_name} is shared by {sorted(owners)}. The per-objective gradient of a "
                f"shared parameter is the SUM over its sites, so the Frobenius product carries "
                f"cross terms; summing per-module Gramians silently drops them and returns a "
                f"structurally plausible wrong answer. Supply shared_handlers[frozenset({sorted(owners)})] "
                f"(design doc II.4), or run on an untied model."
            )
        for owner in owners:
            grouped_names[owner] = group

    target_edges = EdgeRegistry()

    with ModuleHookManager(target_edges) as manager:
        for name, module in modules.items():
            manager.hook_module(name, module)

        losses = compute_losses()
        if losses.ndim != 1:
            raise ValueError(f"compute_losses must return a [m] vector, got shape {tuple(losses.shape)}")
        m = losses.shape[0]

        missing = [name for name, cap in manager.captures.items() if cap.forward_calls == 0]
        if missing:
            raise RuntimeError(
                f"hooked but never called during the forward pass: {missing}. Either they are not "
                f"on the objectives' path, or the forward bypasses them by touching their "
                f"parameters directly instead of calling the module."
            )
        repeated = {
            name: cap.forward_calls
            for name, cap in manager.captures.items()
            if cap.forward_calls > 1
        }
        if repeated:
            raise NotImplementedError(
                f"modules called more than once per forward pass: {repeated}. Handling this needs "
                f"the streaming remaining_counter from TorchJD's GramianComputer, deliberately "
                f"deferred to step 6."
            )

        batched = {
            name: batched_overrides.get(name, _is_batched(cap, m))
            for name, cap in manager.captures.items()
        }

        leaf_edges: list = []
        if use_leaf_edges and len(target_edges) > 0:
            leaf_edges = list(target_edges.get_leaf_edges({get_gradient_edge(losses)}))

        stacked: dict[str, list[Tensor]] = {name: [] for name in modules}

        manager.phase.value = True
        try:
            for i in range(m):
                for capture in manager.captures.values():
                    capture.clear_grads()

                retain = i < m - 1
                if leaf_edges:
                    torch.autograd.grad(
                        outputs=losses[i],
                        inputs=leaf_edges,
                        retain_graph=retain,
                        allow_unused=True,
                    )
                else:
                    model.zero_grad(set_to_none=True)
                    losses[i].backward(retain_graph=retain)

                for name, capture in manager.captures.items():
                    if len(capture.grads) != 1:
                        raise RuntimeError(
                            f"{name}: expected 1 backward capture on objective {i}, got "
                            f"{len(capture.grads)}. Zero means the module is off this objective's "
                            f"reverse path; more than one means it was reached repeatedly."
                        )
                    grad_outputs = capture.grads[0]
                    if len(grad_outputs) != 1:
                        raise NotImplementedError(
                            f"{name}: {len(grad_outputs)} differentiable outputs. Multi-output "
                            f"modules need an identity that consumes all of them."
                        )
                    stacked[name].append(
                        _objective_slice(grad_outputs[0], i, batched[name])
                    )
        finally:
            manager.phase.value = False

        layers: dict[str, LayerCapture] = {}
        for name, capture in manager.captures.items():
            layers[name] = LayerCapture(
                name=name,
                module=capture.module,
                A=torch.stack(stacked[name]),
                X=capture.inputs[0],
            )

    per_module: dict[str, Tensor] = {}
    per_shared_group: dict[frozenset[str], Tensor] = {}

    for name, layer in layers.items():
        if name in grouped_names:
            continue
        handler = handler_overrides.get(name) or dispatch(layer.module)
        per_module[name] = handler(layer.module, layer.A, layer.X)

    for group, handler in shared_handlers.items():
        if not group <= set(layers):
            raise KeyError(f"shared_handlers group {sorted(group)} is not fully hooked")
        per_shared_group[group] = handler({name: layers[name] for name in group})

    contributions = list(per_module.values()) + list(per_shared_group.values())
    total = contributions[0]
    for contribution in contributions[1:]:
        total = total + contribution

    return GramianResult(total=total, per_module=per_module, per_shared_group=per_shared_group)
