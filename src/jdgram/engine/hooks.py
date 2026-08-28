"""Hook-driven Gramian engine: three reverse drivers behind one entry point.

``compute_gramian`` is the whole public surface.  What changes between drivers is
only *how the per-objective upstream gradient ``A`` is obtained*, and *when* each
layer's ``[m, m]`` contribution is formed:

``squashed`` (default)
    One **ordinary** backward seeded with ``ones(m)``.  With per-instance losses
    and batch-independent modules the intermediate Jacobians are block-diagonal,
    so ``d(sum_i L_i)/dz[i] == dL_i/dz[i]``: row ``i`` of the gradient that
    arrives at a layer output *is* objective ``i``'s upstream gradient, with no
    replication anywhere.  Each layer's identity fires inside its own backward and
    its captures are dropped immediately (:mod:`jdgram.engine.accumulate`), so the
    engine holds one layer's ``(A, X)`` at a time.  This is TorchJD autogram's
    strategy; the reason it is 5-10x leaner than what jdgram used to do.

``batched``
    One ``is_grads_batched`` backward seeded with ``eye(m)``.  Correct, and the
    honest way to get ``A`` when the block-diagonal assumption does not hold -- but
    ``vmap`` replicates the *entire* reverse working set ``m`` times, and each
    layer's returned gradient is ``[m, m, ...]`` of which only the diagonal is
    used.  Costs ``O(m)`` more memory than ``squashed`` for the same answer.  Kept
    as the reference the gates A/B against, and as the fallback for models with
    unbatched hooked modules.

``loop``
    ``m`` separate backwards, one objective seeded per pass.  Cheapest in memory,
    ``m`` times the time.  The CPU/debug path.

All three produce the same ``G``; gate 5f pins them together.

Hook injection, the phase flag and gradient-edge bookkeeping are ported from
TorchJD 0.17.0 ``autogram`` (MIT, (c) Valerian Rey, Pierre Quinton).
"""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.overrides import is_tensor_like
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from jdgram.engine.accumulate import GramianAccumulator, SharedGroup
from jdgram.engine.edges import EdgeRegistry
from jdgram.engine.node import GramianNode
from jdgram.engine.registry import (
    IdentityHandler,
    collect_hookable_modules,
    dispatch,
    find_shared_parameters,
)
from jdgram.engine.residual import flatten_residual_row, gramian_from_rows
from jdgram.engine.router import force_route as set_force_route
from jdgram.engine.router import get_force_route
from jdgram.identities.precision import workspace_dtype as workspace_dtype_ctx

Driver = Literal["squashed", "batched", "loop"]


class BoolRef:
    """A mutable boolean, so hooks can see a phase change made from outside."""

    def __init__(self, value: bool) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return self.value


@dataclass
class LayerCapture:
    """Per-module ``(A, X)`` handed to identity and shared-group handlers."""

    name: str
    module: nn.Module
    A: Tensor
    X: PyTree


@dataclass
class ModuleCapture:
    """Everything the engine records for one hooked module.

    Under ``squashed`` this object is also the streaming site: ``record_backward``
    forms the ``[m, m]`` contribution and releases ``inputs`` on the spot, so the
    capture is empty again before the reverse pass reaches the next layer.
    """

    name: str
    module: nn.Module
    forward_calls: int = 0
    inputs: list[PyTree] = field(default_factory=list)
    output_shapes: list[torch.Size] = field(default_factory=list)
    grads: list[tuple[Tensor, ...]] = field(default_factory=list)
    # Wrapped outputs (GramianNode results). The batched driver reads ``A`` from
    # ``autograd.grad(..., inputs=rg_outputs)`` return values, not side effects.
    rg_outputs: list[Tensor] = field(default_factory=list)
    # Set only for the streaming driver.
    stream: "_StreamTarget | None" = None
    # Only the loop driver needs the backward side effect. The batched driver
    # reads ``A`` from returned grads and ignores whatever lands here, so keeping
    # a copy of every module's vmap slice would be pure waste.
    collect: bool = False
    # Loop driver only, set by ``_reverse_loop`` before each pass: which
    # objective is currently seeded, and whether this module's gradients carry a
    # leading objective dimension. Together they let ``record_backward`` keep
    # only the row that is not identically zero -- see there for why that matters.
    objective: int | None = None
    batched: bool = True

    def record_forward(self, args: tuple[PyTree, ...], outputs: list[Tensor]) -> None:
        self.forward_calls += 1
        self.inputs.append(args[0] if args else None)
        self.output_shapes.append(outputs[0].shape)

    def record_backward(self, grad_outputs: tuple[Tensor, ...]) -> None:
        if self.stream is not None:
            self.stream.consume(self, grad_outputs)
        elif self.collect:
            # clone, not detach: the loop driver holds these past the backward
            # call, and aten::detach has no vmap batching rule.
            #
            # Clone the *seeded row*, not the whole gradient. The forward runs
            # once at batch m, so a hooked module's gradient arrives shaped
            # ``[m, ...]`` on every pass -- but ``losses[i]`` depends on batch row
            # ``i`` alone, so the other m-1 rows are identically zero. Slicing
            # here rather than at the end of the pass is what makes the slice
            # cheap: ``g[i]`` is a *view*, and a view pins its parent's entire
            # storage, so holding one per objective kept all m full gradients
            # alive at once and made captures cost m^2 instead of m.
            if self.objective is None:
                self.grads.append(tuple(g.clone() for g in grad_outputs))
            else:
                self.grads.append(
                    tuple(
                        _objective_slice(g, self.objective, self.batched).clone()
                        for g in grad_outputs
                    )
                )

    def take_input(self) -> PyTree:
        """Pop the forward capture, detached. Detaching is load-bearing: ``X``
        still sits on the forward graph, and feeding it to an identity makes ``G``
        require grad, forming a capture->GramianNode->graph cycle that leaks one
        forward per step (~GB on a transformer) until OOM."""
        x = self.inputs.pop(0) if self.inputs else None
        if is_tensor_like(x):
            x = x.detach()
        return x

    def clear_grads(self) -> None:
        self.grads.clear()


@dataclass
class GramianResult:
    total: Tensor
    per_module: dict[str, Tensor]
    per_shared_group: dict[frozenset[str], Tensor]
    #: ``[m, m]`` block for ``residual_params``, already included in ``total``.
    #: ``None`` when no residual parameters were passed.
    residual: Tensor | None = None
    #: Bytes of captures simultaneously pinned at the streaming high-water mark.
    #: Zero for an untied model under ``squashed`` -- every layer frees on the spot.
    peak_held_bytes: int = 0
    driver: str = "squashed"


class _StreamTarget:
    """Fires a module's identity from inside its backward, then frees its captures."""

    def __init__(
        self,
        accumulator: GramianAccumulator,
        handlers: dict[str, IdentityHandler],
        groups: dict[str, SharedGroup],
    ) -> None:
        self.accumulator = accumulator
        self.handlers = handlers
        self.groups = groups
        self.peak_held_bytes = 0
        self.visited: set[str] = set()

    def _note_held(self) -> None:
        held = sum(g.held_bytes() for g in set(self.groups.values()))
        self.peak_held_bytes = max(self.peak_held_bytes, held)

    def consume(self, capture: ModuleCapture, grad_outputs: tuple[Tensor, ...]) -> None:
        name = capture.name
        if len(grad_outputs) != 1:
            raise NotImplementedError(
                f"{name}: {len(grad_outputs)} differentiable outputs. Multi-output "
                f"modules need an identity that consumes all of them."
            )
        if name in self.visited:
            raise NotImplementedError(
                f"{name}: reached twice in one reverse pass. Streaming accumulation "
                f"needs a summed-Jacobian formulation for multi-call modules "
                f"(TorchJD's remaining_counter); use driver='loop' meanwhile."
            )
        self.visited.add(name)

        group = self.groups.get(name)
        A = grad_outputs[0].detach()
        if group is not None:
            # Only a tied group outlives this backward call, so only it pays for
            # a copy; autograd is free to reuse the buffer once we return.
            A = A.clone()
        layer = LayerCapture(
            name=name,
            module=capture.module,
            A=A,
            X=capture.take_input(),
        )
        if group is None:
            handler = self.handlers[name]
            self.accumulator.add_module(name, handler(layer.module, layer.A, layer.X))
            return
        contribution = group.submit(layer)
        self._note_held()
        if contribution is not None:
            self.accumulator.add_group(group.names, contribution)


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
        self.capture.rg_outputs = list(wrapped)
        for i, out in zip(rg_indices, wrapped, strict=True):
            flat_outputs[i] = out
        return tree_unflatten(flat_outputs, output_spec)


class ModuleHookManager:
    """Installs and owns the forward hooks for a set of modules.

    Hooks are removed via ``weakref.finalize``: a live hook keeps the graph alive
    through the nodes that reference it, and those form a reference cycle the
    collector will not break on its own.
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

    A position embedding called with a bare ``[T]`` vector is the exception: its
    ``[T, d]`` output broadcasts over the batch, and autograd's broadcast-backward
    has already summed over the batch by the time the hook sees the gradient.
    Under the ``loop`` driver only row ``i`` was seeded, so that sum *is*
    objective ``i``'s gradient and the capture is taken whole.  Under ``squashed``
    every objective is seeded at once, so the sum is *not* recoverable -- which is
    why that driver rejects unbatched modules outright.

    Requiring both input and output to lead with m keeps that case out.  The rule
    is only ambiguous when the sequence length equals m; pass ``batched_overrides``
    if you hit that.
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


def _assemble_A(grad: Tensor, batched: bool) -> Tensor:
    """Build ``A: [m, ...]`` from one ``is_grads_batched`` capture (leading m).

    Batched modules yield ``[m, m, ...]`` (vmap x batch); take the diagonal
    ``A[i] = grad[i, i, ...]``. Unbatched (e.g. an unbatched wpe) already have
    ``A = grad``.
    """
    m = grad.shape[0]
    if batched:
        if grad.shape[1] != m:
            raise RuntimeError(
                f"batched is_grads_batched capture expected [m, m, ...], got {tuple(grad.shape)}"
            )
        idx = torch.arange(m, device=grad.device)
        return grad[idx, idx]
    return grad


def _differentiation_inputs(leaf_edges: list, manager: ModuleHookManager) -> list:
    """Targets that force the reverse pass through every injected node."""
    if leaf_edges:
        return leaf_edges
    inputs: list[Tensor] = []
    for name, cap in manager.captures.items():
        if not cap.rg_outputs:
            raise RuntimeError(f"{name}: missing rg_outputs; the forward did not run hooked")
        inputs.append(cap.rg_outputs[0])
    return inputs


def _reverse_squashed(
    *,
    losses: Tensor,
    leaf_edges: list,
    manager: ModuleHookManager,
) -> None:
    """One ordinary backward, ``grad_outputs = ones(m)``.

    Nothing is returned: every layer's contribution was accumulated from inside
    its own backward.  ``retain_graph=False`` lets autograd free activations as
    the sweep proceeds, which is a large part of the memory win.
    """
    torch.autograd.grad(
        outputs=losses,
        inputs=_differentiation_inputs(leaf_edges, manager),
        grad_outputs=torch.ones_like(losses),
        retain_graph=False,
        allow_unused=True,
    )


def _reverse_loop(
    *,
    m: int,
    losses: Tensor,
    model: nn.Module,
    leaf_edges: list,
    manager: ModuleHookManager,
    modules: dict[str, nn.Module],
    batched: dict[str, bool],
    residual_params: list[nn.Parameter] | None = None,
) -> tuple[dict[str, Tensor], list[Tensor] | None]:
    """Legacy m reverse passes -- gate/debug path.

    Returns ``(A_by_name, residual_rows)``. ``residual_rows`` is ``None`` unless
    ``residual_params`` was given, in which case it holds one flattened float64
    gradient row per objective, harvested from the same passes.
    """
    stacked: dict[str, list[Tensor]] = {name: [] for name in modules}
    for name, capture in manager.captures.items():
        capture.batched = batched[name]
    tail = list(residual_params or [])
    residual_rows: list[Tensor] | None = [] if tail else None
    for i in range(m):
        for capture in manager.captures.values():
            capture.clear_grads()
            # record_backward keeps only row i; without this it would have to
            # clone the whole [m, ...] gradient and the loop would pin all m.
            capture.objective = i

        retain = i < m - 1
        if leaf_edges:
            # The tail rides along: this pass already walks the graph, and the
            # return value was previously discarded (captures are side effects),
            # so appending the tail parameters costs one extra hop per parameter
            # instead of a whole second forward plus m more backwards.
            grads = torch.autograd.grad(
                outputs=losses[i],
                inputs=list(leaf_edges) + tail,
                retain_graph=retain,
                allow_unused=True,
            )
            if residual_rows is not None:
                residual_rows.append(
                    flatten_residual_row(grads[len(leaf_edges):], tail)
                )
        else:
            model.zero_grad(set_to_none=True)
            losses[i].backward(retain_graph=retain)
            if residual_rows is not None:
                # zero_grad above ran this iteration, so .grad is exactly dL_i/dp.
                residual_rows.append(
                    flatten_residual_row([p.grad for p in tail], tail)
                )

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
            # Already sliced to objective i inside record_backward, and cloned
            # there, so this owns its storage rather than viewing a full [m, ...].
            stacked[name].append(grad_outputs[0])
    A_by_name = {name: torch.stack(parts) for name, parts in stacked.items()}
    return A_by_name, residual_rows


def _reverse_batched(
    *,
    m: int,
    losses: Tensor,
    leaf_edges: list,
    manager: ModuleHookManager,
    batched: dict[str, bool],
    residual_params: list[nn.Parameter] | None = None,
) -> tuple[dict[str, Tensor], list[Tensor] | None]:
    """Single reverse pass: seed all m objectives via ``is_grads_batched``.

    ``A`` is taken from returned grads w.r.t. each module's wrapped outputs.
    Hooks / ``GramianNode.backward`` only see an unbatched slice under
    ``is_grads_batched``, so side-effect capture cannot build ``[m, ...]``.
    ``leaf_edges`` is unused here but kept for call-site symmetry with the loop.
    """
    del leaf_edges  # traversal is forced by differentiating all rg_outputs
    for capture in manager.captures.values():
        capture.clear_grads()

    names = list(manager.captures)
    grad_inputs: list[Tensor] = []
    for name in names:
        cap = manager.captures[name]
        if not cap.rg_outputs:
            raise RuntimeError(f"{name}: missing rg_outputs for batched reverse")
        if len(cap.rg_outputs) != 1:
            raise NotImplementedError(
                f"{name}: {len(cap.rg_outputs)} differentiable outputs. Multi-output "
                f"modules need an identity that consumes all of them."
            )
        grad_inputs.append(cap.rg_outputs[0])

    tail = list(residual_params or [])
    eye = torch.eye(m, device=losses.device, dtype=losses.dtype)
    grads = torch.autograd.grad(
        outputs=losses,
        inputs=grad_inputs + tail,
        grad_outputs=eye,
        is_grads_batched=True,
        retain_graph=False,
        allow_unused=True,
    )
    residual_rows: list[Tensor] | None = None
    if tail:
        # Under is_grads_batched each parameter's gradient comes back [m, *shape],
        # so row i is objective i's -- the same rows the loop driver builds one
        # pass at a time, from a single backward.
        tail_grads = grads[len(grad_inputs):]
        grads = grads[: len(grad_inputs)]
        residual_rows = [
            flatten_residual_row(
                [None if g is None else g[i] for g in tail_grads], tail
            )
            for i in range(m)
        ]
    out: dict[str, Tensor] = {}
    for name, grad in zip(names, grads, strict=True):
        if grad is None:
            raise RuntimeError(
                f"{name}: no gradient under is_grads_batched. The module is off "
                f"every objective's reverse path."
            )
        if grad.shape[0] != m:
            raise RuntimeError(
                f"{name}: expected leading dim m={m} from is_grads_batched, "
                f"got {tuple(grad.shape)}"
            )
        out[name] = _assemble_A(grad, batched[name])
    return out, residual_rows


def _resolve_driver(driver: Driver | None, batched_backward: bool | None) -> Driver:
    """``batched_backward`` is the pre-driver spelling; keep it working."""
    if driver is not None and batched_backward is not None:
        raise ValueError("pass either driver= or batched_backward=, not both")
    if driver is not None:
        if driver not in ("squashed", "batched", "loop"):
            raise ValueError(
                f"unknown driver {driver!r}; expected 'squashed', 'batched' or 'loop'"
            )
        return driver
    if batched_backward is not None:
        return "batched" if batched_backward else "loop"
    return "squashed"


def compute_gramian(
    model: nn.Module,
    compute_losses: Callable[[], Tensor],
    *,
    modules: dict[str, nn.Module] | None = None,
    handler_overrides: dict[str, IdentityHandler] | None = None,
    shared_handlers: dict[frozenset[str], Callable[[dict[str, LayerCapture]], Tensor]] | None = None,
    batched_overrides: dict[str, bool] | None = None,
    use_leaf_edges: bool = True,
    driver: Driver | None = None,
    batched_backward: bool | None = None,
    force_route: str | None = None,
    workspace_dtype: torch.dtype | None = None,
    residual_params: list[nn.Parameter] | None = None,
) -> GramianResult:
    """Exact Gramian of the m objectives, accumulated per hooked module.

    :param compute_losses: runs the forward pass and returns the ``[m]`` vector of
        per-objective losses. Passed as a callable so this package stays
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
    :param driver: ``"squashed"`` (default), ``"batched"`` or ``"loop"``; see the
        module docstring. ``squashed`` requires every hooked module to be batched
        on dim 0 and to treat batch elements independently.
    :param batched_backward: deprecated spelling. ``True`` -> ``"batched"``,
        ``False`` -> ``"loop"``.
    :param force_route: thin adapter over :func:`jdgram.engine.router.force_route`.
        ``None`` leaves the current router force unchanged; ``"tfirst"`` /
        ``"dfirst"`` pins every layer for this call and restores afterward.
    :param workspace_dtype: thin adapter over
        :func:`jdgram.identities.precision.workspace_dtype`. ``None`` leaves the
        process default (fp32, or whatever gates pinned). Final ``[m, m]`` still
        accumulates in float64.
    :param residual_params: parameters with no closed-form identity, whose exact
        ``[m, m]`` block is built by explicit Jacobian and added into ``total``
        (also exposed as ``GramianResult.residual``). Their gradients are taken
        from the reverse pass this call already runs, so the tail costs one extra
        hop per parameter rather than a second forward and ``m`` more backwards
        -- which is what calling :func:`jdgram.engine.residual.residual_gramian`
        separately costs. ``loop`` and ``batched`` only; see there for why
        ``squashed`` cannot.

    ``per_module`` is what per-layer gates diff against brute force's per-parameter
    blocks, so a failure names its layer instead of only reporting that the total
    is wrong.
    """
    resolved = _resolve_driver(driver, batched_backward)
    prev_force = get_force_route()
    try:
        if force_route is not None:
            set_force_route(force_route)  # type: ignore[arg-type]
        ctx = (
            workspace_dtype_ctx(workspace_dtype)
            if workspace_dtype is not None
            else nullcontext()
        )
        with ctx:
            return _compute_gramian_body(
                model,
                compute_losses,
                modules=modules,
                handler_overrides=handler_overrides,
                shared_handlers=shared_handlers,
                batched_overrides=batched_overrides,
                use_leaf_edges=use_leaf_edges,
                driver=resolved,
                residual_params=residual_params,
            )
    finally:
        set_force_route(prev_force)


def _resolve_groups(
    modules: dict[str, nn.Module],
    shared_handlers: dict[frozenset[str], Callable],
) -> dict[str, frozenset[str]]:
    """Map each tied module name to its group, refusing to guess when untold."""
    grouped: dict[str, frozenset[str]] = {}
    for param_name, owners in find_shared_parameters(modules).items():
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
            grouped[owner] = group
    return grouped


def _compute_gramian_body(
    model: nn.Module,
    compute_losses: Callable[[], Tensor],
    *,
    modules: dict[str, nn.Module] | None,
    handler_overrides: dict[str, IdentityHandler] | None,
    shared_handlers: dict[frozenset[str], Callable[[dict[str, LayerCapture]], Tensor]] | None,
    batched_overrides: dict[str, bool] | None,
    use_leaf_edges: bool,
    driver: Driver,
    residual_params: list[nn.Parameter] | None = None,
) -> GramianResult:
    handler_overrides = handler_overrides or {}
    shared_handlers = shared_handlers or {}
    batched_overrides = batched_overrides or {}

    if residual_params and driver == "squashed":
        raise NotImplementedError(
            "driver='squashed' cannot supply per-objective residual rows: its "
            "single ones-seeded backward yields the sum over objectives, not one "
            "row each, and it frees the graph as it sweeps. Use driver='loop' or "
            "'batched', or call jdgram.engine.residual.residual_gramian on a "
            "separate forward."
        )

    if modules is None:
        modules = collect_hookable_modules(model)
    if not modules:
        raise ValueError("no hookable modules: nothing owns trainable parameters")

    grouped_names = _resolve_groups(modules, shared_handlers)
    accumulator = GramianAccumulator()
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

        if driver == "squashed":
            unbatched = sorted(name for name, is_b in batched.items() if not is_b)
            if unbatched:
                raise ValueError(
                    f"driver='squashed' needs every hooked module batched on dim 0, but "
                    f"{unbatched} are not. A single ones-seeded backward sums over objectives "
                    f"at an unbatched module, so per-objective gradients are unrecoverable "
                    f"there. Fix the forward so the module sees a batched input (a position "
                    f"embedding wants pos expanded to [B, T], not [T]), pass batched_overrides "
                    f"if the m == T shape rule misfired, or use driver='batched'."
                )

        leaf_edges: list = []
        if use_leaf_edges and len(target_edges) > 0:
            leaf_edges = list(target_edges.get_leaf_edges({get_gradient_edge(losses)}))

        stream: _StreamTarget | None = None
        if driver == "squashed":
            groups = {}
            built: dict[frozenset[str], SharedGroup] = {}
            for name, group in grouped_names.items():
                if group not in built:
                    built[group] = SharedGroup(group, shared_handlers[group])
                groups[name] = built[group]
            for group in shared_handlers:
                if not group <= set(modules):
                    raise KeyError(f"shared_handlers group {sorted(group)} is not fully hooked")
            handlers = {
                name: (handler_overrides.get(name) or dispatch(mod))
                for name, mod in modules.items()
                if name not in grouped_names
            }
            stream = _StreamTarget(accumulator, handlers, groups)
            for capture in manager.captures.values():
                capture.stream = stream
        elif driver == "loop":
            for capture in manager.captures.values():
                capture.collect = True

        manager.phase.value = True
        try:
            if driver == "squashed":
                # Identities run inside GramianNode.backward, under no_grad.
                _reverse_squashed(losses=losses, leaf_edges=leaf_edges, manager=manager)
                A_by_name = None
                residual_rows = None
            elif driver == "batched":
                A_by_name, residual_rows = _reverse_batched(
                    m=m, losses=losses, leaf_edges=leaf_edges,
                    manager=manager, batched=batched,
                    residual_params=residual_params,
                )
            else:
                A_by_name, residual_rows = _reverse_loop(
                    m=m, losses=losses, model=model, leaf_edges=leaf_edges,
                    manager=manager, modules=modules, batched=batched,
                    residual_params=residual_params,
                )
        finally:
            manager.phase.value = False
            for capture in manager.captures.values():
                capture.stream = None
                capture.collect = False
                capture.objective = None

        if driver == "squashed":
            missed = sorted(set(modules) - stream.visited)  # type: ignore[union-attr]
            if missed:
                raise RuntimeError(
                    f"never reached during the reverse pass: {missed}. They are hooked and were "
                    f"called on the forward, so the loss does not depend on them."
                )
            for capture in manager.captures.values():
                capture.inputs.clear()
                capture.rg_outputs.clear()
                capture.clear_grads()
            if accumulator.total is None:
                raise RuntimeError("no Gramian contributions were accumulated")
            return GramianResult(
                total=accumulator.total.detach(),
                per_module=accumulator.per_module,
                per_shared_group=accumulator.per_shared_group,
                peak_held_bytes=stream.peak_held_bytes,  # type: ignore[union-attr]
                driver=driver,
            )

        layers: dict[str, LayerCapture] = {}
        for name, capture in manager.captures.items():
            layers[name] = LayerCapture(
                name=name,
                module=capture.module,
                A=A_by_name[name].detach(),  # type: ignore[index]
                X=capture.take_input(),
            )
            capture.rg_outputs.clear()
            capture.clear_grads()

    held = 0
    for name, layer in layers.items():
        for tensor in (layer.A, layer.X):
            if torch.is_tensor(tensor):
                held += tensor.numel() * tensor.element_size()

    with torch.no_grad():
        for name, layer in layers.items():
            if name in grouped_names:
                continue
            handler = handler_overrides.get(name) or dispatch(layer.module)
            accumulator.add_module(name, handler(layer.module, layer.A, layer.X))

        for group, handler in shared_handlers.items():
            if not group <= set(layers):
                raise KeyError(f"shared_handlers group {sorted(group)} is not fully hooked")
            accumulator.add_group(group, handler({name: layers[name] for name in group}))

    if accumulator.total is None:
        raise RuntimeError("no Gramian contributions were accumulated")

    total = accumulator.total.detach()
    residual_block: Tensor | None = None
    if residual_rows is not None:
        # Same [m, m] block residual_gramian would have produced from its own
        # forward and its own m backwards -- harvested from the passes above.
        residual_block = gramian_from_rows(residual_rows)
        total = total + residual_block.to(dtype=total.dtype, device=total.device)

    return GramianResult(
        total=total,
        per_module=accumulator.per_module,
        per_shared_group=accumulator.per_shared_group,
        residual=residual_block,
        peak_held_bytes=held,
        driver=driver,
    )
