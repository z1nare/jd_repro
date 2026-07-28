"""Gradient-edge bookkeeping, so the reverse pass runs the nodes and nothing else.

Ported from TorchJD 0.17.0 ``torchjd/autogram/_edge_registry.py``, MIT licence,
(c) Valerian Rey, Pierre Quinton.

Why this exists: a plain ``loss.backward()`` traverses the whole graph and fills
``.grad`` on every parameter, which is pure waste when all the engine wants is
for the injected nodes to fire.  Targeting the nodes' child edges with
``torch.autograd.grad`` instead runs the same traversal without materialising
parameter gradients.

"Leaf edges" are the minimal subset that produces the same traversal: an edge
reachable from another registered edge is dropped, because reaching the outer
one already forces the traversal through it.
"""

from __future__ import annotations

from collections import deque

from torch.autograd.graph import GradientEdge


class EdgeRegistry:
    def __init__(self) -> None:
        self._edges: set[GradientEdge] = set()

    def reset(self) -> None:
        self._edges = set()

    def register(self, edge: GradientEdge) -> None:
        self._edges.add(edge)

    def __len__(self) -> int:
        return len(self._edges)

    def get_leaf_edges(self, roots: set[GradientEdge]) -> set[GradientEdge]:
        """Minimal subset of registered edges giving the same graph traversal.

        ``roots`` is modified in place, so callers should pass a throwaway set.
        """
        nodes_to_traverse = deque((child, root) for root in roots for child in _next_edges(root))
        result = {root for root in roots if root in self._edges}

        excluded = roots
        while nodes_to_traverse:
            node, origin = nodes_to_traverse.popleft()
            if node in self._edges:
                result.add(node)
                result.discard(origin)
                origin = node
            for child in _next_edges(node):
                if child not in excluded:
                    nodes_to_traverse.append((child, origin))
                    excluded.add(child)
        return result


def _next_edges(edge: GradientEdge) -> list[GradientEdge]:
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]
