"""PLACEHOLDER -- B3: convergence overlay, Hadamard vs autogram-style vs scalar.

Same seed, same data order, same aggregator; plot the three loss curves.  They
should coincide within floating-point noise, because all three compute the same
update direction from the same ``G``.

The curves coinciding is the cheap part.  The *explanation* is the deliverable:
the engines differ in how ``G`` is computed, not in what is computed, so an
overlay that separates is a correctness bug rather than an interesting result.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("convergence overlay: after gate 5e")
