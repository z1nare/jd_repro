"""PLACEHOLDER -- model configurations for gates and benchmarks.

``gate_tiny`` (design doc step 0) -- small enough that brute-force per-objective
autograd is cheap, large enough to exercise every layer type::

    n_layer=2, n_head=2, n_embd=64, block_size=16, vocab_size=65,
    dropout=0.0, m=4 per-sequence mean-CE objectives, fp64

Two non-negotiables:
  * ``dropout = 0.0`` -- otherwise the brute-force pass and the hooked pass see
    different networks and the gate compares nothing;
  * gates run with ``bias=True`` **and** ``bias=False``, since the bias terms
    are separate summands in every identity.

fp64 for gates, dtype configurable for runs (A5000 fp64 throughput is
rate-limited, so real-shape runs are fp32).

``nanogpt_small`` and the nanochat/Qwen target shapes belong here too, for
``bench/`` to import.
"""

from __future__ import annotations


def gate_tiny(*args, **kwargs):
    raise NotImplementedError("gate config: step 0 of the execution plan")
