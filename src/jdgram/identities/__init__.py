"""Per-layer Gramian identities. One module per layer family.

Each module returns an ``[m, m]`` block in float64: the contribution one layer
family makes to ``G = J Jt``, computed in closed form from the upstream gradient
``A`` and the layer input ``X`` so the ``[m, P_layer]`` gradient block is never
formed. :mod:`~jdgram.identities.linear` carries both contraction orders;
:mod:`~jdgram.identities.propagation` holds the parameter-free rules, which
contribute no Gramian terms at all and exist so the reverse walk stays explicit
about that.

All of these are gated against brute-force ``autograd.grad`` in float64 at
``rtol=0, atol=1e-10`` before use -- see ``gates/``.
"""
