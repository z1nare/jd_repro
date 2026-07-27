"""Per-layer Gramian identities.  One module per layer family.

Proven: :mod:`~jdgram.identities.conv`, and the rank-1 case of
:mod:`~jdgram.identities.linear` (old preflight gate 4).

Pending gates: the sequence case of :mod:`~jdgram.identities.linear` (II.1),
:mod:`~jdgram.identities.embedding` (II.2/II.3),
:mod:`~jdgram.identities.tied` (II.4), :mod:`~jdgram.identities.norm` (II.5).
:mod:`~jdgram.identities.propagation` holds the parameter-free rules (II.6),
which contribute no Gramian terms at all.
"""
